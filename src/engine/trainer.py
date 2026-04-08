from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.datasets.base import ContinualBenchmark, TaskDefinition
from src.datasets.registry import build_benchmark
from src.models.chu import ConsolidationHomeostasisUnit
from src.models.classifier import IncrementalCosineClassifier
from src.models.losses import (
    feature_retention,
    growth_penalty,
    kd_loss,
    rank_penalty,
    routing_balance_loss,
    slot_orthogonality,
)
from src.models.nh_lora import NHLoRAModel
from src.models.planner import (
    HorizonPlanner,
    MaterializedLayerPlan,
    PlannerControlOutputs,
    PlannerSignals,
    materialize_action,
)
from src.models.task_state import (
    HistoryBank,
    TaskState,
    TaskStateEncoder,
    build_history_entry,
    build_history_summary_vector,
    detach_task_state,
    pool_vector,
)
from src.utils.checkpoint import save_model_artifact
from src.utils.io import ensure_output_dirs

LOSS_COMPONENT_KEYS = ("total", "cls", "kd", "feat", "orth", "rank", "grow", "route")


class NHLoRATrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        logger,
        benchmark: ContinualBenchmark | None = None,
    ):
        self.config = config
        self.logger = logger
        self.benchmark = benchmark if benchmark is not None else build_benchmark(config)
        self.device = self._resolve_device(config["runtime"].get("device", "cpu"))
        self.output_dirs = ensure_output_dirs(config, benchmark_name=self.benchmark.name)

        self.model = NHLoRAModel(config).to(self.device)
        warmup_cfg = config["warmup"]
        planner_cfg = config["planner"]
        history_pool_dim = int(planner_cfg["history_pool_dim"])
        grad_dim = int(warmup_cfg["gradient_sketch_dim"])
        embedding_dim = int(warmup_cfg["task_embedding_dim"])
        history_dim = history_pool_dim * 2 + grad_dim + 3

        self.task_state_encoder = TaskStateEncoder(
            feature_dim=self.model.backbone.embed_dim,
            grad_dim=grad_dim,
            embedding_dim=embedding_dim,
            pool_dim=history_pool_dim,
        ).to(self.device)
        self.planner = HorizonPlanner(
            selected_blocks=self.model.selected_blocks,
            task_embedding_dim=embedding_dim,
            history_dim=history_dim,
            hidden_dim=int(planner_cfg["hidden_dim"]),
            layer_embedding_dim=int(planner_cfg["layer_embedding_dim"]),
            rank_min=int(planner_cfg["rank_min"]),
            rank_max=int(planner_cfg["rank_max"]),
            tau_novelty=float(planner_cfg["tau_novelty"]),
            tau_conflict=float(planner_cfg["tau_conflict"]),
        ).to(self.device)
        self._validate_planner_mode_config()
        self._configure_planner_trainability()
        self.chu = ConsolidationHomeostasisUnit(config["chu"])
        self.history_bank = HistoryBank()
        self.inference_profile = self.model.build_inference_profile()

        self.task_metrics: List[Dict[str, Any]] = []
        self.accuracy_matrix: List[List[float]] = []
        self.total_train_time = 0.0
        self.last_train_state: Dict[str, Any] = {
            "bootstrap_used": False,
            "teacher_used_on_task2": False,
            "planner_used_on_task2": False,
            "materialize_used_on_task2": False,
            "raw_planner_separated": False,
            "task2_materialized_has_candidates": False,
            "task2_history_attention_used": False,
            "router_seen": False,
            "warmup_imprinting_used": False,
            "classifier_imprinting_used": False,
            "classifier_sizes": [],
            "warmup_head_class_counts": [],
            "chu_calls_per_task": [],
            "history_sizes": [],
            "history_summary_dim": history_dim,
            "task_states": [],
            "task2_actions": {},
            "eval_shared_only_layers": [],
            "last_model_artifact_path": None,
        }
        self.training_state: Dict[str, Any] = {
            "seed": None,
            "current_task_id": 0,
            "global_step": 0,
        }
        self._planner_init_snapshot = self._snapshot_named_parameters(self.planner)
        self._planner_post_task1_snapshot = None
        self._planner_control_init_snapshot = self._snapshot_named_parameters(self.planner.control_branch)
        self._planner_control_post_task1_snapshot = None
        self._planner_action_history: Dict[int, List[Dict[str, Any]]] = {
            int(block_id): [] for block_id in self.model.selected_blocks
        }

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested_device)

    def _base_learning_rate(self) -> float:
        return float(self.config["training"]["lr"])

    def _shared_lr_scale(self) -> float:
        return float(self.config["nh_lora"].get("shared_lr_scale", 1.0))

    def _classifier_lr_scale(self) -> float:
        return float(self.config["training"].get("classifier_lr_scale", 1.0))

    def _planner_mode(self) -> str:
        return str(self.config["training"].get("planner_mode", "legacy")).lower()

    def _hybrid_planner_enabled(self) -> bool:
        return self._planner_mode() == "hybrid"

    def _planner_control_recompute_mode(self) -> str:
        return str(self.config["training"].get("planner_control_recompute", "per_batch")).lower()

    def _planner_policy_trainable(self) -> bool:
        return bool(self.config["training"].get("planner_policy_trainable", False))

    def _planner_control_trainable(self) -> bool:
        return bool(self.config["training"].get("planner_control_trainable", True))

    def _planner_use_learned_shared_gate(self) -> bool:
        return bool(self.config["training"].get("planner_use_learned_shared_gate", True))

    def _planner_soft_rank_training_enabled(self) -> bool:
        return bool(self.config["training"].get("planner_soft_rank_training", False))

    def _planner_hard_rank_eval(self) -> bool:
        return bool(self.config["training"].get("planner_hard_rank_eval", True))

    def _validate_planner_mode_config(self) -> None:
        planner_mode = self._planner_mode()
        if planner_mode not in {"legacy", "hybrid"}:
            raise ValueError(f"Unsupported planner_mode: {planner_mode}")
        if planner_mode != "hybrid":
            return
        if self._planner_policy_trainable():
            raise ValueError(
                "Stage 8 hybrid mode does not task-loss-train the planner policy branch. "
                "Set training.planner_policy_trainable=false."
            )
        if not self._planner_control_trainable():
            raise ValueError(
                "Stage 8 hybrid mode requires training.planner_control_trainable=true "
                "to keep the control branch optimizer-updated."
            )
        if not self._planner_use_learned_shared_gate():
            raise ValueError(
                "Stage 8 hybrid mode requires training.planner_use_learned_shared_gate=true."
            )
        if self._planner_soft_rank_training_enabled():
            raise ValueError(
                "Stage 8 Checkpoint A defers training.planner_soft_rank_training=true. "
                "Leave it false for this pass."
            )
        if not self._planner_hard_rank_eval():
            raise ValueError(
                "Stage 8 Checkpoint A requires training.planner_hard_rank_eval=true."
            )
        if self._planner_control_recompute_mode() != "per_batch":
            raise ValueError(
                "Stage 8 Checkpoint A currently supports training.planner_control_recompute=per_batch only."
            )

    def _configure_planner_trainability(self) -> None:
        if self._hybrid_planner_enabled():
            self.planner.freeze_policy_branch()
            if self._planner_control_trainable():
                self.planner.unfreeze_control_branch()
            else:
                self.planner.freeze_control_branch()
            return
        self.planner.unfreeze_policy_branch()
        self.planner.freeze_control_branch()

    def _planner_trainable_parameters(self) -> List[nn.Parameter]:
        if self._hybrid_planner_enabled():
            return [parameter for parameter in self.planner.control_parameters() if parameter.requires_grad]
        return [parameter for parameter in self.planner.parameters() if parameter.requires_grad]

    def _build_optimizer(self):
        shared_params = []
        classifier_params = []
        other_params = []
        classifier_lr_scale = self._classifier_lr_scale()
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".shared_a" in name or ".shared_b" in name:
                shared_params.append(parameter)
            elif name.startswith("classifier.") and classifier_lr_scale != 1.0:
                classifier_params.append(parameter)
            else:
                other_params.append(parameter)
        other_params.extend(self._planner_trainable_parameters())
        other_params.extend([parameter for parameter in self.task_state_encoder.parameters() if parameter.requires_grad])
        parameter_groups = []
        if other_params:
            parameter_groups.append({"params": other_params, "lr": self._base_learning_rate()})
        if shared_params:
            parameter_groups.append(
                {
                    "params": shared_params,
                    "lr": self._base_learning_rate() * self._shared_lr_scale(),
                }
            )
        if classifier_params:
            parameter_groups.append(
                {
                    "params": classifier_params,
                    "lr": self._base_learning_rate() * classifier_lr_scale,
                }
            )
        return AdamW(parameter_groups, lr=self._base_learning_rate(), weight_decay=float(self.config["training"]["weight_decay"]))

    def _build_scheduler(self, optimizer):
        training_cfg = self.config["training"]
        if not bool(training_cfg.get("use_scheduler", False)):
            return None
        scheduler_name = str(training_cfg.get("scheduler", "cosine")).lower()
        if scheduler_name != "cosine":
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        return CosineAnnealingLR(optimizer, T_max=max(int(training_cfg["epochs_per_task"]), 1))

    def _build_train_loader(self, dataset):
        runtime_cfg = self.config["runtime"]
        loader = DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=True,
            num_workers=int(runtime_cfg["num_workers"]),
            pin_memory=bool(runtime_cfg.get("pin_memory", False)),
            persistent_workers=bool(runtime_cfg.get("persistent_workers", False)) and int(runtime_cfg["num_workers"]) > 0,
        )
        return loader

    def _build_eval_loader(self, dataset):
        runtime_cfg = self.config["runtime"]
        return DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(runtime_cfg["num_workers"]),
            pin_memory=bool(runtime_cfg.get("pin_memory", False)),
            persistent_workers=bool(runtime_cfg.get("persistent_workers", False)) and int(runtime_cfg["num_workers"]) > 0,
        )

    def _build_warmup_loader(self, dataset):
        runtime_cfg = self.config["runtime"]
        return DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(runtime_cfg["num_workers"]),
            pin_memory=bool(runtime_cfg.get("pin_memory", False)),
            persistent_workers=bool(runtime_cfg.get("persistent_workers", False)) and int(runtime_cfg["num_workers"]) > 0,
        )

    def _shared_only_profile(self) -> Dict[int, Dict[str, object]]:
        profile = {}
        for block_id in self.model.selected_blocks:
            layer = self.model.layers[str(block_id)]
            profile[block_id] = {
                "action": "reuse_shared",
                "requested_action": "reuse_shared",
                "active_slot_candidates": [],
                "selected_slot": None,
                "rank_cfg": {slot_id: layer.slot_metadata[slot_id].rank for slot_id in layer.live_slot_ids()},
                "shared_gate": 1.0,
                "consolidate_flag": False,
                "deterministic": True,
                "created_new_slot": False,
                "fallback_action": None,
                "strong_retention": False,
                "shared_only": True,
                "compatibility_scores": {},
            }
        return profile

    def _sync_if_cuda(self) -> None:
        if bool(self.config["training"].get("cuda_sync_timing", False)) and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _perf_counter(self) -> float:
        self._sync_if_cuda()
        return time.perf_counter()

    def _format_seconds_human_readable(self, seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        seconds = max(float(seconds), 0.0)
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}h {minutes:02d}m {secs:02d}s"
        return f"{minutes:02d}m {secs:02d}s"

    def _format_per_task_acc(self, per_task_acc: List[float]) -> str:
        return "[" + ", ".join(f"{float(value):.4f}" for value in per_task_acc) + "]"

    def _log_seed_config(self, seed: int) -> None:
        benchmark_cfg = self.config.get("benchmark", {})
        model_cfg = self.config.get("model", {})
        training_cfg = self.config.get("training", {})
        loss_cfg = self.config.get("loss", {})
        planner_cfg = self.config.get("planner", {})
        nh_lora_cfg = self.config.get("nh_lora", {})
        chu_cfg = self.config.get("chu", {})
        self.logger.info("Seed %d config:", seed)
        self.logger.info("  benchmark=%s", self.benchmark.name)
        self.logger.info("  dataset=%s", benchmark_cfg.get("dataset_name", benchmark_cfg.get("name", self.benchmark.name)))
        self.logger.info("  seed=%d", seed)
        self.logger.info("  backbone=%s", model_cfg.get("backbone_name", "n/a"))
        self.logger.info("  selected_blocks=%s", model_cfg.get("selected_blocks", "n/a"))
        self.logger.info("  target_modules=%s", model_cfg.get("insertion_points", "n/a"))
        self.logger.info("  num_tasks=%s", benchmark_cfg.get("num_tasks", len(self.benchmark.tasks)))
        self.logger.info(
            "  epochs_per_task=%s batch_size=%s optimizer=%s lr=%s weight_decay=%s",
            training_cfg.get("epochs_per_task", "n/a"),
            training_cfg.get("batch_size", "n/a"),
            training_cfg.get("optimizer", "n/a"),
            training_cfg.get("lr", "n/a"),
            training_cfg.get("weight_decay", "n/a"),
        )
        self.logger.info(
            "  losses lam_kd=%s lam_feat=%s lam_orth=%s lam_rank=%s lam_grow=%s lam_route=%s retention_feature_representation=%s",
            loss_cfg.get("lambda_kd", "n/a"),
            loss_cfg.get("lambda_feat", "n/a"),
            loss_cfg.get("lambda_orth", "n/a"),
            loss_cfg.get("lambda_rank", "n/a"),
            loss_cfg.get("lambda_grow", "n/a"),
            loss_cfg.get("lambda_route", "n/a"),
            loss_cfg.get("retention_feature_representation", "cls"),
        )
        self.logger.info(
            "  planner tau_novelty=%s tau_conflict=%s tau_consolidate=%s rank_min=%s rank_max=%s",
            planner_cfg.get("tau_novelty", "n/a"),
            planner_cfg.get("tau_conflict", "n/a"),
            planner_cfg.get("tau_consolidate", "n/a"),
            planner_cfg.get("rank_min", "n/a"),
            planner_cfg.get("rank_max", "n/a"),
        )
        self.logger.info(
            "  nh_lora router_topk=%s router_candidate_pool=%s shared_rank=%s bootstrap_slot_rank=%s slot_r_max=%s max_slots_per_block=%s",
            nh_lora_cfg.get("router_topk", "n/a"),
            nh_lora_cfg.get("router_candidate_pool", "n/a"),
            nh_lora_cfg.get("shared_rank", "n/a"),
            nh_lora_cfg.get("bootstrap_slot_rank", "n/a"),
            nh_lora_cfg.get("slot_r_max", "n/a"),
            nh_lora_cfg.get("max_slots_per_block", "n/a"),
        )
        self.logger.info(
            "  chu merge_rate=%s freeze_on_keep=%s usage_high_threshold=%s usage_low_threshold=%s redundancy_threshold=%s stability_threshold=%s",
            chu_cfg.get("merge_rate", "n/a"),
            chu_cfg.get("freeze_on_keep", "n/a"),
            chu_cfg.get("usage_high_threshold", "n/a"),
            chu_cfg.get("usage_low_threshold", "n/a"),
            chu_cfg.get("redundancy_threshold", "n/a"),
            chu_cfg.get("stability_threshold", "n/a"),
        )
        self.logger.info(
            "  logging estimate_eta=%s cuda_sync_timing=%s debug_eval_around_consolidation=%s retention_debug_logging=%s retention_feature_diff_logging=%s routing_debug_logging=%s planner_audit_logging=%s adapter_delta_debug_logging=%s final_feature_diff_debug_logging=%s classifier_drift_debug_logging=%s grad_norm_debug_logging=%s logit_margin_debug_logging=%s freeze_old_classifier_weights=%s classifier_lr_scale=%s freeze_new_classifier_epochs=%s freeze_all_classifier_epochs=%s planner_mode=%s planner_control_recompute=%s planner_policy_trainable=%s planner_control_trainable=%s planner_use_learned_shared_gate=%s planner_soft_rank_training=%s planner_hard_rank_eval=%s",
            training_cfg.get("estimate_eta", True),
            training_cfg.get("cuda_sync_timing", False),
            training_cfg.get("debug_eval_around_consolidation", False),
            training_cfg.get("retention_debug_logging", True),
            training_cfg.get("retention_feature_diff_logging", False),
            training_cfg.get("routing_debug_logging", False),
            training_cfg.get("planner_audit_logging", False),
            training_cfg.get("adapter_delta_debug_logging", False),
            training_cfg.get("final_feature_diff_debug_logging", False),
            training_cfg.get("classifier_drift_debug_logging", False),
            training_cfg.get("grad_norm_debug_logging", False),
            training_cfg.get("logit_margin_debug_logging", False),
            training_cfg.get("freeze_old_classifier_weights", True),
            training_cfg.get("classifier_lr_scale", 1.0),
            training_cfg.get("freeze_new_classifier_epochs", 0),
            training_cfg.get("freeze_all_classifier_epochs", 0),
            training_cfg.get("planner_mode", "legacy"),
            training_cfg.get("planner_control_recompute", "per_batch"),
            training_cfg.get("planner_policy_trainable", False),
            training_cfg.get("planner_control_trainable", True),
            training_cfg.get("planner_use_learned_shared_gate", True),
            training_cfg.get("planner_soft_rank_training", False),
            training_cfg.get("planner_hard_rank_eval", True),
        )
        if str(loss_cfg.get("retention_feature_representation", "cls")).lower() == "full_tokens":
            self.logger.warning(
                "  retention_feature_representation=full_tokens can use substantially more memory and produce heavier debug logs than cls or mean_pool_tokens."
            )

    def _estimate_eta_for_task(self, epoch_durations: List[float], total_epochs: int, current_epoch: int) -> float | None:
        if not bool(self.config["training"].get("estimate_eta", True)) or not epoch_durations:
            return None
        remaining_epochs = max(total_epochs - current_epoch, 0)
        average_epoch = sum(epoch_durations) / len(epoch_durations)
        return remaining_epochs * average_epoch

    def _estimate_eta_for_seed(
        self,
        epoch_durations: List[float],
        total_epochs: int,
        current_epoch: int,
        task_index: int,
        total_tasks: int,
    ) -> float | None:
        if not bool(self.config["training"].get("estimate_eta", True)) or not epoch_durations:
            return None
        eta_task = self._estimate_eta_for_task(epoch_durations, total_epochs, current_epoch)
        if eta_task is None:
            return None
        current_task_projection = sum(epoch_durations)
        if current_epoch < total_epochs:
            average_epoch = sum(epoch_durations) / len(epoch_durations)
            current_task_projection += (total_epochs - current_epoch) * average_epoch
        completed_training_times = [float(metric["training_time"]) for metric in self.task_metrics]
        if completed_training_times:
            average_task_training_time = sum(completed_training_times) / len(completed_training_times)
        else:
            average_task_training_time = current_task_projection
        remaining_tasks = max(total_tasks - (task_index + 1), 0)
        return eta_task + (remaining_tasks * average_task_training_time)

    def _normalize_loss_dict(self, loss_values: Dict[str, Any] | None = None) -> Dict[str, float]:
        normalized = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
        if not loss_values:
            return normalized
        for key in LOSS_COMPONENT_KEYS:
            value = loss_values.get(key, 0.0)
            if isinstance(value, torch.Tensor):
                normalized[key] = float(value.detach().item())
            else:
                normalized[key] = float(value)
        return normalized

    def _summarize_epoch_stats(
        self,
        epoch: int,
        epochs_per_task: int,
        total_samples: int,
        correct_predictions: int,
        loss_sums: Dict[str, float],
        epoch_time: float,
        eta_task_seconds: float | None,
        eta_seed_seconds: float | None,
    ) -> Dict[str, float | int | None]:
        denominator = max(total_samples, 1)
        return {
            "epoch": int(epoch),
            "train_loss": loss_sums["total"] / denominator,
            "train_accuracy": correct_predictions / denominator,
            "loss_total": loss_sums["total"] / denominator,
            "loss_cls": loss_sums["cls"] / denominator,
            "loss_kd": loss_sums["kd"] / denominator,
            "loss_feat": loss_sums["feat"] / denominator,
            "loss_orth": loss_sums["orth"] / denominator,
            "loss_rank": loss_sums["rank"] / denominator,
            "loss_grow": loss_sums["grow"] / denominator,
            "loss_route": loss_sums["route"] / denominator,
            "epoch_time": float(epoch_time),
            "eta_task_seconds": None if eta_task_seconds is None else float(eta_task_seconds),
            "eta_seed_seconds": None if eta_seed_seconds is None else float(eta_seed_seconds),
        }

    def _current_last_task_accuracy(self, per_task_acc: List[float]) -> float:
        if not per_task_acc:
            return 0.0
        return float(per_task_acc[-1])

    def _compute_forgetting(self, current_row: List[float]) -> float:
        if len(current_row) <= 1 or not self.accuracy_matrix:
            return 0.0
        forgetting_values = []
        for task_idx in range(len(current_row) - 1):
            prior_scores = [row[task_idx] for row in self.accuracy_matrix if len(row) > task_idx]
            if not prior_scores:
                continue
            forgetting_values.append(max(prior_scores) - current_row[task_idx])
        if not forgetting_values:
            return 0.0
        return float(sum(forgetting_values) / len(forgetting_values))

    def _mask_old_classifier_gradients(self, context: Dict[str, Any]) -> None:
        training_cfg = self.config["training"]
        current_epoch = int(context.get("current_epoch", 0))
        freeze_all_epochs = int(training_cfg.get("freeze_all_classifier_epochs", 0))
        freeze_new_epochs = int(training_cfg.get("freeze_new_classifier_epochs", 0))
        old_num_classes = int(context.get("old_num_classes", 0))
        classifier = self.model.classifier
        if classifier.weight.grad is not None:
            if current_epoch > 0 and current_epoch <= max(freeze_all_epochs, 0):
                classifier.weight.grad.zero_()
            else:
                if bool(training_cfg.get("freeze_old_classifier_weights", True)) and old_num_classes > 0:
                    # Prevent old classifier prototypes from drifting under new-task CE updates.
                    classifier.weight.grad[:old_num_classes].zero_()
                if current_epoch > 0 and current_epoch <= max(freeze_new_epochs, 0):
                    classifier.weight.grad[old_num_classes:].zero_()
        bias = getattr(classifier, "bias", None)
        if bias is not None and getattr(bias, "grad", None) is not None:
            if current_epoch > 0 and current_epoch <= max(freeze_all_epochs, 0):
                bias.grad.zero_()
            else:
                if bool(training_cfg.get("freeze_old_classifier_weights", True)) and old_num_classes > 0:
                    bias.grad[:old_num_classes].zero_()
                if current_epoch > 0 and current_epoch <= max(freeze_new_epochs, 0):
                    bias.grad[old_num_classes:].zero_()

    def _active_retention_layers(self, context: Dict[str, Any]) -> List[int]:
        configured_layers = self.config["loss"].get("retention_layers")
        if configured_layers:
            retention_layers = {int(layer_id) for layer_id in configured_layers}
        else:
            retention_layers = {int(layer_id) for layer_id in self.model.selected_blocks}
        retention_layers.update(int(layer_id) for layer_id in context["strong_retention_layers"])
        return sorted(retention_layers)

    def _epoch_debug_enabled(self, context: Dict[str, Any], flag_key: str, max_epoch_key: str) -> bool:
        if int(context.get("task_number", 1)) <= 1:
            return False
        if not bool(self.config["training"].get(flag_key, False)):
            return False
        epoch = int(context.get("current_epoch", 0))
        max_epochs = int(self.config["training"].get(max_epoch_key, 3))
        return epoch > 0 and epoch <= max(max_epochs, 0)

    def _routing_audit_enabled(self) -> bool:
        return bool(self.config["training"].get("routing_debug_logging", False))

    def _planner_audit_enabled(self) -> bool:
        return bool(self.config["training"].get("planner_audit_logging", False))

    @staticmethod
    def _snapshot_named_parameters(module: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }

    @staticmethod
    def _parameter_distance_summary(
        current_snapshot: Dict[str, torch.Tensor],
        reference_snapshot: Dict[str, torch.Tensor] | None,
    ) -> Dict[str, float | int] | None:
        if reference_snapshot is None:
            return None
        squared_sum = 0.0
        max_abs = 0.0
        tracked_tensors = 0
        tracked_parameters = 0
        for name, reference in reference_snapshot.items():
            current = current_snapshot.get(name)
            if current is None:
                continue
            diff = current.to(reference.device) - reference
            tracked_tensors += 1
            tracked_parameters += int(diff.numel())
            squared_sum += float(diff.pow(2).sum().item())
            if diff.numel():
                max_abs = max(max_abs, float(diff.abs().max().item()))
        return {
            "l2": squared_sum ** 0.5,
            "max_abs": max_abs,
            "tracked_tensors": tracked_tensors,
            "tracked_parameters": tracked_parameters,
        }

    @staticmethod
    def _history_attention_stats(attention: torch.Tensor | None) -> Dict[str, float | int | bool]:
        if attention is None or attention.numel() == 0:
            return {
                "used": False,
                "count": 0,
                "entropy": 0.0,
                "max_weight": 0.0,
            }
        normalized = attention.detach().float()
        entropy = float(
            -(normalized * normalized.clamp_min(1e-8).log()).sum(dim=-1).mean().item()
        )
        return {
            "used": True,
            "count": int(normalized.shape[-1]),
            "entropy": entropy,
            "max_weight": float(normalized.max().item()),
        }

    @staticmethod
    def _planner_decision_label(
        *,
        novelty: float,
        conflict: float,
        tau_novelty: float,
        tau_conflict: float,
    ) -> str:
        if novelty < tau_novelty and conflict < tau_conflict:
            return "reuse_shared"
        if novelty >= tau_novelty and conflict < tau_conflict:
            return "expand_rank_existing_slot"
        if novelty >= tau_novelty and conflict >= tau_conflict:
            return "open_new_slot"
        return "freeze_old_strong_retention"

    def _planner_optimizer_membership(self, optimizer) -> Dict[str, object]:
        return self._optimizer_membership_for_parameters(optimizer, self.planner.parameters())

    def _optimizer_membership_for_parameters(self, optimizer, parameters) -> Dict[str, object]:
        tracked_parameters = list(parameters)
        planner_param_ids = {id(parameter) for parameter in tracked_parameters}
        total_parameters = len(tracked_parameters)
        requires_grad_parameters = sum(1 for parameter in tracked_parameters if parameter.requires_grad)
        matched_group_indices = []
        matched_group_lrs = []
        matched_parameter_count = 0
        for group_index, group in enumerate(optimizer.param_groups):
            matched_params = [parameter for parameter in group["params"] if id(parameter) in planner_param_ids]
            if not matched_params:
                continue
            matched_group_indices.append(int(group_index))
            matched_group_lrs.append(float(group.get("lr", self._base_learning_rate())))
            matched_parameter_count += len(matched_params)
        return {
            "in_optimizer": bool(matched_group_indices),
            "group_indices": matched_group_indices,
            "group_lrs": matched_group_lrs,
            "total_parameters": int(total_parameters),
            "requires_grad_parameters": int(requires_grad_parameters),
            "matched_parameters": int(matched_parameter_count),
        }

    def _planner_policy_optimizer_membership(self, optimizer) -> Dict[str, object]:
        return self._optimizer_membership_for_parameters(optimizer, self.planner.policy_parameters())

    def _planner_control_optimizer_membership(self, optimizer) -> Dict[str, object]:
        return self._optimizer_membership_for_parameters(optimizer, self.planner.control_parameters())

    @staticmethod
    def _int_list(values) -> List[int]:
        if values is None:
            return []
        return [int(value) for value in values]

    @staticmethod
    def _float_dict(values: Dict[int, float]) -> Dict[int, float]:
        return {int(key): float(value) for key, value in values.items()}

    @staticmethod
    def _candidate_slot_ids(config_like: Dict[str, object] | None) -> List[int]:
        if not config_like:
            return []
        if "active_slot_candidates" in config_like:
            return NHLoRATrainer._int_list(config_like.get("active_slot_candidates"))
        if "candidate_slots" in config_like:
            return NHLoRATrainer._int_list(config_like.get("candidate_slots"))
        return []

    @staticmethod
    def _selected_slot_id(config_like: Dict[str, object] | None) -> int | None:
        if not config_like:
            return None
        selected_slot = config_like.get("selected_slot")
        return None if selected_slot is None else int(selected_slot)

    @staticmethod
    def _fallback_reason(
        *,
        shared_only: bool,
        fallback_action: object,
        candidate_slot_ids: List[int],
        live_slot_ids: List[int],
    ) -> str:
        if fallback_action:
            return str(fallback_action)
        if shared_only:
            return "shared_only"
        if not live_slot_ids:
            return "no_live_slots"
        if not candidate_slot_ids:
            return "empty_candidates"
        return "none"

    def _slot_lifecycle_summary(
        self,
        block_id: int,
        config_like: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        layer = self.model.layers[str(block_id)]
        live_slot_ids = self._int_list(layer.live_slot_ids())
        retained_slot_ids = [
            slot_id
            for slot_id in live_slot_ids
            if layer.slot_metadata[slot_id].retained_for_inference
        ]
        frozen_slot_ids = [
            slot_id
            for slot_id in live_slot_ids
            if layer.slot_metadata[slot_id].frozen
        ]
        pruned_slot_ids = [
            int(slot_id)
            for slot_id, metadata in enumerate(layer.slot_metadata)
            if metadata.pruned
        ]
        candidate_slot_ids = self._candidate_slot_ids(config_like)
        selected_slot_id = self._selected_slot_id(config_like)
        shared_only = bool(config_like.get("shared_only", False)) if config_like else False
        fallback_action = config_like.get("fallback_action") if config_like else None
        return {
            "live_slot_ids": live_slot_ids,
            "retained_slot_ids": retained_slot_ids,
            "frozen_slot_ids": frozen_slot_ids,
            "pruned_slot_ids": pruned_slot_ids,
            "candidate_slot_ids": candidate_slot_ids,
            "selected_slot_id": selected_slot_id,
            "rank_by_slot": {slot_id: int(layer.slot_metadata[slot_id].rank) for slot_id in live_slot_ids},
            "usage_ema_by_slot": self._float_dict({slot_id: layer.slot_metadata[slot_id].usage_ema for slot_id in live_slot_ids}),
            "cumulative_usage_by_slot": self._float_dict(
                {slot_id: layer.slot_metadata[slot_id].cumulative_usage for slot_id in live_slot_ids}
            ),
            "shared_only": shared_only,
            "fallback_action": None if fallback_action is None else str(fallback_action),
            "fallback_reason": self._fallback_reason(
                shared_only=shared_only,
                fallback_action=fallback_action,
                candidate_slot_ids=candidate_slot_ids,
                live_slot_ids=live_slot_ids,
            ),
        }

    def _log_slot_lifecycle_summary(
        self,
        *,
        context: Dict[str, Any],
        label: str,
        block_id: int,
        config_like: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        summary = self._slot_lifecycle_summary(block_id, config_like=config_like)
        self.logger.info(
            "[SlotLifecycle][Task %d][%s][Layer %d] live=%s retained=%s candidates=%s selected=%s shared_only=%s fallback=%s frozen=%s pruned=%s ranks=%s usage_ema=%s cumulative_usage=%s",
            context["task_number"],
            label,
            int(block_id),
            summary["live_slot_ids"],
            summary["retained_slot_ids"],
            summary["candidate_slot_ids"],
            summary["selected_slot_id"],
            summary["shared_only"],
            summary["fallback_reason"],
            summary["frozen_slot_ids"],
            summary["pruned_slot_ids"],
            summary["rank_by_slot"],
            summary["usage_ema_by_slot"],
            summary["cumulative_usage_by_slot"],
        )
        return summary

    def _log_stage5_plan_debug(self, context: Dict[str, Any]) -> None:
        if not self._routing_audit_enabled():
            return
        raw_planner = context.get("raw_planner", {})
        materialized_plans = context.get("materialized_plans", {})
        applied_plans = context.get("applied_plans", {})
        context.setdefault("stage5_slot_lifecycle", {})
        for block_id in self.model.selected_blocks:
            signals = raw_planner.get(block_id)
            if signals is not None:
                action = self.planner.decide_action(signals)
                self.logger.info(
                    "[RawPlanner][Task %d][Layer %d] action=%s novelty=%.4f conflict=%.4f shared_gate=%.4f consolidate=%.4f rank_budget=%d",
                    context["task_number"],
                    int(block_id),
                    action,
                    float(signals.novelty.item()),
                    float(signals.conflict.item()),
                    float(signals.shared_gate.item()),
                    float(signals.consolidate.item()),
                    int(signals.rank_budget),
                )
            plan = materialized_plans.get(block_id)
            if plan is not None:
                plan_config = {
                    "candidate_slots": list(plan.candidate_slots),
                    "selected_slot": plan.selected_slot,
                    "shared_only": plan.shared_only,
                    "fallback_action": plan.fallback_action,
                }
                live_slot_ids = self._slot_lifecycle_summary(block_id)["live_slot_ids"]
                fallback_reason = self._fallback_reason(
                    shared_only=bool(plan.shared_only),
                    fallback_action=plan.fallback_action,
                    candidate_slot_ids=self._int_list(plan.candidate_slots),
                    live_slot_ids=live_slot_ids,
                )
                self.logger.info(
                    "[MaterializedPlan][Task %d][Layer %d] action=%s requested=%s shared_only=%s fallback=%s live=%s candidates=%s selected=%s create_new_slot=%s",
                    context["task_number"],
                    int(block_id),
                    plan.action,
                    plan.requested_action,
                    bool(plan.shared_only),
                    fallback_reason,
                    live_slot_ids,
                    self._int_list(plan.candidate_slots),
                    None if plan.selected_slot is None else int(plan.selected_slot),
                    bool(plan.create_new_slot),
                )
                context["stage5_slot_lifecycle"].setdefault("materialized", {})[int(block_id)] = self._log_slot_lifecycle_summary(
                    context=context,
                    label="Materialized",
                    block_id=int(block_id),
                    config_like=plan_config,
                )
            applied = applied_plans.get(block_id)
            if applied is not None:
                self.logger.info(
                    "[AppliedPlan][Task %d][Layer %d] action=%s requested=%s shared_only=%s deterministic=%s fallback=%s candidates=%s selected=%s created_new_slot=%s",
                    context["task_number"],
                    int(block_id),
                    applied.get("action"),
                    applied.get("requested_action"),
                    bool(applied.get("shared_only", False)),
                    bool(applied.get("deterministic", False)),
                    self._slot_lifecycle_summary(block_id, applied)["fallback_reason"],
                    self._candidate_slot_ids(applied),
                    self._selected_slot_id(applied),
                    bool(applied.get("created_new_slot", False)),
                )
                context["stage5_slot_lifecycle"].setdefault("applied", {})[int(block_id)] = self._log_slot_lifecycle_summary(
                    context=context,
                    label="Applied",
                    block_id=int(block_id),
                    config_like=applied,
                )

    def _planner_margin_record(
        self,
        *,
        block_id: int,
        task_number: int,
        signals: PlannerSignals,
    ) -> Dict[str, object]:
        planner_cfg = self.config["planner"]
        tau_novelty = float(planner_cfg["tau_novelty"])
        tau_conflict = float(planner_cfg["tau_conflict"])
        novelty = float(signals.novelty.item())
        conflict = float(signals.conflict.item())
        novelty_margin = novelty - tau_novelty
        conflict_margin = conflict - tau_conflict
        attention_stats = self._history_attention_stats(signals.history_attention)
        return {
            "task_number": int(task_number),
            "block_id": int(block_id),
            "action": self._planner_decision_label(
                novelty=novelty,
                conflict=conflict,
                tau_novelty=tau_novelty,
                tau_conflict=tau_conflict,
            ),
            "novelty": novelty,
            "conflict": conflict,
            "tau_novelty": tau_novelty,
            "tau_conflict": tau_conflict,
            "novelty_margin": novelty_margin,
            "conflict_margin": conflict_margin,
            "shared_gate": float(signals.shared_gate.item()),
            "consolidate": float(signals.consolidate.item()),
            "rank_budget": int(signals.rank_budget),
            "planner_input_norm": float(signals.planner_input.detach().norm().item())
            if signals.planner_input is not None
            else 0.0,
            "planner_representation_norm": float(signals.planner_representation.detach().norm().item())
            if signals.planner_representation is not None
            else 0.0,
            "history_attention_used": bool(attention_stats["used"]),
            "history_attention_count": int(attention_stats["count"]),
            "history_attention_entropy": float(attention_stats["entropy"]),
            "history_attention_max_weight": float(attention_stats["max_weight"]),
        }

    def _log_planner_layer_focus(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        records = context.get("planner_audit_records", {})
        if len(records) <= 1:
            return
        focus_layer = 6 if 6 in records else None
        if focus_layer is None:
            open_layers = [block_id for block_id, record in records.items() if record["action"] == "open_new_slot"]
            focus_layer = open_layers[0] if open_layers else None
        if focus_layer is None or focus_layer not in records:
            return
        others = [record for block_id, record in records.items() if int(block_id) != int(focus_layer)]
        if not others:
            return
        action_counts: Dict[str, int] = {}
        for record in others:
            action = str(record["action"])
            action_counts[action] = action_counts.get(action, 0) + 1
        other_novelty_mean = sum(float(record["novelty_margin"]) for record in others) / len(others)
        other_conflict_mean = sum(float(record["conflict_margin"]) for record in others) / len(others)
        focus_record = records[int(focus_layer)]
        self.logger.info(
            "[PlannerLayerCompare][Task %d] focus_layer=%d action=%s novelty_margin=%.4f conflict_margin=%.4f other_action_counts=%s other_mean_novelty_margin=%.4f other_mean_conflict_margin=%.4f",
            context["task_number"],
            int(focus_layer),
            focus_record["action"],
            float(focus_record["novelty_margin"]),
            float(focus_record["conflict_margin"]),
            action_counts,
            other_novelty_mean,
            other_conflict_mean,
        )

    def _log_planner_audit_prepare(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        task_state = context["task_state"]
        optimizer_summary = context.setdefault(
            "planner_optimizer_summary",
            self._planner_optimizer_membership(context["optimizer"]),
        )
        self.logger.info(
            "[PlannerInputAudit][Task %d] embedding_norm=%.4e similarity_mean=%.4e entropy_mean=%.4e gradient_sketch_norm=%.4e has_history=%s history_entries=%d",
            context["task_number"],
            float(task_state.embedding.norm().item()),
            float(task_state.similarity.mean().item()),
            float(task_state.entropy.mean().item()),
            float(task_state.gradient_sketch.norm().item()),
            bool(context["warmup_info"].get("has_history", False)),
            len(self.history_bank.entries),
        )
        self.logger.info(
            "[PlannerTrainPath][Task %d] in_optimizer=%s group_indices=%s group_lrs=%s requires_grad=%d/%d matched_parameters=%d",
            context["task_number"],
            bool(optimizer_summary["in_optimizer"]),
            optimizer_summary["group_indices"],
            optimizer_summary["group_lrs"],
            int(optimizer_summary["requires_grad_parameters"]),
            int(optimizer_summary["total_parameters"]),
            int(optimizer_summary["matched_parameters"]),
        )
        raw_planner = context.get("raw_planner", {})
        if not raw_planner:
            self.logger.info(
                "[PlannerAudit][Task %d] no raw planner signals available (bootstrap task or planner bypass).",
                context["task_number"],
            )
            return
        records = {}
        for block_id in self.model.selected_blocks:
            signals = raw_planner.get(block_id)
            if signals is None:
                continue
            record = self._planner_margin_record(
                block_id=int(block_id),
                task_number=int(context["task_number"]),
                signals=signals,
            )
            records[int(block_id)] = record
            self._planner_action_history.setdefault(int(block_id), []).append(
                {
                    "task_number": int(context["task_number"]),
                    "action": str(record["action"]),
                    "novelty_margin": float(record["novelty_margin"]),
                    "conflict_margin": float(record["conflict_margin"]),
                }
            )
            self.logger.info(
                "[PlannerAudit][Task %d][Layer %d] action=%s novelty=%.4f tau_novelty=%.4f novelty_margin=%.4f conflict=%.4f tau_conflict=%.4f conflict_margin=%.4f shared_gate=%.4f consolidate=%.4f rank_budget=%d planner_input_norm=%.4e planner_representation_norm=%.4e history_used=%s history_count=%d history_entropy=%.4f history_max_weight=%.4f",
                context["task_number"],
                int(block_id),
                record["action"],
                float(record["novelty"]),
                float(record["tau_novelty"]),
                float(record["novelty_margin"]),
                float(record["conflict"]),
                float(record["tau_conflict"]),
                float(record["conflict_margin"]),
                float(record["shared_gate"]),
                float(record["consolidate"]),
                int(record["rank_budget"]),
                float(record["planner_input_norm"]),
                float(record["planner_representation_norm"]),
                bool(record["history_attention_used"]),
                int(record["history_attention_count"]),
                float(record["history_attention_entropy"]),
                float(record["history_attention_max_weight"]),
            )
        context["planner_audit_records"] = records
        for block_id in sorted(records):
            history = self._planner_action_history.get(int(block_id), [])
            self.logger.info(
                "[PlannerTrajectory][Layer %d] tasks=%s actions=%s novelty_margins=%s conflict_margins=%s",
                int(block_id),
                [int(entry["task_number"]) for entry in history],
                [str(entry["action"]) for entry in history],
                [round(float(entry["novelty_margin"]), 4) for entry in history],
                [round(float(entry["conflict_margin"]), 4) for entry in history],
            )
        self._log_planner_layer_focus(context)

    def _log_hybrid_planner_prepare(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        optimizer = context["optimizer"]
        policy_summary = context.setdefault(
            "planner_policy_optimizer_summary",
            self._planner_policy_optimizer_membership(optimizer),
        )
        control_summary = context.setdefault(
            "planner_control_optimizer_summary",
            self._planner_control_optimizer_membership(optimizer),
        )
        self.logger.info(
            "[HybridPlannerConfig][Task %d] planner_mode=%s policy_trainable=%s control_trainable=%s control_recompute=%s learned_shared_gate=%s soft_rank_enabled=%s soft_rank_temperature=%s hard_rank_eval=%s",
            context["task_number"],
            self._planner_mode(),
            self._planner_policy_trainable(),
            self._planner_control_trainable(),
            self._planner_control_recompute_mode(),
            self._planner_use_learned_shared_gate(),
            self._planner_soft_rank_training_enabled(),
            float(self.config["training"].get("planner_soft_rank_temperature", 0.5)),
            self._planner_hard_rank_eval(),
        )
        self.logger.info(
            "[PlannerPolicyTrainPath][Task %d] task_loss_trainable=%s in_optimizer=%s group_indices=%s group_lrs=%s requires_grad=%d/%d matched_parameters=%d",
            context["task_number"],
            False,
            bool(policy_summary["in_optimizer"]),
            policy_summary["group_indices"],
            policy_summary["group_lrs"],
            int(policy_summary["requires_grad_parameters"]),
            int(policy_summary["total_parameters"]),
            int(policy_summary["matched_parameters"]),
        )
        self.logger.info(
            "[PlannerControlTrainPath][Task %d] task_loss_trainable=%s in_optimizer=%s group_indices=%s group_lrs=%s requires_grad=%d/%d matched_parameters=%d",
            context["task_number"],
            True,
            bool(control_summary["in_optimizer"]),
            control_summary["group_indices"],
            control_summary["group_lrs"],
            int(control_summary["requires_grad_parameters"]),
            int(control_summary["total_parameters"]),
            int(control_summary["matched_parameters"]),
        )

    @staticmethod
    def _compute_module_grad_stats(module: nn.Module) -> Dict[str, float | int]:
        squared_sum = 0.0
        max_abs = 0.0
        present_params = 0
        nonzero_params = 0
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            present_params += 1
            squared_sum += float(grad.pow(2).sum().item())
            if grad.numel():
                grad_max = float(grad.abs().max().item())
                max_abs = max(max_abs, grad_max)
                if grad_max > 0.0:
                    nonzero_params += 1
        return {
            "l2": squared_sum ** 0.5,
            "max_abs": max_abs,
            "present_params": present_params,
            "nonzero_params": nonzero_params,
        }

    def _compute_planner_grad_stats(self) -> Dict[str, float | int]:
        return self._compute_module_grad_stats(self.planner)

    def _accumulate_planner_audit_pre_step(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        accumulator = context.setdefault(
            "planner_audit_epoch_accumulator",
            {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
            },
        )
        stats = self._compute_planner_grad_stats()
        accumulator["batches"] += 1
        accumulator["grad_l2_sum"] += float(stats["l2"])
        accumulator["grad_l2_max"] = max(float(accumulator["grad_l2_max"]), float(stats["max_abs"]))
        accumulator["grad_present_batches"] += int(int(stats["present_params"]) > 0)
        accumulator["grad_nonzero_batches"] += int(int(stats["nonzero_params"]) > 0)

    def _record_planner_optimizer_step(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        accumulator = context.setdefault(
            "planner_audit_epoch_accumulator",
            {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
            },
        )
        accumulator["optimizer_steps"] += 1

    def _accumulate_planner_control_pre_step(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        accumulator = context.setdefault(
            "planner_control_epoch_accumulator",
            {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
                "shared_gate": {},
            },
        )
        stats = self._compute_module_grad_stats(self.planner.control_branch)
        accumulator["batches"] += 1
        accumulator["grad_l2_sum"] += float(stats["l2"])
        accumulator["grad_l2_max"] = max(float(accumulator["grad_l2_max"]), float(stats["max_abs"]))
        accumulator["grad_present_batches"] += int(int(stats["present_params"]) > 0)
        accumulator["grad_nonzero_batches"] += int(int(stats["nonzero_params"]) > 0)
        control_outputs = context.get("planner_control_last_outputs", {})
        for block_id, outputs in control_outputs.items():
            gate_value = float(outputs.shared_gate.detach().mean().item())
            gate_stats = accumulator["shared_gate"].setdefault(
                int(block_id),
                {"sum": 0.0, "min": gate_value, "max": gate_value, "count": 0},
            )
            gate_stats["sum"] += gate_value
            gate_stats["min"] = min(float(gate_stats["min"]), gate_value)
            gate_stats["max"] = max(float(gate_stats["max"]), gate_value)
            gate_stats["count"] += 1

    def _record_planner_control_optimizer_step(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        accumulator = context.setdefault(
            "planner_control_epoch_accumulator",
            {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
                "shared_gate": {},
            },
        )
        accumulator["optimizer_steps"] += 1

    def _log_planner_epoch_audit(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        accumulator = context.get("planner_audit_epoch_accumulator", {})
        batches = max(int(accumulator.get("batches", 0)), 1)
        optimizer_summary = context.get("planner_optimizer_summary") or self._planner_optimizer_membership(context["optimizer"])
        self.logger.info(
            "[PlannerTrainPath][Task %d][Epoch %d] in_optimizer=%s group_indices=%s group_lrs=%s requires_grad=%d/%d grad_present_batches=%d/%d grad_nonzero_batches=%d/%d optimizer_steps=%d pre_step_grad_l2_mean=%.4e pre_step_grad_l2_max=%.4e",
            context["task_number"],
            int(context["current_epoch"]),
            bool(optimizer_summary["in_optimizer"]),
            optimizer_summary["group_indices"],
            optimizer_summary["group_lrs"],
            int(optimizer_summary["requires_grad_parameters"]),
            int(optimizer_summary["total_parameters"]),
            int(accumulator.get("grad_present_batches", 0)),
            batches,
            int(accumulator.get("grad_nonzero_batches", 0)),
            batches,
            int(accumulator.get("optimizer_steps", 0)),
            float(accumulator.get("grad_l2_sum", 0.0)) / batches,
            float(accumulator.get("grad_l2_max", 0.0)),
        )

    def _log_planner_control_epoch(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        accumulator = context.get("planner_control_epoch_accumulator", {})
        batches = max(int(accumulator.get("batches", 0)), 1)
        optimizer_summary = context.get("planner_control_optimizer_summary") or self._planner_control_optimizer_membership(
            context["optimizer"]
        )
        self.logger.info(
            "[PlannerControlTrainPath][Task %d][Epoch %d] in_optimizer=%s group_indices=%s group_lrs=%s requires_grad=%d/%d grad_present_batches=%d/%d grad_nonzero_batches=%d/%d optimizer_steps=%d pre_step_grad_l2_mean=%.4e pre_step_grad_l2_max=%.4e",
            context["task_number"],
            int(context["current_epoch"]),
            bool(optimizer_summary["in_optimizer"]),
            optimizer_summary["group_indices"],
            optimizer_summary["group_lrs"],
            int(optimizer_summary["requires_grad_parameters"]),
            int(optimizer_summary["total_parameters"]),
            int(accumulator.get("grad_present_batches", 0)),
            batches,
            int(accumulator.get("grad_nonzero_batches", 0)),
            batches,
            int(accumulator.get("optimizer_steps", 0)),
            float(accumulator.get("grad_l2_sum", 0.0)) / batches,
            float(accumulator.get("grad_l2_max", 0.0)),
        )
        for block_id in sorted(accumulator.get("shared_gate", {})):
            stats = accumulator["shared_gate"][block_id]
            count = max(int(stats.get("count", 0)), 1)
            self.logger.info(
                "[PlannerControlValues][Task %d][Epoch %d][Layer %d] beta_mean=%.4f beta_min=%.4f beta_max=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(stats.get("sum", 0.0)) / count,
                float(stats.get("min", 0.0)),
                float(stats.get("max", 0.0)),
            )

    def _log_planner_parameter_drift(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        current_snapshot = self._snapshot_named_parameters(self.planner)
        from_init = self._parameter_distance_summary(current_snapshot, self._planner_init_snapshot)
        from_post_task1 = self._parameter_distance_summary(current_snapshot, self._planner_post_task1_snapshot)
        self.logger.info(
            "[PlannerParamDrift][Task %d] from_init_l2=%.4e from_init_max_abs=%.4e from_post_task1_l2=%s from_post_task1_max_abs=%s",
            context["task_number"],
            0.0 if from_init is None else float(from_init["l2"]),
            0.0 if from_init is None else float(from_init["max_abs"]),
            "n/a" if from_post_task1 is None else f"{float(from_post_task1['l2']):.4e}",
            "n/a" if from_post_task1 is None else f"{float(from_post_task1['max_abs']):.4e}",
        )
        if int(context["task_number"]) == 1 and self._planner_post_task1_snapshot is None:
            self._planner_post_task1_snapshot = current_snapshot
            self.logger.info("[PlannerParamDrift][Task 1] stored_post_task1_reference=True")

    def _log_planner_control_parameter_drift(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        current_snapshot = self._snapshot_named_parameters(self.planner.control_branch)
        from_init = self._parameter_distance_summary(current_snapshot, self._planner_control_init_snapshot)
        from_post_task1 = self._parameter_distance_summary(current_snapshot, self._planner_control_post_task1_snapshot)
        self.logger.info(
            "[PlannerControlParamDrift][Task %d] from_init_l2=%.4e from_init_max_abs=%.4e from_post_task1_l2=%s from_post_task1_max_abs=%s",
            context["task_number"],
            0.0 if from_init is None else float(from_init["l2"]),
            0.0 if from_init is None else float(from_init["max_abs"]),
            "n/a" if from_post_task1 is None else f"{float(from_post_task1['l2']):.4e}",
            "n/a" if from_post_task1 is None else f"{float(from_post_task1['max_abs']):.4e}",
        )
        if int(context["task_number"]) == 1 and self._planner_control_post_task1_snapshot is None:
            self._planner_control_post_task1_snapshot = current_snapshot
            self.logger.info("[PlannerControlParamDrift][Task 1] stored_post_task1_reference=True")

    def _log_stage5_lifecycle_for_profile(self, context: Dict[str, Any], label: str, profile: Dict[int, Dict[str, object]]) -> None:
        if not self._routing_audit_enabled():
            return
        lifecycle = context.setdefault("stage5_slot_lifecycle", {}).setdefault(label, {})
        for block_id in self.model.selected_blocks:
            layer_profile = profile.get(block_id, {})
            summary = self._log_slot_lifecycle_summary(
                context=context,
                label=label,
                block_id=int(block_id),
                config_like=layer_profile,
            )
            lifecycle[int(block_id)] = summary
            self.logger.info(
                "[InferenceProfile][Task %d][%s][Layer %d] active_candidates=%s selected=%s retained=%s retained_count=%d shared_only=%s deterministic=%s",
                context["task_number"],
                label,
                int(block_id),
                summary["candidate_slot_ids"],
                summary["selected_slot_id"],
                summary["retained_slot_ids"],
                len(summary["retained_slot_ids"]),
                bool(layer_profile.get("shared_only", False)),
                bool(layer_profile.get("deterministic", False)),
            )

    def _dropout_training_count(self) -> int:
        return sum(1 for module in self.model.modules() if isinstance(module, nn.Dropout) and module.training)

    @staticmethod
    def _route_comparison_flags(
        *,
        applied_candidates: List[int],
        train_candidates: List[int],
        eval_applied_candidates: List[int],
        profile_candidates: List[int],
        eval_profile_candidates: List[int],
    ) -> Dict[str, bool]:
        return {
            "applied_empty_profile_nonempty": bool(not applied_candidates and profile_candidates),
            "train_eval_applied_mismatch": bool(train_candidates != eval_applied_candidates),
            "eval_profile_mismatch": bool(profile_candidates != eval_profile_candidates),
        }

    def _route_probe_forward(
        self,
        *,
        images: torch.Tensor,
        planner_out: Dict[int, Dict[str, object]],
        task_state: TaskState | None,
        train_mode: bool,
    ) -> Dict[str, object]:
        previous_mode = self.model.training
        try:
            self.model.train(mode=train_mode)
            dropout_training_count = self._dropout_training_count()
            with torch.no_grad():
                grad_enabled = torch.is_grad_enabled()
                outputs = self.model.forward_with_state(images, task_state=task_state, planner_out=planner_out)
            return {
                "model_training": bool(self.model.training),
                "grad_enabled": bool(grad_enabled),
                "dropout_training_count": int(dropout_training_count),
                "route_info": outputs["route_info"],
            }
        finally:
            self.model.train(mode=previous_mode)

    def _log_stage5_train_eval_comparison(self, context: Dict[str, Any]) -> None:
        if not self._routing_audit_enabled() or int(context.get("task_number", 1)) <= 1:
            return
        probe_images = context.get("routing_debug_probe_images")
        if not isinstance(probe_images, torch.Tensor):
            self.logger.warning(
                "[RouteModeCompare][Task %d] skipped: no same-input train batch was captured.",
                context["task_number"],
            )
            return
        input_source = str(context.get("routing_debug_probe_source", "unknown"))
        images = probe_images.to(self.device)
        applied_plans = context["applied_plans"]
        inference_profile = self.inference_profile
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            train_probe = self._route_probe_forward(
                images=images,
                planner_out=applied_plans,
                task_state=context["task_state"],
                train_mode=True,
            )
            eval_applied_probe = self._route_probe_forward(
                images=images,
                planner_out=applied_plans,
                task_state=context["task_state"],
                train_mode=False,
            )
            eval_profile_probe = self._route_probe_forward(
                images=images,
                planner_out=inference_profile,
                task_state=None,
                train_mode=False,
            )
        finally:
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        self.logger.info(
            "[RouteModeCompare][Task %d] input_source=%s train_mode=%s train_grad_enabled=%s train_dropout_modules=%d eval_applied_mode=%s eval_applied_grad_enabled=%s eval_applied_dropout_modules=%d eval_profile_mode=%s eval_profile_grad_enabled=%s eval_profile_dropout_modules=%d",
            context["task_number"],
            input_source,
            train_probe["model_training"],
            train_probe["grad_enabled"],
            train_probe["dropout_training_count"],
            eval_applied_probe["model_training"],
            eval_applied_probe["grad_enabled"],
            eval_applied_probe["dropout_training_count"],
            eval_profile_probe["model_training"],
            eval_profile_probe["grad_enabled"],
            eval_profile_probe["dropout_training_count"],
        )
        train_route = train_probe["route_info"]
        eval_applied_route = eval_applied_probe["route_info"]
        eval_profile_route = eval_profile_probe["route_info"]
        for block_id in self.model.selected_blocks:
            train_layer = train_route.get(block_id, {})
            eval_applied_layer = eval_applied_route.get(block_id, {})
            eval_profile_layer = eval_profile_route.get(block_id, {})
            materialized_plan = context["materialized_plans"].get(block_id)
            materialized_candidates = self._int_list(materialized_plan.candidate_slots) if materialized_plan is not None else []
            applied_candidates = self._candidate_slot_ids(applied_plans.get(block_id, {}))
            profile_candidates = self._candidate_slot_ids(inference_profile.get(block_id, {}))
            train_candidates = self._candidate_slot_ids(train_layer)
            eval_applied_candidates = self._candidate_slot_ids(eval_applied_layer)
            eval_profile_candidates = self._candidate_slot_ids(eval_profile_layer)
            retained_slots = self._slot_lifecycle_summary(block_id, inference_profile.get(block_id, {}))["retained_slot_ids"]
            comparison_flags = self._route_comparison_flags(
                applied_candidates=applied_candidates,
                train_candidates=train_candidates,
                eval_applied_candidates=eval_applied_candidates,
                profile_candidates=profile_candidates,
                eval_profile_candidates=eval_profile_candidates,
            )
            self.logger.info(
                "[RouteModeCompare][Task %d][Layer %d] materialized_candidates=%s applied_candidates=%s train_candidates=%s eval_applied_candidates=%s profile_candidates=%s eval_profile_candidates=%s retained=%s selected_train=%s selected_eval_profile=%s shared_only_train=%s shared_only_profile=%s mismatch_applied_empty_profile_nonempty=%s mismatch_train_eval_applied=%s mismatch_eval_profile=%s",
                context["task_number"],
                int(block_id),
                materialized_candidates,
                applied_candidates,
                train_candidates,
                eval_applied_candidates,
                profile_candidates,
                eval_profile_candidates,
                retained_slots,
                self._int_list(train_layer.get("selected_slots", [])),
                self._int_list(eval_profile_layer.get("selected_slots", [])),
                bool(applied_plans.get(block_id, {}).get("shared_only", False)),
                bool(inference_profile.get(block_id, {}).get("shared_only", False)),
                comparison_flags["applied_empty_profile_nonempty"],
                comparison_flags["train_eval_applied_mismatch"],
                comparison_flags["eval_profile_mismatch"],
            )

    def _decorate_plans_for_debug(
        self,
        context: Dict[str, Any],
        plans: Dict[int, Dict[str, object]],
    ) -> Dict[int, Dict[str, object]]:
        if not self._epoch_debug_enabled(
            context,
            flag_key="adapter_delta_debug_logging",
            max_epoch_key="adapter_delta_debug_max_epochs",
        ):
            return plans
        accumulator = context.setdefault("adapter_delta_debug_accumulator", {})
        debug_plans = {}
        for block_id, planner_cfg in plans.items():
            layer_cfg = dict(planner_cfg)
            layer_cfg["_debug_delta_stats"] = accumulator
            layer_cfg["_debug_block_id"] = int(block_id)
            debug_plans[int(block_id)] = layer_cfg
        return debug_plans

    def _hybrid_plans_for_training_forward(self, context: Dict[str, Any]) -> Dict[int, Dict[str, object]]:
        control_context = context.get("planner_control_context")
        if not isinstance(control_context, dict):
            return context["applied_plans"]
        task_embedding = control_context.get("task_embedding")
        history_summary = control_context.get("history_summary")
        if not isinstance(task_embedding, torch.Tensor):
            return context["applied_plans"]
        control_outputs_by_block: Dict[int, PlannerControlOutputs] = {}
        hybrid_plans: Dict[int, Dict[str, object]] = {}
        for block_id, planner_cfg in context["applied_plans"].items():
            layer_cfg = dict(planner_cfg)
            control_outputs = self.planner.forward_control(
                int(block_id),
                task_embedding=task_embedding,
                history_summary=history_summary,
            )
            control_outputs_by_block[int(block_id)] = control_outputs
            if self._planner_use_learned_shared_gate():
                layer_cfg["shared_gate"] = control_outputs.shared_gate
            hybrid_plans[int(block_id)] = layer_cfg
        context["planner_control_last_outputs"] = control_outputs_by_block
        return hybrid_plans

    def _plans_for_training_forward(self, context: Dict[str, Any]) -> Dict[int, Dict[str, object]]:
        if self._hybrid_planner_enabled():
            return self._decorate_plans_for_debug(
                context,
                self._hybrid_plans_for_training_forward(context),
            )
        return self._decorate_plans_for_debug(context, context["applied_plans"])

    def _log_adapter_delta_debug(self, context: Dict[str, Any]) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="adapter_delta_debug_logging",
            max_epoch_key="adapter_delta_debug_max_epochs",
        ):
            return
        epoch = int(context["current_epoch"])
        if context.get("last_adapter_delta_debug_epoch") == epoch:
            return
        context["last_adapter_delta_debug_epoch"] = epoch
        accumulator = context.get("adapter_delta_debug_accumulator", {})
        context["last_adapter_delta_debug"] = accumulator
        if not accumulator:
            self.logger.warning(
                "[AdapterDelta][Task %d][Epoch %d] no adapter delta records were collected.",
                context["task_number"],
                epoch,
            )
            return
        for block_id in sorted(accumulator):
            for point_name in sorted(accumulator[block_id]):
                point_stats = accumulator[block_id][point_name]
                for kind in ("shared", "slot"):
                    stats = point_stats.get(kind)
                    if not stats:
                        continue
                    calls = max(int(stats.get("calls", 0)), 1)
                    self.logger.info(
                        "[AdapterDelta][Task %d][Epoch %d][Layer %d][%s][%s] calls=%d norm_mean=%.4e mean_abs=%.4e max_abs=%.4e",
                        context["task_number"],
                        epoch,
                        int(block_id),
                        point_name,
                        kind,
                        int(stats.get("calls", 0)),
                        float(stats.get("norm_sum", 0.0)) / calls,
                        float(stats.get("mean_abs_sum", 0.0)) / calls,
                        float(stats.get("max_abs", 0.0)),
                    )

    def _compute_grad_group_norms(self) -> Dict[str, float]:
        squared_sums = {
            "shared": 0.0,
            "slot": 0.0,
            "router": 0.0,
            "planner": 0.0,
            "classifier": 0.0,
        }
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue
            if ".shared_a" in name or ".shared_b" in name:
                group = "shared"
            elif ".slot_a" in name or ".slot_b" in name:
                group = "slot"
            elif ".query_proj" in name or ".slot_keys" in name:
                group = "router"
            elif name.startswith("classifier."):
                group = "classifier"
            else:
                continue
            squared_sums[group] += float(parameter.grad.detach().pow(2).sum().item())
        for parameter in self.planner.parameters():
            if parameter.grad is not None:
                squared_sums["planner"] += float(parameter.grad.detach().pow(2).sum().item())
        return {group: value ** 0.5 for group, value in squared_sums.items()}

    def _accumulate_grad_norm_debug(self, context: Dict[str, Any]) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="grad_norm_debug_logging",
            max_epoch_key="grad_norm_debug_max_epochs",
        ):
            return
        accumulator = context.setdefault(
            "grad_norm_debug_accumulator",
            {group: {"sum": 0.0, "max": 0.0, "batches": 0} for group in ("shared", "slot", "router", "planner", "classifier")},
        )
        for group, norm_value in self._compute_grad_group_norms().items():
            stats = accumulator[group]
            stats["sum"] += float(norm_value)
            stats["max"] = max(float(stats["max"]), float(norm_value))
            stats["batches"] += 1

    def _log_grad_norm_debug(self, context: Dict[str, Any]) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="grad_norm_debug_logging",
            max_epoch_key="grad_norm_debug_max_epochs",
        ):
            return
        epoch = int(context["current_epoch"])
        if context.get("last_grad_norm_debug_epoch") == epoch:
            return
        context["last_grad_norm_debug_epoch"] = epoch
        accumulator = context.get("grad_norm_debug_accumulator", {})
        context["last_grad_norm_debug"] = accumulator
        for group in ("shared", "slot", "router", "planner", "classifier"):
            stats = accumulator.get(group, {"sum": 0.0, "max": 0.0, "batches": 0})
            batches = max(int(stats.get("batches", 0)), 1)
            self.logger.info(
                "[GradNorm][Task %d][Epoch %d][%s] mean=%.4e max=%.4e batches=%d",
                context["task_number"],
                epoch,
                group,
                float(stats.get("sum", 0.0)) / batches,
                float(stats.get("max", 0.0)),
                int(stats.get("batches", 0)),
            )

    def _classifier_drift_stats(self, context: Dict[str, Any]) -> Dict[str, float] | None:
        snapshot = context.get("old_classifier_weight_snapshot")
        old_num_classes = int(context.get("old_num_classes", 0))
        if old_num_classes <= 0 or not isinstance(snapshot, torch.Tensor):
            return None
        current = self.model.classifier.weight[:old_num_classes].detach()
        snapshot = snapshot.to(device=current.device, dtype=current.dtype)
        diff = current - snapshot
        return {
            "mean_abs": float(diff.abs().mean().item()),
            "max_abs": float(diff.abs().max().item()),
            "norm": float(diff.norm().item()),
            "current_norm_mean": float(current.norm(dim=-1).mean().item()),
            "snapshot_norm_mean": float(snapshot.norm(dim=-1).mean().item()),
        }

    def _log_classifier_drift_debug(self, context: Dict[str, Any], final: bool = False) -> None:
        if not bool(self.config["training"].get("classifier_drift_debug_logging", False)):
            return
        epoch = int(context.get("current_epoch", 0))
        max_epochs = int(self.config["training"].get("classifier_drift_debug_max_epochs", 3))
        if not final and (int(context.get("task_number", 1)) <= 1 or epoch <= 0 or epoch > max(max_epochs, 0)):
            return
        label = "Final" if final else f"Epoch {epoch}"
        if not final and context.get("last_classifier_drift_debug_epoch") == epoch:
            return
        if not final:
            context["last_classifier_drift_debug_epoch"] = epoch
        stats = self._classifier_drift_stats(context)
        if stats is None:
            return
        context["last_classifier_drift_debug"] = stats
        self.logger.info(
            "[ClassifierDrift][Task %d][%s] old_classes=%d mean_abs=%.4e max_abs=%.4e norm=%.4e old_weight_norm_mean=%.4e snapshot_norm_mean=%.4e",
            context["task_number"],
            label,
            int(context.get("old_num_classes", 0)),
            stats["mean_abs"],
            stats["max_abs"],
            stats["norm"],
            stats["current_norm_mean"],
            stats["snapshot_norm_mean"],
        )

    def _log_final_feature_diff_debug(
        self,
        context: Dict[str, Any],
        outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
    ) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="final_feature_diff_debug_logging",
            max_epoch_key="final_feature_diff_debug_max_epochs",
        ):
            return
        epoch = int(context["current_epoch"])
        if context.get("last_final_feature_diff_debug_epoch") == epoch:
            return
        context["last_final_feature_diff_debug_epoch"] = epoch
        student_features = F.normalize(outputs["features"].detach(), dim=-1)
        teacher_features = F.normalize(teacher_outputs["features"].detach(), dim=-1)
        diff = (student_features - teacher_features).abs()
        cosine = (student_features * teacher_features).sum(dim=-1)
        stats = {
            "mean_abs": float(diff.mean().item()),
            "max_abs": float(diff.max().item()),
            "mean_cosine": float(cosine.mean().item()),
            "min_cosine": float(cosine.min().item()),
        }
        context["last_final_feature_diff_debug"] = stats
        self.logger.info(
            "[FinalFeatureDiff][Task %d][Epoch %d] mean_abs=%.4e max_abs=%.4e mean_cosine=%.4e min_cosine=%.4e",
            context["task_number"],
            epoch,
            stats["mean_abs"],
            stats["max_abs"],
            stats["mean_cosine"],
            stats["min_cosine"],
        )

    def _log_logit_margin_debug(self, context: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="logit_margin_debug_logging",
            max_epoch_key="logit_margin_debug_max_epochs",
        ):
            return
        epoch = int(context["current_epoch"])
        if context.get("last_logit_margin_debug_epoch") == epoch:
            return
        old_num_classes = int(context.get("old_num_classes", 0))
        logits = outputs["logits"].detach()
        if old_num_classes <= 0 or logits.size(-1) <= old_num_classes:
            return
        context["last_logit_margin_debug_epoch"] = epoch
        old_max = logits[:, :old_num_classes].max(dim=-1).values
        new_max = logits[:, old_num_classes:].max(dim=-1).values
        margin = new_max - old_max
        stats = {
            "old_max_mean": float(old_max.mean().item()),
            "new_max_mean": float(new_max.mean().item()),
            "new_minus_old_mean": float(margin.mean().item()),
            "new_wins_ratio": float((margin > 0).float().mean().item()),
        }
        context["last_logit_margin_debug"] = stats
        self.logger.info(
            "[LogitMargin][Task %d][Epoch %d] old_classes=%d new_classes=%d old_max_mean=%.4e new_max_mean=%.4e new_minus_old_mean=%.4e new_wins_ratio=%.4f",
            context["task_number"],
            epoch,
            old_num_classes,
            logits.size(-1) - old_num_classes,
            stats["old_max_mean"],
            stats["new_max_mean"],
            stats["new_minus_old_mean"],
            stats["new_wins_ratio"],
        )

    def _retention_feature_layer_match(
        self,
        current_features: Dict[int, torch.Tensor],
        teacher_features: Dict[int, torch.Tensor],
        retention_layers: List[int],
    ) -> Dict[str, List[int]]:
        student_keys = sorted(int(layer_id) for layer_id in current_features.keys())
        teacher_keys = sorted(int(layer_id) for layer_id in teacher_features.keys())
        matched = [
            int(layer_id)
            for layer_id in retention_layers
            if layer_id in current_features and layer_id in teacher_features
        ]
        missing_student = [int(layer_id) for layer_id in retention_layers if layer_id not in current_features]
        missing_teacher = [int(layer_id) for layer_id in retention_layers if layer_id not in teacher_features]
        return {
            "requested": [int(layer_id) for layer_id in retention_layers],
            "student_keys": student_keys,
            "teacher_keys": teacher_keys,
            "matched": matched,
            "missing_student": missing_student,
            "missing_teacher": missing_teacher,
        }

    def _log_feature_retention_diff(
        self,
        context: Dict[str, Any],
        current_features: Dict[int, torch.Tensor],
        teacher_features: Dict[int, torch.Tensor],
        retention_layers: List[int],
    ) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="retention_feature_diff_logging",
            max_epoch_key="retention_feature_diff_max_epochs",
        ):
            return
        epoch = int(context["current_epoch"])
        if context.get("last_feature_diff_epoch") == epoch:
            return
        context["last_feature_diff_epoch"] = epoch
        match_info = self._retention_feature_layer_match(current_features, teacher_features, retention_layers)
        context["last_feature_diff_debug"] = match_info
        self.logger.info(
            "[RetentionFeatures][Task %d][Epoch %d] requested_layers=%s student_keys=%s teacher_keys=%s matched_layers=%s missing_student=%s missing_teacher=%s",
            context["task_number"],
            epoch,
            match_info["requested"],
            match_info["student_keys"],
            match_info["teacher_keys"],
            match_info["matched"],
            match_info["missing_student"],
            match_info["missing_teacher"],
        )
        if not match_info["matched"]:
            self.logger.warning(
                "[RetentionFeatures][Task %d][Epoch %d] no matched retention layers; feature_retention will be zero unless configuration changes.",
                context["task_number"],
                epoch,
            )
            return
        for layer_id in match_info["matched"]:
            student_feature = current_features[layer_id].detach()
            teacher_feature = teacher_features[layer_id].detach()
            if student_feature.shape != teacher_feature.shape:
                self.logger.warning(
                    "[RetentionFeatures][Task %d][Epoch %d][Layer %d] shape mismatch student_shape=%s teacher_shape=%s; feature_retention may fail if this configuration is used.",
                    context["task_number"],
                    epoch,
                    layer_id,
                    tuple(student_feature.shape),
                    tuple(teacher_feature.shape),
                )
                continue
            diff = (student_feature - teacher_feature).abs()
            self.logger.info(
                "[RetentionFeatures][Task %d][Epoch %d][Layer %d] student_shape=%s teacher_shape=%s student_norm=%.4e teacher_norm=%.4e mean_abs_diff=%.4e max_abs_diff=%.4e",
                context["task_number"],
                epoch,
                layer_id,
                tuple(student_feature.shape),
                tuple(teacher_feature.shape),
                float(student_feature.norm().item()),
                float(teacher_feature.norm().item()),
                float(diff.mean().item()),
                float(diff.max().item()),
            )

    def _update_retention_debug(
        self,
        context: Dict[str, Any],
        outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        teacher_num_classes: int,
        retention_layers: List[int],
        kd_term: torch.Tensor,
        feat_term: torch.Tensor,
    ) -> None:
        if not bool(self.config["training"].get("retention_debug_logging", True)):
            return
        if teacher_num_classes <= 0:
            return
        teacher_old_logits = teacher_outputs["logits"].detach()
        student_old_logits = outputs["logits"][:, :teacher_num_classes].detach()
        old_logit_diff = (student_old_logits - teacher_old_logits).abs()
        context["last_retention_debug"] = {
            "teacher_num_classes": int(teacher_num_classes),
            "old_num_classes": int(context.get("old_num_classes", 0)),
            "retention_layers": list(retention_layers),
            "teacher_old_logits_norm": float(teacher_old_logits.norm(dim=-1).mean().item()),
            "student_old_logits_norm": float(student_old_logits.norm(dim=-1).mean().item()),
            "old_logits_mean_abs_diff": float(old_logit_diff.mean().item()),
            "old_logits_max_abs_diff": float(old_logit_diff.max().item()),
            "kd_raw": float(kd_term.detach().item()),
            "feat_raw": float(feat_term.detach().item()),
        }

    def _summarize_task_metrics(
        self,
        context: Dict[str, Any],
        eval_metrics: Dict[str, Any],
        chu_report: Dict[str, int],
        epoch_history: List[Dict[str, Any]],
        training_time: float,
        task_wall_time: float,
        inference_overhead: float,
    ) -> Dict[str, Any]:
        current_row = list(eval_metrics["per_task_acc"])
        parameter_growth = self._estimate_parameter_growth()
        previous_growth = int(self.task_metrics[-1]["parameter_growth"]) if self.task_metrics else 0
        mean_losses = self._normalize_loss_dict(
            {
                "total": sum(float(epoch["loss_total"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "cls": sum(float(epoch["loss_cls"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "kd": sum(float(epoch["loss_kd"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "feat": sum(float(epoch["loss_feat"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "orth": sum(float(epoch["loss_orth"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "rank": sum(float(epoch["loss_rank"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "grow": sum(float(epoch["loss_grow"]) for epoch in epoch_history) / max(len(epoch_history), 1),
                "route": sum(float(epoch["loss_route"]) for epoch in epoch_history) / max(len(epoch_history), 1),
            }
        )
        return {
            "task_id": int(context["task_number"]),
            "avg_acc": float(eval_metrics["avg_acc"]),
            "last_task_accuracy": self._current_last_task_accuracy(current_row),
            "forgetting": self._compute_forgetting(current_row),
            "per_task_acc": current_row,
            "per_task_accuracy": current_row,
            "accuracy_matrix_row": current_row,
            "opened_slots": int(chu_report["opened_slots"]),
            "pruned_slots": int(chu_report["pruned_slots"]),
            "merged_slots": int(chu_report["merged_slots"]),
            "frozen_slots": int(chu_report["frozen_slots"]),
            "kept_slots": int(chu_report["kept_slots"]),
            "parameter_growth": parameter_growth,
            "parameter_growth_delta": parameter_growth - previous_growth,
            "total_active_rank": self._total_active_rank(),
            # training_time tracks only the epoch update loop for the task.
            "training_time": float(training_time),
            # task_wall_time tracks the end-to-end task wall-clock duration.
            "task_wall_time": float(task_wall_time),
            "inference_overhead_ratio": float(inference_overhead),
            "mean_loss": mean_losses["total"],
            "mean_loss_cls": mean_losses["cls"],
            "mean_loss_kd": mean_losses["kd"],
            "mean_loss_feat": mean_losses["feat"],
            "mean_loss_orth": mean_losses["orth"],
            "mean_loss_rank": mean_losses["rank"],
            "mean_loss_grow": mean_losses["grow"],
            "mean_loss_route": mean_losses["route"],
            "epoch_history": epoch_history,
        }

    def _prepare_images_labels(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        _, images, labels = batch
        return images.to(self.device), labels.to(self.device)

    def _collect_class_prototypes(self, features: torch.Tensor, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        prototypes: Dict[int, torch.Tensor] = {}
        for class_id in torch.unique(labels).tolist():
            mask = labels == int(class_id)
            if mask.any():
                prototypes[int(class_id)] = F.normalize(features[mask].mean(dim=0), dim=-1)
        return prototypes

    def _extract_gradient_sketch(self, grad_dim: int) -> torch.Tensor:
        gradient_values = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad or parameter.grad is None:
                continue
            if name.startswith("backbone") or name.startswith("classifier"):
                continue
            gradient_values.append(parameter.grad.detach().norm().view(1, 1))
        if not gradient_values:
            return torch.zeros(1, grad_dim, device=self.device)
        stacked = torch.cat(gradient_values, dim=-1)
        return pool_vector(stacked, grad_dim)

    def _run_warmup_sensing(self, task_definition: TaskDefinition) -> Tuple[TaskState, Dict[str, Any]]:
        warmup_cfg = self.config["warmup"]
        num_batches = int(warmup_cfg["num_batches"])
        grad_dim = int(warmup_cfg["gradient_sketch_dim"])
        pool_dim = int(self.config["planner"]["history_pool_dim"])
        has_history = len(self.history_bank) > 0
        warmup_profile = deepcopy(self.inference_profile) if has_history else self._shared_only_profile()

        train_dataset, _ = self.benchmark.build_task_datasets(task_definition.task_id)
        warmup_loader = self._build_warmup_loader(train_dataset)
        images_buffer = []
        labels_buffer = []
        with torch.no_grad():
            for batch_index, batch in enumerate(warmup_loader):
                if batch_index >= num_batches:
                    break
                images, labels = self._prepare_images_labels(batch)
                images_buffer.append(images)
                labels_buffer.append(labels)
        if not images_buffer:
            raise RuntimeError("Warm-up sensing could not collect any batches.")

        warmup_images = torch.cat(images_buffer, dim=0)
        warmup_labels = torch.cat(labels_buffer, dim=0)

        with torch.no_grad():
            warmup_features = self.model.encode(warmup_images, task_state=None, planner_out=warmup_profile)["features"]
        class_prototypes = self._collect_class_prototypes(warmup_features.detach(), warmup_labels)

        unique_labels = sorted(int(label) for label in torch.unique(warmup_labels).tolist())
        label_to_local = {class_id: local_index for local_index, class_id in enumerate(unique_labels)}
        local_labels = torch.tensor([label_to_local[int(label.item())] for label in warmup_labels], device=self.device)
        aux_head = IncrementalCosineClassifier(feature_dim=self.model.backbone.embed_dim, tau=self.model.classifier.tau).to(self.device)
        aux_head.expand(len(unique_labels))
        aux_head.imprint_from_prototypes(
            {label_to_local[class_id]: prototype for class_id, prototype in class_prototypes.items()},
            range(len(unique_labels)),
        )
        self.last_train_state["warmup_imprinting_used"] = True
        self.last_train_state["warmup_head_class_counts"].append(len(unique_labels))

        self.model.zero_grad(set_to_none=True)
        features_with_grad = self.model.encode(warmup_images, task_state=None, planner_out=warmup_profile)["features"]
        warmup_logits = aux_head(features_with_grad)
        warmup_loss = F.cross_entropy(warmup_logits, local_labels)
        warmup_loss.backward()
        gradient_sketch = self._extract_gradient_sketch(grad_dim)

        probs = torch.softmax(warmup_logits.detach(), dim=-1)
        entropy = (-(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()).view(1, 1)
        feature_mean = warmup_features.detach().mean(dim=0, keepdim=True)
        feature_var = warmup_features.detach().var(dim=0, unbiased=False, keepdim=True)
        pooled_feature_mean = pool_vector(feature_mean, pool_dim)
        pooled_feature_var = pool_vector(feature_var, pool_dim)
        similarity_anchor = F.normalize(torch.cat([pooled_feature_mean, pooled_feature_var], dim=-1), dim=-1)
        provisional_summary = build_history_summary_vector(
            pooled_feature_mean=pooled_feature_mean,
            pooled_feature_var=pooled_feature_var,
            gradient_sketch=gradient_sketch.detach(),
            usage_summary=torch.zeros(1, 1, device=self.device),
            active_rank_summary=torch.zeros(1, 1, device=self.device),
            entropy_summary=entropy.detach(),
        )
        similarity = (
            self.history_bank.mean_similarity(provisional_summary, similarity_anchor)
            if has_history
            else torch.zeros(1, 1, device=self.device)
        )
        task_state = self.task_state_encoder(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch.detach(),
            similarity=similarity.detach(),
            entropy=entropy.detach(),
            similarity_anchor=similarity_anchor.detach(),
            summary_vector=provisional_summary.detach(),
            class_prototypes={class_id: prototype.detach() for class_id, prototype in class_prototypes.items()},
            warmup_logits=warmup_logits.detach(),
        )
        self.model.zero_grad(set_to_none=True)
        return task_state, {
            "has_history": has_history,
            "pooled_feature_mean": pooled_feature_mean.detach(),
            "pooled_feature_var": pooled_feature_var.detach(),
            "warmup_profile": warmup_profile,
            "class_ids_seen": unique_labels,
        }

    def _compute_raw_planner(
        self,
        task_state: TaskState,
        history_summary: torch.Tensor | None = None,
    ) -> Dict[int, PlannerSignals]:
        if history_summary is None:
            history_summary = self.history_bank.aggregate()
        raw_planner = {}
        for block_id in self.model.selected_blocks:
            raw_planner[block_id] = self.planner(
                block_id,
                task_embedding=task_state.embedding,
                history_summary=history_summary,
            )
        return raw_planner

    def _materialize_structure(
        self,
        task_state: TaskState,
        raw_planner: Dict[int, PlannerSignals],
        task_id: int,
    ) -> Dict[int, MaterializedLayerPlan]:
        plans = {}
        for block_id, signals in raw_planner.items():
            action = self.planner.decide_action(signals)
            plans[block_id] = materialize_action(
                action=action,
                signals=signals,
                slot_bank=self.model.layers[str(block_id)],
                task_embedding=task_state.embedding.detach(),
                task_id=task_id,
                max_slots_per_block=int(self.config["nh_lora"]["max_slots_per_block"]),
                tau_consolidate=float(self.config["planner"].get("tau_consolidate", 0.5)),
                router_candidate_pool=int(
                    self.config["nh_lora"].get(
                        "router_candidate_pool",
                        max(int(self.config["nh_lora"].get("router_topk", 1)), 3),
                    )
                ),
            )
        return plans

    def _expand_classifier_for_task(self, task_definition: TaskDefinition, task_state: TaskState) -> None:
        target_num_classes = max(self.benchmark.seen_classes_up_to(task_definition.task_id)) + 1
        if target_num_classes > self.model.classifier.num_classes:
            self.model.expand_classifier(target_num_classes - self.model.classifier.num_classes)
        init_mode = str(self.config["classifier"].get("init_mode", "imprint")).lower()
        if init_mode == "imprint":
            self.model.classifier.imprint_from_prototypes(task_state.class_prototypes, task_definition.class_ids)
            self.last_train_state["classifier_imprinting_used"] = True
        self.last_train_state["classifier_sizes"].append(self.model.classifier.num_classes)

    def _build_teacher_payload(self):
        teacher_model = self.model.make_teacher_snapshot()
        teacher_profile = teacher_model.build_inference_profile()
        return teacher_model, teacher_profile

    def _prepare_task_context(self, task_definition: TaskDefinition) -> Dict[str, Any]:
        task_index = task_definition.task_id
        task_number = task_index + 1
        task_state, warmup_info = self._run_warmup_sensing(task_definition)
        planner_history_summary = self.history_bank.aggregate()
        self.last_train_state["task_states"].append(
            {
                "task_id": task_number,
                "has_history": warmup_info["has_history"],
                "similarity_mean": float(task_state.similarity.mean().item()),
            }
        )

        if task_index == 0:
            teacher_model = None
            teacher_profile = None
            raw_planner: Dict[int, PlannerSignals] = {}
            materialized_plans = self.model.build_bootstrap_plan(task_state.embedding.detach())
            self.last_train_state["bootstrap_used"] = True
        else:
            teacher_model, teacher_profile = self._build_teacher_payload()
            raw_planner = self._compute_raw_planner(task_state, history_summary=planner_history_summary)
            materialized_plans = self._materialize_structure(task_state, raw_planner, task_id=task_number)
            self.last_train_state["planner_used_on_task2"] = task_number == 2
            self.last_train_state["materialize_used_on_task2"] = task_number == 2
            self.last_train_state["raw_planner_separated"] = bool(raw_planner) and bool(materialized_plans)
            if task_number == 2:
                self.last_train_state["teacher_used_on_task2"] = True
                self.last_train_state["task2_actions"] = {
                    block_id: plan.action for block_id, plan in materialized_plans.items()
                }
                self.last_train_state["task2_materialized_has_candidates"] = any(
                    bool(plan.candidate_slots) or bool(plan.create_new_slot) for plan in materialized_plans.values()
                )
                self.last_train_state["task2_history_attention_used"] = any(
                    signals.history_attention is not None for signals in raw_planner.values()
                )

        applied_plans = self.model.apply_structure_changes(
            materialized_plans,
            task_embedding=task_state.embedding.detach(),
            task_id=task_number,
        )
        strong_retention_layers = {
            int(block_id)
            for block_id, runtime_cfg in applied_plans.items()
            if bool(runtime_cfg.get("strong_retention", False))
        }
        self.model.capture_pre_task_snapshots()
        old_num_classes = self.model.classifier.num_classes
        old_classifier_weight_snapshot = (
            self.model.classifier.weight[:old_num_classes].detach().clone()
            if old_num_classes > 0
            else None
        )
        self._expand_classifier_for_task(task_definition, task_state)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        context = {
            "task_definition": task_definition,
            "task_number": task_number,
            "task_state": detach_task_state(task_state),
            "raw_planner": raw_planner,
            "materialized_plans": materialized_plans,
            "applied_plans": applied_plans,
            "teacher_model": teacher_model,
            "teacher_profile": teacher_profile,
            "strong_retention_layers": strong_retention_layers,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "warmup_info": warmup_info,
            "old_num_classes": old_num_classes,
            "old_classifier_weight_snapshot": old_classifier_weight_snapshot,
            "planner_control_context": {
                "task_embedding": task_state.embedding.detach(),
                "history_summary": None if planner_history_summary is None else planner_history_summary.detach(),
            },
        }
        if self._planner_audit_enabled():
            context["planner_optimizer_summary"] = self._planner_optimizer_membership(optimizer)
        if self._hybrid_planner_enabled():
            context["planner_policy_optimizer_summary"] = self._planner_policy_optimizer_membership(optimizer)
            context["planner_control_optimizer_summary"] = self._planner_control_optimizer_membership(optimizer)
        self._log_stage5_plan_debug(context)
        self._log_planner_audit_prepare(context)
        self._log_hybrid_planner_prepare(context)
        return context

    def _accumulate_usage(self, accumulator: Dict[int, Dict[int, float]], route_info: Dict[int, Dict[str, object]]) -> None:
        for block_id, layer_state in route_info.items():
            candidate_slots = layer_state.get("candidate_slots", [])
            distribution = layer_state.get("routing_distribution")
            if distribution is None or distribution.numel() == 0 or not candidate_slots:
                continue
            usage_vector = distribution.mean(dim=0)
            block_usage = accumulator.setdefault(int(block_id), {})
            for index, slot_id in enumerate(candidate_slots):
                block_usage[int(slot_id)] = block_usage.get(int(slot_id), 0.0) + float(usage_vector[index].item())

    def _finalize_usage(self, accumulator: Dict[int, Dict[int, float]], num_batches: int) -> Dict[int, Dict[int, float]]:
        finalized = {}
        for block_id, usage in accumulator.items():
            finalized[block_id] = {
                slot_id: value / max(num_batches, 1)
                for slot_id, value in usage.items()
            }
        return finalized

    def _update_routing_debug(self, context: Dict[str, Any], route_info: Dict[int, Dict[str, object]]) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="routing_debug_logging",
            max_epoch_key="routing_debug_max_epochs",
        ):
            return
        accumulator = context.setdefault("routing_debug_accumulator", {})
        for block_id, layer_state in route_info.items():
            candidate_slots = list(layer_state.get("candidate_slots", []))
            distribution = layer_state.get("routing_distribution")
            weights = layer_state.get("routing_weights")
            if isinstance(weights, torch.Tensor) and weights.ndim > 1:
                topk_size = int(weights.shape[-1])
            else:
                topk_size = 0
            entropy = 0.0
            mean_distribution: List[float] = []
            if isinstance(distribution, torch.Tensor) and distribution.numel() > 0:
                mean_tensor = distribution.detach().mean(dim=0)
                normalized = mean_tensor / mean_tensor.sum().clamp_min(1e-8)
                entropy = float(-(normalized * normalized.clamp_min(1e-8).log()).sum().item())
                mean_distribution = [float(value) for value in normalized.cpu().tolist()]
            entry = accumulator.setdefault(
                int(block_id),
                {
                    "batches": 0,
                    "candidate_count_sum": 0.0,
                    "shared_only_count": 0.0,
                    "topk_sum": 0.0,
                    "entropy_sum": 0.0,
                    "last_candidates": [],
                    "last_mean_distribution": [],
                },
            )
            entry["batches"] += 1
            entry["candidate_count_sum"] += float(len(candidate_slots))
            entry["shared_only_count"] += float(bool(layer_state.get("shared_only", False)))
            entry["topk_sum"] += float(topk_size)
            entry["entropy_sum"] += float(entropy)
            entry["last_candidates"] = [int(slot_id) for slot_id in candidate_slots]
            entry["last_mean_distribution"] = mean_distribution

    def _log_routing_debug(self, context: Dict[str, Any]) -> None:
        if not self._epoch_debug_enabled(
            context,
            flag_key="routing_debug_logging",
            max_epoch_key="routing_debug_max_epochs",
        ):
            return
        epoch = int(context["current_epoch"])
        accumulator = context.get("routing_debug_accumulator", {})
        for block_id in sorted(accumulator):
            entry = accumulator[block_id]
            batches = max(int(entry.get("batches", 0)), 1)
            distribution = "[" + ", ".join(f"{float(value):.4f}" for value in entry.get("last_mean_distribution", [])) + "]"
            self.logger.info(
                "[Routing][Task %d][Epoch %d][Layer %d] avg_candidate_count=%.2f shared_only_ratio=%.2f avg_topk=%.2f avg_entropy=%.4f last_candidates=%s mean_distribution=%s",
                context["task_number"],
                epoch,
                int(block_id),
                float(entry.get("candidate_count_sum", 0.0)) / batches,
                float(entry.get("shared_only_count", 0.0)) / batches,
                float(entry.get("topk_sum", 0.0)) / batches,
                float(entry.get("entropy_sum", 0.0)) / batches,
                entry.get("last_candidates", []),
                distribution,
            )

    def _compute_loss(
        self,
        context: Dict[str, Any],
        outputs: Dict[str, Any],
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_cfg = self.config["loss"]
        task_number = int(context["task_number"])
        cls_loss = F.cross_entropy(outputs["logits"], labels)
        orth_loss = slot_orthogonality(self.model)
        rank_loss = rank_penalty(self.model, self.device)
        route_loss = routing_balance_loss(outputs["route_info"], self.device)
        total_loss = cls_loss
        total_loss = total_loss + float(loss_cfg["lambda_orth"]) * orth_loss
        total_loss = total_loss + float(loss_cfg["lambda_rank"]) * rank_loss
        total_loss = total_loss + float(loss_cfg["lambda_route"]) * route_loss

        kd_term = torch.zeros((), device=self.device)
        feat_term = torch.zeros((), device=self.device)
        grow_term = torch.zeros((), device=self.device)

        teacher_model = context.get("teacher_model")
        if teacher_model is not None and task_number > 1:
            teacher_profile = context["teacher_profile"]
            with torch.no_grad():
                teacher_outputs = teacher_model.forward_with_state(
                    outputs["images"],
                    task_state=None,
                    planner_out=teacher_profile,
                )
            self._log_final_feature_diff_debug(context, outputs, teacher_outputs)
            self._log_logit_margin_debug(context, outputs)
            teacher_num_classes = teacher_model.classifier.num_classes
            if teacher_num_classes > 0:
                kd_term = kd_loss(
                    outputs["logits"][:, :teacher_num_classes],
                    teacher_outputs["logits"],
                    temperature=float(loss_cfg["kd_temperature"]),
                )
                total_loss = total_loss + float(loss_cfg["lambda_kd"]) * kd_term
            retention_layers = self._active_retention_layers(context)
            self._log_feature_retention_diff(
                context=context,
                current_features=outputs["layer_features"],
                teacher_features=teacher_outputs["layer_features"],
                retention_layers=retention_layers,
            )
            feat_term = feature_retention(
                outputs["layer_features"],
                teacher_outputs["layer_features"],
                layers=retention_layers,
                device=self.device,
            )
            total_loss = total_loss + float(loss_cfg["lambda_feat"]) * feat_term
            grow_term = growth_penalty(context["applied_plans"], self.device)
            total_loss = total_loss + float(loss_cfg["lambda_grow"]) * grow_term
            self._update_retention_debug(
                context=context,
                outputs=outputs,
                teacher_outputs=teacher_outputs,
                teacher_num_classes=teacher_num_classes,
                retention_layers=retention_layers,
                kd_term=kd_term,
                feat_term=feat_term,
            )

        return {
            "total": total_loss,
            "cls": cls_loss,
            "orth": orth_loss,
            "rank": rank_loss,
            "route": route_loss,
            "kd": kd_term,
            "feat": feat_term,
            "grow": grow_term,
        }

    def _train_single_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        task_definition = context["task_definition"]
        task_index = task_definition.task_id
        optimizer = context["optimizer"]
        scheduler = context["scheduler"]
        train_dataset, _ = self.benchmark.build_task_datasets(task_index)
        epochs_per_task = int(self.config["training"]["epochs_per_task"])
        grad_clip_norm = float(self.config["training"]["grad_clip_norm"])

        usage_accumulator: Dict[int, Dict[int, float]] = {}
        batch_count = 0
        epoch_history: List[Dict[str, Any]] = []
        epoch_durations: List[float] = []
        training_time_start = self._perf_counter()

        for epoch in range(1, epochs_per_task + 1):
            epoch_start = self._perf_counter()
            context["current_epoch"] = epoch
            context["routing_debug_accumulator"] = {}
            context["adapter_delta_debug_accumulator"] = {}
            context["grad_norm_debug_accumulator"] = {
                group: {"sum": 0.0, "max": 0.0, "batches": 0}
                for group in ("shared", "slot", "router", "planner", "classifier")
            }
            context["planner_audit_epoch_accumulator"] = {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
            }
            context["planner_control_epoch_accumulator"] = {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
                "shared_gate": {},
            }
            train_loader = self._build_train_loader(train_dataset)
            self.model.train()
            self.planner.train()
            self.task_state_encoder.train()
            epoch_loss_sums = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
            epoch_correct = 0
            epoch_total = 0
            for batch in train_loader:
                images, labels = self._prepare_images_labels(batch)
                if self._routing_audit_enabled() and "routing_debug_probe_images" not in context:
                    context["routing_debug_probe_images"] = images.detach().clone()
                    context["routing_debug_probe_source"] = (
                        f"task={context['task_number']} epoch={epoch} batch_index={batch_count}"
                    )
                optimizer.zero_grad(set_to_none=True)
                planner_out = self._plans_for_training_forward(context)
                outputs = self.model.forward_with_state(images, context["task_state"], planner_out)
                outputs["images"] = images
                losses = self._compute_loss(context, outputs, labels)
                losses["total"].backward()
                self._mask_old_classifier_gradients(context)
                self._accumulate_planner_audit_pre_step(context)
                self._accumulate_planner_control_pre_step(context)
                self._accumulate_grad_norm_debug(context)
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + self._planner_trainable_parameters() + list(self.task_state_encoder.parameters()),
                        grad_clip_norm,
                    )
                optimizer.step()
                self._record_planner_optimizer_step(context)
                self._record_planner_control_optimizer_step(context)
                self._accumulate_usage(usage_accumulator, outputs["route_info"])
                self._update_routing_debug(context, outputs["route_info"])
                if outputs["route_info"]:
                    self.last_train_state["router_seen"] = True

                predictions = outputs["logits"].argmax(dim=-1)
                batch_size = int(labels.size(0))
                normalized_losses = self._normalize_loss_dict(losses)
                for key in LOSS_COMPONENT_KEYS:
                    epoch_loss_sums[key] += normalized_losses[key] * batch_size
                epoch_correct += int((predictions == labels).sum().item())
                epoch_total += batch_size
                self.training_state["global_step"] += 1
                batch_count += 1

            if scheduler is not None:
                scheduler.step()

            epoch_time = self._perf_counter() - epoch_start
            epoch_durations.append(epoch_time)
            eta_task_seconds = self._estimate_eta_for_task(epoch_durations, epochs_per_task, epoch)
            eta_seed_seconds = self._estimate_eta_for_seed(
                epoch_durations,
                epochs_per_task,
                epoch,
                task_index=task_index,
                total_tasks=len(self.benchmark.tasks),
            )
            epoch_metrics = self._summarize_epoch_stats(
                epoch=epoch,
                epochs_per_task=epochs_per_task,
                total_samples=epoch_total,
                correct_predictions=epoch_correct,
                loss_sums=epoch_loss_sums,
                epoch_time=epoch_time,
                eta_task_seconds=eta_task_seconds,
                eta_seed_seconds=eta_seed_seconds,
            )
            epoch_history.append(epoch_metrics)
            if bool(self.config["training"].get("log_every_epoch", True)):
                self.logger.info(
                    "[Task %d][Epoch %d/%d] loss=%.4f acc=%.4f cls=%.4f kd=%.4e feat=%.4e orth=%.4f rank=%.4f grow=%.4f route=%.4f epoch_time=%.2fs eta_task=%s eta_seed=%s",
                    context["task_number"],
                    epoch,
                    epochs_per_task,
                    epoch_metrics["loss_total"],
                    epoch_metrics["train_accuracy"],
                    epoch_metrics["loss_cls"],
                    epoch_metrics["loss_kd"],
                    epoch_metrics["loss_feat"],
                    epoch_metrics["loss_orth"],
                    epoch_metrics["loss_rank"],
                    epoch_metrics["loss_grow"],
                    epoch_metrics["loss_route"],
                    epoch_metrics["epoch_time"],
                    self._format_seconds_human_readable(epoch_metrics["eta_task_seconds"]),
                    self._format_seconds_human_readable(epoch_metrics["eta_seed_seconds"]),
                )
                if bool(self.config["training"].get("retention_debug_logging", True)) and context.get("last_retention_debug"):
                    debug = context["last_retention_debug"]
                    self.logger.info(
                        "[Retention][Task %d][Epoch %d] teacher_classes=%d old_classes=%d layers=%s teacher_logit_norm=%.4e student_logit_norm=%.4e old_logit_mean_abs_diff=%.4e old_logit_max_abs_diff=%.4e kd_raw=%.4e feat_raw=%.4e",
                        context["task_number"],
                        epoch,
                        debug["teacher_num_classes"],
                        debug["old_num_classes"],
                        debug["retention_layers"],
                        debug["teacher_old_logits_norm"],
                        debug["student_old_logits_norm"],
                        debug["old_logits_mean_abs_diff"],
                        debug["old_logits_max_abs_diff"],
                        debug["kd_raw"],
                        debug["feat_raw"],
                    )
                self._log_classifier_drift_debug(context)
                self._log_grad_norm_debug(context)
                self._log_adapter_delta_debug(context)
                self._log_routing_debug(context)
                self._log_planner_epoch_audit(context)
                self._log_planner_control_epoch(context)

        training_time = self._perf_counter() - training_time_start
        self._log_classifier_drift_debug(context, final=True)
        usage_stats = self._finalize_usage(usage_accumulator, batch_count)
        for block_id, stats in usage_stats.items():
            self.model.layers[str(block_id)].update_usage_statistics(stats)
        if self._routing_audit_enabled():
            self._log_stage5_lifecycle_for_profile(
                context=context,
                label="PreCHUProfile",
                profile=self.model.build_inference_profile(),
            )

        debug_eval = bool(self.config["training"].get("debug_eval_around_consolidation", False))
        if debug_eval:
            pre_profile = self.model.build_inference_profile()
            pre_metrics = self._evaluate_up_to(task_index, planner_out=pre_profile)
            self.logger.info(
                "[Debug][Task %d] pre-consolidation avg_acc=%.4f per_task_acc=%s",
                context["task_number"],
                pre_metrics["avg_acc"],
                self._format_per_task_acc(pre_metrics["per_task_acc"]),
            )

        chu_report = self._run_consolidation(context, usage_stats)
        self.last_train_state["chu_calls_per_task"].append(1)
        self.inference_profile = self.model.build_inference_profile()
        self.last_train_state["eval_shared_only_layers"] = [
            block_id
            for block_id, layer_profile in self.inference_profile.items()
            if bool(layer_profile.get("shared_only", False))
        ]
        self._log_stage5_lifecycle_for_profile(
            context=context,
            label="PostCHUProfile",
            profile=self.inference_profile,
        )
        self._log_stage5_train_eval_comparison(context)
        self._log_planner_parameter_drift(context)
        self._log_planner_control_parameter_drift(context)
        self._append_history(context, usage_stats)
        self.last_train_state["history_sizes"].append(len(self.history_bank.entries))

        eval_metrics = self._evaluate_up_to(task_index)
        if debug_eval:
            self.logger.info(
                "[Debug][Task %d] post-consolidation avg_acc=%.4f per_task_acc=%s",
                context["task_number"],
                eval_metrics["avg_acc"],
                self._format_per_task_acc(eval_metrics["per_task_acc"]),
            )
        inference_overhead = self._estimate_inference_overhead(task_index)
        return self._summarize_task_metrics(
            context=context,
            eval_metrics=eval_metrics,
            chu_report=chu_report,
            epoch_history=epoch_history,
            training_time=training_time,
            task_wall_time=0.0,
            inference_overhead=inference_overhead,
        )

    def _run_consolidation(self, context: Dict[str, Any], usage_stats: Dict[int, Dict[int, float]]) -> Dict[str, int]:
        merged_slots = 0
        pruned_slots = 0
        kept_slots = 0
        frozen_slots = 0
        opened_slots = 0
        for block_id, runtime_cfg in context["applied_plans"].items():
            layer = self.model.layers[str(block_id)]
            if bool(runtime_cfg.get("created_new_slot", False)):
                opened_slots += 1
            report = self.chu.consolidate_layer(layer, runtime_cfg, usage_stats.get(block_id, {}))
            merged_slots += report.merged_slots
            pruned_slots += report.pruned_slots
            kept_slots += report.kept_slots
            frozen_slots += report.frozen_slots
        return {
            "merged_slots": merged_slots,
            "pruned_slots": pruned_slots,
            "kept_slots": kept_slots,
            "frozen_slots": frozen_slots,
            "opened_slots": opened_slots,
        }

    def _append_history(self, context: Dict[str, Any], usage_stats: Dict[int, Dict[int, float]]) -> None:
        task_state = context["task_state"]
        pooled_feature_mean = pool_vector(task_state.feature_mean, int(self.config["planner"]["history_pool_dim"]))
        pooled_feature_var = pool_vector(task_state.feature_var, int(self.config["planner"]["history_pool_dim"]))
        usage_values = []
        rank_values = []
        for block_id, layer in self.model.layers.items():
            live_slots = layer.live_slot_ids()
            if live_slots:
                for slot_id in live_slots:
                    usage_values.append(float(usage_stats.get(int(block_id), {}).get(slot_id, 0.0)))
                    rank_values.append(float(layer.slot_metadata[slot_id].rank) / float(layer.slot_r_max))
            usage_values.append(float(layer.last_shared_gate))
        usage_summary = torch.tensor([[sum(usage_values) / max(len(usage_values), 1)]], device=self.device)
        active_rank_summary = torch.tensor([[sum(rank_values) / max(len(rank_values), 1)]], device=self.device)
        entry = build_history_entry(
            task_id=context["task_number"],
            task_state=task_state,
            usage_summary=usage_summary,
            active_rank_summary=active_rank_summary,
            pooled_feature_mean=pooled_feature_mean,
            pooled_feature_var=pooled_feature_var,
        )
        self.history_bank.append(entry)

    def _evaluate_up_to(self, task_index: int, planner_out: Dict[int, Dict[str, object]] | None = None) -> Dict[str, Any]:
        self.model.eval()
        per_task_acc = []
        eval_profile = self.inference_profile if planner_out is None else planner_out
        with torch.no_grad():
            for eval_task_id in range(task_index + 1):
                _, test_dataset = self.benchmark.build_task_datasets(eval_task_id)
                loader = self._build_eval_loader(test_dataset)
                correct = 0
                total = 0
                for batch in loader:
                    images, labels = self._prepare_images_labels(batch)
                    outputs = self.model.forward_with_state(images, task_state=None, planner_out=eval_profile)
                    predictions = outputs["logits"].argmax(dim=-1)
                    correct += int((predictions == labels).sum().item())
                    total += int(labels.numel())
                per_task_acc.append(correct / max(total, 1))
        avg_acc = sum(per_task_acc) / max(len(per_task_acc), 1)
        return {"per_task_acc": per_task_acc, "avg_acc": avg_acc}

    def _estimate_parameter_growth(self) -> int:
        total = 0
        for layer in self.model.layers.values():
            total += len(layer.live_slot_ids()) * layer.router_dim
            for slot_id in layer.live_slot_ids():
                rank = layer.slot_metadata[slot_id].rank
                for bank in layer.point_banks.values():
                    total += rank * (bank.input_dim + bank.output_dim)
        return total

    def _total_active_rank(self) -> int:
        total = 0
        for layer in self.model.layers.values():
            total += sum(layer.slot_metadata[slot_id].rank for slot_id in layer.live_slot_ids())
        return total

    def _estimate_inference_overhead(self, task_index: int) -> float:
        _, test_dataset = self.benchmark.build_task_datasets(task_index)
        loader = self._build_eval_loader(test_dataset)
        try:
            batch = next(iter(loader))
        except StopIteration:
            return 1.0
        images, _ = self._prepare_images_labels(batch)
        self.model.eval()
        with torch.no_grad():
            base_start = time.perf_counter()
            self.model.backbone.forward_features(images)
            base_elapsed = max(time.perf_counter() - base_start, 1e-8)
            nh_start = time.perf_counter()
            self.model.forward_with_state(images, task_state=None, planner_out=self.inference_profile)
            nh_elapsed = max(time.perf_counter() - nh_start, 1e-8)
        return nh_elapsed / base_elapsed

    def _final_model_artifact_payload(self, seed: int, final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark.name,
            "seed": int(seed),
            "model_state": self.model.state_dict(),
            "model_structure_state": self.model.export_structure_state(),
            "inference_profile": deepcopy(self.inference_profile),
            "final_metrics": deepcopy(final_metrics),
        }

    def _save_final_model_artifact(self, seed: int, final_metrics: Dict[str, Any]) -> Path | None:
        if not bool(self.config["experiment"].get("save_checkpoints", False)):
            self.last_train_state["last_model_artifact_path"] = None
            return None
        output_path = Path(self.output_dirs["benchmark_checkpoints"]) / f"{self.benchmark.name}_seed{seed}_final.pt"
        payload = self._final_model_artifact_payload(seed=seed, final_metrics=final_metrics)
        save_model_artifact(payload, output_path)
        self.last_train_state["last_model_artifact_path"] = str(output_path)
        return output_path

    def train(self, seed: int) -> Dict[str, Any]:
        if self.training_state["seed"] is None:
            self.training_state["seed"] = int(seed)

        self._log_seed_config(seed)
        seed_wall_start = self._perf_counter()
        for task_index in range(int(self.training_state["current_task_id"]), len(self.benchmark.tasks)):
            task_wall_start = self._perf_counter()
            context = self._prepare_task_context(self.benchmark.tasks[task_index])
            metrics = self._train_single_task(context)
            metrics["task_wall_time"] = self._perf_counter() - task_wall_start
            self.task_metrics.append(metrics)
            self.accuracy_matrix.append(metrics["accuracy_matrix_row"])
            self.total_train_time += float(metrics["training_time"])
            self.training_state["current_task_id"] = task_index + 1
            self.logger.info(
                "Finished task %d | avg_acc=%.4f | last_task_acc=%.4f | forgetting=%.4f | per_task_acc=%s | active_rank=%d | opened=%d | pruned=%d | merged=%d | frozen=%d | kept=%d | param_growth=%d | param_growth_delta=%d | loss=%.4f | task_train_time=%.2fs | task_wall_time=%.2fs",
                context["task_number"],
                metrics["avg_acc"],
                metrics["last_task_accuracy"],
                metrics["forgetting"],
                self._format_per_task_acc(metrics["per_task_acc"]),
                metrics["total_active_rank"],
                metrics["opened_slots"],
                metrics["pruned_slots"],
                metrics["merged_slots"],
                metrics["frozen_slots"],
                metrics["kept_slots"],
                metrics["parameter_growth"],
                metrics["parameter_growth_delta"],
                metrics["mean_loss"],
                metrics["training_time"],
                metrics["task_wall_time"],
            )

        final_avg_acc = self.task_metrics[-1]["avg_acc"] if self.task_metrics else 0.0
        final_last_task_accuracy = self.task_metrics[-1]["last_task_accuracy"] if self.task_metrics else 0.0
        final_forgetting = self.task_metrics[-1]["forgetting"] if self.task_metrics else 0.0
        seed_wall_time_total = self._perf_counter() - seed_wall_start
        final_metrics = {
            "benchmark": self.benchmark.name,
            "final_avg_acc": final_avg_acc,
            "final_last_task_accuracy": final_last_task_accuracy,
            "final_forgetting": final_forgetting,
            "opened_slots": sum(metric["opened_slots"] for metric in self.task_metrics),
            "pruned_slots": sum(metric["pruned_slots"] for metric in self.task_metrics),
            "merged_slots": sum(metric["merged_slots"] for metric in self.task_metrics),
            "frozen_slots": sum(metric["frozen_slots"] for metric in self.task_metrics),
            "kept_slots": sum(metric["kept_slots"] for metric in self.task_metrics),
            "parameter_growth": self.task_metrics[-1]["parameter_growth"] if self.task_metrics else 0,
            "total_active_rank": self.task_metrics[-1]["total_active_rank"] if self.task_metrics else 0,
            "training_time_total": self.total_train_time,
            "seed_wall_time_total": seed_wall_time_total,
            "inference_overhead_ratio": self.task_metrics[-1]["inference_overhead_ratio"] if self.task_metrics else 1.0,
            "task_metrics": deepcopy(self.task_metrics),
            "accuracy_matrix": deepcopy(self.accuracy_matrix),
        }
        self._save_final_model_artifact(seed=seed, final_metrics=final_metrics)
        average_task_wall_time = (
            sum(float(metric["task_wall_time"]) for metric in self.task_metrics) / len(self.task_metrics)
            if self.task_metrics
            else 0.0
        )
        self.logger.info(
            "Finished seed %d | final_avg_acc=%.4f | final_last_task_acc=%.4f | final_forgetting=%.4f | active_rank=%d | param_growth=%d | training_time_total=%.2fs | seed_wall_time_total=%.2fs | avg_task_wall_time=%.2fs",
            seed,
            final_metrics["final_avg_acc"],
            final_metrics["final_last_task_accuracy"],
            final_metrics["final_forgetting"],
            final_metrics["total_active_rank"],
            final_metrics["parameter_growth"],
            final_metrics["training_time_total"],
            final_metrics["seed_wall_time_total"],
            average_task_wall_time,
        )
        return final_metrics
