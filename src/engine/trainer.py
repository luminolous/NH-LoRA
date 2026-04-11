from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
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
        self._planner_policy_record_history: Dict[int, List[Dict[str, Any]]] = {
            int(block_id): [] for block_id in self.model.selected_blocks
        }
        self._planner_growth_history: List[Dict[str, Any]] = []
        self._planner_layer_structure_history: Dict[int, List[Dict[str, Any]]] = {
            int(block_id): [] for block_id in self.model.selected_blocks
        }
        self._planner_realization_history: Dict[int, List[Dict[str, Any]]] = {
            int(block_id): [] for block_id in self.model.selected_blocks
        }
        self._retention_layer_audit_history: Dict[int, List[Dict[str, Any]]] = {
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

<<<<<<< HEAD
    def _optimizer_name(self) -> str:
        return str(self.config["training"].get("optimizer", "adamw")).strip().lower()
=======
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

    def _planner_control_delta_logit_scale(self) -> float:
        return float(self.config["training"].get("planner_control_delta_logit_scale", 2.0))

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
        if self._planner_control_delta_logit_scale() <= 0.0:
            raise ValueError("Hybrid planner requires training.planner_control_delta_logit_scale > 0.")
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
>>>>>>> 860378ac0afde0cbf4d45b63b6aca6d5315df287

    def _build_optimizer(self):
        training_cfg = self.config["training"]
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
<<<<<<< HEAD
        optimizer_name = self._optimizer_name()
        weight_decay = float(training_cfg["weight_decay"])
        if optimizer_name == "adamw":
            return AdamW(parameter_groups, lr=self._base_learning_rate(), weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return SGD(
                parameter_groups,
                lr=self._base_learning_rate(),
                momentum=float(training_cfg.get("sgd_momentum", 0.9)),
                nesterov=bool(training_cfg.get("sgd_nesterov", False)),
                weight_decay=weight_decay,
            )
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported optimizers: ['adamw', 'sgd']")
=======
        if classifier_params:
            parameter_groups.append(
                {
                    "params": classifier_params,
                    "lr": self._base_learning_rate() * classifier_lr_scale,
                }
            )
        return AdamW(parameter_groups, lr=self._base_learning_rate(), weight_decay=float(self.config["training"]["weight_decay"]))
>>>>>>> 860378ac0afde0cbf4d45b63b6aca6d5315df287

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
            self._optimizer_name(),
            training_cfg.get("lr", "n/a"),
            training_cfg.get("weight_decay", "n/a"),
        )
        if self._optimizer_name() == "sgd":
            self.logger.info(
                "  sgd_momentum=%s sgd_nesterov=%s",
                training_cfg.get("sgd_momentum", 0.9),
                training_cfg.get("sgd_nesterov", False),
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
            "  logging estimate_eta=%s cuda_sync_timing=%s debug_eval_around_consolidation=%s retention_debug_logging=%s retention_feature_diff_logging=%s routing_debug_logging=%s planner_audit_logging=%s adapter_delta_debug_logging=%s final_feature_diff_debug_logging=%s classifier_drift_debug_logging=%s grad_norm_debug_logging=%s logit_margin_debug_logging=%s freeze_old_classifier_weights=%s classifier_lr_scale=%s freeze_new_classifier_epochs=%s freeze_all_classifier_epochs=%s planner_mode=%s planner_control_recompute=%s planner_policy_trainable=%s planner_control_trainable=%s planner_use_learned_shared_gate=%s planner_control_delta_logit_scale=%s planner_soft_rank_training=%s planner_hard_rank_eval=%s",
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
            training_cfg.get("planner_control_delta_logit_scale", 2.0),
            training_cfg.get("planner_soft_rank_training", False),
            training_cfg.get("planner_hard_rank_eval", True),
        )
        if str(loss_cfg.get("retention_feature_representation", "cls")).lower() == "full_tokens":
            self.logger.warning(
                "  retention_feature_representation=full_tokens can use substantially more memory and produce heavier debug logs than cls or mean_pool_tokens."
            )

    @staticmethod
    def _benchmark_class_name_preview(class_names: List[str], limit: int = 5) -> List[str]:
        preview = [str(name) for name in class_names[:limit]]
        if len(class_names) > limit:
            preview.append(f"...(+{len(class_names) - limit} more)")
        return preview

    def _benchmark_sanity_summary(self) -> Dict[str, Any]:
        benchmark_metadata = self.benchmark.metadata if isinstance(self.benchmark.metadata, dict) else {}
        split_class_names = benchmark_metadata.get("split_class_names", {})
        if not isinstance(split_class_names, dict):
            split_class_names = {}
        summary: Dict[str, Any] = {
            "name": self.benchmark.name,
            "num_classes": int(self.benchmark.num_classes),
            "num_tasks": len(self.benchmark.tasks),
            "dataset_type": benchmark_metadata.get("dataset_type"),
            "class_name_count": benchmark_metadata.get("class_name_count"),
            "split_class_names": {
                str(split): [str(name) for name in class_names]
                for split, class_names in split_class_names.items()
                if isinstance(class_names, list)
            },
            "tasks": [],
        }
        for task in self.benchmark.tasks:
            task_metadata = task.metadata if isinstance(task.metadata, dict) else {}
            task_class_names = task_metadata.get("class_names", [])
            if not isinstance(task_class_names, list):
                task_class_names = []
            summary["tasks"].append(
                {
                    "task_number": int(task.task_id) + 1,
                    "class_count": int(task_metadata.get("class_count", len(task.class_ids))),
                    "train_sample_count": int(task_metadata.get("train_sample_count", len(task.train_records))),
                    "test_sample_count": int(task_metadata.get("test_sample_count", len(task.test_records))),
                    "class_ids": [int(class_id) for class_id in task.class_ids],
                    "class_names": [str(name) for name in task_class_names],
                }
            )
        return summary

    def _log_benchmark_sanity_summary(self) -> None:
        summary = self._benchmark_sanity_summary()
        self.logger.info(
            "[BenchmarkSummary] name=%s num_classes=%d num_tasks=%d",
            summary["name"],
            summary["num_classes"],
            summary["num_tasks"],
        )
        split_class_names = summary["split_class_names"]
        if split_class_names:
            train_class_names = split_class_names.get("train", [])
            test_class_names = split_class_names.get("test", [])
            self.logger.info(
                "[BenchmarkSummary][ImageFolder] class_name_count=%s train_class_count=%d test_class_count=%d split_class_sets_match=%s train_class_preview=%s test_class_preview=%s",
                summary.get("class_name_count", "n/a"),
                len(train_class_names),
                len(test_class_names),
                train_class_names == test_class_names,
                self._benchmark_class_name_preview(train_class_names),
                self._benchmark_class_name_preview(test_class_names),
            )
        for task_summary in summary["tasks"]:
            self.logger.info(
                "[BenchmarkTaskSummary][Task %d] class_count=%d train_samples=%d test_samples=%d class_ids=%s class_name_preview=%s",
                task_summary["task_number"],
                task_summary["class_count"],
                task_summary["train_sample_count"],
                task_summary["test_sample_count"],
                task_summary["class_ids"],
                self._benchmark_class_name_preview(task_summary["class_names"], limit=8),
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

    def _retention_audit_enabled(self) -> bool:
        return bool(self.config["training"].get("retention_debug_logging", True))

    def _loss_component_weights(self) -> Dict[str, float]:
        loss_cfg = self.config["loss"]
        return {
            "cls": 1.0,
            "kd": float(loss_cfg.get("lambda_kd", 0.0)),
            "feat": float(loss_cfg.get("lambda_feat", 0.0)),
            "orth": float(loss_cfg.get("lambda_orth", 0.0)),
            "rank": float(loss_cfg.get("lambda_rank", 0.0)),
            "grow": float(loss_cfg.get("lambda_grow", 0.0)),
            "route": float(loss_cfg.get("lambda_route", 0.0)),
        }

    def _weighted_loss_balance_summary(self, loss_values: Dict[str, Any] | None = None) -> Dict[str, Any]:
        normalized = self._normalize_loss_dict(loss_values)
        weights = self._loss_component_weights()
        weighted = {
            key: float(normalized.get(key, 0.0)) * float(weights.get(key, 0.0))
            for key in weights
        }
        weighted_total = float(sum(weighted.values()))
        weighted_shares = {
            key: (float(value) / weighted_total if weighted_total > 0.0 else 0.0)
            for key, value in weighted.items()
        }
        retention_weighted = float(weighted["kd"] + weighted["feat"])
        cls_weighted = float(weighted["cls"])
        return {
            "weighted_losses": weighted,
            "weighted_shares": weighted_shares,
            "weighted_total": weighted_total,
            "retention_weighted_total": retention_weighted,
            "retention_to_cls_ratio": retention_weighted / max(cls_weighted, 1e-8),
        }

    def _forgetting_decomposition(
        self,
        current_row: List[float],
        prior_rows: List[List[float]] | None = None,
    ) -> Dict[str, Any]:
        rows = self.accuracy_matrix if prior_rows is None else prior_rows
        entries: List[Dict[str, float | int]] = []
        for task_idx in range(max(len(current_row) - 1, 0)):
            prior_scores = [float(row[task_idx]) for row in rows if len(row) > task_idx]
            if not prior_scores:
                continue
            current_accuracy = float(current_row[task_idx])
            best_prior = float(max(prior_scores))
            latest_prior = float(prior_scores[-1])
            entries.append(
                {
                    "task_number": int(task_idx + 1),
                    "current_accuracy": current_accuracy,
                    "best_prior_accuracy": best_prior,
                    "latest_prior_accuracy": latest_prior,
                    "drop_from_best": best_prior - current_accuracy,
                    "drop_from_latest": latest_prior - current_accuracy,
                }
            )
        return {
            "tasks": entries,
            "drop_from_best_summary": self._scalar_series_summary(
                [float(entry["drop_from_best"]) for entry in entries]
            ),
            "drop_from_latest_summary": self._scalar_series_summary(
                [float(entry["drop_from_latest"]) for entry in entries]
            ),
        }

    @staticmethod
    def _feature_diff_summary(
        student_features: torch.Tensor | None,
        teacher_features: torch.Tensor | None,
    ) -> Dict[str, float]:
        if not isinstance(student_features, torch.Tensor) or not isinstance(teacher_features, torch.Tensor):
            return {
                "mean_abs_diff": 0.0,
                "max_abs_diff": 0.0,
                "mean_cosine": 0.0,
                "min_cosine": 0.0,
            }
        if student_features.shape != teacher_features.shape or student_features.numel() == 0:
            return {
                "mean_abs_diff": 0.0,
                "max_abs_diff": 0.0,
                "mean_cosine": 0.0,
                "min_cosine": 0.0,
            }
        student_flat = student_features.detach().float().reshape(student_features.shape[0], -1)
        teacher_flat = teacher_features.detach().float().reshape(teacher_features.shape[0], -1)
        diff = (student_flat - teacher_flat).abs()
        cosine = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
        return {
            "mean_abs_diff": float(diff.mean().item()),
            "max_abs_diff": float(diff.max().item()),
            "mean_cosine": float(cosine.mean().item()),
            "min_cosine": float(cosine.min().item()),
        }

    @staticmethod
    def _classifier_calibration_summary(
        old_logits: torch.Tensor,
        new_logits: torch.Tensor | None,
        old_classifier_weights: torch.Tensor | None = None,
        new_classifier_weights: torch.Tensor | None = None,
        teacher_old_logits: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        old_logits = old_logits.detach().float()
        if old_logits.ndim == 1:
            old_logits = old_logits.unsqueeze(0)
        old_logit_norm_mean = float(old_logits.norm(dim=-1).mean().item()) if old_logits.numel() > 0 else 0.0
        old_max = old_logits.max(dim=-1).values if old_logits.numel() > 0 else old_logits.new_zeros(old_logits.size(0))
        if isinstance(new_logits, torch.Tensor) and new_logits.numel() > 0:
            new_logits = new_logits.detach().float()
            if new_logits.ndim == 1:
                new_logits = new_logits.unsqueeze(0)
            new_logit_norm_mean = float(new_logits.norm(dim=-1).mean().item())
            new_max = new_logits.max(dim=-1).values
            old_minus_new_mean = float((old_max - new_max).mean().item())
            new_wins_ratio = float((new_max > old_max).float().mean().item())
        else:
            new_logit_norm_mean = 0.0
            old_minus_new_mean = float(old_max.mean().item()) if old_max.numel() > 0 else 0.0
            new_wins_ratio = 0.0
        old_weight_norm_mean = (
            float(old_classifier_weights.detach().float().norm(dim=-1).mean().item())
            if isinstance(old_classifier_weights, torch.Tensor) and old_classifier_weights.numel() > 0
            else 0.0
        )
        new_weight_norm_mean = (
            float(new_classifier_weights.detach().float().norm(dim=-1).mean().item())
            if isinstance(new_classifier_weights, torch.Tensor) and new_classifier_weights.numel() > 0
            else 0.0
        )
        teacher_old_logit_norm_mean = (
            float(teacher_old_logits.detach().float().norm(dim=-1).mean().item())
            if isinstance(teacher_old_logits, torch.Tensor) and teacher_old_logits.numel() > 0
            else 0.0
        )
        old_logit_compression_ratio = (
            old_logit_norm_mean / max(teacher_old_logit_norm_mean, 1e-8)
            if teacher_old_logit_norm_mean > 0.0
            else 0.0
        )
        return {
            "old_logit_norm_mean": old_logit_norm_mean,
            "new_logit_norm_mean": new_logit_norm_mean,
            "old_max_mean": float(old_max.mean().item()) if old_max.numel() > 0 else 0.0,
            "new_max_mean": float(old_max.new_tensor(0.0).item()) if not (isinstance(new_logits, torch.Tensor) and new_logits.numel() > 0) else float(new_logits.max(dim=-1).values.mean().item()),
            "old_minus_new_mean": old_minus_new_mean,
            "new_wins_ratio": new_wins_ratio,
            "old_weight_norm_mean": old_weight_norm_mean,
            "new_weight_norm_mean": new_weight_norm_mean,
            "teacher_old_logit_norm_mean": teacher_old_logit_norm_mean,
            "old_logit_compression_ratio": old_logit_compression_ratio,
        }

    def _route_profile_retention_summary(
        self,
        route_infos: List[Dict[int, Dict[str, object]]],
        selected_blocks: List[int] | None = None,
    ) -> Dict[str, Any]:
        blocks = [int(block_id) for block_id in (selected_blocks or self.model.selected_blocks)]
        per_layer = {
            int(block_id): {
                "batches": 0,
                "candidate_count_sum": 0.0,
                "shared_only_count": 0.0,
                "route_available_count": 0.0,
                "multi_slot_available_count": 0.0,
                "usage_top1_share_sum": 0.0,
            }
            for block_id in blocks
        }
        batch_count = 0
        shared_only_layer_count_sum = 0.0
        route_available_layer_count_sum = 0.0
        multi_slot_available_layer_count_sum = 0.0
        candidate_slot_count_sum = 0.0
        usage_top1_share_sum = 0.0
        usage_top1_share_count = 0
        for route_info in route_infos:
            batch_count += 1
            for block_id in blocks:
                layer_state = route_info.get(int(block_id), {}) if isinstance(route_info, dict) else {}
                candidate_slots = self._candidate_slot_ids(layer_state)
                candidate_count = len(candidate_slots)
                shared_only = bool(layer_state.get("shared_only", False))
                route_available = candidate_count > 0 and not shared_only
                multi_slot_available = candidate_count > 1 and not shared_only
                shared_only_layer_count_sum += float(shared_only)
                route_available_layer_count_sum += float(route_available)
                multi_slot_available_layer_count_sum += float(multi_slot_available)
                candidate_slot_count_sum += float(candidate_count)
                distribution = layer_state.get("routing_distribution")
                top1_share = 0.0
                if isinstance(distribution, torch.Tensor) and distribution.numel() > 0:
                    mean_distribution = distribution.detach().float().mean(dim=0)
                    normalized = mean_distribution / mean_distribution.sum().clamp_min(1e-8)
                    top1_share = float(normalized.max().item())
                    usage_top1_share_sum += top1_share
                    usage_top1_share_count += 1
                layer_summary = per_layer[int(block_id)]
                layer_summary["batches"] += 1
                layer_summary["candidate_count_sum"] += float(candidate_count)
                layer_summary["shared_only_count"] += float(shared_only)
                layer_summary["route_available_count"] += float(route_available)
                layer_summary["multi_slot_available_count"] += float(multi_slot_available)
                layer_summary["usage_top1_share_sum"] += float(top1_share)
        denominator = max(batch_count * max(len(blocks), 1), 1)
        finalized_per_layer: Dict[int, Dict[str, float]] = {}
        for block_id, layer_summary in per_layer.items():
            layer_batches = max(int(layer_summary["batches"]), 1)
            finalized_per_layer[int(block_id)] = {
                "candidate_count_mean": float(layer_summary["candidate_count_sum"]) / layer_batches,
                "shared_only_frequency": float(layer_summary["shared_only_count"]) / layer_batches,
                "route_available_frequency": float(layer_summary["route_available_count"]) / layer_batches,
                "multi_slot_available_frequency": float(layer_summary["multi_slot_available_count"]) / layer_batches,
                "usage_top1_share_mean": float(layer_summary["usage_top1_share_sum"]) / layer_batches,
            }
        return {
            "batch_count": int(batch_count),
            "shared_only_layer_count_mean": shared_only_layer_count_sum / max(batch_count, 1),
            "route_available_layer_count_mean": route_available_layer_count_sum / max(batch_count, 1),
            "multi_slot_available_layer_count_mean": multi_slot_available_layer_count_sum / max(batch_count, 1),
            "candidate_slot_count_mean": candidate_slot_count_sum / denominator,
            "usage_top1_share_mean": usage_top1_share_sum / max(usage_top1_share_count, 1),
            "shared_only_layer_frequency": shared_only_layer_count_sum / denominator,
            "route_available_layer_frequency": route_available_layer_count_sum / denominator,
            "multi_slot_available_layer_frequency": multi_slot_available_layer_count_sum / denominator,
            "per_layer": finalized_per_layer,
        }

    def _retention_pre_post_diff(
        self,
        pre_eval: Dict[str, Any] | None,
        post_eval: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        pre_tasks = {
            int(task_summary["task_number"]): task_summary
            for task_summary in (pre_eval or {}).get("retention_audit", {}).get("task_summaries", [])
        }
        post_tasks = {
            int(task_summary["task_number"]): task_summary
            for task_summary in (post_eval or {}).get("retention_audit", {}).get("task_summaries", [])
        }
        deltas: List[Dict[str, float | int]] = []
        for task_number in sorted(set(pre_tasks).intersection(post_tasks)):
            pre_task = pre_tasks[task_number]
            post_task = post_tasks[task_number]
            pre_route = pre_task.get("route_summary", {})
            post_route = post_task.get("route_summary", {})
            deltas.append(
                {
                    "task_number": int(task_number),
                    "accuracy_delta": float(post_task.get("accuracy", 0.0) - pre_task.get("accuracy", 0.0)),
                    "old_logit_mean_abs_diff_delta": float(
                        post_task.get("old_logit_mean_abs_diff", 0.0) - pre_task.get("old_logit_mean_abs_diff", 0.0)
                    ),
                    "feature_mean_abs_diff_delta": float(
                        post_task.get("feature_mean_abs_diff", 0.0) - pre_task.get("feature_mean_abs_diff", 0.0)
                    ),
                    "shared_only_layer_count_mean_delta": float(
                        post_route.get("shared_only_layer_count_mean", 0.0)
                        - pre_route.get("shared_only_layer_count_mean", 0.0)
                    ),
                    "multi_slot_available_layer_count_mean_delta": float(
                        post_route.get("multi_slot_available_layer_count_mean", 0.0)
                        - pre_route.get("multi_slot_available_layer_count_mean", 0.0)
                    ),
                }
            )
        return {
            "tasks": deltas,
            "accuracy_delta_summary": self._scalar_series_summary(
                [float(entry["accuracy_delta"]) for entry in deltas]
            ),
            "old_logit_delta_summary": self._scalar_series_summary(
                [float(entry["old_logit_mean_abs_diff_delta"]) for entry in deltas]
            ),
            "feature_delta_summary": self._scalar_series_summary(
                [float(entry["feature_mean_abs_diff_delta"]) for entry in deltas]
            ),
        }

    def _profile_layer_count_summary(self, profile: Dict[int, Dict[str, object]] | None) -> Dict[str, int]:
        if not isinstance(profile, dict):
            return {"shared_only_layers": 0, "multi_slot_layers": 0}
        shared_only_layers = 0
        multi_slot_layers = 0
        for block_id in self.model.selected_blocks:
            layer_profile = profile.get(int(block_id), {})
            if bool(layer_profile.get("shared_only", False)):
                shared_only_layers += 1
            if len(self._candidate_slot_ids(layer_profile)) > 1:
                multi_slot_layers += 1
        return {
            "shared_only_layers": int(shared_only_layers),
            "multi_slot_layers": int(multi_slot_layers),
        }

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

    def _planner_control_saturation_audit_enabled(self) -> bool:
        return self._planner_audit_enabled() and self._hybrid_planner_enabled()

    @staticmethod
    def _snapshot_named_parameters(module: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }

    @staticmethod
    def _scalar_series_summary(values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "first": 0.0,
                "last": 0.0,
                "count": 0.0,
            }
        samples = torch.tensor(values, dtype=torch.float32)
        quantiles = torch.quantile(samples, torch.tensor([0.5, 0.9, 0.99], dtype=samples.dtype))
        return {
            "mean": float(samples.mean().item()),
            "min": float(samples.min().item()),
            "max": float(samples.max().item()),
            "p50": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
            "first": float(samples[0].item()),
            "last": float(samples[-1].item()),
            "count": float(samples.numel()),
        }

    @staticmethod
    def _fraction_above(values: List[float], threshold: float) -> float:
        if not values:
            return 0.0
        return float(sum(1 for value in values if float(value) > threshold) / len(values))

    @staticmethod
    def _fraction_below(values: List[float], threshold: float) -> float:
        if not values:
            return 0.0
        return float(sum(1 for value in values if float(value) < threshold) / len(values))

    @staticmethod
    def _fraction_abs_above(values: List[float], threshold: float) -> float:
        if not values:
            return 0.0
        return float(sum(1 for value in values if abs(float(value)) > threshold) / len(values))

    @staticmethod
    def _fraction_within(values: List[float], target: float, atol: float) -> float:
        if not values:
            return 0.0
        return float(sum(1 for value in values if abs(float(value) - target) <= atol) / len(values))

    @staticmethod
    def _mean_abs_gap_to_target(values: List[float], target: float) -> float:
        if not values:
            return 0.0
        return float(sum(abs(float(value) - target) for value in values) / len(values))

    @staticmethod
    def _paired_series_correlation(left: List[float], right: List[float]) -> float:
        if len(left) <= 1 or len(left) != len(right):
            return 0.0
        left_tensor = torch.tensor(left, dtype=torch.float32)
        right_tensor = torch.tensor(right, dtype=torch.float32)
        left_centered = left_tensor - left_tensor.mean()
        right_centered = right_tensor - right_tensor.mean()
        denominator = float(left_centered.norm().item() * right_centered.norm().item())
        if denominator <= 1e-12:
            return 0.0
        return float(torch.dot(left_centered, right_centered).item() / denominator)

    @staticmethod
    def _pairwise_cosine_summary(vectors: Dict[int, torch.Tensor]) -> Dict[str, float]:
        if len(vectors) <= 1:
            return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        normalized_vectors = []
        for _, vector in sorted(vectors.items()):
            flat = vector.detach().float().reshape(-1)
            denom = float(flat.norm().item())
            if denom <= 1e-12:
                normalized_vectors.append(flat.new_zeros(flat.shape))
            else:
                normalized_vectors.append(flat / flat.norm())
        pairwise = []
        for left_index in range(len(normalized_vectors)):
            for right_index in range(left_index + 1, len(normalized_vectors)):
                pairwise.append(float((normalized_vectors[left_index] * normalized_vectors[right_index]).sum().item()))
        if not pairwise:
            return {"count": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
        pairwise_tensor = torch.tensor(pairwise, dtype=torch.float32)
        return {
            "count": float(pairwise_tensor.numel()),
            "mean": float(pairwise_tensor.mean().item()),
            "min": float(pairwise_tensor.min().item()),
            "max": float(pairwise_tensor.max().item()),
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
    def _planner_vector_collection_summary(vectors: Dict[int, torch.Tensor]) -> Dict[str, float]:
        if not vectors:
            return {
                "count": 0.0,
                "norm_mean": 0.0,
                "var_mean": 0.0,
                "pairwise_cosine_mean": 0.0,
                "pairwise_cosine_min": 0.0,
                "pairwise_cosine_max": 0.0,
            }
        norms = []
        variances = []
        flattened = {}
        for block_id, vector in vectors.items():
            flat = vector.detach().float().reshape(-1)
            flattened[int(block_id)] = flat
            norms.append(float(flat.norm().item()))
            variances.append(float(flat.var(unbiased=False).item()) if flat.numel() > 1 else 0.0)
        pairwise = NHLoRATrainer._pairwise_cosine_summary(flattened)
        return {
            "count": float(len(flattened)),
            "norm_mean": sum(norms) / max(len(norms), 1),
            "var_mean": sum(variances) / max(len(variances), 1),
            "pairwise_cosine_mean": float(pairwise["mean"]),
            "pairwise_cosine_min": float(pairwise["min"]),
            "pairwise_cosine_max": float(pairwise["max"]),
        }

    @staticmethod
    def _new_planner_control_epoch_accumulator() -> Dict[str, object]:
        return {
            "batches": 0,
            "grad_l2_sum": 0.0,
            "grad_l2_max": 0.0,
            "grad_present_batches": 0,
            "grad_nonzero_batches": 0,
            "optimizer_steps": 0,
            "layers": {},
            "input_pairwise_cosine": [],
            "representation_pairwise_cosine": [],
        }

    @staticmethod
    def _planner_control_layer_accumulator(
        accumulator: Dict[str, object],
        block_id: int,
    ) -> Dict[str, object]:
        layers = accumulator.setdefault("layers", {})
        return layers.setdefault(
            int(block_id),
            {
                "beta_values": [],
                "beta_logits": [],
                "anchor_beta_values": [],
                "anchor_logit_values": [],
                "delta_raw_values": [],
                "delta_logit_values": [],
                "delta_bound_derivative_values": [],
                "beta_gap_abs": [],
                "logit_gap_abs": [],
                "delta_from_representation_values": [],
                "delta_from_representation_abs_values": [],
                "delta_bias_values": [],
                "delta_bias_abs_values": [],
                "delta_bias_share_values": [],
                "head_weight_norms": [],
                "head_bias_norms": [],
                "head_weight_drift_from_init": [],
                "head_bias_drift_from_init": [],
                "beta_grad_abs": [],
                "logit_grad_abs": [],
                "delta_raw_grad_abs": [],
                "delta_logit_grad_abs": [],
                "beta_grad_present_batches": 0,
                "beta_grad_nonzero_batches": 0,
                "logit_grad_present_batches": 0,
                "logit_grad_nonzero_batches": 0,
                "delta_raw_grad_present_batches": 0,
                "delta_raw_grad_nonzero_batches": 0,
                "delta_logit_grad_present_batches": 0,
                "delta_logit_grad_nonzero_batches": 0,
                "bridge_grad_lost_batches": 0,
                "loss_output_zero_batches": 0,
                "input_norms": [],
                "input_vars": [],
                "representation_norms": [],
                "representation_vars": [],
                "normalized_representation_norms": [],
                "normalized_representation_vars": [],
            },
        )

    @staticmethod
    def _gate_tensor_like(anchor_beta: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if isinstance(anchor_beta, torch.Tensor):
            gate = anchor_beta.to(device=reference.device, dtype=reference.dtype)
        else:
            gate = reference.new_full(reference.shape, float(anchor_beta))
        while gate.dim() < reference.dim():
            gate = gate.unsqueeze(-1)
        if gate.shape != reference.shape:
            gate = gate.expand_as(reference)
        return gate

    @staticmethod
    def _bounded_residual_transform(delta_raw: torch.Tensor) -> torch.Tensor:
        return F.softsign(delta_raw)

    @staticmethod
    def _bounded_residual_derivative(delta_raw: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + delta_raw.abs()).pow(2)

    def _compose_hybrid_control_outputs(
        self,
        raw_outputs: PlannerControlOutputs,
        *,
        anchor_beta: float | torch.Tensor,
    ) -> PlannerControlOutputs:
        anchor_beta_tensor = self._gate_tensor_like(anchor_beta, raw_outputs.delta_raw)
        clamped_anchor_beta = anchor_beta_tensor.clamp(min=1e-4, max=1.0 - 1e-4)
        anchor_logit = torch.logit(clamped_anchor_beta)
        delta_logit = self._planner_control_delta_logit_scale() * self._bounded_residual_transform(raw_outputs.delta_raw)
        effective_logit = anchor_logit + delta_logit
        effective_beta = torch.sigmoid(effective_logit)
        return PlannerControlOutputs(
            shared_gate=effective_beta,
            shared_gate_logit=effective_logit,
            delta_raw=raw_outputs.delta_raw,
            delta_logit=delta_logit,
            anchor_beta=anchor_beta_tensor,
            anchor_logit=anchor_logit,
            delta_from_representation=raw_outputs.delta_from_representation,
            delta_bias=raw_outputs.delta_bias,
            control_head_weight_norm=raw_outputs.control_head_weight_norm,
            control_head_bias_norm=raw_outputs.control_head_bias_norm,
            history_attention=raw_outputs.history_attention,
            history_context=raw_outputs.history_context,
            planner_input=raw_outputs.planner_input,
            planner_representation=raw_outputs.planner_representation,
            normalized_planner_representation=raw_outputs.normalized_planner_representation,
        )

    def _planner_control_head_summary(self, block_id: int) -> Dict[str, float]:
        output_head = self.planner.control_branch.output_heads[str(int(block_id))]
        weight = output_head.weight.detach().cpu()
        bias = None if output_head.bias is None else output_head.bias.detach().cpu()
        init_weight = self._planner_control_init_snapshot.get(f"output_heads.{int(block_id)}.weight")
        init_bias = self._planner_control_init_snapshot.get(f"output_heads.{int(block_id)}.bias")
        weight_drift = 0.0
        bias_drift = 0.0
        if init_weight is not None:
            weight_drift = float((weight - init_weight).norm().item())
        if bias is not None and init_bias is not None:
            bias_drift = float((bias - init_bias).norm().item())
        return {
            "weight_norm": float(weight.norm().item()),
            "bias_norm": 0.0 if bias is None else float(bias.norm().item()),
            "weight_l2_from_init": weight_drift,
            "bias_l2_from_init": bias_drift,
        }

    def _capture_planner_control_task_start(
        self,
        task_embedding: torch.Tensor,
        history_summary: torch.Tensor | None,
        applied_plans: Dict[int, Dict[str, object]],
    ) -> Dict[int, Dict[str, object]]:
        if not self._hybrid_planner_enabled():
            return {}
        snapshot: Dict[int, Dict[str, object]] = {}
        with torch.no_grad():
            for block_id in self.model.selected_blocks:
                raw_outputs = self.planner.forward_control(
                    int(block_id),
                    task_embedding=task_embedding,
                    history_summary=history_summary,
                )
                outputs = self._compose_hybrid_control_outputs(
                    raw_outputs,
                    anchor_beta=float(applied_plans[int(block_id)]["shared_gate"]),
                )
                snapshot[int(block_id)] = {
                    "beta": float(outputs.shared_gate.detach().mean().item()),
                    "logit": float(outputs.shared_gate_logit.detach().mean().item()),
                    "anchor_beta": float(outputs.anchor_beta.detach().mean().item()),
                    "anchor_logit": float(outputs.anchor_logit.detach().mean().item()),
                    "delta_raw": float(outputs.delta_raw.detach().mean().item()),
                    "delta_logit": 0.0
                    if outputs.delta_logit is None
                    else float(outputs.delta_logit.detach().mean().item()),
                    "planner_input": outputs.planner_input.detach().float().cpu() if outputs.planner_input is not None else None,
                    "planner_representation": outputs.planner_representation.detach().float().cpu()
                    if outputs.planner_representation is not None
                    else None,
                }
        return snapshot

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

    @staticmethod
    def _policy_output_index(name: str) -> int:
        indices = {
            "novelty": 0,
            "conflict": 1,
            "rank": 2,
            "consolidate": 3,
            "shared_gate": 4,
        }
        return int(indices[name])

    @staticmethod
    def _policy_tensor_scalar(tensor: torch.Tensor | None, index: int) -> float:
        if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return 0.0
        detached = tensor.detach().float()
        if detached.dim() == 1:
            if index >= detached.numel():
                return 0.0
            return float(detached[index].item())
        if detached.size(-1) <= index:
            return 0.0
        return float(detached[..., index].mean().item())

    @staticmethod
    def _component_bias_share(activation_value: float, bias_value: float) -> float:
        total = abs(float(activation_value)) + abs(float(bias_value))
        if total <= 1e-12:
            return 0.0
        return float(abs(float(bias_value)) / total)

    @staticmethod
    def _requested_growth(action: object) -> bool:
        return str(action) in {"open_new_slot", "expand_rank_existing_slot"}

    @staticmethod
    def _threshold_proximity_flags(
        novelty_margin: float,
        conflict_margin: float,
        *,
        near_eps: float = 0.05,
    ) -> Dict[str, bool]:
        return {
            "near_novelty_miss": bool(-near_eps <= novelty_margin < 0.0),
            "near_conflict_miss": bool(-near_eps <= conflict_margin < 0.0),
            "far_below_novelty": bool(novelty_margin < -near_eps),
            "far_below_conflict": bool(conflict_margin < -near_eps),
            "far_below_both": bool(novelty_margin < -near_eps and conflict_margin < -near_eps),
        }

    @staticmethod
    def _planner_threshold_proximity_summary(
        entries: List[Dict[str, object]],
        *,
        near_eps: float = 0.05,
    ) -> Dict[str, object]:
        if not entries:
            return {
                "requested_growth_frequency": 0.0,
                "open_requested_frequency": 0.0,
                "expand_requested_frequency": 0.0,
                "near_novelty_miss_frequency": 0.0,
                "near_conflict_miss_frequency": 0.0,
                "far_below_both_frequency": 0.0,
            }
        total = len(entries)
        requested_growth = 0
        open_requested = 0
        expand_requested = 0
        near_novelty = 0
        near_conflict = 0
        far_below_both = 0
        for entry in entries:
            action = str(entry.get("action", ""))
            if NHLoRATrainer._requested_growth(action):
                requested_growth += 1
            if action == "open_new_slot":
                open_requested += 1
            if action == "expand_rank_existing_slot":
                expand_requested += 1
            flags = NHLoRATrainer._threshold_proximity_flags(
                float(entry.get("novelty_margin", 0.0)),
                float(entry.get("conflict_margin", 0.0)),
                near_eps=near_eps,
            )
            near_novelty += int(flags["near_novelty_miss"])
            near_conflict += int(flags["near_conflict_miss"])
            far_below_both += int(flags["far_below_both"])
        return {
            "requested_growth_frequency": float(requested_growth / total),
            "open_requested_frequency": float(open_requested / total),
            "expand_requested_frequency": float(expand_requested / total),
            "near_novelty_miss_frequency": float(near_novelty / total),
            "near_conflict_miss_frequency": float(near_conflict / total),
            "far_below_both_frequency": float(far_below_both / total),
        }

    @staticmethod
    def _planner_rank_gap(ranking: List[Dict[str, object]], *, action: str) -> float:
        if len(ranking) < 2:
            return 0.0
        top = ranking[0]
        runner_up = ranking[1]
        if action == "open_new_slot":
            return float(top.get("conflict_margin", 0.0)) - float(runner_up.get("conflict_margin", 0.0))
        if action == "expand_rank_existing_slot":
            return float(top.get("novelty_margin", 0.0)) - float(runner_up.get("novelty_margin", 0.0))
        return 0.0

    @staticmethod
    def _planner_rank_stability_summary(task_summaries: List[Dict[str, object]]) -> Dict[str, object]:
        open_runner_ups: Dict[int, int] = {}
        expand_runner_ups: Dict[int, int] = {}
        open_gap_values: List[float] = []
        expand_gap_values: List[float] = []
        for summary in task_summaries:
            open_ranking = summary.get("open_ranking", [])
            expand_ranking = summary.get("expand_ranking", [])
            if len(open_ranking) >= 2:
                runner_up = int(open_ranking[1]["block_id"])
                open_runner_ups[runner_up] = open_runner_ups.get(runner_up, 0) + 1
            if len(expand_ranking) >= 2:
                runner_up = int(expand_ranking[1]["block_id"])
                expand_runner_ups[runner_up] = expand_runner_ups.get(runner_up, 0) + 1
            open_gap_values.append(NHLoRATrainer._planner_rank_gap(open_ranking, action="open_new_slot"))
            expand_gap_values.append(
                NHLoRATrainer._planner_rank_gap(expand_ranking, action="expand_rank_existing_slot")
            )
        winner_distribution = NHLoRATrainer._growth_winner_distribution(task_summaries)
        return {
            "open_winners": winner_distribution["open_winners"],
            "expand_winners": winner_distribution["expand_winners"],
            "open_runner_ups": open_runner_ups,
            "expand_runner_ups": expand_runner_ups,
            "avg_open_gap_to_runner_up": float(sum(open_gap_values) / len(open_gap_values)) if open_gap_values else 0.0,
            "avg_expand_gap_to_runner_up": float(sum(expand_gap_values) / len(expand_gap_values))
            if expand_gap_values
            else 0.0,
        }

    @staticmethod
    def _planner_history_conditioning_summary(entries: List[Dict[str, object]]) -> Dict[str, object]:
        if not entries:
            return {
                "history_used_frequency": 0.0,
                "history_entropy_mean": 0.0,
                "history_max_weight_mean": 0.0,
                "requested_growth_history_entropy_mean": 0.0,
                "requested_growth_history_max_weight_mean": 0.0,
                "history_max_weight_vs_novelty_corr": 0.0,
                "history_max_weight_vs_conflict_corr": 0.0,
            }
        total = len(entries)
        entropy_values = [float(entry.get("history_attention_entropy", 0.0)) for entry in entries]
        max_weight_values = [float(entry.get("history_attention_max_weight", 0.0)) for entry in entries]
        growth_entries = [entry for entry in entries if NHLoRATrainer._requested_growth(entry.get("action"))]
        growth_entropy_values = [float(entry.get("history_attention_entropy", 0.0)) for entry in growth_entries]
        growth_max_weight_values = [float(entry.get("history_attention_max_weight", 0.0)) for entry in growth_entries]
        novelty_values = [float(entry.get("novelty_margin", 0.0)) for entry in entries]
        conflict_values = [float(entry.get("conflict_margin", 0.0)) for entry in entries]
        return {
            "history_used_frequency": float(
                sum(1 for entry in entries if bool(entry.get("history_attention_used", False))) / total
            ),
            "history_entropy_mean": float(sum(entropy_values) / len(entropy_values)) if entropy_values else 0.0,
            "history_max_weight_mean": float(sum(max_weight_values) / len(max_weight_values))
            if max_weight_values
            else 0.0,
            "requested_growth_history_entropy_mean": float(sum(growth_entropy_values) / len(growth_entropy_values))
            if growth_entropy_values
            else 0.0,
            "requested_growth_history_max_weight_mean": float(sum(growth_max_weight_values) / len(growth_max_weight_values))
            if growth_max_weight_values
            else 0.0,
            "history_max_weight_vs_novelty_corr": NHLoRATrainer._paired_series_correlation(
                max_weight_values,
                novelty_values,
            ),
            "history_max_weight_vs_conflict_corr": NHLoRATrainer._paired_series_correlation(
                max_weight_values,
                conflict_values,
            ),
        }

    @staticmethod
    def _planner_post_task_outcome(
        *,
        requested_growth: bool,
        materialized_growth: bool,
        applied_growth: bool,
        post_shared_only: bool,
    ) -> str:
        if applied_growth and not post_shared_only:
            return "live_and_growing"
        if applied_growth and post_shared_only:
            return "temporarily_expanded_then_collapsed"
        if requested_growth and not materialized_growth:
            return "never_really_materialized"
        if post_shared_only:
            return "shared_only"
        return "live_and_growing"

    @staticmethod
    def _final_outcome_from_trajectory(history_summary: Dict[str, object]) -> str:
        tasks = history_summary.get("tasks", [])
        if not tasks:
            return "never_really_materialized"
        if bool(history_summary.get("opened_then_shared_only", False)):
            return "temporarily_expanded_then_collapsed"
        post_shared_only_flags = history_summary.get("post_shared_only_flags", [])
        final_shared_only = bool(post_shared_only_flags[-1]) if post_shared_only_flags else True
        if final_shared_only:
            return "shared_only"
        return "live_and_growing"

    @staticmethod
    def _top_reason_counts(reason_counts: Dict[str, int], *, limit: int = 3) -> List[Tuple[str, int]]:
        return sorted(
            [(str(reason), int(count)) for reason, count in reason_counts.items()],
            key=lambda item: (-item[1], item[0]),
        )[:limit]

    def _planner_realization_trace_summary(self, block_ids: List[int]) -> Dict[str, object]:
        selected_block_ids = [
            int(block_id)
            for block_id in block_ids
            if int(block_id) in self._planner_policy_record_history
        ]
        policy_entries = [
            entry
            for block_id in selected_block_ids
            for entry in self._planner_policy_record_history.get(int(block_id), [])
        ]
        realization_entries = [
            entry
            for block_id in selected_block_ids
            for entry in self._planner_realization_history.get(int(block_id), [])
        ]
        threshold_summary = self._planner_threshold_proximity_summary(policy_entries)
        history_summary = self._planner_history_conditioning_summary(policy_entries)
        fallback_counts: Dict[str, int] = {}
        outcome_counts: Dict[str, int] = {}
        requested_growth_count = 0
        materialized_growth_count = 0
        applied_growth_count = 0
        for entry in realization_entries:
            requested_growth_count += int(bool(entry.get("requested_growth", False)))
            materialized_growth_count += int(bool(entry.get("materialized_growth", False)))
            applied_growth_count += int(bool(entry.get("applied_growth", False)))
            fallback_reason = str(entry.get("fallback_reason", "none"))
            fallback_counts[fallback_reason] = fallback_counts.get(fallback_reason, 0) + 1
            post_outcome = str(entry.get("post_outcome", "shared_only"))
            outcome_counts[post_outcome] = outcome_counts.get(post_outcome, 0) + 1
        total_realization = len(realization_entries)
        novelty_margin_values = [float(entry.get("novelty_margin", 0.0)) for entry in policy_entries]
        conflict_margin_values = [float(entry.get("conflict_margin", 0.0)) for entry in policy_entries]
        novelty_bias_share_values = [float(entry.get("novelty_bias_share", 0.0)) for entry in policy_entries]
        conflict_bias_share_values = [float(entry.get("conflict_bias_share", 0.0)) for entry in policy_entries]
        final_shared_only_layers: List[int] = []
        final_non_shared_layers: List[int] = []
        final_outcomes_by_layer: Dict[int, str] = {}
        for block_id in selected_block_ids:
            history = self._aggregate_layer_lifecycle_history(self._planner_layer_structure_history.get(int(block_id), []))
            final_outcome = self._final_outcome_from_trajectory(history)
            final_outcomes_by_layer[int(block_id)] = final_outcome
            if final_outcome == "shared_only" or final_outcome == "temporarily_expanded_then_collapsed":
                final_shared_only_layers.append(int(block_id))
            else:
                final_non_shared_layers.append(int(block_id))
        return {
            "policy_entry_count": len(policy_entries),
            "realization_entry_count": total_realization,
            "requested_growth_frequency": float(requested_growth_count / total_realization) if total_realization else 0.0,
            "materialized_growth_frequency": float(materialized_growth_count / total_realization)
            if total_realization
            else 0.0,
            "applied_growth_frequency": float(applied_growth_count / total_realization) if total_realization else 0.0,
            "novelty_margin_mean": float(sum(novelty_margin_values) / len(novelty_margin_values))
            if novelty_margin_values
            else 0.0,
            "conflict_margin_mean": float(sum(conflict_margin_values) / len(conflict_margin_values))
            if conflict_margin_values
            else 0.0,
            "novelty_bias_share_mean": float(sum(novelty_bias_share_values) / len(novelty_bias_share_values))
            if novelty_bias_share_values
            else 0.0,
            "conflict_bias_share_mean": float(sum(conflict_bias_share_values) / len(conflict_bias_share_values))
            if conflict_bias_share_values
            else 0.0,
            "threshold_summary": threshold_summary,
            "history_summary": history_summary,
            "fallback_counts": fallback_counts,
            "top_fallbacks": self._top_reason_counts(fallback_counts),
            "outcome_counts": outcome_counts,
            "final_outcomes_by_layer": final_outcomes_by_layer,
            "final_shared_only_layers": sorted(final_shared_only_layers),
            "final_non_shared_layers": sorted(final_non_shared_layers),
        }

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
        novelty_index = self._policy_output_index("novelty")
        conflict_index = self._policy_output_index("conflict")
        rank_index = self._policy_output_index("rank")
        consolidate_index = self._policy_output_index("consolidate")
        shared_gate_index = self._policy_output_index("shared_gate")
        novelty_from_representation = self._policy_tensor_scalar(signals.output_from_representation, novelty_index)
        novelty_bias = self._policy_tensor_scalar(signals.output_bias, novelty_index)
        conflict_from_representation = self._policy_tensor_scalar(signals.output_from_representation, conflict_index)
        conflict_bias = self._policy_tensor_scalar(signals.output_bias, conflict_index)
        threshold_flags = self._threshold_proximity_flags(
            novelty_margin,
            conflict_margin,
        )
        action = self._planner_decision_label(
            novelty=novelty,
            conflict=conflict,
            tau_novelty=tau_novelty,
            tau_conflict=tau_conflict,
        )
        return {
            "task_number": int(task_number),
            "block_id": int(block_id),
            "action": action,
            "novelty": novelty,
            "conflict": conflict,
            "tau_novelty": tau_novelty,
            "tau_conflict": tau_conflict,
            "novelty_margin": novelty_margin,
            "conflict_margin": conflict_margin,
            "requested_growth": bool(self._requested_growth(action)),
            "shared_gate": float(signals.shared_gate.item()),
            "consolidate": float(signals.consolidate.item()),
            "rank_budget": int(signals.rank_budget),
            "novelty_logit": self._policy_tensor_scalar(signals.raw_outputs, novelty_index),
            "conflict_logit": self._policy_tensor_scalar(signals.raw_outputs, conflict_index),
            "rank_logit": self._policy_tensor_scalar(signals.raw_outputs, rank_index),
            "consolidate_logit": self._policy_tensor_scalar(signals.raw_outputs, consolidate_index),
            "shared_gate_logit": self._policy_tensor_scalar(signals.raw_outputs, shared_gate_index),
            "novelty_from_representation": novelty_from_representation,
            "novelty_bias": novelty_bias,
            "novelty_bias_share": self._component_bias_share(novelty_from_representation, novelty_bias),
            "conflict_from_representation": conflict_from_representation,
            "conflict_bias": self._policy_tensor_scalar(signals.output_bias, conflict_index),
            "conflict_bias_share": self._component_bias_share(conflict_from_representation, conflict_bias),
            "rank_from_representation": self._policy_tensor_scalar(signals.output_from_representation, rank_index),
            "rank_bias": self._policy_tensor_scalar(signals.output_bias, rank_index),
            "consolidate_from_representation": self._policy_tensor_scalar(
                signals.output_from_representation,
                consolidate_index,
            ),
            "consolidate_bias": self._policy_tensor_scalar(signals.output_bias, consolidate_index),
            "shared_gate_from_representation": self._policy_tensor_scalar(
                signals.output_from_representation,
                shared_gate_index,
            ),
            "shared_gate_bias": self._policy_tensor_scalar(signals.output_bias, shared_gate_index),
            "policy_head_weight_norm": float(signals.policy_head_weight_norm or 0.0),
            "policy_head_bias_norm": float(signals.policy_head_bias_norm or 0.0),
            "novelty_head_row_norm": self._policy_tensor_scalar(signals.policy_head_row_norms, novelty_index),
            "conflict_head_row_norm": self._policy_tensor_scalar(signals.policy_head_row_norms, conflict_index),
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
            "near_novelty_miss": bool(threshold_flags["near_novelty_miss"]),
            "near_conflict_miss": bool(threshold_flags["near_conflict_miss"]),
            "far_below_novelty": bool(threshold_flags["far_below_novelty"]),
            "far_below_conflict": bool(threshold_flags["far_below_conflict"]),
            "far_below_both": bool(threshold_flags["far_below_both"]),
        }

    @staticmethod
    def _planner_candidate_ranking(
        records: Dict[int, Dict[str, object]],
        *,
        action: str,
    ) -> List[Dict[str, object]]:
        ranked: List[Dict[str, object]] = []
        for block_id, record in records.items():
            if str(record.get("action", "")) != str(action):
                continue
            ranked.append(
                {
                    "block_id": int(block_id),
                    "novelty_margin": float(record.get("novelty_margin", 0.0)),
                    "conflict_margin": float(record.get("conflict_margin", 0.0)),
                    "shared_gate": float(record.get("shared_gate", 0.0)),
                    "rank_budget": int(record.get("rank_budget", 0)),
                }
            )
        if action == "open_new_slot":
            ranked.sort(
                key=lambda item: (
                    -float(item["conflict_margin"]),
                    -float(item["novelty_margin"]),
                    int(item["block_id"]),
                )
            )
        elif action == "expand_rank_existing_slot":
            ranked.sort(
                key=lambda item: (
                    -float(item["novelty_margin"]),
                    float(item["conflict_margin"]),
                    int(item["block_id"]),
                )
            )
        else:
            ranked.sort(
                key=lambda item: (
                    -float(item["novelty_margin"]),
                    -float(item["conflict_margin"]),
                    int(item["block_id"]),
                )
            )
        return ranked

    @staticmethod
    def _ranking_position(ranking: List[Dict[str, object]], block_id: int) -> int | None:
        for index, entry in enumerate(ranking, start=1):
            if int(entry["block_id"]) == int(block_id):
                return int(index)
        return None

    def _planner_growth_task_summary(
        self,
        *,
        task_number: int,
        records: Dict[int, Dict[str, object]],
        materialized_plans: Dict[int, MaterializedLayerPlan],
        applied_plans: Dict[int, Dict[str, object]],
    ) -> Dict[str, object]:
        open_ranking = self._planner_candidate_ranking(records, action="open_new_slot")
        expand_ranking = self._planner_candidate_ranking(records, action="expand_rank_existing_slot")
        growth_layers: List[int] = []
        opened_layers: List[int] = []
        expanded_layers: List[int] = []
        fallback_layers: List[int] = []
        layer_outcomes: Dict[int, Dict[str, object]] = {}
        for block_id, plan in materialized_plans.items():
            opened = bool(plan.create_new_slot)
            expanded = int(plan.rank_delta) > 0
            if opened or expanded:
                growth_layers.append(int(block_id))
            if opened:
                opened_layers.append(int(block_id))
            if expanded and not opened:
                expanded_layers.append(int(block_id))
            if plan.fallback_action is not None:
                fallback_layers.append(int(block_id))
            applied = applied_plans.get(int(block_id), {})
            layer_outcomes[int(block_id)] = {
                "requested_action": str(plan.requested_action),
                "materialized_action": str(plan.action),
                "applied_action": str(applied.get("action", plan.action)),
                "opened": opened,
                "expanded": expanded and not opened,
                "rank_delta": int(plan.rank_delta),
                "fallback_action": None if plan.fallback_action is None else str(plan.fallback_action),
                "applied_fallback_action": None
                if applied.get("fallback_action") is None
                else str(applied.get("fallback_action")),
                "candidate_count": len(plan.candidate_slots),
                "applied_candidate_count": len(self._candidate_slot_ids(applied)),
                "shared_only": bool(plan.shared_only),
                "applied_shared_only": bool(applied.get("shared_only", plan.shared_only)),
                "created_new_slot": bool(plan.create_new_slot),
                "applied_created_new_slot": bool(applied.get("created_new_slot", False)),
            }
        growth_layers.sort()
        opened_layers.sort()
        expanded_layers.sort()
        fallback_layers.sort()
        return {
            "task_number": int(task_number),
            "open_ranking": open_ranking,
            "expand_ranking": expand_ranking,
            "open_winner": None if not open_ranking else int(open_ranking[0]["block_id"]),
            "expand_winner": None if not expand_ranking else int(expand_ranking[0]["block_id"]),
            "growth_layers": growth_layers,
            "opened_layers": opened_layers,
            "expanded_layers": expanded_layers,
            "fallback_layers": fallback_layers,
            "layer6_open_rank": self._ranking_position(open_ranking, 6),
            "layer6_expand_rank": self._ranking_position(expand_ranking, 6),
            "layer_outcomes": layer_outcomes,
        }

    @staticmethod
    def _growth_winner_distribution(task_summaries: List[Dict[str, object]]) -> Dict[str, object]:
        open_winners: Dict[int, int] = {}
        expand_winners: Dict[int, int] = {}
        actual_growth_counts: Dict[int, int] = {}
        zero_growth_tasks: List[int] = []
        for summary in task_summaries:
            task_number = int(summary.get("task_number", 0))
            open_winner = summary.get("open_winner")
            if open_winner is not None:
                block_id = int(open_winner)
                open_winners[block_id] = open_winners.get(block_id, 0) + 1
            expand_winner = summary.get("expand_winner")
            if expand_winner is not None:
                block_id = int(expand_winner)
                expand_winners[block_id] = expand_winners.get(block_id, 0) + 1
            growth_layers = [int(block_id) for block_id in summary.get("growth_layers", [])]
            if not growth_layers:
                zero_growth_tasks.append(task_number)
            for block_id in growth_layers:
                actual_growth_counts[block_id] = actual_growth_counts.get(block_id, 0) + 1
        return {
            "open_winners": open_winners,
            "expand_winners": expand_winners,
            "actual_growth_counts": actual_growth_counts,
            "zero_growth_tasks": zero_growth_tasks,
            "layer6_open_wins": int(open_winners.get(6, 0)),
            "layer6_expand_wins": int(expand_winners.get(6, 0)),
            "layer6_growth_tasks": int(actual_growth_counts.get(6, 0)),
        }

    @staticmethod
    def _new_structure_route_accumulator() -> Dict[int, Dict[str, object]]:
        return {}

    @staticmethod
    def _structure_route_layer_accumulator(
        accumulator: Dict[int, Dict[str, object]],
        block_id: int,
    ) -> Dict[str, object]:
        return accumulator.setdefault(
            int(block_id),
            {
                "batches": 0,
                "empty_candidate_batches": 0,
                "candidate_count_sum": 0.0,
                "selected_count_sum": 0.0,
                "nonzero_usage_slot_count_sum": 0.0,
                "usage_top1_share_sum": 0.0,
                "usage_entropy_sum": 0.0,
                "usage_mass_by_slot": {},
            },
        )

    def _accumulate_planner_structure_route_usage(
        self,
        context: Dict[str, Any],
        route_info: Dict[int, Dict[str, object]],
    ) -> None:
        if not self._planner_audit_enabled():
            return
        accumulator = context.setdefault(
            "planner_structure_route_accumulator",
            self._new_structure_route_accumulator(),
        )
        for block_id, layer_state in route_info.items():
            layer_stats = self._structure_route_layer_accumulator(accumulator, int(block_id))
            candidate_slots = self._int_list(layer_state.get("candidate_slots", []))
            selected_slots = self._int_list(layer_state.get("selected_slots", []))
            distribution = layer_state.get("routing_distribution")
            usage_vector = layer_state.get("usage_vector")
            if usage_vector is None and isinstance(distribution, torch.Tensor) and distribution.numel() > 0:
                usage_vector = distribution.detach().mean(dim=0)
            layer_stats["batches"] += 1
            layer_stats["candidate_count_sum"] += float(len(candidate_slots))
            layer_stats["selected_count_sum"] += float(len(selected_slots))
            if not candidate_slots:
                layer_stats["empty_candidate_batches"] += 1
                continue
            if isinstance(usage_vector, torch.Tensor) and usage_vector.numel() > 0:
                detached_usage = usage_vector.detach().float().reshape(-1).cpu()
                total_mass = float(detached_usage.sum().item())
                active_count = int((detached_usage > 1e-8).sum().item())
                if total_mass > 1e-12:
                    normalized_usage = detached_usage / total_mass
                    top_share = float(normalized_usage.max().item())
                    entropy = float(
                        -(normalized_usage * normalized_usage.clamp_min(1e-8).log()).sum().item()
                    )
                else:
                    top_share = 0.0
                    entropy = 0.0
                layer_stats["nonzero_usage_slot_count_sum"] += float(active_count)
                layer_stats["usage_top1_share_sum"] += top_share
                layer_stats["usage_entropy_sum"] += entropy
                usage_mass_by_slot = layer_stats.setdefault("usage_mass_by_slot", {})
                for slot_id, slot_mass in zip(candidate_slots, detached_usage.tolist()):
                    usage_mass_by_slot[int(slot_id)] = usage_mass_by_slot.get(int(slot_id), 0.0) + float(slot_mass)

    @staticmethod
    def _route_usage_concentration_summary(
        route_stats: Dict[str, object],
        *,
        lifecycle_summary: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        batches = max(int(route_stats.get("batches", 0)), 1)
        usage_mass_by_slot = {
            int(slot_id): float(value)
            for slot_id, value in route_stats.get("usage_mass_by_slot", {}).items()
        }
        usage_total = sum(usage_mass_by_slot.values())
        usage_top1_share = 0.0
        usage_entropy = 0.0
        if usage_total > 1e-12:
            normalized_usage = torch.tensor(
                [value / usage_total for _, value in sorted(usage_mass_by_slot.items())],
                dtype=torch.float32,
            )
            usage_top1_share = float(normalized_usage.max().item())
            usage_entropy = float(
                -(normalized_usage * normalized_usage.clamp_min(1e-8).log()).sum().item()
            )
        cumulative_usage_by_slot = {}
        usage_ema_by_slot = {}
        profile_candidate_count = 0
        profile_shared_only = False
        if lifecycle_summary:
            cumulative_usage_by_slot = {
                int(slot_id): float(value)
                for slot_id, value in lifecycle_summary.get("cumulative_usage_by_slot", {}).items()
            }
            usage_ema_by_slot = {
                int(slot_id): float(value)
                for slot_id, value in lifecycle_summary.get("usage_ema_by_slot", {}).items()
            }
            profile_candidate_count = len(lifecycle_summary.get("candidate_slot_ids", []))
            profile_shared_only = bool(lifecycle_summary.get("shared_only", False))

        def _top_share(values: Dict[int, float]) -> float:
            total = sum(values.values())
            if total <= 1e-12:
                return 0.0
            return float(max(values.values()) / total)

        train_candidate_count_mean = float(route_stats.get("candidate_count_sum", 0.0)) / batches
        return {
            "train_candidate_count_mean": train_candidate_count_mean,
            "train_selected_count_mean": float(route_stats.get("selected_count_sum", 0.0)) / batches,
            "train_nonzero_usage_slot_count_mean": float(route_stats.get("nonzero_usage_slot_count_sum", 0.0)) / batches,
            "empty_candidate_fraction": float(route_stats.get("empty_candidate_batches", 0)) / batches,
            "usage_top1_share_mean": float(route_stats.get("usage_top1_share_sum", 0.0)) / batches,
            "usage_entropy_mean": float(route_stats.get("usage_entropy_sum", 0.0)) / batches,
            "aggregated_usage_top1_share": usage_top1_share,
            "aggregated_usage_entropy": usage_entropy,
            "usage_nonzero_slot_count": int(sum(1 for value in usage_mass_by_slot.values() if value > 1e-8)),
            "usage_ema_top1_share": _top_share(usage_ema_by_slot),
            "cumulative_usage_top1_share": _top_share(cumulative_usage_by_slot),
            "profile_candidate_count": int(profile_candidate_count),
            "profile_shared_only": bool(profile_shared_only),
            "candidate_but_zero_usage": bool(train_candidate_count_mean > 0.0 and usage_total <= 1e-12),
        }

    @staticmethod
    def _profile_contraction_diff(
        pre_summary: Dict[str, object],
        post_summary: Dict[str, object],
    ) -> Dict[str, object]:
        pre_live = len(pre_summary.get("live_slot_ids", []))
        post_live = len(post_summary.get("live_slot_ids", []))
        pre_retained = len(pre_summary.get("retained_slot_ids", []))
        post_retained = len(post_summary.get("retained_slot_ids", []))
        pre_candidates = len(pre_summary.get("candidate_slot_ids", []))
        post_candidates = len(post_summary.get("candidate_slot_ids", []))
        pre_frozen = len(pre_summary.get("frozen_slot_ids", []))
        post_frozen = len(post_summary.get("frozen_slot_ids", []))
        pre_pruned = len(pre_summary.get("pruned_slot_ids", []))
        post_pruned = len(post_summary.get("pruned_slot_ids", []))
        pre_shared_only = bool(pre_summary.get("shared_only", False))
        post_shared_only = bool(post_summary.get("shared_only", False))
        return {
            "live_delta": int(post_live - pre_live),
            "retained_delta": int(post_retained - pre_retained),
            "candidate_delta": int(post_candidates - pre_candidates),
            "frozen_delta": int(post_frozen - pre_frozen),
            "pruned_delta": int(post_pruned - pre_pruned),
            "shared_only_changed": bool(pre_shared_only != post_shared_only),
            "contracted": bool(
                post_live < pre_live
                or post_retained < pre_retained
                or post_candidates < pre_candidates
                or (not pre_shared_only and post_shared_only)
            ),
        }

    @staticmethod
    def _aggregate_layer_lifecycle_history(entries: List[Dict[str, object]]) -> Dict[str, object]:
        if not entries:
            return {
                "tasks": [],
                "post_live_counts": [],
                "post_retained_counts": [],
                "post_shared_only_flags": [],
                "opened_tasks": [],
                "expanded_tasks": [],
                "grew_tasks": [],
                "never_grew": True,
                "ever_multi_slot": False,
                "opened_then_shared_only": False,
                "collapsed_after_growth_tasks": [],
            }
        ordered_entries = sorted(entries, key=lambda item: int(item.get("task_number", 0)))
        tasks = [int(entry["task_number"]) for entry in ordered_entries]
        post_live_counts = [int(entry.get("post_live_count", 0)) for entry in ordered_entries]
        post_retained_counts = [int(entry.get("post_retained_count", 0)) for entry in ordered_entries]
        post_shared_only_flags = [bool(entry.get("post_shared_only", False)) for entry in ordered_entries]
        opened_tasks = [int(entry["task_number"]) for entry in ordered_entries if bool(entry.get("opened", False))]
        expanded_tasks = [int(entry["task_number"]) for entry in ordered_entries if bool(entry.get("expanded", False))]
        grew_tasks = [int(entry["task_number"]) for entry in ordered_entries if bool(entry.get("opened", False) or entry.get("expanded", False))]
        collapsed_after_growth_tasks = [
            int(entry["task_number"])
            for entry in ordered_entries
            if bool(entry.get("opened", False) or entry.get("expanded", False))
            and bool(entry.get("post_shared_only", False))
        ]
        ever_multi_slot = any(
            int(entry.get("pre_live_count", 0)) > 1 or int(entry.get("post_live_count", 0)) > 1
            for entry in ordered_entries
        )
        final_post_shared_only = bool(ordered_entries[-1].get("post_shared_only", False))
        return {
            "tasks": tasks,
            "post_live_counts": post_live_counts,
            "post_retained_counts": post_retained_counts,
            "post_shared_only_flags": post_shared_only_flags,
            "opened_tasks": opened_tasks,
            "expanded_tasks": expanded_tasks,
            "grew_tasks": grew_tasks,
            "never_grew": not bool(grew_tasks),
            "ever_multi_slot": bool(ever_multi_slot),
            "opened_then_shared_only": bool(grew_tasks and final_post_shared_only),
            "collapsed_after_growth_tasks": collapsed_after_growth_tasks,
        }

    @staticmethod
    def _final_epoch_contribution_ratio(contribution_accumulator: Dict[int, Dict[str, object]], block_id: int) -> float:
        contribution_stats = contribution_accumulator.get(int(block_id), {})
        shared_post_count = max(int(contribution_stats.get("shared_post_beta_norm_count", 0)), 1)
        slot_count = max(int(contribution_stats.get("slot_norm_count", 0)), 1)
        mean_shared_post = float(contribution_stats.get("shared_post_beta_norm_sum", 0.0)) / shared_post_count
        mean_slot = float(contribution_stats.get("slot_norm_sum", 0.0)) / slot_count
        return mean_shared_post / max(mean_slot, 1e-12)

    def _log_stage13_policy_growth_summary(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        records = context.get("planner_audit_records", {})
        materialized_plans = context.get("materialized_plans", {})
        applied_plans = context.get("applied_plans", {})
        if not records or not materialized_plans or not applied_plans:
            return
        summary = self._planner_growth_task_summary(
            task_number=int(context["task_number"]),
            records=records,
            materialized_plans=materialized_plans,
            applied_plans=applied_plans,
        )
        context["planner_growth_task_summary"] = summary
        self._planner_growth_history.append(summary)
        open_ranked_layers = [
            (
                int(entry["block_id"]),
                round(float(entry["novelty_margin"]), 4),
                round(float(entry["conflict_margin"]), 4),
            )
            for entry in summary["open_ranking"]
        ]
        expand_ranked_layers = [
            (
                int(entry["block_id"]),
                round(float(entry["novelty_margin"]), 4),
                round(float(entry["conflict_margin"]), 4),
            )
            for entry in summary["expand_ranking"]
        ]
        self.logger.info(
            "[PlannerGrowthTask][Task %d] open_ranked_layers=%s expand_ranked_layers=%s growth_layers=%s opened_layers=%s expanded_layers=%s fallback_layers=%s layer6_open_rank=%s layer6_expand_rank=%s",
            context["task_number"],
            open_ranked_layers,
            expand_ranked_layers,
            summary["growth_layers"],
            summary["opened_layers"],
            summary["expanded_layers"],
            summary["fallback_layers"],
            "n/a" if summary["layer6_open_rank"] is None else int(summary["layer6_open_rank"]),
            "n/a" if summary["layer6_expand_rank"] is None else int(summary["layer6_expand_rank"]),
        )
        distribution = self._growth_winner_distribution(self._planner_growth_history)
        self.logger.info(
            "[PlannerGrowthConcentration][Task %d] open_winners=%s expand_winners=%s actual_growth_counts=%s zero_growth_tasks=%s layer6_open_wins=%d layer6_expand_wins=%d layer6_growth_tasks=%d",
            context["task_number"],
            distribution["open_winners"],
            distribution["expand_winners"],
            distribution["actual_growth_counts"],
            distribution["zero_growth_tasks"],
            int(distribution["layer6_open_wins"]),
            int(distribution["layer6_expand_wins"]),
            int(distribution["layer6_growth_tasks"]),
        )

    def _log_stage14_policy_signal_audit(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        records = context.get("planner_audit_records", {})
        if not records:
            return
        for block_id in sorted(records):
            record = dict(records[int(block_id)])
            self._planner_policy_record_history.setdefault(int(block_id), []).append(record)
            self.logger.info(
                "[PlannerPolicyLogits][Task %d][Layer %d] novelty_logit=%.4f conflict_logit=%.4f rank_logit=%.4f consolidate_logit=%.4f shared_gate_logit=%.4f head_weight_norm=%.4e head_bias_norm=%.4e novelty_head_row_norm=%.4e conflict_head_row_norm=%.4e",
                context["task_number"],
                int(block_id),
                float(record["novelty_logit"]),
                float(record["conflict_logit"]),
                float(record["rank_logit"]),
                float(record["consolidate_logit"]),
                float(record["shared_gate_logit"]),
                float(record["policy_head_weight_norm"]),
                float(record["policy_head_bias_norm"]),
                float(record["novelty_head_row_norm"]),
                float(record["conflict_head_row_norm"]),
            )
            self.logger.info(
                "[PlannerPolicyDecomposition][Task %d][Layer %d] novelty_from_rep=%.4f novelty_bias=%.4f novelty_bias_share=%.4f conflict_from_rep=%.4f conflict_bias=%.4f conflict_bias_share=%.4f rank_from_rep=%.4f rank_bias=%.4f shared_gate_from_rep=%.4f shared_gate_bias=%.4f",
                context["task_number"],
                int(block_id),
                float(record["novelty_from_representation"]),
                float(record["novelty_bias"]),
                float(record["novelty_bias_share"]),
                float(record["conflict_from_representation"]),
                float(record["conflict_bias"]),
                float(record["conflict_bias_share"]),
                float(record["rank_from_representation"]),
                float(record["rank_bias"]),
                float(record["shared_gate_from_representation"]),
                float(record["shared_gate_bias"]),
            )
            self.logger.info(
                "[PlannerThresholdProximity][Task %d][Layer %d] requested_growth=%s near_novelty_miss=%s near_conflict_miss=%s far_below_novelty=%s far_below_conflict=%s far_below_both=%s",
                context["task_number"],
                int(block_id),
                bool(record["requested_growth"]),
                bool(record["near_novelty_miss"]),
                bool(record["near_conflict_miss"]),
                bool(record["far_below_novelty"]),
                bool(record["far_below_conflict"]),
                bool(record["far_below_both"]),
            )
        rank_summary = self._planner_rank_stability_summary(self._planner_growth_history)
        self.logger.info(
            "[PlannerPolicyRankStability][Task %d] open_winners=%s open_runner_ups=%s avg_open_gap=%.4f expand_winners=%s expand_runner_ups=%s avg_expand_gap=%.4f",
            context["task_number"],
            rank_summary["open_winners"],
            rank_summary["open_runner_ups"],
            float(rank_summary["avg_open_gap_to_runner_up"]),
            rank_summary["expand_winners"],
            rank_summary["expand_runner_ups"],
            float(rank_summary["avg_expand_gap_to_runner_up"]),
        )

    def _log_stage14_policy_realization_audit(self, context: Dict[str, Any]) -> None:
        if not self._planner_audit_enabled():
            return
        growth_summary = context.get("planner_growth_task_summary", {})
        lifecycle = context.get("stage5_slot_lifecycle", {})
        materialized_lifecycle = lifecycle.get("materialized", {})
        applied_lifecycle = lifecycle.get("applied", {})
        post_profile = lifecycle.get("PostCHUProfile", {})
        if not growth_summary or not post_profile:
            return
        for block_id in self.model.selected_blocks:
            block_id = int(block_id)
            layer_outcome = growth_summary.get("layer_outcomes", {}).get(block_id, {})
            if not layer_outcome:
                continue
            materialized_summary = materialized_lifecycle.get(block_id, {})
            applied_summary = applied_lifecycle.get(block_id, {})
            post_summary = post_profile.get(block_id, {})
            requested_action = str(layer_outcome.get("requested_action", "n/a"))
            materialized_action = str(layer_outcome.get("materialized_action", "n/a"))
            applied_action = str(layer_outcome.get("applied_action", materialized_action))
            requested_growth = bool(self._requested_growth(requested_action))
            materialized_growth = bool(layer_outcome.get("opened", False) or layer_outcome.get("expanded", False))
            applied_growth = bool(materialized_growth or layer_outcome.get("applied_created_new_slot", False))
            materialized_fallback = str(materialized_summary.get("fallback_reason", "none"))
            applied_fallback = str(applied_summary.get("fallback_reason", materialized_fallback))
            fallback_reason = applied_fallback if applied_fallback != "none" else materialized_fallback
            post_outcome = self._planner_post_task_outcome(
                requested_growth=requested_growth,
                materialized_growth=materialized_growth,
                applied_growth=applied_growth,
                post_shared_only=bool(post_summary.get("shared_only", False)),
            )
            realization_entry = {
                "task_number": int(context["task_number"]),
                "block_id": block_id,
                "requested_action": requested_action,
                "materialized_action": materialized_action,
                "applied_action": applied_action,
                "requested_growth": requested_growth,
                "materialized_growth": materialized_growth,
                "applied_growth": applied_growth,
                "materialized_fallback_reason": materialized_fallback,
                "applied_fallback_reason": applied_fallback,
                "fallback_reason": fallback_reason,
                "post_outcome": post_outcome,
                "post_shared_only": bool(post_summary.get("shared_only", False)),
                "applied_live_count": len(applied_summary.get("live_slot_ids", [])),
                "post_live_count": len(post_summary.get("live_slot_ids", [])),
            }
            self._planner_realization_history.setdefault(block_id, []).append(realization_entry)
            self.logger.info(
                "[PlannerRealizationTrace][Task %d][Layer %d] requested=%s materialized=%s applied=%s requested_growth=%s materialized_growth=%s applied_growth=%s materialized_fallback=%s applied_fallback=%s post_outcome=%s applied_live=%d post_live=%d post_shared_only=%s",
                context["task_number"],
                block_id,
                requested_action,
                materialized_action,
                applied_action,
                requested_growth,
                materialized_growth,
                applied_growth,
                materialized_fallback,
                applied_fallback,
                post_outcome,
                int(realization_entry["applied_live_count"]),
                int(realization_entry["post_live_count"]),
                bool(realization_entry["post_shared_only"]),
            )
        comparison_groups = [
            ("Layer6", [6]),
            ("Layer9", [9]),
            ("Layers7_8_10_11", [7, 8, 10, 11]),
        ]
        for label, block_ids in comparison_groups:
            available_block_ids = [int(block_id) for block_id in block_ids if int(block_id) in self.model.selected_blocks]
            if not available_block_ids:
                continue
            summary = self._planner_realization_trace_summary(available_block_ids)
            threshold_summary = summary["threshold_summary"]
            history_summary = summary["history_summary"]
            self.logger.info(
                "[PlannerPolicyDeconcentration][Task %d][Group %s] novelty_margin_mean=%.4f conflict_margin_mean=%.4f requested_growth_freq=%.4f near_novelty_miss_freq=%.4f near_conflict_miss_freq=%.4f far_below_both_freq=%.4f history_used_freq=%.4f history_entropy_mean=%.4f history_max_weight_mean=%.4f novelty_bias_share_mean=%.4f conflict_bias_share_mean=%.4f",
                context["task_number"],
                label,
                float(summary["novelty_margin_mean"]),
                float(summary["conflict_margin_mean"]),
                float(threshold_summary["requested_growth_frequency"]),
                float(threshold_summary["near_novelty_miss_frequency"]),
                float(threshold_summary["near_conflict_miss_frequency"]),
                float(threshold_summary["far_below_both_frequency"]),
                float(history_summary["history_used_frequency"]),
                float(history_summary["history_entropy_mean"]),
                float(history_summary["history_max_weight_mean"]),
                float(summary["novelty_bias_share_mean"]),
                float(summary["conflict_bias_share_mean"]),
            )
            self.logger.info(
                "[PlannerRequestedAppliedTrace][Task %d][Group %s] requested_growth_freq=%.4f materialized_growth_freq=%.4f applied_growth_freq=%.4f main_fallbacks=%s final_shared_only_layers=%s final_non_shared_layers=%s final_outcomes=%s",
                context["task_number"],
                label,
                float(summary["requested_growth_frequency"]),
                float(summary["materialized_growth_frequency"]),
                float(summary["applied_growth_frequency"]),
                summary["top_fallbacks"],
                summary["final_shared_only_layers"],
                summary["final_non_shared_layers"],
                summary["final_outcomes_by_layer"],
            )

    def _log_stage13_structural_shared_only_audit(
        self,
        context: Dict[str, Any],
        _usage_stats: Dict[int, Dict[int, float]],
    ) -> None:
        if not self._planner_audit_enabled():
            return
        lifecycle = context.get("stage5_slot_lifecycle", {})
        pre_profile = lifecycle.get("PreCHUProfile", {})
        post_profile = lifecycle.get("PostCHUProfile", {})
        if not pre_profile or not post_profile:
            return
        route_accumulator = context.get("planner_structure_route_accumulator", {})
        growth_summary = context.get("planner_growth_task_summary", {})
        contribution_accumulator = context.get("planner_control_contribution_epoch_accumulator", {})
        shared_only_layers_post: List[int] = []
        multi_slot_layers_post: List[int] = []
        contracted_layers: List[int] = []
        for block_id in self.model.selected_blocks:
            block_id = int(block_id)
            pre_summary = pre_profile.get(block_id)
            post_summary = post_profile.get(block_id)
            if not isinstance(pre_summary, dict) or not isinstance(post_summary, dict):
                continue
            route_summary = self._route_usage_concentration_summary(
                route_accumulator.get(block_id, {}),
                lifecycle_summary=post_summary,
            )
            profile_diff = self._profile_contraction_diff(pre_summary, post_summary)
            layer_outcome = growth_summary.get("layer_outcomes", {}).get(block_id, {})
            layer_entry = {
                "task_number": int(context["task_number"]),
                "opened": bool(layer_outcome.get("opened", False)),
                "expanded": bool(layer_outcome.get("expanded", False)),
                "pre_live_count": len(pre_summary.get("live_slot_ids", [])),
                "post_live_count": len(post_summary.get("live_slot_ids", [])),
                "post_retained_count": len(post_summary.get("retained_slot_ids", [])),
                "post_shared_only": bool(post_summary.get("shared_only", False)),
            }
            self._planner_layer_structure_history.setdefault(block_id, []).append(layer_entry)
            if bool(post_summary.get("shared_only", False)):
                shared_only_layers_post.append(block_id)
            if len(post_summary.get("live_slot_ids", [])) > 1:
                multi_slot_layers_post.append(block_id)
            if bool(profile_diff["contracted"]):
                contracted_layers.append(block_id)
            self.logger.info(
                "[PlannerStructureTask][Task %d][Layer %d] requested_action=%s materialized_action=%s opened=%s expanded=%s pre_live=%d post_live=%d pre_retained=%d post_retained=%d pre_shared_only=%s post_shared_only=%s train_candidate_count_mean=%.2f train_selected_count_mean=%.2f train_nonzero_usage_slot_count_mean=%.2f usage_top1_share_mean=%.4f aggregated_usage_top1_share=%.4f usage_ema_top1_share=%.4f cumulative_usage_top1_share=%.4f candidate_but_zero_usage=%s",
                context["task_number"],
                block_id,
                layer_outcome.get("requested_action", "n/a"),
                layer_outcome.get("materialized_action", "n/a"),
                bool(layer_outcome.get("opened", False)),
                bool(layer_outcome.get("expanded", False)),
                len(pre_summary.get("live_slot_ids", [])),
                len(post_summary.get("live_slot_ids", [])),
                len(pre_summary.get("retained_slot_ids", [])),
                len(post_summary.get("retained_slot_ids", [])),
                bool(pre_summary.get("shared_only", False)),
                bool(post_summary.get("shared_only", False)),
                float(route_summary["train_candidate_count_mean"]),
                float(route_summary["train_selected_count_mean"]),
                float(route_summary["train_nonzero_usage_slot_count_mean"]),
                float(route_summary["usage_top1_share_mean"]),
                float(route_summary["aggregated_usage_top1_share"]),
                float(route_summary["usage_ema_top1_share"]),
                float(route_summary["cumulative_usage_top1_share"]),
                bool(route_summary["candidate_but_zero_usage"]),
            )
            self.logger.info(
                "[PlannerCHUDiff][Task %d][Layer %d] live_delta=%d retained_delta=%d candidate_delta=%d frozen_delta=%d pruned_delta=%d shared_only_changed=%s contracted=%s",
                context["task_number"],
                block_id,
                int(profile_diff["live_delta"]),
                int(profile_diff["retained_delta"]),
                int(profile_diff["candidate_delta"]),
                int(profile_diff["frozen_delta"]),
                int(profile_diff["pruned_delta"]),
                bool(profile_diff["shared_only_changed"]),
                bool(profile_diff["contracted"]),
            )
            history_summary = self._aggregate_layer_lifecycle_history(
                self._planner_layer_structure_history.get(block_id, [])
            )
            self.logger.info(
                "[PlannerStructureTrajectory][Layer %d] tasks=%s post_live=%s post_retained=%s post_shared_only=%s opened_tasks=%s expanded_tasks=%s never_grew=%s ever_multi_slot=%s opened_then_shared_only=%s collapsed_after_growth_tasks=%s",
                block_id,
                history_summary["tasks"],
                history_summary["post_live_counts"],
                history_summary["post_retained_counts"],
                history_summary["post_shared_only_flags"],
                history_summary["opened_tasks"],
                history_summary["expanded_tasks"],
                bool(history_summary["never_grew"]),
                bool(history_summary["ever_multi_slot"]),
                bool(history_summary["opened_then_shared_only"]),
                history_summary["collapsed_after_growth_tasks"],
            )
        focus_block_id = 6 if 6 in self.model.selected_blocks else int(self.model.selected_blocks[0])
        focus_pre = pre_profile.get(focus_block_id, {})
        focus_post = post_profile.get(focus_block_id, {})
        focus_route_summary = self._route_usage_concentration_summary(
            route_accumulator.get(focus_block_id, {}),
            lifecycle_summary=focus_post,
        )
        focus_outcome = growth_summary.get("layer_outcomes", {}).get(focus_block_id, {})
        focus_contribution_ratio = self._final_epoch_contribution_ratio(contribution_accumulator, focus_block_id)
        self.logger.info(
            "[PlannerLayer6Audit][Task %d] policy_open_rank=%s policy_expand_rank=%s opened=%s expanded=%s pre_live=%d post_live=%d pre_shared_only=%s post_shared_only=%s train_candidate_count_mean=%.2f train_selected_count_mean=%.2f usage_ema_top1_share=%.4f cumulative_usage_top1_share=%.4f final_epoch_shared_to_slot_ratio=%.4e profile_candidates_pre=%d profile_candidates_post=%d",
            context["task_number"],
            "n/a" if growth_summary.get("layer6_open_rank") is None else int(growth_summary["layer6_open_rank"]),
            "n/a" if growth_summary.get("layer6_expand_rank") is None else int(growth_summary["layer6_expand_rank"]),
            bool(focus_outcome.get("opened", False)),
            bool(focus_outcome.get("expanded", False)),
            len(focus_pre.get("live_slot_ids", [])),
            len(focus_post.get("live_slot_ids", [])),
            bool(focus_pre.get("shared_only", False)),
            bool(focus_post.get("shared_only", False)),
            float(focus_route_summary["train_candidate_count_mean"]),
            float(focus_route_summary["train_selected_count_mean"]),
            float(focus_route_summary["usage_ema_top1_share"]),
            float(focus_route_summary["cumulative_usage_top1_share"]),
            float(focus_contribution_ratio),
            len(focus_pre.get("candidate_slot_ids", [])),
            len(focus_post.get("candidate_slot_ids", [])),
        )
        self.logger.info(
            "[PlannerStructuralConcentration][Task %d] shared_only_layers_post=%s multi_slot_layers_post=%s contracted_layers=%s",
            context["task_number"],
            sorted(shared_only_layers_post),
            sorted(multi_slot_layers_post),
            sorted(contracted_layers),
        )
        self._log_stage14_policy_realization_audit(context)

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
        policy_inputs = {
            int(block_id): signals.planner_input.detach().float().cpu()
            for block_id, signals in raw_planner.items()
            if signals.planner_input is not None
        }
        policy_representations = {
            int(block_id): signals.planner_representation.detach().float().cpu()
            for block_id, signals in raw_planner.items()
            if signals.planner_representation is not None
        }
        policy_input_summary = self._planner_vector_collection_summary(policy_inputs)
        policy_representation_summary = self._planner_vector_collection_summary(policy_representations)
        context["planner_policy_input_summary"] = {
            "inputs": policy_input_summary,
            "representations": policy_representation_summary,
        }
        task_context_stats = context.get("planner_control_task_context_stats", {"norm": 0.0, "var": 0.0})
        self.logger.info(
            "[PlannerPolicyInputs][Task %d] task_context_norm=%.4e task_context_var=%.4e input_norm_mean=%.4e input_var_mean=%.4e input_pairwise_cosine_mean=%.4f input_pairwise_cosine_min=%.4f input_pairwise_cosine_max=%.4f representation_norm_mean=%.4e representation_var_mean=%.4e representation_pairwise_cosine_mean=%.4f representation_pairwise_cosine_min=%.4f representation_pairwise_cosine_max=%.4f",
            context["task_number"],
            float(task_context_stats.get("norm", 0.0)),
            float(task_context_stats.get("var", 0.0)),
            float(policy_input_summary["norm_mean"]),
            float(policy_input_summary["var_mean"]),
            float(policy_input_summary["pairwise_cosine_mean"]),
            float(policy_input_summary["pairwise_cosine_min"]),
            float(policy_input_summary["pairwise_cosine_max"]),
            float(policy_representation_summary["norm_mean"]),
            float(policy_representation_summary["var_mean"]),
            float(policy_representation_summary["pairwise_cosine_mean"]),
            float(policy_representation_summary["pairwise_cosine_min"]),
            float(policy_representation_summary["pairwise_cosine_max"]),
        )
        self._log_planner_layer_focus(context)
        self._log_stage13_policy_growth_summary(context)
        self._log_stage14_policy_signal_audit(context)

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
            "[HybridPlannerConfig][Task %d] planner_mode=%s policy_trainable=%s control_trainable=%s control_recompute=%s learned_shared_gate=%s delta_logit_scale=%.4f soft_rank_enabled=%s soft_rank_temperature=%s hard_rank_eval=%s",
            context["task_number"],
            self._planner_mode(),
            self._planner_policy_trainable(),
            self._planner_control_trainable(),
            self._planner_control_recompute_mode(),
            self._planner_use_learned_shared_gate(),
            self._planner_control_delta_logit_scale(),
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

    def _record_planner_control_forward_stats(
        self,
        context: Dict[str, Any],
        control_outputs_by_block: Dict[int, PlannerControlOutputs],
    ) -> None:
        accumulator = context.setdefault(
            "planner_control_epoch_accumulator",
            self._new_planner_control_epoch_accumulator(),
        )
        input_vectors: Dict[int, torch.Tensor] = {}
        representation_vectors: Dict[int, torch.Tensor] = {}
        for block_id, outputs in control_outputs_by_block.items():
            layer_stats = self._planner_control_layer_accumulator(accumulator, int(block_id))
            layer_stats["beta_values"].append(float(outputs.shared_gate.detach().mean().item()))
            layer_stats["beta_logits"].append(float(outputs.shared_gate_logit.detach().mean().item()))
            if outputs.anchor_beta is not None:
                layer_stats["anchor_beta_values"].append(float(outputs.anchor_beta.detach().mean().item()))
                layer_stats["beta_gap_abs"].append(
                    float((outputs.shared_gate.detach() - outputs.anchor_beta.detach()).abs().mean().item())
                )
            if outputs.anchor_logit is not None:
                layer_stats["anchor_logit_values"].append(float(outputs.anchor_logit.detach().mean().item()))
                layer_stats["logit_gap_abs"].append(
                    float((outputs.shared_gate_logit.detach() - outputs.anchor_logit.detach()).abs().mean().item())
                )
            layer_stats["delta_raw_values"].append(float(outputs.delta_raw.detach().mean().item()))
            layer_stats["delta_bound_derivative_values"].append(
                float(self._bounded_residual_derivative(outputs.delta_raw.detach()).mean().item())
            )
            if outputs.delta_logit is not None:
                layer_stats["delta_logit_values"].append(float(outputs.delta_logit.detach().mean().item()))
            if outputs.delta_from_representation is not None:
                delta_from_representation = outputs.delta_from_representation.detach()
                layer_stats["delta_from_representation_values"].append(float(delta_from_representation.mean().item()))
                layer_stats["delta_from_representation_abs_values"].append(
                    float(delta_from_representation.abs().mean().item())
                )
            if outputs.delta_bias is not None:
                delta_bias = outputs.delta_bias.detach()
                bias_abs_mean = float(delta_bias.abs().mean().item())
                activation_abs_mean = 0.0
                if outputs.delta_from_representation is not None:
                    activation_abs_mean = float(outputs.delta_from_representation.detach().abs().mean().item())
                layer_stats["delta_bias_values"].append(float(delta_bias.mean().item()))
                layer_stats["delta_bias_abs_values"].append(bias_abs_mean)
                layer_stats["delta_bias_share_values"].append(
                    bias_abs_mean / max(bias_abs_mean + activation_abs_mean, 1e-12)
                )
            head_summary = self._planner_control_head_summary(int(block_id))
            layer_stats["head_weight_norms"].append(float(head_summary["weight_norm"]))
            layer_stats["head_bias_norms"].append(float(head_summary["bias_norm"]))
            layer_stats["head_weight_drift_from_init"].append(float(head_summary["weight_l2_from_init"]))
            layer_stats["head_bias_drift_from_init"].append(float(head_summary["bias_l2_from_init"]))
            if outputs.planner_input is not None:
                planner_input = outputs.planner_input.detach().float().reshape(-1).cpu()
                layer_stats["input_norms"].append(float(planner_input.norm().item()))
                layer_stats["input_vars"].append(
                    float(planner_input.var(unbiased=False).item()) if planner_input.numel() > 1 else 0.0
                )
                input_vectors[int(block_id)] = planner_input
            if outputs.planner_representation is not None:
                planner_representation = outputs.planner_representation.detach().float().reshape(-1).cpu()
                layer_stats["representation_norms"].append(float(planner_representation.norm().item()))
                layer_stats["representation_vars"].append(
                    float(planner_representation.var(unbiased=False).item()) if planner_representation.numel() > 1 else 0.0
                )
                representation_vectors[int(block_id)] = planner_representation
            if outputs.normalized_planner_representation is not None:
                normalized_planner_representation = (
                    outputs.normalized_planner_representation.detach().float().reshape(-1).cpu()
                )
                layer_stats["normalized_representation_norms"].append(
                    float(normalized_planner_representation.norm().item())
                )
                layer_stats["normalized_representation_vars"].append(
                    float(normalized_planner_representation.var(unbiased=False).item())
                    if normalized_planner_representation.numel() > 1
                    else 0.0
                )
        if self._planner_control_saturation_audit_enabled():
            input_pairwise = self._pairwise_cosine_summary(input_vectors)
            representation_pairwise = self._pairwise_cosine_summary(representation_vectors)
            accumulator.setdefault("input_pairwise_cosine", []).append(float(input_pairwise["mean"]))
            accumulator.setdefault("representation_pairwise_cosine", []).append(float(representation_pairwise["mean"]))

    def _accumulate_planner_control_pre_step(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        accumulator = context.setdefault(
            "planner_control_epoch_accumulator",
            self._new_planner_control_epoch_accumulator(),
        )
        stats = self._compute_module_grad_stats(self.planner.control_branch)
        accumulator["batches"] += 1
        accumulator["grad_l2_sum"] += float(stats["l2"])
        accumulator["grad_l2_max"] = max(float(accumulator["grad_l2_max"]), float(stats["max_abs"]))
        accumulator["grad_present_batches"] += int(int(stats["present_params"]) > 0)
        accumulator["grad_nonzero_batches"] += int(int(stats["nonzero_params"]) > 0)
        if not self._planner_control_saturation_audit_enabled():
            return
        control_outputs = context.get("planner_control_last_outputs", {})
        grad_zero_epsilon = 1e-12
        for block_id, outputs in control_outputs.items():
            layer_stats = self._planner_control_layer_accumulator(accumulator, int(block_id))
            beta_grad = outputs.shared_gate.grad
            beta_grad_abs = 0.0 if beta_grad is None else abs(float(beta_grad.detach().mean().item()))
            layer_stats["beta_grad_abs"].append(beta_grad_abs)
            layer_stats["beta_grad_present_batches"] += int(beta_grad is not None)
            layer_stats["beta_grad_nonzero_batches"] += int(beta_grad_abs > grad_zero_epsilon)
            logit_grad = outputs.shared_gate_logit.grad
            logit_grad_abs = 0.0 if logit_grad is None else abs(float(logit_grad.detach().mean().item()))
            layer_stats["logit_grad_abs"].append(logit_grad_abs)
            layer_stats["logit_grad_present_batches"] += int(logit_grad is not None)
            layer_stats["logit_grad_nonzero_batches"] += int(logit_grad_abs > grad_zero_epsilon)
            delta_raw_grad = outputs.delta_raw.grad
            delta_raw_grad_abs = 0.0 if delta_raw_grad is None else abs(float(delta_raw_grad.detach().mean().item()))
            layer_stats["delta_raw_grad_abs"].append(delta_raw_grad_abs)
            layer_stats["delta_raw_grad_present_batches"] += int(delta_raw_grad is not None)
            layer_stats["delta_raw_grad_nonzero_batches"] += int(delta_raw_grad_abs > grad_zero_epsilon)
            delta_logit_grad = None if outputs.delta_logit is None else outputs.delta_logit.grad
            delta_logit_grad_abs = (
                0.0 if delta_logit_grad is None else abs(float(delta_logit_grad.detach().mean().item()))
            )
            layer_stats["delta_logit_grad_abs"].append(delta_logit_grad_abs)
            layer_stats["delta_logit_grad_present_batches"] += int(delta_logit_grad is not None)
            layer_stats["delta_logit_grad_nonzero_batches"] += int(delta_logit_grad_abs > grad_zero_epsilon)
            layer_stats["bridge_grad_lost_batches"] += int(
                delta_logit_grad_abs > grad_zero_epsilon and delta_raw_grad_abs <= grad_zero_epsilon
            )
            layer_stats["loss_output_zero_batches"] += int(logit_grad_abs <= grad_zero_epsilon)

    def _record_planner_control_optimizer_step(self, context: Dict[str, Any]) -> None:
        if not self._hybrid_planner_enabled():
            return
        accumulator = context.setdefault(
            "planner_control_epoch_accumulator",
            self._new_planner_control_epoch_accumulator(),
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
        if not self._planner_control_saturation_audit_enabled():
            for block_id in sorted(accumulator.get("layers", {})):
                layer_stats = accumulator["layers"][block_id]
                beta_summary = self._scalar_series_summary(layer_stats.get("beta_values", []))
                self.logger.info(
                    "[PlannerControlValues][Task %d][Epoch %d][Layer %d] beta_mean=%.4f beta_min=%.4f beta_max=%.4f",
                    context["task_number"],
                    int(context["current_epoch"]),
                    int(block_id),
                    float(beta_summary["mean"]),
                    float(beta_summary["min"]),
                    float(beta_summary["max"]),
                )
            return
        task_start = context.get("planner_control_task_start", {})
        contribution_accumulator = context.get("planner_control_contribution_epoch_accumulator", {})
        logit_threshold_099 = math.log(0.99 / 0.01)
        logit_threshold_0999 = math.log(0.999 / 0.001)
        task_context_stats = context.get("planner_control_task_context_stats", {"norm": 0.0, "var": 0.0})
        policy_input_summary = context.get("planner_policy_input_summary", {})
        control_input_pairwise = self._scalar_series_summary(accumulator.get("input_pairwise_cosine", []))
        control_rep_pairwise = self._scalar_series_summary(accumulator.get("representation_pairwise_cosine", []))
        self.logger.info(
            "[PlannerControlInputCompare][Task %d][Epoch %d] task_context_norm=%.4e task_context_var=%.4e control_input_pairwise_cosine_mean=%.4f control_input_pairwise_cosine_min=%.4f control_input_pairwise_cosine_max=%.4f control_representation_pairwise_cosine_mean=%.4f control_representation_pairwise_cosine_min=%.4f control_representation_pairwise_cosine_max=%.4f policy_input_pairwise_cosine_mean=%s policy_representation_pairwise_cosine_mean=%s",
            context["task_number"],
            int(context["current_epoch"]),
            float(task_context_stats.get("norm", 0.0)),
            float(task_context_stats.get("var", 0.0)),
            float(control_input_pairwise["mean"]),
            float(control_input_pairwise["min"]),
            float(control_input_pairwise["max"]),
            float(control_rep_pairwise["mean"]),
            float(control_rep_pairwise["min"]),
            float(control_rep_pairwise["max"]),
            "n/a"
            if not policy_input_summary
            else f"{float(policy_input_summary['inputs']['pairwise_cosine_mean']):.4f}",
            "n/a"
            if not policy_input_summary
            else f"{float(policy_input_summary['representations']['pairwise_cosine_mean']):.4f}",
        )
        for block_id in sorted(accumulator.get("layers", {})):
            layer_stats = accumulator["layers"][block_id]
            beta_summary = self._scalar_series_summary(layer_stats.get("beta_values", []))
            logit_summary = self._scalar_series_summary(layer_stats.get("beta_logits", []))
            anchor_beta_summary = self._scalar_series_summary(layer_stats.get("anchor_beta_values", []))
            anchor_logit_summary = self._scalar_series_summary(layer_stats.get("anchor_logit_values", []))
            delta_raw_summary = self._scalar_series_summary(layer_stats.get("delta_raw_values", []))
            delta_logit_summary = self._scalar_series_summary(layer_stats.get("delta_logit_values", []))
            delta_bound_derivative_summary = self._scalar_series_summary(layer_stats.get("delta_bound_derivative_values", []))
            beta_gap_summary = self._scalar_series_summary(layer_stats.get("beta_gap_abs", []))
            logit_gap_summary = self._scalar_series_summary(layer_stats.get("logit_gap_abs", []))
            beta_grad_summary = self._scalar_series_summary(layer_stats.get("beta_grad_abs", []))
            logit_grad_summary = self._scalar_series_summary(layer_stats.get("logit_grad_abs", []))
            delta_raw_grad_summary = self._scalar_series_summary(layer_stats.get("delta_raw_grad_abs", []))
            delta_logit_grad_summary = self._scalar_series_summary(layer_stats.get("delta_logit_grad_abs", []))
            head_weight_norm_summary = self._scalar_series_summary(layer_stats.get("head_weight_norms", []))
            head_bias_norm_summary = self._scalar_series_summary(layer_stats.get("head_bias_norms", []))
            head_weight_drift_summary = self._scalar_series_summary(layer_stats.get("head_weight_drift_from_init", []))
            head_bias_drift_summary = self._scalar_series_summary(layer_stats.get("head_bias_drift_from_init", []))
            delta_from_representation_summary = self._scalar_series_summary(
                layer_stats.get("delta_from_representation_values", [])
            )
            delta_from_representation_abs_summary = self._scalar_series_summary(
                layer_stats.get("delta_from_representation_abs_values", [])
            )
            delta_bias_summary = self._scalar_series_summary(layer_stats.get("delta_bias_values", []))
            delta_bias_abs_summary = self._scalar_series_summary(layer_stats.get("delta_bias_abs_values", []))
            delta_bias_share_summary = self._scalar_series_summary(layer_stats.get("delta_bias_share_values", []))
            task_start_stats = task_start.get(int(block_id), {})
            task_start_beta = float(task_start_stats.get("beta", 0.0))
            task_start_logit = float(task_start_stats.get("logit", 0.0))
            task_start_anchor_beta = float(task_start_stats.get("anchor_beta", 0.0))
            task_start_anchor_logit = float(task_start_stats.get("anchor_logit", 0.0))
            task_start_delta_raw = float(task_start_stats.get("delta_raw", 0.0))
            task_start_delta_logit = float(task_start_stats.get("delta_logit", 0.0))
            delta_abs_values = [abs(float(value)) for value in layer_stats.get("delta_logit_values", [])]
            delta_scale = self._planner_control_delta_logit_scale()
            delta_cap_atol = 1e-3
            frac_near_pos_cap = self._fraction_within(layer_stats.get("delta_logit_values", []), delta_scale, delta_cap_atol)
            frac_near_neg_cap = self._fraction_within(layer_stats.get("delta_logit_values", []), -delta_scale, delta_cap_atol)
            frac_near_any_cap = self._fraction_within(delta_abs_values, delta_scale, delta_cap_atol)
            mean_abs_gap_to_cap = self._mean_abs_gap_to_target(delta_abs_values, delta_scale)
            if frac_near_any_cap >= 0.95:
                cap_state = "hard_pinned"
            elif abs(float(delta_logit_summary["mean"])) >= 0.75 * delta_scale:
                cap_state = "high_but_movable"
            else:
                cap_state = "bounded_active"
            delta_representation_correlation = self._paired_series_correlation(
                layer_stats.get("representation_norms", []),
                [abs(float(value)) for value in layer_stats.get("delta_raw_values", [])],
            )
            self.logger.info(
                "[PlannerControlAnchor][Task %d][Epoch %d][Layer %d] anchor_beta_mean=%.4f anchor_beta_min=%.4f anchor_beta_max=%.4f frac_anchor_gt_099=%.4f frac_anchor_gt_0999=%.4f anchor_logit_mean=%.4e anchor_logit_p90=%.4e anchor_logit_p99=%.4e task_start_anchor_beta=%.4f task_start_anchor_logit=%.4e",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(anchor_beta_summary["mean"]),
                float(anchor_beta_summary["min"]),
                float(anchor_beta_summary["max"]),
                self._fraction_above(layer_stats.get("anchor_beta_values", []), 0.99),
                self._fraction_above(layer_stats.get("anchor_beta_values", []), 0.999),
                float(anchor_logit_summary["mean"]),
                float(anchor_logit_summary["p90"]),
                float(anchor_logit_summary["p99"]),
                task_start_anchor_beta,
                task_start_anchor_logit,
            )
            self.logger.info(
                "[PlannerControlDelta][Task %d][Epoch %d][Layer %d] delta_raw_mean=%.4e delta_raw_min=%.4e delta_raw_max=%.4e delta_logit_mean=%.4e delta_logit_min=%.4e delta_logit_max=%.4e beta_anchor_gap_mean_abs=%.4e logit_anchor_gap_mean_abs=%.4e",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(delta_raw_summary["mean"]),
                float(delta_raw_summary["min"]),
                float(delta_raw_summary["max"]),
                float(delta_logit_summary["mean"]),
                float(delta_logit_summary["min"]),
                float(delta_logit_summary["max"]),
                float(beta_gap_summary["mean"]),
                float(logit_gap_summary["mean"]),
            )
            self.logger.info(
                "[PlannerResidualRaw][Task %d][Epoch %d][Layer %d] delta_raw_mean=%.4e delta_raw_min=%.4e delta_raw_max=%.4e delta_raw_p50=%.4e delta_raw_p90=%.4e delta_raw_p99=%.4e frac_abs_gt_2=%.4f frac_abs_gt_4=%.4f frac_abs_gt_6=%.4f bound_derivative_mean=%.4e bound_derivative_min=%.4e bound_derivative_max=%.4e epoch_delta_raw_drift=%.4e task_delta_raw_drift=%.4e task_start_delta_raw=%.4e task_start_delta_logit=%.4e",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(delta_raw_summary["mean"]),
                float(delta_raw_summary["min"]),
                float(delta_raw_summary["max"]),
                float(delta_raw_summary["p50"]),
                float(delta_raw_summary["p90"]),
                float(delta_raw_summary["p99"]),
                self._fraction_abs_above(layer_stats.get("delta_raw_values", []), 2.0),
                self._fraction_abs_above(layer_stats.get("delta_raw_values", []), 4.0),
                self._fraction_abs_above(layer_stats.get("delta_raw_values", []), 6.0),
                float(delta_bound_derivative_summary["mean"]),
                float(delta_bound_derivative_summary["min"]),
                float(delta_bound_derivative_summary["max"]),
                float(delta_raw_summary["last"]) - float(delta_raw_summary["first"]),
                float(delta_raw_summary["mean"]) - task_start_delta_raw,
                task_start_delta_raw,
                task_start_delta_logit,
            )
            self.logger.info(
                "[PlannerControlLogits][Task %d][Epoch %d][Layer %d] logit_mean=%.4e logit_min=%.4e logit_max=%.4e logit_p50=%.4e logit_p90=%.4e logit_p99=%.4e frac_gt_0=%.4f frac_gt_2=%.4f frac_gt_logit099=%.4f frac_gt_logit0999=%.4f epoch_logit_drift=%.4e task_logit_drift=%.4e task_start_logit=%.4e",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(logit_summary["mean"]),
                float(logit_summary["min"]),
                float(logit_summary["max"]),
                float(logit_summary["p50"]),
                float(logit_summary["p90"]),
                float(logit_summary["p99"]),
                self._fraction_above(layer_stats.get("beta_logits", []), 0.0),
                self._fraction_above(layer_stats.get("beta_logits", []), 2.0),
                self._fraction_above(layer_stats.get("beta_logits", []), logit_threshold_099),
                self._fraction_above(layer_stats.get("beta_logits", []), logit_threshold_0999),
                float(logit_summary["last"]) - float(logit_summary["first"]),
                float(logit_summary["mean"]) - task_start_logit,
                task_start_logit,
            )
            self.logger.info(
                "[PlannerControlValues][Task %d][Epoch %d][Layer %d] beta_mean=%.4f beta_min=%.4f beta_max=%.4f beta_p50=%.4f beta_p90=%.4f beta_p99=%.4f frac_gt_099=%.4f frac_gt_0999=%.4f frac_lt_001=%.4f epoch_beta_drift=%.4e task_beta_drift=%.4e task_start_beta=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(beta_summary["mean"]),
                float(beta_summary["min"]),
                float(beta_summary["max"]),
                float(beta_summary["p50"]),
                float(beta_summary["p90"]),
                float(beta_summary["p99"]),
                self._fraction_above(layer_stats.get("beta_values", []), 0.99),
                self._fraction_above(layer_stats.get("beta_values", []), 0.999),
                self._fraction_below(layer_stats.get("beta_values", []), 0.01),
                float(beta_summary["last"]) - float(beta_summary["first"]),
                float(beta_summary["mean"]) - task_start_beta,
                task_start_beta,
            )
            self.logger.info(
                "[PlannerControlGradients][Task %d][Epoch %d][Layer %d] beta_grad_mean=%.4e beta_grad_max=%.4e beta_grad_present_batches=%d/%d beta_grad_nonzero_batches=%d/%d beta_grad_effectively_zero_fraction=%.4f logit_grad_mean=%.4e logit_grad_max=%.4e logit_grad_present_batches=%d/%d logit_grad_nonzero_batches=%d/%d logit_grad_effectively_zero_fraction=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(beta_grad_summary["mean"]),
                float(beta_grad_summary["max"]),
                int(layer_stats.get("beta_grad_present_batches", 0)),
                max(int(beta_summary["count"]), 1),
                int(layer_stats.get("beta_grad_nonzero_batches", 0)),
                max(int(beta_summary["count"]), 1),
                self._fraction_below(layer_stats.get("beta_grad_abs", []), 1e-12),
                float(logit_grad_summary["mean"]),
                float(logit_grad_summary["max"]),
                int(layer_stats.get("logit_grad_present_batches", 0)),
                max(int(logit_summary["count"]), 1),
                int(layer_stats.get("logit_grad_nonzero_batches", 0)),
                max(int(logit_summary["count"]), 1),
                self._fraction_below(layer_stats.get("logit_grad_abs", []), 1e-12),
            )
            self.logger.info(
                "[PlannerResidualGradients][Task %d][Epoch %d][Layer %d] delta_raw_grad_mean=%.4e delta_raw_grad_max=%.4e delta_raw_grad_present_batches=%d/%d delta_raw_grad_nonzero_batches=%d/%d delta_logit_grad_mean=%.4e delta_logit_grad_max=%.4e delta_logit_grad_present_batches=%d/%d delta_logit_grad_nonzero_batches=%d/%d bridge_grad_lost_fraction=%.4f loss_output_zero_fraction=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(delta_raw_grad_summary["mean"]),
                float(delta_raw_grad_summary["max"]),
                int(layer_stats.get("delta_raw_grad_present_batches", 0)),
                max(int(delta_raw_grad_summary["count"]), 1),
                int(layer_stats.get("delta_raw_grad_nonzero_batches", 0)),
                max(int(delta_raw_grad_summary["count"]), 1),
                float(delta_logit_grad_summary["mean"]),
                float(delta_logit_grad_summary["max"]),
                int(layer_stats.get("delta_logit_grad_present_batches", 0)),
                max(int(delta_logit_grad_summary["count"]), 1),
                int(layer_stats.get("delta_logit_grad_nonzero_batches", 0)),
                max(int(delta_logit_grad_summary["count"]), 1),
                float(layer_stats.get("bridge_grad_lost_batches", 0)) / max(int(delta_raw_grad_summary["count"]), 1),
                float(layer_stats.get("loss_output_zero_batches", 0)) / max(int(logit_summary["count"]), 1),
            )
            input_norm_summary = self._scalar_series_summary(layer_stats.get("input_norms", []))
            input_var_summary = self._scalar_series_summary(layer_stats.get("input_vars", []))
            representation_norm_summary = self._scalar_series_summary(layer_stats.get("representation_norms", []))
            representation_var_summary = self._scalar_series_summary(layer_stats.get("representation_vars", []))
            normalized_representation_norm_summary = self._scalar_series_summary(
                layer_stats.get("normalized_representation_norms", [])
            )
            normalized_representation_var_summary = self._scalar_series_summary(
                layer_stats.get("normalized_representation_vars", [])
            )
            policy_layer_input = None if not task_start_stats else task_start_stats.get("planner_input")
            latest_input_similarity_to_task_start = 0.0
            if policy_layer_input is not None and layer_stats.get("input_norms"):
                current_input = None
                control_outputs = context.get("planner_control_last_outputs", {})
                if int(block_id) in control_outputs and control_outputs[int(block_id)].planner_input is not None:
                    current_input = control_outputs[int(block_id)].planner_input.detach().float().reshape(-1).cpu()
                if current_input is not None:
                    current_norm = float(current_input.norm().item())
                    reference_input = policy_layer_input.detach().float().reshape(-1)
                    reference_norm = float(reference_input.norm().item())
                    if current_norm > 1e-12 and reference_norm > 1e-12:
                        latest_input_similarity_to_task_start = float(
                            torch.dot(current_input / current_norm, reference_input / reference_norm).item()
                        )
            self.logger.info(
                "[PlannerControlInputs][Task %d][Epoch %d][Layer %d] input_norm_mean=%.4e input_var_mean=%.4e representation_norm_mean=%.4e representation_var_mean=%.4e normalized_representation_norm_mean=%.4e normalized_representation_var_mean=%.4e input_similarity_to_task_start=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(input_norm_summary["mean"]),
                float(input_var_summary["mean"]),
                float(representation_norm_summary["mean"]),
                float(representation_var_summary["mean"]),
                float(normalized_representation_norm_summary["mean"]),
                float(normalized_representation_var_summary["mean"]),
                latest_input_similarity_to_task_start,
            )
            self.logger.info(
                "[PlannerControlHead][Task %d][Epoch %d][Layer %d] head_weight_norm_mean=%.4e head_bias_norm_mean=%.4e head_weight_l2_from_init_mean=%.4e head_bias_l2_from_init_mean=%.4e delta_from_representation_mean=%.4e delta_from_representation_abs_mean=%.4e delta_bias_mean=%.4e delta_bias_abs_mean=%.4e delta_bias_share_mean=%.4f delta_abs_vs_representation_norm_corr=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(head_weight_norm_summary["mean"]),
                float(head_bias_norm_summary["mean"]),
                float(head_weight_drift_summary["mean"]),
                float(head_bias_drift_summary["mean"]),
                float(delta_from_representation_summary["mean"]),
                float(delta_from_representation_abs_summary["mean"]),
                float(delta_bias_summary["mean"]),
                float(delta_bias_abs_summary["mean"]),
                float(delta_bias_share_summary["mean"]),
                delta_representation_correlation,
            )
            contribution_stats = contribution_accumulator.get(int(block_id), {})
            shared_pre_count = max(int(contribution_stats.get("shared_pre_beta_norm_count", 0)), 1)
            shared_post_count = max(int(contribution_stats.get("shared_post_beta_norm_count", 0)), 1)
            slot_count = max(int(contribution_stats.get("slot_norm_count", 0)), 1)
            structural_calls = max(int(contribution_stats.get("slot_structurally_available_calls", 0)), 1)
            mean_shared_pre = float(contribution_stats.get("shared_pre_beta_norm_sum", 0.0)) / shared_pre_count
            mean_shared_post = float(contribution_stats.get("shared_post_beta_norm_sum", 0.0)) / shared_post_count
            mean_slot = float(contribution_stats.get("slot_norm_sum", 0.0)) / slot_count
            shared_to_slot_ratio = mean_shared_post / max(mean_slot, 1e-12)
            self.logger.info(
                "[PlannerContribution][Task %d][Epoch %d][Layer %d] shared_pre_beta_norm_mean=%.4e shared_post_beta_norm_mean=%.4e slot_norm_mean=%.4e shared_to_slot_ratio=%.4e slot_structurally_available_fraction=%.4f slot_nontrivial_fraction=%.4f slot_zero_contribution_fraction=%.4f beta_gt_099_with_structural_slot_fraction=%.4f beta_gt_0999_with_structural_slot_fraction=%.4f",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                mean_shared_pre,
                mean_shared_post,
                mean_slot,
                shared_to_slot_ratio,
                float(contribution_stats.get("slot_structurally_available_calls", 0)) / max(slot_count, 1),
                float(contribution_stats.get("slot_nontrivial_calls", 0)) / structural_calls,
                float(contribution_stats.get("slot_zero_contribution_calls", 0)) / structural_calls,
                float(contribution_stats.get("beta_gt_099_with_structural_slot_calls", 0)) / structural_calls,
                float(contribution_stats.get("beta_gt_0999_with_structural_slot_calls", 0)) / structural_calls,
            )
            self.logger.info(
                "[PlannerResidualCap][Task %d][Epoch %d][Layer %d] delta_scale=%.4f frac_near_pos_cap=%.4f frac_near_neg_cap=%.4f frac_near_any_cap=%.4f mean_abs_gap_to_cap=%.4e cap_state=%s",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                delta_scale,
                frac_near_pos_cap,
                frac_near_neg_cap,
                frac_near_any_cap,
                mean_abs_gap_to_cap,
                cap_state,
            )
            self.logger.info(
                "[PlannerResidualSummary][Task %d][Epoch %d][Layer %d] anchor_beta_mean=%.4f effective_beta_mean=%.4f delta_logit_mean=%.4e cap_state=%s slot_structurally_available_fraction=%.4f shared_to_slot_ratio=%.4e layer6_focus=%s",
                context["task_number"],
                int(context["current_epoch"]),
                int(block_id),
                float(anchor_beta_summary["mean"]),
                float(beta_summary["mean"]),
                float(delta_logit_summary["mean"]),
                cap_state,
                float(contribution_stats.get("slot_structurally_available_calls", 0)) / max(slot_count, 1),
                shared_to_slot_ratio,
                "yes" if int(block_id) == 6 else "no",
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
        if not (self._routing_audit_enabled() or self._planner_audit_enabled()):
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
        adapter_delta_enabled = self._epoch_debug_enabled(
            context,
            flag_key="adapter_delta_debug_logging",
            max_epoch_key="adapter_delta_debug_max_epochs",
        )
        planner_control_contribution_enabled = self._planner_control_saturation_audit_enabled()
        if not adapter_delta_enabled and not planner_control_contribution_enabled:
            return plans
        accumulator = context.setdefault("adapter_delta_debug_accumulator", {})
        control_contribution_accumulator = context.setdefault("planner_control_contribution_epoch_accumulator", {})
        debug_plans = {}
        for block_id, planner_cfg in plans.items():
            layer_cfg = dict(planner_cfg)
            layer_cfg["_debug_block_id"] = int(block_id)
            if adapter_delta_enabled:
                layer_cfg["_debug_delta_stats"] = accumulator
            if planner_control_contribution_enabled:
                layer_cfg["_planner_control_contribution_accumulator"] = control_contribution_accumulator
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
            raw_outputs = self.planner.forward_control(
                int(block_id),
                task_embedding=task_embedding,
                history_summary=history_summary,
            )
            control_outputs = self._compose_hybrid_control_outputs(
                raw_outputs,
                anchor_beta=float(planner_cfg["shared_gate"]),
            )
            if self._planner_control_saturation_audit_enabled():
                if control_outputs.shared_gate.requires_grad:
                    control_outputs.shared_gate.retain_grad()
                if control_outputs.shared_gate_logit.requires_grad:
                    control_outputs.shared_gate_logit.retain_grad()
                if control_outputs.delta_raw.requires_grad:
                    control_outputs.delta_raw.retain_grad()
                if control_outputs.delta_logit is not None and control_outputs.delta_logit.requires_grad:
                    control_outputs.delta_logit.retain_grad()
            control_outputs_by_block[int(block_id)] = control_outputs
            if self._planner_use_learned_shared_gate():
                layer_cfg["shared_gate"] = control_outputs.shared_gate
            hybrid_plans[int(block_id)] = layer_cfg
        context["planner_control_last_outputs"] = control_outputs_by_block
        self._record_planner_control_forward_stats(context, control_outputs_by_block)
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
            "planner_policy": 0.0,
            "planner_control": 0.0,
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
        for parameter in self.planner.policy_parameters():
            if parameter.grad is not None:
                squared_sums["planner_policy"] += float(parameter.grad.detach().pow(2).sum().item())
        for parameter in self.planner.control_parameters():
            if parameter.grad is not None:
                squared_sums["planner_control"] += float(parameter.grad.detach().pow(2).sum().item())
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
            {
                group: {"sum": 0.0, "max": 0.0, "batches": 0}
                for group in ("shared", "slot", "router", "planner_policy", "planner_control", "planner", "classifier")
            },
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
        for group in ("shared", "slot", "router", "planner_policy", "planner_control", "planner", "classifier"):
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

    def _evaluate_task_retention_summary(
        self,
        *,
        current_task_index: int,
        eval_task_id: int,
        accuracy: float,
        sample_count: int,
        route_infos: List[Dict[int, Dict[str, object]]],
        teacher_old_logits: List[torch.Tensor],
        student_old_logits: List[torch.Tensor],
        student_new_logits: List[torch.Tensor],
        feature_diff_values: List[Dict[str, float]],
        layer_feature_summaries: Dict[int, List[Dict[str, float]]],
        old_num_classes: int,
    ) -> Dict[str, Any]:
        task_number = int(eval_task_id) + 1
        route_summary = self._route_profile_retention_summary(route_infos)
        old_logit_mean_abs_diff = 0.0
        old_logit_max_abs_diff = 0.0
        calibration = {
            "old_logit_norm_mean": 0.0,
            "new_logit_norm_mean": 0.0,
            "old_max_mean": 0.0,
            "new_max_mean": 0.0,
            "old_minus_new_mean": 0.0,
            "new_wins_ratio": 0.0,
            "old_weight_norm_mean": 0.0,
            "new_weight_norm_mean": 0.0,
            "teacher_old_logit_norm_mean": 0.0,
            "old_logit_compression_ratio": 0.0,
        }
        if old_num_classes > 0 and teacher_old_logits and student_old_logits:
            teacher_logits = torch.cat(teacher_old_logits, dim=0)
            student_logits = torch.cat(student_old_logits, dim=0)
            new_logits = torch.cat(student_new_logits, dim=0) if student_new_logits else None
            diff = (student_logits - teacher_logits).abs()
            old_logit_mean_abs_diff = float(diff.mean().item())
            old_logit_max_abs_diff = float(diff.max().item())
            classifier_weights = self.model.classifier.weight.detach()
            calibration = self._classifier_calibration_summary(
                old_logits=student_logits,
                new_logits=new_logits,
                old_classifier_weights=classifier_weights[:old_num_classes],
                new_classifier_weights=classifier_weights[old_num_classes:],
                teacher_old_logits=teacher_logits,
            )
        feature_summary = {
            "mean_abs_diff": 0.0,
            "max_abs_diff": 0.0,
            "mean_cosine": 0.0,
            "min_cosine": 0.0,
        }
        if feature_diff_values:
            for key in feature_summary:
                feature_summary[key] = float(
                    sum(float(item[key]) for item in feature_diff_values) / len(feature_diff_values)
                )
        per_layer_feature_summary: Dict[int, Dict[str, float]] = {}
        for block_id, values in layer_feature_summaries.items():
            if not values:
                continue
            per_layer_feature_summary[int(block_id)] = {
                "mean_abs_diff": float(sum(float(item["mean_abs_diff"]) for item in values) / len(values)),
                "max_abs_diff": float(max(float(item["max_abs_diff"]) for item in values)),
                "mean_cosine": float(sum(float(item["mean_cosine"]) for item in values) / len(values)),
                "min_cosine": float(min(float(item["min_cosine"]) for item in values)),
            }
        return {
            "task_number": task_number,
            "accuracy": float(accuracy),
            "sample_count": int(sample_count),
            "is_old_task": bool(old_num_classes > 0 and eval_task_id < int(current_task_index)),
            "old_logit_mean_abs_diff": old_logit_mean_abs_diff,
            "old_logit_max_abs_diff": old_logit_max_abs_diff,
            "feature_mean_abs_diff": float(feature_summary["mean_abs_diff"]),
            "feature_max_abs_diff": float(feature_summary["max_abs_diff"]),
            "feature_mean_cosine": float(feature_summary["mean_cosine"]),
            "feature_min_cosine": float(feature_summary["min_cosine"]),
            "classifier_calibration": calibration,
            "route_summary": route_summary,
            "layer_feature_summary": per_layer_feature_summary,
        }

    def _record_retention_layer_history(
        self,
        *,
        task_number: int,
        task_summaries: List[Dict[str, Any]],
    ) -> None:
        old_task_summaries = [summary for summary in task_summaries if bool(summary.get("is_old_task", False))]
        if not old_task_summaries:
            return
        for block_id in self.model.selected_blocks:
            block_id = int(block_id)
            layer_entries = [
                task_summary["layer_feature_summary"][block_id]
                for task_summary in old_task_summaries
                if block_id in task_summary.get("layer_feature_summary", {})
            ]
            route_entries = [
                task_summary.get("route_summary", {}).get("per_layer", {}).get(block_id, {})
                for task_summary in old_task_summaries
                if block_id in task_summary.get("route_summary", {}).get("per_layer", {})
            ]
            if not layer_entries and not route_entries:
                continue
            self._retention_layer_audit_history.setdefault(block_id, []).append(
                {
                    "task_number": int(task_number),
                    "feature_mean_abs_diff": float(
                        sum(float(entry.get("mean_abs_diff", 0.0)) for entry in layer_entries) / max(len(layer_entries), 1)
                    ),
                    "feature_mean_cosine": float(
                        sum(float(entry.get("mean_cosine", 0.0)) for entry in layer_entries) / max(len(layer_entries), 1)
                    ),
                    "route_available_frequency": float(
                        sum(float(entry.get("route_available_frequency", 0.0)) for entry in route_entries)
                        / max(len(route_entries), 1)
                    ),
                    "shared_only_frequency": float(
                        sum(float(entry.get("shared_only_frequency", 0.0)) for entry in route_entries)
                        / max(len(route_entries), 1)
                    ),
                    "multi_slot_available_frequency": float(
                        sum(float(entry.get("multi_slot_available_frequency", 0.0)) for entry in route_entries)
                        / max(len(route_entries), 1)
                    ),
                    "old_task_accuracy_mean": float(
                        sum(float(task_summary.get("accuracy", 0.0)) for task_summary in old_task_summaries)
                        / len(old_task_summaries)
                    ),
                }
            )

    def _log_retention_boundary_audit(
        self,
        context: Dict[str, Any],
        *,
        epoch_history: List[Dict[str, Any]],
        pre_eval: Dict[str, Any] | None,
        post_eval: Dict[str, Any],
    ) -> None:
        if not self._retention_audit_enabled() or int(context.get("task_number", 1)) <= 1:
            return
        current_row = list(post_eval.get("per_task_acc", []))
        current_train_accuracy = float(epoch_history[-1]["train_accuracy"]) if epoch_history else 0.0
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
        weighted_balance = self._weighted_loss_balance_summary(mean_losses)
        forgetting = self._compute_forgetting(current_row)
        decomposition = self._forgetting_decomposition(current_row)
        current_eval_accuracy = self._current_last_task_accuracy(current_row)
        old_task_accuracies = [float(value) for value in current_row[:-1]]
        old_task_mean_accuracy = float(sum(old_task_accuracies) / len(old_task_accuracies)) if old_task_accuracies else 0.0
        self.logger.info(
            "[RetentionAudit][Task %d] current_train_acc=%.4f current_eval_acc=%.4f seen_eval_avg_acc=%.4f old_task_avg_acc=%.4f forgetting=%.4f weighted_cls=%.4e weighted_kd=%.4e weighted_feat=%.4e weighted_orth=%.4e weighted_rank=%.4e weighted_grow=%.4e weighted_route=%.4e retention_to_cls_ratio=%.4f cls_share=%.4f kd_share=%.4f feat_share=%.4f",
            context["task_number"],
            current_train_accuracy,
            current_eval_accuracy,
            float(post_eval.get("avg_acc", 0.0)),
            old_task_mean_accuracy,
            forgetting,
            float(weighted_balance["weighted_losses"]["cls"]),
            float(weighted_balance["weighted_losses"]["kd"]),
            float(weighted_balance["weighted_losses"]["feat"]),
            float(weighted_balance["weighted_losses"]["orth"]),
            float(weighted_balance["weighted_losses"]["rank"]),
            float(weighted_balance["weighted_losses"]["grow"]),
            float(weighted_balance["weighted_losses"]["route"]),
            float(weighted_balance["retention_to_cls_ratio"]),
            float(weighted_balance["weighted_shares"]["cls"]),
            float(weighted_balance["weighted_shares"]["kd"]),
            float(weighted_balance["weighted_shares"]["feat"]),
        )
        decomposition_by_task = {
            int(entry["task_number"]): entry
            for entry in decomposition.get("tasks", [])
        }
        pre_post_diff = self._retention_pre_post_diff(pre_eval, post_eval)
        pre_post_by_task = {
            int(entry["task_number"]): entry
            for entry in pre_post_diff.get("tasks", [])
        }
        pre_tasks = {
            int(task_summary["task_number"]): task_summary
            for task_summary in (pre_eval or {}).get("retention_audit", {}).get("task_summaries", [])
        }
        post_tasks = post_eval.get("retention_audit", {}).get("task_summaries", [])
        for task_summary in post_tasks:
            task_number = int(task_summary["task_number"])
            if task_number >= int(context["task_number"]):
                continue
            decomposition_entry = decomposition_by_task.get(task_number, {})
            calibration = task_summary.get("classifier_calibration", {})
            route_summary = task_summary.get("route_summary", {})
            delta_entry = pre_post_by_task.get(task_number, {})
            self.logger.info(
                "[RetentionOldTask][Task %d][SeenTask %d] acc=%.4f best_prior_acc=%.4f latest_prior_acc=%.4f drop_from_best=%.4f drop_from_latest=%.4f old_logit_mean_abs_diff=%.4e old_logit_max_abs_diff=%.4e feature_mean_abs_diff=%.4e feature_mean_cosine=%.4f",
                context["task_number"],
                task_number,
                float(task_summary.get("accuracy", 0.0)),
                float(decomposition_entry.get("best_prior_accuracy", 0.0)),
                float(decomposition_entry.get("latest_prior_accuracy", 0.0)),
                float(decomposition_entry.get("drop_from_best", 0.0)),
                float(decomposition_entry.get("drop_from_latest", 0.0)),
                float(task_summary.get("old_logit_mean_abs_diff", 0.0)),
                float(task_summary.get("old_logit_max_abs_diff", 0.0)),
                float(task_summary.get("feature_mean_abs_diff", 0.0)),
                float(task_summary.get("feature_mean_cosine", 0.0)),
            )
            self.logger.info(
                "[RetentionCalibration][Task %d][SeenTask %d] old_logit_norm_mean=%.4e new_logit_norm_mean=%.4e old_max_mean=%.4e new_max_mean=%.4e old_minus_new_mean=%.4e new_wins_ratio=%.4f old_weight_norm_mean=%.4e new_weight_norm_mean=%.4e old_logit_compression_ratio=%.4f classifier_drift_mean_abs=%.4e",
                context["task_number"],
                task_number,
                float(calibration.get("old_logit_norm_mean", 0.0)),
                float(calibration.get("new_logit_norm_mean", 0.0)),
                float(calibration.get("old_max_mean", 0.0)),
                float(calibration.get("new_max_mean", 0.0)),
                float(calibration.get("old_minus_new_mean", 0.0)),
                float(calibration.get("new_wins_ratio", 0.0)),
                float(calibration.get("old_weight_norm_mean", 0.0)),
                float(calibration.get("new_weight_norm_mean", 0.0)),
                float(calibration.get("old_logit_compression_ratio", 0.0)),
                float(self._classifier_drift_stats(context).get("mean_abs", 0.0) if self._classifier_drift_stats(context) else 0.0),
            )
            self.logger.info(
                "[RetentionRoute][Task %d][SeenTask %d] shared_only_layer_count_mean=%.4f route_available_layer_count_mean=%.4f multi_slot_available_layer_count_mean=%.4f candidate_slot_count_mean=%.4f retained_usage_top1_share_mean=%.4f shared_only_layer_frequency=%.4f route_available_layer_frequency=%.4f multi_slot_available_layer_frequency=%.4f",
                context["task_number"],
                task_number,
                float(route_summary.get("shared_only_layer_count_mean", 0.0)),
                float(route_summary.get("route_available_layer_count_mean", 0.0)),
                float(route_summary.get("multi_slot_available_layer_count_mean", 0.0)),
                float(route_summary.get("candidate_slot_count_mean", 0.0)),
                float(route_summary.get("usage_top1_share_mean", 0.0)),
                float(route_summary.get("shared_only_layer_frequency", 0.0)),
                float(route_summary.get("route_available_layer_frequency", 0.0)),
                float(route_summary.get("multi_slot_available_layer_frequency", 0.0)),
            )
            self.logger.info(
                "[RetentionCHUDiff][Task %d][SeenTask %d] pre_acc=%.4f post_acc=%.4f acc_delta=%.4e pre_old_logit_mean_abs_diff=%.4e post_old_logit_mean_abs_diff=%.4e old_logit_diff_delta=%.4e feature_diff_delta=%.4e shared_only_layer_count_mean_delta=%.4e multi_slot_available_layer_count_mean_delta=%.4e",
                context["task_number"],
                task_number,
                float(pre_tasks.get(task_number, {}).get("accuracy", 0.0)),
                float(task_summary.get("accuracy", 0.0)),
                float(delta_entry.get("accuracy_delta", 0.0)),
                float(pre_tasks.get(task_number, {}).get("old_logit_mean_abs_diff", 0.0)),
                float(task_summary.get("old_logit_mean_abs_diff", 0.0)),
                float(delta_entry.get("old_logit_mean_abs_diff_delta", 0.0)),
                float(delta_entry.get("feature_mean_abs_diff_delta", 0.0)),
                float(delta_entry.get("shared_only_layer_count_mean_delta", 0.0)),
                float(delta_entry.get("multi_slot_available_layer_count_mean_delta", 0.0)),
            )
        pre_profile_summary = self._profile_layer_count_summary(
            context.get("stage5_slot_lifecycle", {}).get("PreCHUProfile", {})
        )
        post_profile_summary = self._profile_layer_count_summary(
            context.get("stage5_slot_lifecycle", {}).get("PostCHUProfile", {})
        )
        self.logger.info(
            "[RetentionCHUSummary][Task %d] pre_shared_only_layers=%d post_shared_only_layers=%d pre_multi_slot_layers=%d post_multi_slot_layers=%d old_task_acc_delta_mean=%.4e old_logit_delta_mean=%.4e feature_delta_mean=%.4e",
            context["task_number"],
            int(pre_profile_summary["shared_only_layers"]),
            int(post_profile_summary["shared_only_layers"]),
            int(pre_profile_summary["multi_slot_layers"]),
            int(post_profile_summary["multi_slot_layers"]),
            float(pre_post_diff["accuracy_delta_summary"]["mean"]),
            float(pre_post_diff["old_logit_delta_summary"]["mean"]),
            float(pre_post_diff["feature_delta_summary"]["mean"]),
        )
        self._record_retention_layer_history(
            task_number=int(context["task_number"]),
            task_summaries=post_tasks,
        )
        for block_id in self.model.selected_blocks:
            history = self._retention_layer_audit_history.get(int(block_id), [])
            if not history:
                continue
            self.logger.info(
                "[RetentionLayerAttribution][Task %d][Layer %d] feature_mean_abs_diff=%.4e feature_mean_cosine=%.4f route_available_frequency=%.4f shared_only_frequency=%.4f multi_slot_available_frequency=%.4f old_task_accuracy_mean=%.4f observations=%d",
                context["task_number"],
                int(block_id),
                float(sum(float(entry["feature_mean_abs_diff"]) for entry in history) / len(history)),
                float(sum(float(entry["feature_mean_cosine"]) for entry in history) / len(history)),
                float(sum(float(entry["route_available_frequency"]) for entry in history) / len(history)),
                float(sum(float(entry["shared_only_frequency"]) for entry in history) / len(history)),
                float(sum(float(entry["multi_slot_available_frequency"]) for entry in history) / len(history)),
                float(sum(float(entry["old_task_accuracy_mean"]) for entry in history) / len(history)),
                len(history),
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
        weighted_loss_balance = self._weighted_loss_balance_summary(mean_losses)
        forgetting_decomposition = self._forgetting_decomposition(current_row)
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
            "weighted_loss_balance": weighted_loss_balance,
            "forgetting_decomposition": forgetting_decomposition,
            "current_train_accuracy": float(epoch_history[-1].get("train_accuracy", 0.0)) if epoch_history else 0.0,
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
            "planner_control_task_context_stats": {
                "norm": float(task_state.embedding.detach().norm().item()),
                "var": float(task_state.embedding.detach().float().var(unbiased=False).item())
                if task_state.embedding.numel() > 1
                else 0.0,
            },
        }
        if self._planner_audit_enabled():
            context["planner_optimizer_summary"] = self._planner_optimizer_membership(optimizer)
            context["planner_structure_route_accumulator"] = self._new_structure_route_accumulator()
        if self._hybrid_planner_enabled():
            context["planner_policy_optimizer_summary"] = self._planner_policy_optimizer_membership(optimizer)
            context["planner_control_optimizer_summary"] = self._planner_control_optimizer_membership(optimizer)
            context["planner_control_task_start"] = self._capture_planner_control_task_start(
                task_state.embedding.detach(),
                None if planner_history_summary is None else planner_history_summary.detach(),
                applied_plans,
            )
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
                for group in ("shared", "slot", "router", "planner_policy", "planner_control", "planner", "classifier")
            }
            context["planner_audit_epoch_accumulator"] = {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
            }
            context["planner_control_epoch_accumulator"] = self._new_planner_control_epoch_accumulator()
            context["planner_control_contribution_epoch_accumulator"] = {}
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
                self._accumulate_planner_structure_route_usage(context, outputs["route_info"])
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

        retention_eval_audit = self._retention_audit_enabled() and int(context["task_number"]) > 1
        debug_eval = bool(self.config["training"].get("debug_eval_around_consolidation", False))
        pre_metrics = None
        if debug_eval or retention_eval_audit:
            pre_profile = self.model.build_inference_profile()
            pre_metrics = self._evaluate_up_to(
                task_index,
                planner_out=pre_profile,
                teacher_model=context.get("teacher_model") if retention_eval_audit else None,
                teacher_profile=context.get("teacher_profile") if retention_eval_audit else None,
                collect_retention_audit=retention_eval_audit,
            )
            if debug_eval:
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
        self._log_stage13_structural_shared_only_audit(context, usage_stats)
        self._log_stage5_train_eval_comparison(context)
        self._log_planner_parameter_drift(context)
        self._log_planner_control_parameter_drift(context)
        self._append_history(context, usage_stats)
        self.last_train_state["history_sizes"].append(len(self.history_bank.entries))

        eval_metrics = self._evaluate_up_to(
            task_index,
            teacher_model=context.get("teacher_model"),
            teacher_profile=context.get("teacher_profile"),
            collect_retention_audit=retention_eval_audit,
        )
        if debug_eval:
            self.logger.info(
                "[Debug][Task %d] post-consolidation avg_acc=%.4f per_task_acc=%s",
                context["task_number"],
                eval_metrics["avg_acc"],
                self._format_per_task_acc(eval_metrics["per_task_acc"]),
            )
        self._log_retention_boundary_audit(
            context,
            epoch_history=epoch_history,
            pre_eval=pre_metrics,
            post_eval=eval_metrics,
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

    def _evaluate_up_to(
        self,
        task_index: int,
        planner_out: Dict[int, Dict[str, object]] | None = None,
        *,
        teacher_model=None,
        teacher_profile: Dict[int, Dict[str, object]] | None = None,
        collect_retention_audit: bool = False,
    ) -> Dict[str, Any]:
        self.model.eval()
        per_task_acc = []
        eval_profile = self.inference_profile if planner_out is None else planner_out
        retention_task_summaries: List[Dict[str, Any]] = []
        old_num_classes = int(teacher_model.classifier.num_classes) if teacher_model is not None else 0
        if teacher_model is not None:
            teacher_model.eval()
        with torch.no_grad():
            for eval_task_id in range(task_index + 1):
                _, test_dataset = self.benchmark.build_task_datasets(eval_task_id)
                loader = self._build_eval_loader(test_dataset)
                correct = 0
                total = 0
                route_infos: List[Dict[int, Dict[str, object]]] = []
                teacher_old_logits: List[torch.Tensor] = []
                student_old_logits: List[torch.Tensor] = []
                student_new_logits: List[torch.Tensor] = []
                feature_diff_values: List[Dict[str, float]] = []
                layer_feature_summaries: Dict[int, List[Dict[str, float]]] = {
                    int(block_id): [] for block_id in self.model.selected_blocks
                }
                for batch in loader:
                    images, labels = self._prepare_images_labels(batch)
                    outputs = self.model.forward_with_state(images, task_state=None, planner_out=eval_profile)
                    predictions = outputs["logits"].argmax(dim=-1)
                    correct += int((predictions == labels).sum().item())
                    total += int(labels.numel())
                    if collect_retention_audit:
                        route_infos.append(outputs["route_info"])
                        if teacher_model is not None and teacher_profile is not None and old_num_classes > 0:
                            teacher_outputs = teacher_model.forward_with_state(
                                images,
                                task_state=None,
                                planner_out=teacher_profile,
                            )
                            teacher_old_logits.append(teacher_outputs["logits"].detach().cpu())
                            student_old_logits.append(outputs["logits"][:, :old_num_classes].detach().cpu())
                            if outputs["logits"].size(-1) > old_num_classes:
                                student_new_logits.append(outputs["logits"][:, old_num_classes:].detach().cpu())
                            feature_diff_values.append(
                                self._feature_diff_summary(
                                    outputs["features"],
                                    teacher_outputs["features"],
                                )
                            )
                            for block_id in self.model.selected_blocks:
                                student_layer_feature = outputs["layer_features"].get(int(block_id))
                                teacher_layer_feature = teacher_outputs["layer_features"].get(int(block_id))
                                if student_layer_feature is None or teacher_layer_feature is None:
                                    continue
                                layer_feature_summaries[int(block_id)].append(
                                    self._feature_diff_summary(
                                        student_layer_feature,
                                        teacher_layer_feature,
                                    )
                                )
                per_task_acc.append(correct / max(total, 1))
                if collect_retention_audit:
                    retention_task_summaries.append(
                        self._evaluate_task_retention_summary(
                            current_task_index=int(task_index),
                            eval_task_id=eval_task_id,
                            accuracy=per_task_acc[-1],
                            sample_count=total,
                            route_infos=route_infos,
                            teacher_old_logits=teacher_old_logits,
                            student_old_logits=student_old_logits,
                            student_new_logits=student_new_logits,
                            feature_diff_values=feature_diff_values,
                            layer_feature_summaries=layer_feature_summaries,
                            old_num_classes=old_num_classes,
                        )
                    )
        avg_acc = sum(per_task_acc) / max(len(per_task_acc), 1)
        return {
            "per_task_acc": per_task_acc,
            "avg_acc": avg_acc,
            "retention_audit": {
                "task_summaries": retention_task_summaries,
            }
            if collect_retention_audit
            else {},
        }

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
        self._log_benchmark_sanity_summary()
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
