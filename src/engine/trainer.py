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
from src.models.planner import HorizonPlanner, MaterializedLayerPlan, PlannerSignals, materialize_action
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

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device(requested_device)

    def _base_learning_rate(self) -> float:
        return float(self.config["training"]["lr"])

    def _shared_lr_scale(self) -> float:
        return float(self.config["nh_lora"].get("shared_lr_scale", 1.0))

    def _build_optimizer(self):
        shared_params = []
        other_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".shared_a" in name or ".shared_b" in name:
                shared_params.append(parameter)
            else:
                other_params.append(parameter)
        other_params.extend([parameter for parameter in self.planner.parameters() if parameter.requires_grad])
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
            "  losses lam_kd=%s lam_feat=%s lam_orth=%s lam_rank=%s lam_grow=%s lam_route=%s",
            loss_cfg.get("lambda_kd", "n/a"),
            loss_cfg.get("lambda_feat", "n/a"),
            loss_cfg.get("lambda_orth", "n/a"),
            loss_cfg.get("lambda_rank", "n/a"),
            loss_cfg.get("lambda_grow", "n/a"),
            loss_cfg.get("lambda_route", "n/a"),
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
            "  logging estimate_eta=%s cuda_sync_timing=%s debug_eval_around_consolidation=%s retention_debug_logging=%s freeze_old_classifier_weights=%s",
            training_cfg.get("estimate_eta", True),
            training_cfg.get("cuda_sync_timing", False),
            training_cfg.get("debug_eval_around_consolidation", False),
            training_cfg.get("retention_debug_logging", True),
            training_cfg.get("freeze_old_classifier_weights", True),
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
        if not bool(self.config["training"].get("freeze_old_classifier_weights", True)):
            return
        old_num_classes = int(context.get("old_num_classes", 0))
        if old_num_classes <= 0:
            return
        classifier = self.model.classifier
        if classifier.weight.grad is not None:
            # Prevent old classifier prototypes from drifting under new-task CE updates.
            classifier.weight.grad[:old_num_classes].zero_()
        bias = getattr(classifier, "bias", None)
        if bias is not None and getattr(bias, "grad", None) is not None:
            bias.grad[:old_num_classes].zero_()

    def _active_retention_layers(self, context: Dict[str, Any]) -> List[int]:
        configured_layers = self.config["loss"].get("retention_layers")
        if configured_layers:
            retention_layers = {int(layer_id) for layer_id in configured_layers}
        else:
            retention_layers = {int(layer_id) for layer_id in self.model.selected_blocks}
        retention_layers.update(int(layer_id) for layer_id in context["strong_retention_layers"])
        return sorted(retention_layers)

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
        context["last_retention_debug"] = {
            "teacher_num_classes": int(teacher_num_classes),
            "old_num_classes": int(context.get("old_num_classes", 0)),
            "retention_layers": list(retention_layers),
            "teacher_old_logits_norm": float(teacher_outputs["logits"].detach().norm(dim=-1).mean().item()),
            "student_old_logits_norm": float(outputs["logits"][:, :teacher_num_classes].detach().norm(dim=-1).mean().item()),
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

    def _compute_raw_planner(self, task_state: TaskState) -> Dict[int, PlannerSignals]:
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
            raw_planner = self._compute_raw_planner(task_state)
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
        self._expand_classifier_for_task(task_definition, task_state)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        return {
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
        }

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
            teacher_num_classes = teacher_model.classifier.num_classes
            if teacher_num_classes > 0:
                kd_term = kd_loss(
                    outputs["logits"][:, :teacher_num_classes],
                    teacher_outputs["logits"],
                    temperature=float(loss_cfg["kd_temperature"]),
                )
                total_loss = total_loss + float(loss_cfg["lambda_kd"]) * kd_term
            retention_layers = self._active_retention_layers(context)
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
            train_loader = self._build_train_loader(train_dataset)
            self.model.train()
            self.planner.train()
            self.task_state_encoder.train()
            epoch_loss_sums = {key: 0.0 for key in LOSS_COMPONENT_KEYS}
            epoch_correct = 0
            epoch_total = 0
            for batch in train_loader:
                images, labels = self._prepare_images_labels(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model.forward_with_state(images, context["task_state"], context["applied_plans"])
                outputs["images"] = images
                losses = self._compute_loss(context, outputs, labels)
                losses["total"].backward()
                self._mask_old_classifier_gradients(context)
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.planner.parameters()) + list(self.task_state_encoder.parameters()),
                        grad_clip_norm,
                    )
                optimizer.step()
                self._accumulate_usage(usage_accumulator, outputs["route_info"])
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
                        "[Retention][Task %d][Epoch %d] teacher_classes=%d old_classes=%d layers=%s teacher_logit_norm=%.4e student_logit_norm=%.4e kd_raw=%.4e feat_raw=%.4e",
                        context["task_number"],
                        epoch,
                        debug["teacher_num_classes"],
                        debug["old_num_classes"],
                        debug["retention_layers"],
                        debug["teacher_old_logits_norm"],
                        debug["student_old_logits_norm"],
                        debug["kd_raw"],
                        debug["feat_raw"],
                    )

        training_time = self._perf_counter() - training_time_start
        usage_stats = self._finalize_usage(usage_accumulator, batch_count)
        for block_id, stats in usage_stats.items():
            self.model.layers[str(block_id)].update_usage_statistics(stats)

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
