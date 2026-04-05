from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict
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
    deserialize_task_state,
    detach_task_state,
    pool_vector,
    serialize_task_state,
)
from src.utils.checkpoint import (
    capture_rng_state,
    load_checkpoint as load_checkpoint_file,
    restore_rng_state,
    save_checkpoint as save_checkpoint_file,
)
from src.utils.io import ensure_output_dirs
from src.utils.sampler import StatefulIndexSampler


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
        self.current_task_context: Dict[str, Any] | None = None
        self.current_optimizer = None
        self.current_scheduler = None

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
            "last_checkpoint_path": None,
        }
        self.training_state: Dict[str, Any] = {
            "seed": None,
            "current_task_id": 0,
            "epoch": 0,
            "global_step": 0,
            "step_in_epoch": 0,
            "sampler_state": None,
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

    def _build_train_loader(self, dataset, task_id: int, epoch: int, sampler_state=None):
        runtime_cfg = self.config["runtime"]
        seed = int(self.training_state["seed"] or 0) + task_id * 10_000 + epoch
        sampler = StatefulIndexSampler(len(dataset), shuffle=True, seed=seed)
        if sampler_state is not None:
            sampler.load_state_dict(sampler_state)
        loader = DataLoader(
            dataset,
            batch_size=int(self.config["training"]["batch_size"]),
            sampler=sampler,
            num_workers=int(runtime_cfg["num_workers"]),
            pin_memory=bool(runtime_cfg.get("pin_memory", False)),
            persistent_workers=bool(runtime_cfg.get("persistent_workers", False)) and int(runtime_cfg["num_workers"]) > 0,
        )
        return loader, sampler

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
        teacher_profile = deepcopy(self.inference_profile)
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
            retention_layers = set(int(layer_id) for layer_id in loss_cfg.get("retention_layers", []))
            retention_layers.update(int(layer_id) for layer_id in context["strong_retention_layers"])
            feat_term = feature_retention(
                outputs["layer_features"],
                teacher_outputs["layer_features"],
                layers=sorted(retention_layers),
                device=self.device,
            )
            total_loss = total_loss + float(loss_cfg["lambda_feat"]) * feat_term
            grow_term = growth_penalty(context["applied_plans"], self.device)
            total_loss = total_loss + float(loss_cfg["lambda_grow"]) * grow_term

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
        self.current_optimizer = optimizer
        self.current_scheduler = scheduler
        train_dataset, _ = self.benchmark.build_task_datasets(task_index)
        epochs_per_task = int(self.config["training"]["epochs_per_task"])
        grad_clip_norm = float(self.config["training"]["grad_clip_norm"])

        epoch_start = int(self.training_state["epoch"])
        sampler_state = deepcopy(self.training_state.get("sampler_state"))
        usage_accumulator: Dict[int, Dict[int, float]] = {}
        batch_count = 0
        loss_history = []
        start_time = time.perf_counter()

        for epoch in range(epoch_start, epochs_per_task):
            train_loader, sampler = self._build_train_loader(
                train_dataset,
                task_id=task_index,
                epoch=epoch,
                sampler_state=sampler_state,
            )
            self.training_state["epoch"] = epoch
            self.training_state["step_in_epoch"] = 0 if sampler_state is None else int(self.training_state["step_in_epoch"])
            self.training_state["sampler_state"] = sampler.state_dict()
            sampler_state = None

            self.model.train()
            self.planner.train()
            self.task_state_encoder.train()
            for batch in train_loader:
                images, labels = self._prepare_images_labels(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model.forward_with_state(images, context["task_state"], context["applied_plans"])
                outputs["images"] = images
                losses = self._compute_loss(context, outputs, labels)
                losses["total"].backward()
                if grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.planner.parameters()) + list(self.task_state_encoder.parameters()),
                        grad_clip_norm,
                    )
                optimizer.step()
                self._accumulate_usage(usage_accumulator, outputs["route_info"])
                if outputs["route_info"]:
                    self.last_train_state["router_seen"] = True

                batch_size = int(labels.size(0))
                sampler.mark_consumed(batch_size)
                self.training_state["sampler_state"] = sampler.state_dict()
                self.training_state["global_step"] += 1
                self.training_state["step_in_epoch"] += 1
                batch_count += 1
                loss_history.append(float(losses["total"].detach().item()))

            if scheduler is not None:
                scheduler.step()
            self.training_state["step_in_epoch"] = 0
            self.training_state["sampler_state"] = None
            self._auto_checkpoint(task_definition.task_id, epoch + 1)

        usage_stats = self._finalize_usage(usage_accumulator, batch_count)
        for block_id, stats in usage_stats.items():
            self.model.layers[str(block_id)].update_usage_statistics(stats)

        task_train_time = time.perf_counter() - start_time
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
        inference_overhead = self._estimate_inference_overhead(task_index)
        metrics = {
            "task_id": context["task_number"],
            "avg_acc": eval_metrics["avg_acc"],
            "per_task_acc": eval_metrics["per_task_acc"],
            "opened_slots": chu_report["opened_slots"],
            "pruned_slots": chu_report["pruned_slots"],
            "merged_slots": chu_report["merged_slots"],
            "frozen_slots": chu_report["frozen_slots"],
            "parameter_growth": self._estimate_parameter_growth(),
            "total_active_rank": self._total_active_rank(),
            "training_time": task_train_time,
            "inference_overhead_ratio": inference_overhead,
            "mean_loss": sum(loss_history) / max(len(loss_history), 1),
        }
        self.task_metrics.append(metrics)
        self.accuracy_matrix.append(eval_metrics["per_task_acc"])
        self.total_train_time += task_train_time
        self.training_state["epoch"] = 0
        self.training_state["step_in_epoch"] = 0
        self.training_state["sampler_state"] = None
        return metrics

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

    def _evaluate_up_to(self, task_index: int) -> Dict[str, Any]:
        self.model.eval()
        per_task_acc = []
        with torch.no_grad():
            for eval_task_id in range(task_index + 1):
                _, test_dataset = self.benchmark.build_task_datasets(eval_task_id)
                loader = self._build_eval_loader(test_dataset)
                correct = 0
                total = 0
                for batch in loader:
                    images, labels = self._prepare_images_labels(batch)
                    outputs = self.model.forward_with_state(images, task_state=None, planner_out=self.inference_profile)
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

    def _serialize_planner_signals(self, raw_planner: Dict[int, PlannerSignals]) -> Dict[int, Dict[str, Any]]:
        payload = {}
        for block_id, signals in raw_planner.items():
            payload[int(block_id)] = {
                "novelty": signals.novelty.detach(),
                "conflict": signals.conflict.detach(),
                "rank_score": signals.rank_score.detach(),
                "rank_budget": int(signals.rank_budget),
                "consolidate": signals.consolidate.detach(),
                "shared_gate": signals.shared_gate.detach(),
                "history_attention": None if signals.history_attention is None else signals.history_attention.detach(),
                "history_context": None if signals.history_context is None else signals.history_context.detach(),
                "planner_input": None if signals.planner_input is None else signals.planner_input.detach(),
                "planner_representation": None if signals.planner_representation is None else signals.planner_representation.detach(),
            }
        return payload

    def _deserialize_planner_signals(self, payload: Dict[int, Dict[str, Any]]) -> Dict[int, PlannerSignals]:
        restored = {}
        for block_id, signals in payload.items():
            restored[int(block_id)] = PlannerSignals(
                novelty=signals["novelty"],
                conflict=signals["conflict"],
                rank_score=signals["rank_score"],
                rank_budget=int(signals["rank_budget"]),
                consolidate=signals["consolidate"],
                shared_gate=signals["shared_gate"],
                history_attention=signals.get("history_attention"),
                history_context=signals.get("history_context"),
                planner_input=signals.get("planner_input"),
                planner_representation=signals.get("planner_representation"),
            )
        return restored

    def _serialize_materialized_plans(self, plans: Dict[int, MaterializedLayerPlan]) -> Dict[int, Dict[str, Any]]:
        return {int(block_id): asdict(plan) for block_id, plan in plans.items()}

    def _deserialize_materialized_plans(self, payload: Dict[int, Dict[str, Any]]) -> Dict[int, MaterializedLayerPlan]:
        return {
            int(block_id): MaterializedLayerPlan(**plan_dict)
            for block_id, plan_dict in payload.items()
        }

    def _serialize_current_task_context(self) -> Dict[str, Any] | None:
        if self.current_task_context is None:
            return None
        context = self.current_task_context
        teacher_model = context.get("teacher_model")
        teacher_payload = None
        if teacher_model is not None:
            teacher_payload = {
                "structure_state": teacher_model.export_structure_state(),
                "model_state": teacher_model.state_dict(),
                "inference_profile": context.get("teacher_profile"),
            }
        return {
            "task_id": int(context["task_definition"].task_id),
            "task_number": int(context["task_number"]),
            "task_state": serialize_task_state(context["task_state"]),
            "raw_planner": self._serialize_planner_signals(context["raw_planner"]),
            "materialized_plans": self._serialize_materialized_plans(context["materialized_plans"]),
            "applied_plans": deepcopy(context["applied_plans"]),
            "strong_retention_layers": sorted(int(layer_id) for layer_id in context["strong_retention_layers"]),
            "optimizer_state": None if context["optimizer"] is None else context["optimizer"].state_dict(),
            "scheduler_state": None if context["scheduler"] is None else context["scheduler"].state_dict(),
            "teacher_payload": teacher_payload,
            "warmup_info": deepcopy(context["warmup_info"]),
        }

    def _restore_current_task_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_definition = self.benchmark.tasks[int(payload["task_id"])]
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        if payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])

        teacher_model = None
        teacher_profile = None
        teacher_payload = payload.get("teacher_payload")
        if teacher_payload is not None:
            teacher_model = NHLoRAModel(self.config).to(self.device)
            teacher_model.load_structure_state(teacher_payload["structure_state"])
            teacher_model.load_state_dict(teacher_payload["model_state"])
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad = False
            teacher_profile = teacher_payload["inference_profile"]

        return {
            "task_definition": task_definition,
            "task_number": int(payload["task_number"]),
            "task_state": detach_task_state(deserialize_task_state(payload["task_state"])),
            "raw_planner": self._deserialize_planner_signals(payload["raw_planner"]),
            "materialized_plans": self._deserialize_materialized_plans(payload["materialized_plans"]),
            "applied_plans": payload["applied_plans"],
            "teacher_model": teacher_model,
            "teacher_profile": teacher_profile,
            "strong_retention_layers": set(int(layer_id) for layer_id in payload["strong_retention_layers"]),
            "optimizer": optimizer,
            "scheduler": scheduler,
            "warmup_info": payload["warmup_info"],
        }

    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "model_structure_state": self.model.export_structure_state(),
            "model_state": self.model.state_dict(),
            "planner_state": self.planner.state_dict(),
            "task_state_encoder_state": self.task_state_encoder.state_dict(),
            "classifier_state": self.model.classifier.state_dict(),
            "history_bank_state": self.history_bank.state_dict(),
            "trainer_state": deepcopy(self.training_state),
            "current_task_context": self._serialize_current_task_context(),
            "rng_state": capture_rng_state(),
            "config_snapshot": deepcopy(self.config),
            "metrics_summary": {
                "task_metrics": deepcopy(self.task_metrics),
                "accuracy_matrix": deepcopy(self.accuracy_matrix),
                "total_train_time": float(self.total_train_time),
                "last_train_state": deepcopy(self.last_train_state),
            },
            "inference_profile": deepcopy(self.inference_profile),
        }

    def save_checkpoint(self, name: str = "latest.pt", output_path: str | Path | None = None) -> Path:
        if output_path is None:
            output_path = Path(self.output_dirs["benchmark_checkpoints"]) / name
        output_path = Path(output_path)
        payload = self._checkpoint_payload()
        save_checkpoint_file(payload, output_path)
        self.last_train_state["last_checkpoint_path"] = str(output_path)
        return output_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        payload = load_checkpoint_file(checkpoint_path, map_location=self.device)
        self.model.load_structure_state(payload["model_structure_state"])
        self.model.load_state_dict(payload["model_state"])
        self.planner.load_state_dict(payload["planner_state"])
        self.task_state_encoder.load_state_dict(payload["task_state_encoder_state"])
        self.model.classifier.load_state_dict(payload["classifier_state"])
        self.history_bank.load_state_dict(payload["history_bank_state"])
        self.training_state = payload["trainer_state"]
        self.inference_profile = payload["inference_profile"]
        self.task_metrics = payload["metrics_summary"]["task_metrics"]
        self.accuracy_matrix = payload["metrics_summary"]["accuracy_matrix"]
        self.total_train_time = float(payload["metrics_summary"]["total_train_time"])
        self.last_train_state = payload["metrics_summary"]["last_train_state"]
        restore_rng_state(payload["rng_state"])
        current_task_context = payload.get("current_task_context")
        if current_task_context is not None:
            self.current_task_context = self._restore_current_task_context(current_task_context)
            self.current_optimizer = self.current_task_context["optimizer"]
            self.current_scheduler = self.current_task_context["scheduler"]
        else:
            self.current_task_context = None
            self.current_optimizer = None
            self.current_scheduler = None

    def _auto_checkpoint(self, task_id: int, epoch: int) -> None:
        if not bool(self.config["experiment"].get("save_checkpoints", False)):
            return
        checkpoint_name = f"task{task_id + 1}_epoch{epoch}.pt"
        self.save_checkpoint(name=checkpoint_name)
        self.save_checkpoint(name="latest.pt")

    def train(self, seed: int) -> Dict[str, Any]:
        if self.training_state["seed"] is None:
            self.training_state["seed"] = int(seed)

        for task_index in range(int(self.training_state["current_task_id"]), len(self.benchmark.tasks)):
            if self.current_task_context is not None and self.current_task_context["task_definition"].task_id == task_index:
                context = self.current_task_context
            else:
                context = self._prepare_task_context(self.benchmark.tasks[task_index])
                self.current_task_context = context
            metrics = self._train_single_task(context)
            self.current_task_context = None
            self.current_optimizer = None
            self.current_scheduler = None
            self.training_state["current_task_id"] = task_index + 1
            self.save_checkpoint(name="latest.pt")
            self.logger.info(
                "Finished task %d | avg_acc=%.4f | active_rank=%d",
                context["task_number"],
                metrics["avg_acc"],
                metrics["total_active_rank"],
            )

        final_avg_acc = self.task_metrics[-1]["avg_acc"] if self.task_metrics else 0.0
        return {
            "benchmark": self.benchmark.name,
            "final_avg_acc": final_avg_acc,
            "opened_slots": sum(metric["opened_slots"] for metric in self.task_metrics),
            "pruned_slots": sum(metric["pruned_slots"] for metric in self.task_metrics),
            "parameter_growth": self.task_metrics[-1]["parameter_growth"] if self.task_metrics else 0,
            "total_active_rank": self.task_metrics[-1]["total_active_rank"] if self.task_metrics else 0,
            "training_time_total": self.total_train_time,
            "inference_overhead_ratio": self.task_metrics[-1]["inference_overhead_ratio"] if self.task_metrics else 1.0,
            "task_metrics": deepcopy(self.task_metrics),
            "accuracy_matrix": deepcopy(self.accuracy_matrix),
        }
