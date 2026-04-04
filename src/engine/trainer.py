from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

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
from src.models.planner import HorizonPlanner, materialize_action
from src.models.task_state import (
    HistoryBank,
    TaskStateEncoder,
    build_history_entry,
    build_history_summary_vector,
    detach_task_state,
    pool_vector,
)
from src.utils.io import ensure_output_dirs


@dataclass
class TaskRunRecord:
    avg_seen_accuracy: float
    per_task_accuracy: List[float]
    train_time_sec: float


class NHLoRATrainer:
    def __init__(self, config: Dict[str, object], logger: logging.Logger, benchmark=None):
        self.config = config
        self.logger = logger
        runtime_device = str(config["runtime"].get("device", "cpu"))
        if runtime_device.startswith("cuda") and not torch.cuda.is_available():
            runtime_device = "cpu"
        self.device = torch.device(runtime_device)
        self.output_dirs = ensure_output_dirs(config, benchmark_name=str(config["benchmark"]["name"]))
        self.benchmark = benchmark or build_benchmark(config)
        self.model = NHLoRAModel(config).to(self.device)
        self.history_bank = HistoryBank()
        self.history_pool_dim = int(config["planner"].get("history_pool_dim", 8))
        self.task_state_encoder = TaskStateEncoder(
            feature_dim=self.model.backbone.embed_dim,
            grad_dim=int(config["warmup"].get("gradient_sketch_dim", 16)),
            embedding_dim=int(config["warmup"].get("task_embedding_dim", 128)),
            pool_dim=self.history_pool_dim,
        ).to(self.device)
        history_dim = 2 * self.history_pool_dim + self.task_state_encoder.grad_dim + 3 * len(self.model.selected_blocks) + 1
        self.planner = HorizonPlanner(
            selected_blocks=self.model.selected_blocks,
            task_embedding_dim=self.task_state_encoder.embedding_dim,
            history_dim=history_dim,
            hidden_dim=int(config["planner"]["hidden_dim"]),
            layer_embedding_dim=int(config["planner"]["layer_embedding_dim"]),
            rank_min=int(config["planner"]["rank_min"]),
            rank_max=int(config["planner"]["rank_max"]),
            tau_novelty=float(config["planner"]["tau_novelty"]),
            tau_conflict=float(config["planner"]["tau_conflict"]),
        ).to(self.device)
        self.chu = ConsolidationHomeostasisUnit(config["chu"])
        self.last_train_state = {
            "task_states": [],
            "bootstrap_used": False,
            "teacher_used_on_task2": False,
            "planner_used_on_task2": False,
            "materialize_used_on_task2": False,
            "router_seen": False,
            "classifier_sizes": [],
            "classifier_imprinting_used": False,
            "warmup_imprinting_used": False,
            "warmup_head_class_counts": [],
            "history_summary_dim": 0,
            "raw_planner_separated": False,
            "task2_materialized_has_candidates": False,
            "task2_history_attention_used": False,
            "task2_actions": {},
            "chu_calls": 0,
            "chu_calls_per_task": [],
            "history_sizes": [],
        }
        self.last_teacher_planner_out = None

    def train(self, seed: int) -> Dict[str, object]:
        task_records: List[TaskRunRecord] = []
        accuracy_matrix: List[List[float]] = []
        total_train_time = 0.0
        final_chu_report = None

        for task in self.benchmark.tasks:
            train_loader = self._make_loader(
                self.benchmark.build_task_datasets(task.task_id)[0],
                shuffle=True,
                batch_size=int(self.config["training"]["batch_size"]),
            )
            task_state = detach_task_state(self._build_task_state(train_loader, task))
            self.last_train_state["task_states"].append(
                {
                    "task_id": task.task_id + 1,
                    "similarity_mean": float(task_state.similarity.mean().item()),
                    "has_history": len(self.history_bank) > 0,
                }
            )
            self.model.capture_pre_task_snapshots()
            if task.task_id == 0:
                planning_bundle = {
                    "raw": {},
                    "materialized": self.model.initialize_bootstrap_structure(task_id=task.task_id + 1),
                }
                teacher_model = None
                teacher_planning_bundle = None
                self.last_train_state["bootstrap_used"] = True
            else:
                teacher_model = self.model.make_teacher_snapshot()
                teacher_planning_bundle = self._build_eval_planning(teacher_model)
                planning_bundle = self._plan_structure(task_state, task.task_id + 1)
                self.last_train_state["raw_planner_separated"] = bool(planning_bundle["raw"])
                if task.task_id == 1:
                    self.last_train_state["teacher_used_on_task2"] = teacher_model is not None
                    self.last_train_state["planner_used_on_task2"] = True
                    self.last_train_state["materialize_used_on_task2"] = True
                    self.last_train_state["task2_history_attention_used"] = all(
                        signals.history_attention is not None for signals in planning_bundle["raw"].values()
                    )
                    self.last_train_state["task2_materialized_has_candidates"] = all(
                        "active_slot_candidates" in cfg for cfg in planning_bundle["materialized"].values()
                    )
                    self.last_train_state["task2_actions"] = {
                        str(block_id): cfg["action"] for block_id, cfg in planning_bundle["materialized"].items()
                    }

            required_classes = max(task.class_ids) + 1
            num_new_classes = required_classes - self.model.classifier.num_classes
            self.model.expand_classifier(num_new_classes)
            if str(self.config["classifier"].get("init_mode", "imprint")).lower() == "imprint":
                self.model.classifier.imprint_from_prototypes(task_state.class_prototypes, task.class_ids)
                self.last_train_state["classifier_imprinting_used"] = True
            self.last_train_state["classifier_sizes"].append(self.model.classifier.num_classes)
            optimizer = self._build_optimizer()

            start_time = time.time()
            self._train_single_task(
                train_loader=train_loader,
                task=task,
                task_state=task_state,
                planning_bundle=planning_bundle,
                teacher_model=teacher_model,
                teacher_planning_bundle=teacher_planning_bundle,
                optimizer=optimizer,
            )
            train_time = time.time() - start_time
            total_train_time += train_time

            eval_planning = self._build_eval_planning(self.model)
            usage_stats = self._estimate_slot_usage(task, task_state, eval_planning["materialized"])
            final_chu_report = self._run_chu(planning_bundle["materialized"], usage_stats)
            self.last_train_state["chu_calls_per_task"].append(1)
            self._append_history(task.task_id + 1, task_state, usage_stats, planning_bundle["materialized"])
            self.last_train_state["history_sizes"].append(len(self.history_bank))

            accuracies = self._evaluate_seen_tasks(task.task_id, eval_planning["materialized"])
            accuracy_matrix.append(accuracies)
            task_records.append(
                TaskRunRecord(
                    avg_seen_accuracy=sum(accuracies) / max(len(accuracies), 1),
                    per_task_accuracy=accuracies,
                    train_time_sec=train_time,
                )
            )

        metrics = self._build_metrics(task_records, accuracy_matrix, total_train_time, final_chu_report)
        return metrics

    def _make_loader(self, dataset, shuffle: bool, batch_size: int) -> DataLoader:
        num_workers = int(self.config["runtime"].get("num_workers", 0))
        persistent = bool(self.config["runtime"].get("persistent_workers", False)) and num_workers > 0
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=bool(self.config["runtime"].get("pin_memory", False)),
            persistent_workers=persistent,
        )

    def _build_optimizer(self):
        base_lr = float(self.config["training"]["lr"])
        shared_lr_scale = float(self.config["nh_lora"].get("shared_lr_scale", 1.0))
        shared_params = []
        default_params = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "point_banks" in name and ("shared_a" in name or "shared_b" in name):
                shared_params.append(parameter)
            else:
                default_params.append(parameter)
        default_params.extend([p for p in self.planner.parameters() if p.requires_grad])
        default_params.extend([p for p in self.task_state_encoder.parameters() if p.requires_grad])
        param_groups = []
        if default_params:
            param_groups.append({"params": default_params, "lr": base_lr})
        if shared_params:
            param_groups.append({"params": shared_params, "lr": base_lr * shared_lr_scale})
        return torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=float(self.config["training"]["weight_decay"]),
        )

    def _compress_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        grad_dim = self.task_state_encoder.grad_dim
        flattened = gradient.flatten()
        if flattened.numel() < grad_dim:
            padded = torch.zeros(grad_dim, device=gradient.device)
            padded[: flattened.numel()] = flattened
            return padded.unsqueeze(0)
        chunks = torch.chunk(flattened, grad_dim)
        stats = torch.stack([chunk.mean() for chunk in chunks], dim=0)
        return stats.unsqueeze(0)

    def _build_persistent_warmup_head(self, features: torch.Tensor, labels: torch.Tensor):
        local_classes = sorted({int(label.item()) for label in labels})
        class_to_local = {class_id: offset for offset, class_id in enumerate(local_classes)}
        local_targets = torch.tensor([class_to_local[int(label.item())] for label in labels], device=self.device)
        aux_head = IncrementalCosineClassifier(
            feature_dim=features.size(-1),
            tau=float(self.config["classifier"]["tau_cls"]),
        ).to(self.device)
        aux_head.expand(len(local_classes))
        prototypes = {}
        for class_id in local_classes:
            mask = labels == class_id
            prototypes[class_id] = features[mask].mean(dim=0).detach()
        local_prototypes = {class_to_local[class_id]: prototype for class_id, prototype in prototypes.items()}
        aux_head.imprint_from_prototypes(local_prototypes, range(len(local_classes)))
        self.last_train_state["warmup_imprinting_used"] = True
        self.last_train_state["warmup_head_class_counts"].append(len(local_classes))
        return aux_head, local_targets, prototypes

    def _build_warmup_planning(self):
        if len(self.history_bank) == 0:
            return None
        return self._build_eval_planning(self.model)["materialized"]

    def _build_task_state(self, train_loader: DataLoader, task):
        warm_batches = int(self.config["warmup"]["num_batches"])
        feature_chunks = []
        label_chunks = []
        warmup_planning = self._build_warmup_planning()

        for batch_idx, (_, images, labels) in enumerate(train_loader):
            if batch_idx >= warm_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                if warmup_planning is None:
                    features = self.model.extract_features(images)
                else:
                    features = self.model.encode(images, planner_out=warmup_planning)["features"]
            feature_chunks.append(features)
            label_chunks.append(labels)

        feature_tensor = torch.cat(feature_chunks, dim=0)
        label_tensor = torch.cat(label_chunks, dim=0)
        aux_head, local_targets, prototypes = self._build_persistent_warmup_head(feature_tensor.detach(), label_tensor)
        aux_logits = aux_head(feature_tensor.detach())
        entropy = (-torch.softmax(aux_logits, dim=-1) * torch.log_softmax(aux_logits, dim=-1)).sum(dim=-1).mean().view(1, 1)
        aux_loss = F.cross_entropy(aux_logits, local_targets)
        aux_head.zero_grad()
        aux_loss.backward()

        feature_mean = feature_tensor.mean(dim=0, keepdim=True)
        feature_var = feature_tensor.var(dim=0, unbiased=False, keepdim=True)
        gradient_sketch = self._compress_gradient(aux_head.weight.grad.detach())
        pooled_feature_mean = pool_vector(feature_mean, self.history_pool_dim)
        pooled_feature_var = pool_vector(feature_var, self.history_pool_dim)
        zero_usage = feature_mean.new_zeros(1, 2 * len(self.model.selected_blocks))
        zero_rank = feature_mean.new_zeros(1, len(self.model.selected_blocks))
        summary_vector = build_history_summary_vector(
            pooled_feature_mean=pooled_feature_mean,
            pooled_feature_var=pooled_feature_var,
            gradient_sketch=gradient_sketch,
            usage_summary=zero_usage,
            active_rank_summary=zero_rank,
            entropy_summary=entropy,
        )
        similarity_anchor = F.normalize(torch.cat([pooled_feature_mean, pooled_feature_var], dim=-1), dim=-1)

        provisional = self.task_state_encoder(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=feature_mean.new_zeros(1, 1),
            entropy=entropy,
            similarity_anchor=similarity_anchor,
            summary_vector=summary_vector,
            class_prototypes=prototypes,
            warmup_logits=aux_logits.detach(),
        )
        if len(self.history_bank) == 0:
            similarity = feature_mean.new_zeros(1, 1)
        else:
            similarity = self.history_bank.mean_similarity(provisional.summary_vector, provisional.similarity_anchor)
        task_state = self.task_state_encoder(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=similarity,
            entropy=entropy,
            similarity_anchor=similarity_anchor,
            summary_vector=summary_vector,
            class_prototypes=prototypes,
            warmup_logits=aux_logits.detach(),
        )
        return task_state

    def _plan_structure(self, task_state, task_id: int):
        history_summary = self.history_bank.aggregate()
        raw_outputs = {}
        materialized = {}
        for block_id in self.model.selected_blocks:
            layer = self.model.layers[str(block_id)]
            signals = self.planner(block_id, task_state.embedding, history_summary)
            raw_outputs[block_id] = signals
            action = self.planner.decide_action(signals)
            materialized[block_id] = materialize_action(
                action=action,
                signals=signals,
                slot_bank=layer,
                task_embedding=task_state.embedding.detach(),
                task_id=task_id,
                max_slots_per_block=layer.max_slots,
                tau_consolidate=float(self.config["planner"].get("tau_consolidate", 0.5)),
            )
        return {"raw": raw_outputs, "materialized": materialized}

    def _compute_raw_planner(self, task_state):
        history_summary = self.history_bank.aggregate()
        raw_outputs = {}
        for block_id in self.model.selected_blocks:
            raw_outputs[block_id] = self.planner(block_id, task_state.embedding, history_summary)
        return raw_outputs

    def _build_eval_planning(self, model):
        materialized = {}
        for block_id in model.selected_blocks:
            layer = model.layers[str(block_id)]
            live_slots = layer.live_slot_ids()
            materialized[block_id] = {
                "action": "eval_all_live_slots",
                "active_slot_candidates": live_slots,
                "selected_slot": live_slots[0] if live_slots else None,
                "rank_cfg": {slot_id: layer.slot_metadata[slot_id].rank for slot_id in live_slots},
                "shared_gate": 1.0,
                "consolidate_flag": False,
                "deterministic": len(live_slots) <= 1,
                "created_new_slot": False,
                "fallback_action": None,
                "strong_retention": False,
            }
        return {"raw": {}, "materialized": materialized}

    def _train_single_task(self, train_loader, task, task_state, planning_bundle, teacher_model, teacher_planning_bundle, optimizer):
        epochs = int(self.config["training"]["epochs_per_task"])
        grad_clip = float(self.config["training"]["grad_clip_norm"])
        device = self.device
        old_class_cutoff = min(task.class_ids)
        retention_layers = [int(layer_id) for layer_id in self.config["loss"].get("retention_layers", [])]
        for _ in range(epochs):
            self.model.train()
            for _, images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                current_state = self.model.forward_with_state(images, task_state, planning_bundle["materialized"])
                logits = current_state["logits"]
                features = current_state["features"]
                route_info = current_state["route_info"]
                if route_info:
                    self.last_train_state["router_seen"] = True
                loss_cls = F.cross_entropy(logits, labels)
                batch_raw_planner = self._compute_raw_planner(task_state) if planning_bundle["raw"] else None
                loss_rank = rank_penalty(self.model, batch_raw_planner, device)
                loss_route = routing_balance_loss(route_info, device)
                if task.task_id == 0 or teacher_model is None or teacher_planning_bundle is None or old_class_cutoff == 0:
                    loss_orth = slot_orthogonality(self.model)
                    loss = (
                        loss_cls
                        + float(self.config["loss"]["lambda_orth"]) * loss_orth
                        + float(self.config["loss"]["lambda_rank"]) * loss_rank
                        + float(self.config["loss"]["lambda_route"]) * loss_route
                    )
                else:
                    with torch.no_grad():
                        teacher_state = teacher_model.forward_with_state(images, None, teacher_planning_bundle["materialized"])
                    old_classes = list(range(old_class_cutoff))
                    loss_kd = kd_loss(
                        logits[:, old_classes],
                        teacher_state["logits"][:, old_classes],
                        temperature=float(self.config["loss"]["kd_temperature"]),
                    ) if old_classes else torch.zeros((), device=device)
                    loss_feat = feature_retention(
                        current_state["layer_features"],
                        teacher_state["layer_features"],
                        layers=retention_layers,
                        device=device,
                    )
                    loss_orth = slot_orthogonality(self.model)
                    loss_grow = growth_penalty(batch_raw_planner, device)
                    loss = (
                        loss_cls
                        + float(self.config["loss"]["lambda_kd"]) * loss_kd
                        + float(self.config["loss"]["lambda_feat"]) * loss_feat
                        + float(self.config["loss"]["lambda_orth"]) * loss_orth
                        + float(self.config["loss"]["lambda_rank"]) * loss_rank
                        + float(self.config["loss"]["lambda_grow"]) * loss_grow
                        + float(self.config["loss"]["lambda_route"]) * loss_route
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    grad_clip,
                )
                optimizer.step()

    def _estimate_slot_usage(self, task, task_state, planner_out):
        _, test_dataset = self.benchmark.build_task_datasets(task.task_id)
        loader = self._make_loader(test_dataset, shuffle=False, batch_size=int(self.config["training"]["batch_size"]))
        usage_stats = {block_id: {} for block_id in self.model.selected_blocks}
        total_examples = 0
        self.model.eval()
        with torch.no_grad():
            for _, images, _ in loader:
                images = images.to(self.device)
                total_examples += images.size(0)
                _, _, route_info = self.model(images, task_state, planner_out)
                for block_id, layer_route in route_info.items():
                    active_slots = layer_route.get("candidate_slots", layer_route.get("active_slots", []))
                    weights = layer_route.get("routing_weights")
                    if weights is None or weights.numel() == 0:
                        for slot_id in active_slots:
                            usage_stats[block_id][slot_id] = usage_stats[block_id].get(slot_id, 0.0) + 1.0
                        continue
                    if "topk_indices" in layer_route:
                        for row, indices in enumerate(layer_route["topk_indices"]):
                            for col, candidate_index in enumerate(indices.tolist()):
                                slot_id = active_slots[candidate_index]
                                usage_stats[block_id][slot_id] = usage_stats[block_id].get(slot_id, 0.0) + float(weights[row, col].item())
                    else:
                        for slot_id in active_slots:
                            usage_stats[block_id][slot_id] = usage_stats[block_id].get(slot_id, 0.0) + float(weights.mean().item())
        if total_examples > 0:
            for block_usage in usage_stats.values():
                for slot_id in list(block_usage.keys()):
                    block_usage[slot_id] /= float(total_examples)
        return usage_stats

    def _run_chu(self, planner_out, usage_stats):
        report = {}
        for block_id in self.model.selected_blocks:
            layer = self.model.layers[str(block_id)]
            report[block_id] = self.chu.consolidate_layer(layer, planner_out[block_id], usage_stats.get(block_id, {}))
        self.last_train_state["chu_calls"] += 1
        return report

    def _append_history(self, task_id: int, task_state, usage_stats, planner_out):
        slot_usage_values = []
        shared_usage_values = []
        rank_values = []
        for block_id in self.model.selected_blocks:
            block_usage = usage_stats.get(block_id, {})
            layer = self.model.layers[str(block_id)]
            slot_usage_values.append(sum(block_usage.values()) / max(len(block_usage), 1))
            shared_usage_values.append(float(planner_out.get(block_id, {}).get("shared_gate", 1.0)))
            live_slots = layer.live_slot_ids()
            if live_slots:
                rank_values.append(
                    sum(layer.slot_metadata[slot_id].rank / layer.slot_r_max for slot_id in live_slots) / len(live_slots)
                )
            else:
                rank_values.append(0.0)
        usage_summary = task_state.embedding.new_tensor([slot_usage_values + shared_usage_values])
        active_rank_summary = task_state.embedding.new_tensor([rank_values])
        pooled_feature_mean = pool_vector(task_state.feature_mean, self.history_pool_dim)
        pooled_feature_var = pool_vector(task_state.feature_var, self.history_pool_dim)
        entry = build_history_entry(
            task_id=task_id,
            task_state=task_state,
            usage_summary=usage_summary,
            active_rank_summary=active_rank_summary,
            pooled_feature_mean=pooled_feature_mean,
            pooled_feature_var=pooled_feature_var,
        )
        self.last_train_state["history_summary_dim"] = entry.summary_vector.size(-1)
        self.history_bank.append(entry)

    def _evaluate_seen_tasks(self, current_task_id: int, planner_out):
        accuracies = []
        self.model.eval()
        with torch.no_grad():
            for task_id in range(current_task_id + 1):
                _, dataset = self.benchmark.build_task_datasets(task_id)
                loader = self._make_loader(dataset, shuffle=False, batch_size=int(self.config["training"]["batch_size"]))
                correct = 0
                total = 0
                for _, images, labels in loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    logits, _, _ = self.model(images, None, planner_out)
                    predictions = logits.argmax(dim=-1)
                    correct += int((predictions == labels).sum().item())
                    total += int(labels.numel())
                accuracies.append(correct / max(total, 1))
        return accuracies

    def _build_metrics(self, task_records, accuracy_matrix, total_train_time, final_chu_report):
        final_per_task = accuracy_matrix[-1] if accuracy_matrix else []
        final_avg_acc = sum(final_per_task) / max(len(final_per_task), 1)
        avg_inc_acc = sum(record.avg_seen_accuracy for record in task_records) / max(len(task_records), 1)
        first_learned = [row[idx] for idx, row in enumerate(accuracy_matrix)]
        forgetting_terms = []
        bwt_terms = []
        for task_id in range(max(len(final_per_task) - 1, 0)):
            historical = [row[task_id] for row in accuracy_matrix[task_id:]]
            max_hist = max(historical) if historical else 0.0
            forgetting_terms.append(max_hist - final_per_task[task_id])
            bwt_terms.append(final_per_task[task_id] - first_learned[task_id])
        parameter_growth = sum(len(layer.slot_metadata) for layer in self.model.layers.values())
        total_active_rank = sum(meta.rank for layer in self.model.layers.values() for meta in layer.slot_metadata if not meta.pruned)
        opened_slots = sum(len(layer.slot_metadata) for layer in self.model.layers.values())
        pruned_slots = sum(1 for layer in self.model.layers.values() for meta in layer.slot_metadata if meta.pruned)
        return {
            "benchmark": self.benchmark.name,
            "final_avg_acc": final_avg_acc,
            "avg_inc_acc": avg_inc_acc,
            "forgetting": sum(forgetting_terms) / max(len(forgetting_terms), 1),
            "backward_transfer": sum(bwt_terms) / max(len(bwt_terms), 1),
            "last_task_acc": final_per_task[-1] if final_per_task else 0.0,
            "parameter_growth": parameter_growth,
            "total_active_rank": total_active_rank,
            "opened_slots": opened_slots,
            "pruned_slots": pruned_slots,
            "train_time_sec": total_train_time,
            "notes": {
                "num_tasks": len(self.benchmark.tasks),
                "history_entries": len(self.history_bank),
                "chu_report": {str(block_id): vars(report) for block_id, report in (final_chu_report or {}).items()},
            },
        }
