from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from src.datasets.registry import build_benchmark
from src.models.chu import ConsolidationHomeostasisUnit
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
from src.models.task_state import HistoryBank, TaskStateEncoder, build_history_entry
from src.utils.io import ensure_output_dirs, write_json


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
        self.task_state_encoder = TaskStateEncoder(
            feature_dim=self.model.backbone.embed_dim,
            grad_dim=int(config["warmup"].get("gradient_sketch_dim", 16)),
            embedding_dim=int(config["warmup"].get("task_embedding_dim", 128)),
        ).to(self.device)
        history_dim = self.task_state_encoder.embedding_dim + 4
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
            task_state = self._build_task_state(train_loader)
            self.last_train_state["task_states"].append(
                {
                    "task_id": task.task_id + 1,
                    "similarity_mean": float(task_state.similarity.mean().item()),
                    "has_history": len(self.history_bank) > 0,
                }
            )
            self.model.capture_pre_task_snapshots()
            if task.task_id == 0:
                planner_out = self.model.initialize_bootstrap_structure(task_id=task.task_id + 1)
                teacher_model = None
                teacher_planner_out = None
                self.last_train_state["bootstrap_used"] = True
            else:
                teacher_model = self.model.make_teacher_snapshot()
                teacher_planner_out = self._build_eval_planner_out(teacher_model)
                planner_out = self._plan_structure(task_state, task.task_id + 1)
                if task.task_id == 1:
                    self.last_train_state["teacher_used_on_task2"] = teacher_model is not None
                    self.last_train_state["planner_used_on_task2"] = True
                    self.last_train_state["materialize_used_on_task2"] = True

            required_classes = max(task.class_ids) + 1
            num_new_classes = required_classes - self.model.classifier.num_classes
            self.model.expand_classifier(num_new_classes)
            self.last_train_state["classifier_sizes"].append(self.model.classifier.num_classes)
            optimizer = self._build_optimizer()

            start_time = time.time()
            self._train_single_task(
                train_loader=train_loader,
                task=task,
                task_state=task_state,
                planner_out=planner_out,
                teacher_model=teacher_model,
                teacher_planner_out=teacher_planner_out,
                optimizer=optimizer,
            )
            train_time = time.time() - start_time
            total_train_time += train_time

            usage_stats = self._estimate_slot_usage(task, task_state, self._build_eval_planner_out(self.model))
            final_chu_report = self._run_chu(planner_out, usage_stats)
            self.last_train_state["chu_calls_per_task"].append(1)
            self._append_history(task.task_id + 1, task_state, usage_stats)
            self.last_train_state["history_sizes"].append(len(self.history_bank))

            accuracies = self._evaluate_seen_tasks(task.task_id, self._build_eval_planner_out(self.model))
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
        params = [p for p in list(self.model.parameters()) + list(self.planner.parameters()) + list(self.task_state_encoder.parameters()) if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=float(self.config["training"]["lr"]),
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

    def _build_task_state(self, train_loader: DataLoader):
        warm_batches = int(self.config["warmup"]["num_batches"])
        feature_chunks = []
        entropy_values = []
        grad_sketches = []
        labels_for_aux = []
        features_for_aux = []

        for batch_idx, (_, images, labels) in enumerate(train_loader):
            if batch_idx >= warm_batches:
                break
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                features = self.model.extract_features(images)
            feature_chunks.append(features)
            local_classes = sorted(labels.unique().tolist())
            class_to_local = {int(class_id): offset for offset, class_id in enumerate(local_classes)}
            local_targets = torch.tensor([class_to_local[int(label.item())] for label in labels], device=self.device)
            aux_head = nn.Linear(features.size(-1), max(len(local_classes), 2), device=self.device)
            aux_logits = aux_head(features.detach())
            entropy = (-torch.softmax(aux_logits, dim=-1) * torch.log_softmax(aux_logits, dim=-1)).sum(dim=-1).mean()
            entropy_values.append(entropy.detach())
            aux_loss = F.cross_entropy(aux_logits, local_targets)
            aux_head.zero_grad()
            aux_loss.backward()
            grad_sketches.append(self._compress_gradient(aux_head.weight.grad.detach()))
            labels_for_aux.append(labels)
            features_for_aux.append(features.detach())

        feature_tensor = torch.cat(feature_chunks, dim=0)
        feature_mean = feature_tensor.mean(dim=0, keepdim=True)
        feature_var = feature_tensor.var(dim=0, unbiased=False, keepdim=True)
        gradient_sketch = torch.cat(grad_sketches, dim=0).mean(dim=0, keepdim=True)
        entropy = torch.stack(entropy_values).mean().view(1, 1)

        provisional = self.task_state_encoder(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=feature_mean.new_zeros(1, 1),
            entropy=entropy,
        )
        if len(self.history_bank) == 0:
            similarity = feature_mean.new_zeros(1, 1)
        else:
            similarity = self.history_bank.mean_similarity(provisional.embedding)
        task_state = self.task_state_encoder(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=similarity,
            entropy=entropy,
        )
        return task_state

    def _plan_structure(self, task_state, task_id: int):
        history_summary = self.history_bank.aggregate()
        planner_out = {}
        for block_id in self.model.selected_blocks:
            layer = self.model.layers[str(block_id)]
            signals = self.planner(block_id, task_state.embedding, history_summary)
            action = self.planner.decide_action(signals)
            planner_out[block_id] = materialize_action(
                action=action,
                signals=signals,
                slot_bank=layer,
                task_embedding=task_state.embedding.detach(),
                task_id=task_id,
                max_slots_per_block=layer.max_slots,
            )
        return planner_out

    def _build_eval_planner_out(self, model):
        planner_out = {}
        for block_id in model.selected_blocks:
            layer = model.layers[str(block_id)]
            live_slots = layer.live_slot_ids()
            planner_out[block_id] = {
                "action": "eval_all_live_slots",
                "active_slots": live_slots,
                "rank_cfg": {slot_id: layer.slot_metadata[slot_id].rank for slot_id in live_slots},
                "shared_gate": 1.0,
                "consolidate_flag": False,
                "deterministic": len(live_slots) <= 1,
                "created_new_slot": False,
            }
        return planner_out

    def _train_single_task(self, train_loader, task, task_state, planner_out, teacher_model, teacher_planner_out, optimizer):
        epochs = int(self.config["training"]["epochs_per_task"])
        grad_clip = float(self.config["training"]["grad_clip_norm"])
        device = self.device
        old_class_cutoff = min(task.class_ids)
        for _ in range(epochs):
            self.model.train()
            for _, images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits, features, route_info = self.model(images, task_state, planner_out)
                if route_info:
                    self.last_train_state["router_seen"] = True
                loss_cls = F.cross_entropy(logits, labels)
                loss_rank = rank_penalty(self.model)
                loss_route = routing_balance_loss(route_info, device)
                if task.task_id == 0 or teacher_model is None or teacher_planner_out is None or old_class_cutoff == 0:
                    loss_orth = slot_orthogonality(self.model)
                    loss = (
                        loss_cls
                        + float(self.config["loss"]["lambda_orth"]) * loss_orth
                        + float(self.config["loss"]["lambda_rank"]) * loss_rank
                        + float(self.config["loss"]["lambda_route"]) * loss_route
                    )
                else:
                    with torch.no_grad():
                        teacher_logits, teacher_features, _ = teacher_model(images, None, teacher_planner_out)
                    old_classes = list(range(old_class_cutoff))
                    loss_kd = kd_loss(
                        logits[:, old_classes],
                        teacher_logits[:, old_classes],
                        temperature=float(self.config["loss"]["kd_temperature"]),
                    ) if old_classes else torch.zeros((), device=device)
                    loss_feat = feature_retention(features, teacher_features)
                    loss_orth = slot_orthogonality(self.model)
                    loss_grow = growth_penalty(planner_out, device)
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
        self.model.eval()
        with torch.no_grad():
            for _, images, _ in loader:
                images = images.to(self.device)
                _, _, route_info = self.model(images, task_state, planner_out)
                for block_id, layer_route in route_info.items():
                    active_slots = layer_route.get("active_slots", [])
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
        return usage_stats

    def _run_chu(self, planner_out, usage_stats):
        report = {}
        for block_id in self.model.selected_blocks:
            layer = self.model.layers[str(block_id)]
            report[block_id] = self.chu.consolidate_layer(layer, planner_out[block_id], usage_stats.get(block_id, {}))
        self.last_train_state["chu_calls"] += 1
        return report

    def _append_history(self, task_id: int, task_state, usage_stats):
        flat_usage = []
        flat_ranks = []
        for block_id in self.model.selected_blocks:
            block_usage = usage_stats.get(block_id, {})
            flat_usage.extend(block_usage.values())
            layer = self.model.layers[str(block_id)]
            flat_ranks.extend([layer.slot_metadata[slot_id].rank for slot_id in layer.live_slot_ids()])
        mean_usage = float(sum(flat_usage) / max(len(flat_usage), 1))
        mean_rank = float(sum(flat_ranks) / max(len(flat_ranks), 1))
        self.history_bank.append(build_history_entry(task_id, task_state, mean_usage, mean_rank))

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
