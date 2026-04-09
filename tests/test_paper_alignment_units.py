from __future__ import annotations

import logging
import os
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition
from src.datasets.transforms import build_cifar_test_transform
from src.engine.trainer import NHLoRATrainer
from src.models.chu import ConsolidationHomeostasisUnit
from src.models.lora import NHLoRALayer
from src.models.losses import feature_retention, growth_penalty, rank_penalty, routing_balance_loss, slot_orthogonality
from src.models.nh_lora import NHLoRAModel
from src.models.planner import HorizonPlanner, MaterializedLayerPlan, PlannerControlOutputs, PlannerSignals, materialize_action
from src.utils.logging_utils import configure_logger
from src.utils.seeding import seed_everything


def _make_image_bytes(label: int, sample_id: int, size: int = 32) -> bytes:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = (label * 17 + sample_id * 3) % 255
    image[..., 1] = (label * 29 + sample_id * 5) % 255
    image[..., 2] = (label * 43 + sample_id * 7) % 255
    return image.tobytes()


def _make_records(class_ids, split: str, samples_per_class: int):
    records = []
    for class_id in class_ids:
        for sample_id in range(samples_per_class):
            records.append(
                SampleRecord(
                    path=None,
                    label=class_id,
                    split=split,
                    image_bytes=_make_image_bytes(class_id, sample_id),
                    metadata={"size": (32, 32)},
                )
            )
    return records


def _build_tiny_benchmark(num_tasks: int = 1):
    tasks = []
    all_tasks = [
        TaskDefinition(
            task_id=0,
            class_ids=[0, 1],
            train_records=_make_records([0, 1], "train", 4),
            test_records=_make_records([0, 1], "test", 2),
        ),
        TaskDefinition(
            task_id=1,
            class_ids=[2, 3],
            train_records=_make_records([2, 3], "train", 4),
            test_records=_make_records([2, 3], "test", 2),
        ),
    ]
    tasks.extend(all_tasks[:num_tasks])
    transform = build_cifar_test_transform(32)
    return ContinualBenchmark(
        name="tiny_alignment",
        num_classes=4,
        tasks=tasks,
        train_transform=transform,
        test_transform=transform,
    )


def _build_test_config(output_root: str):
    return {
        "experiment": {
            "name": "nh_lora_unit",
            "project_name": "nh_lora_unit",
            "output_root": output_root,
            "log_dir": str(Path(output_root) / "logs"),
            "metrics_dir": str(Path(output_root) / "metrics"),
            "summaries_dir": str(Path(output_root) / "summaries"),
            "checkpoints_dir": str(Path(output_root) / "checkpoints"),
            "save_checkpoints": True,
        },
        "runtime": {
            "device": "cpu",
            "num_workers": 0,
            "deterministic": True,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "benchmark": {
            "name": "tiny_alignment",
            "dataset_name": "synthetic_smoke",
            "data_root": output_root,
            "num_tasks": 1,
            "classes_per_task": 2,
            "image_size": 32,
        },
        "model": {
            "name": "nh_lora",
            "backbone_name": "toy_vit_tiny",
            "backbone_source": "internal",
            "freeze_backbone": True,
            "selected_blocks": [1],
            "insertion_points": ["q_proj", "v_proj"],
        },
        "nh_lora": {
            "shared_rank": 2,
            "shared_lr_scale": 0.5,
            "slot_r_max": 4,
            "slot_init_rank": 2,
            "bootstrap_slot_rank": 1,
            "max_slots_per_block": 2,
            "router_topk": 1,
            "router_candidate_pool": 3,
            "router_temperature": 1.0,
            "use_rank_mask": True,
        },
        "warmup": {
            "enabled": True,
            "num_batches": 2,
            "task_embedding_dim": 32,
            "gradient_sketch_dim": 8,
        },
        "planner": {
            "type": "mlp_per_layer",
            "hidden_dim": 32,
            "layer_embedding_dim": 8,
            "history_pool_dim": 4,
            "tau_novelty": 0.4,
            "tau_conflict": 0.4,
            "tau_consolidate": 0.5,
            "rank_min": 1,
            "rank_max": 4,
            "use_history_aggregation": True,
        },
        "chu": {
            "merge_rate": 0.1,
            "usage_high_threshold": 0.05,
            "usage_low_threshold": 0.0,
            "stability_threshold": 0.2,
            "redundancy_threshold": 0.99,
            "freeze_on_keep": True,
        },
        "classifier": {
            "type": "cosine",
            "tau_cls": 8.0,
            "init_mode": "imprint",
        },
        "training": {
            "optimizer": "adamw",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs_per_task": 1,
            "batch_size": 4,
            "grad_clip_norm": 5.0,
            "use_scheduler": True,
            "scheduler": "cosine",
            "freeze_old_classifier_weights": True,
            "classifier_lr_scale": 1.0,
            "freeze_new_classifier_epochs": 0,
            "freeze_all_classifier_epochs": 0,
            "retention_debug_logging": True,
            "retention_feature_diff_logging": False,
            "retention_feature_diff_max_epochs": 3,
            "routing_debug_logging": False,
            "routing_debug_max_epochs": 3,
            "planner_audit_logging": False,
            "planner_mode": "legacy",
            "planner_control_recompute": "per_batch",
            "planner_policy_trainable": False,
            "planner_control_trainable": True,
            "planner_use_learned_shared_gate": True,
            "planner_control_delta_logit_scale": 2.0,
            "planner_soft_rank_training": False,
            "planner_soft_rank_temperature": 0.5,
            "planner_hard_rank_eval": True,
            "adapter_delta_debug_logging": False,
            "adapter_delta_debug_max_epochs": 3,
            "final_feature_diff_debug_logging": False,
            "final_feature_diff_debug_max_epochs": 3,
            "classifier_drift_debug_logging": False,
            "classifier_drift_debug_max_epochs": 3,
            "grad_norm_debug_logging": False,
            "grad_norm_debug_max_epochs": 3,
            "logit_margin_debug_logging": False,
            "logit_margin_debug_max_epochs": 3,
        },
        "loss": {
            "lambda_kd": 0.5,
            "lambda_feat": 0.5,
            "lambda_orth": 0.05,
            "lambda_rank": 1e-4,
            "lambda_grow": 0.01,
            "lambda_route": 0.01,
            "kd_temperature": 2.0,
            "retention_layers": [1],
            "retention_feature_representation": "cls",
        },
    }


class _DummyModel(nn.Module):
    def __init__(self, layer: NHLoRALayer):
        super().__init__()
        self.layers = nn.ModuleDict({"0": layer})


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


class PaperAlignmentUnitTests(unittest.TestCase):
    def _build_stage4_model(self, insertion_points):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / ("stage4_" + "_".join(insertion_points))
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["model"]["insertion_points"] = list(insertion_points)
        return NHLoRAModel(config)

    def _stage4_planner_cfg(self, slot_id: int, accumulator=None):
        planner_cfg = {
            "active_slot_candidates": [slot_id],
            "selected_slot": slot_id,
            "rank_cfg": {slot_id: 1},
            "shared_gate": 1.0,
            "deterministic": True,
            "shared_only": False,
        }
        if accumulator is not None:
            planner_cfg["_debug_delta_stats"] = accumulator
            planner_cfg["_debug_block_id"] = 1
        return planner_cfg

    def test_dynamic_slot_params_follow_layer_device(self):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        ).to(target_device)
        slot_id = layer.add_slot(initial_rank=1, task_id=1)

        self.assertEqual(layer.slot_keys[slot_id].device, layer.query_proj.weight.device)
        bank = layer.point_banks["q_proj"]
        self.assertEqual(bank.slot_a[slot_id].device, target_device)
        self.assertEqual(bank.slot_b[slot_id].device, target_device)

        hidden_states = torch.randn(2, 5, 8, device=target_device)
        delta = layer._slot_delta(
            "q_proj",
            hidden_states,
            route_state={"candidate_slots": [slot_id]},
            planner_cfg={"rank_cfg": {slot_id: 1}},
        )
        self.assertEqual(delta.device, hidden_states.device)

    def test_planner_input_contains_all_terms(self):
        planner = HorizonPlanner(
            selected_blocks=[0],
            task_embedding_dim=4,
            history_dim=6,
            hidden_dim=8,
            layer_embedding_dim=3,
            rank_min=1,
            rank_max=4,
            tau_novelty=0.5,
            tau_conflict=0.5,
        )
        task_embedding = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        history_summary = torch.tensor([[0.2, 0.1, 0.3, 0.4, 0.0, 0.5]])
        signals = planner(0, task_embedding=task_embedding, history_summary=history_summary)
        layer_embedding = planner.layer_embeddings(torch.tensor([0])).expand(1, -1)
        expected = torch.cat(
            [
                task_embedding,
                signals.history_context,
                torch.abs(task_embedding - signals.history_context),
                task_embedding * signals.history_context,
                layer_embedding,
            ],
            dim=-1,
        )
        self.assertTrue(torch.allclose(signals.planner_input, expected, atol=1e-6))

    def test_hybrid_planner_exposes_separate_policy_and_control_params(self):
        planner = HorizonPlanner(
            selected_blocks=[0],
            task_embedding_dim=4,
            history_dim=6,
            hidden_dim=8,
            layer_embedding_dim=3,
            rank_min=1,
            rank_max=4,
            tau_novelty=0.5,
            tau_conflict=0.5,
        )
        policy_param_ids = {id(parameter) for parameter in planner.policy_parameters()}
        control_param_ids = {id(parameter) for parameter in planner.control_parameters()}

        self.assertTrue(policy_param_ids)
        self.assertTrue(control_param_ids)
        self.assertTrue(policy_param_ids.isdisjoint(control_param_ids))

        task_embedding = torch.randn(1, 4)
        history_summary = torch.randn(1, 6)
        policy_outputs = planner.forward_policy(0, task_embedding=task_embedding, history_summary=history_summary)
        control_outputs = planner.forward_control(0, task_embedding=task_embedding, history_summary=history_summary)

        self.assertEqual(tuple(policy_outputs.shared_gate.shape), (1, 1))
        self.assertEqual(tuple(control_outputs.shared_gate.shape), (1, 1))
        self.assertEqual(tuple(control_outputs.delta_raw.shape), (1, 1))
        self.assertEqual(tuple(control_outputs.delta_from_representation.shape), (1, 1))
        self.assertEqual(tuple(control_outputs.delta_bias.shape), (1, 1))
        self.assertTrue(torch.allclose(control_outputs.delta_raw, torch.zeros_like(control_outputs.delta_raw)))
        self.assertTrue(
            torch.allclose(control_outputs.delta_from_representation, torch.zeros_like(control_outputs.delta_from_representation))
        )
        self.assertTrue(torch.allclose(control_outputs.delta_bias, torch.zeros_like(control_outputs.delta_bias)))
        self.assertGreaterEqual(float(control_outputs.control_head_weight_norm), 0.0)
        self.assertAlmostEqual(float(control_outputs.control_head_bias_norm), 0.0, places=6)

    def test_hybrid_anchored_gate_stays_on_anchor_at_zero_init_and_moves_monotonically(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "hybrid_anchor_gate_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_mode"] = "hybrid"
        benchmark = _build_tiny_benchmark(num_tasks=1)
        trainer = NHLoRATrainer(config, configure_logger(level=logging.WARNING), benchmark=benchmark)

        raw_outputs = PlannerControlOutputs(
            shared_gate=torch.tensor([[0.5]], dtype=torch.float32),
            shared_gate_logit=torch.zeros(1, 1),
            delta_raw=torch.zeros(1, 1),
        )
        anchored = trainer._compose_hybrid_control_outputs(raw_outputs, anchor_beta=0.7)
        self.assertAlmostEqual(float(anchored.shared_gate.item()), 0.7, places=6)
        self.assertAlmostEqual(float(anchored.anchor_beta.item()), 0.7, places=6)
        self.assertAlmostEqual(float(anchored.delta_logit.item()), 0.0, places=6)

        positive = trainer._compose_hybrid_control_outputs(
            PlannerControlOutputs(
                shared_gate=torch.sigmoid(torch.tensor([[1.0]], dtype=torch.float32)),
                shared_gate_logit=torch.tensor([[1.0]], dtype=torch.float32),
                delta_raw=torch.tensor([[1.0]], dtype=torch.float32),
            ),
            anchor_beta=0.5,
        )
        negative = trainer._compose_hybrid_control_outputs(
            PlannerControlOutputs(
                shared_gate=torch.sigmoid(torch.tensor([[-1.0]], dtype=torch.float32)),
                shared_gate_logit=torch.tensor([[-1.0]], dtype=torch.float32),
                delta_raw=torch.tensor([[-1.0]], dtype=torch.float32),
            ),
            anchor_beta=0.5,
        )
        self.assertGreater(float(positive.shared_gate.item()), 0.5)
        self.assertLess(float(negative.shared_gate.item()), 0.5)
        self.assertLessEqual(
            float(positive.delta_logit.abs().max().item()),
            float(config["training"]["planner_control_delta_logit_scale"]) + 1e-6,
        )
        expected_positive_delta = float(
            config["training"]["planner_control_delta_logit_scale"]
            * F.softsign(torch.tensor([[1.0]], dtype=torch.float32)).item()
        )
        self.assertAlmostEqual(float(positive.delta_logit.item()), expected_positive_delta, places=6)

    def test_planner_control_representation_normalization_is_rms_stable(self):
        planner = HorizonPlanner(
            selected_blocks=[0],
            task_embedding_dim=4,
            history_dim=6,
            hidden_dim=8,
            layer_embedding_dim=3,
            rank_min=1,
            rank_max=4,
            tau_novelty=0.5,
            tau_conflict=0.5,
        )
        representation = torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]], dtype=torch.float32)
        scaled_representation = representation * 11.0
        normalized = planner.control_branch._normalize_control_representation(representation)
        normalized_scaled = planner.control_branch._normalize_control_representation(scaled_representation)
        rms = torch.sqrt(normalized.pow(2).mean(dim=-1))
        self.assertTrue(torch.allclose(normalized, normalized_scaled, atol=1e-6))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-6))

        output_head = planner.control_branch.output_heads["0"]
        with torch.no_grad():
            output_head.weight.fill_(1.0)
            if output_head.bias is not None:
                output_head.bias.zero_()
        delta_from_representation = F.linear(normalized, output_head.weight, bias=None)
        delta_from_scaled_representation = F.linear(normalized_scaled, output_head.weight, bias=None)
        self.assertTrue(torch.allclose(delta_from_representation, delta_from_scaled_representation, atol=1e-6))

    def test_softsign_bounded_residual_keeps_gradient_on_large_delta_raw(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "hybrid_softsign_gradient_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_mode"] = "hybrid"
        benchmark = _build_tiny_benchmark(num_tasks=1)
        trainer = NHLoRATrainer(config, configure_logger(level=logging.WARNING), benchmark=benchmark)

        delta_raw = torch.tensor([[10.0]], dtype=torch.float32, requires_grad=True)
        outputs = trainer._compose_hybrid_control_outputs(
            PlannerControlOutputs(
                shared_gate=torch.sigmoid(delta_raw.detach()),
                shared_gate_logit=delta_raw.detach(),
                delta_raw=delta_raw,
            ),
            anchor_beta=0.5,
        )
        loss = outputs.shared_gate.sum()
        loss.backward()

        self.assertIsNotNone(delta_raw.grad)
        self.assertGreater(abs(float(delta_raw.grad.item())), 0.0)
        self.assertAlmostEqual(
            float(outputs.delta_logit.item()),
            float(config["training"]["planner_control_delta_logit_scale"] * F.softsign(torch.tensor([[10.0]])).item()),
            places=6,
        )

    def test_materialize_action_is_pure_and_reuse_shared_is_shared_only(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
        )
        layer.add_slot(initial_rank=1, task_id=1)
        before_slots = len(layer.slot_metadata)
        before_rank = layer.slot_metadata[0].rank
        signals = PlannerSignals(
            novelty=torch.tensor([[0.2]]),
            conflict=torch.tensor([[0.2]]),
            rank_score=torch.tensor([[0.1]]),
            rank_budget=1,
            consolidate=torch.tensor([[0.3]]),
            shared_gate=torch.tensor([[0.7]]),
        )
        materialized = materialize_action(
            action="reuse_shared",
            signals=signals,
            slot_bank=layer,
            task_embedding=torch.randn(1, 8),
            task_id=2,
            max_slots_per_block=2,
        )
        self.assertEqual(len(layer.slot_metadata), before_slots)
        self.assertEqual(layer.slot_metadata[0].rank, before_rank)
        self.assertTrue(materialized.shared_only)
        self.assertEqual(materialized.candidate_slots, [])

    def test_apply_structure_changes_mutates_and_freezes(self):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        ).to(target_device)
        open_plan = MaterializedLayerPlan(
            requested_action="open_new_slot",
            action="open_new_slot",
            selected_slot=None,
            target_rank=None,
            rank_delta=0,
            create_new_slot=True,
            new_slot_rank=2,
        )
        runtime = layer.apply_structure_change(open_plan, torch.randn(1, 8, device=target_device), task_id=1)
        self.assertEqual(len(layer.slot_metadata), 1)
        self.assertTrue(runtime["created_new_slot"])
        self.assertEqual(layer.slot_keys[0].device, target_device)
        self.assertEqual(layer.point_banks["q_proj"].slot_a[0].device, target_device)

        freeze_plan = MaterializedLayerPlan(
            requested_action="freeze_old_strong_retention",
            action="freeze_old_strong_retention",
            selected_slot=None,
            target_rank=None,
            rank_delta=0,
            create_new_slot=False,
            new_slot_rank=None,
            strong_retention=True,
            shared_only=True,
        )
        frozen_runtime = layer.apply_structure_change(freeze_plan, torch.randn(1, 8, device=target_device), task_id=2)
        self.assertTrue(layer.slot_metadata[0].frozen)
        self.assertEqual(frozen_runtime["active_slot_candidates"], [])

    def test_load_structure_state_keeps_dynamic_params_on_layer_device(self):
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        ).to(target_device)
        slot_id = layer.add_slot(initial_rank=2, task_id=1)
        layer.initialize_slot_key(slot_id, torch.randn(1, 8, device=target_device))
        state = layer.export_structure_state()

        restored = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        ).to(target_device)
        restored.load_structure_state(state)

        self.assertEqual(restored.slot_keys[0].device, target_device)
        self.assertEqual(restored.point_banks["q_proj"].slot_a[0].device, target_device)
        self.assertEqual(restored.point_banks["q_proj"].slot_b[0].device, target_device)

    def test_slot_full_fallback_uses_compatible_existing_slot(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=1,
            router_topk=1,
            router_temperature=1.0,
        )
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        task_embedding = torch.randn(1, 8)
        layer.initialize_slot_key(slot_id, task_embedding)
        signals = PlannerSignals(
            novelty=torch.tensor([[0.9]]),
            conflict=torch.tensor([[0.9]]),
            rank_score=torch.tensor([[0.75]]),
            rank_budget=3,
            consolidate=torch.tensor([[0.6]]),
            shared_gate=torch.tensor([[0.4]]),
        )
        materialized = materialize_action(
            action="open_new_slot",
            signals=signals,
            slot_bank=layer,
            task_embedding=task_embedding,
            task_id=2,
            max_slots_per_block=1,
        )
        self.assertEqual(materialized.fallback_action, "expand_rank_existing_slot")
        self.assertEqual(materialized.action, "expand_rank_existing_slot")
        self.assertEqual(materialized.selected_slot, 0)
        self.assertIn(0, materialized.compatibility_scores)

    def test_materialize_action_preserves_candidate_pool(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=4,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        )
        task_embedding = torch.randn(1, 8)
        for task_id in range(3):
            slot_id = layer.add_slot(initial_rank=1, task_id=task_id + 1)
            layer.initialize_slot_key(slot_id, task_embedding + float(task_id) * 0.01)
        signals = PlannerSignals(
            novelty=torch.tensor([[0.9]]),
            conflict=torch.tensor([[0.2]]),
            rank_score=torch.tensor([[0.75]]),
            rank_budget=2,
            consolidate=torch.tensor([[0.6]]),
            shared_gate=torch.tensor([[0.4]]),
        )

        materialized = materialize_action(
            action="expand_rank_existing_slot",
            signals=signals,
            slot_bank=layer,
            task_embedding=task_embedding,
            task_id=4,
            max_slots_per_block=4,
            router_candidate_pool=3,
        )

        self.assertEqual(len(materialized.candidate_slots), 3)
        self.assertIn(materialized.selected_slot, materialized.candidate_slots)

    def test_apply_structure_change_keeps_created_slot_in_candidate_pool(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=4,
            router_topk=2,
            router_temperature=1.0,
            task_embedding_dim=8,
        )
        task_embedding = torch.randn(1, 8)
        first_slot = layer.add_slot(initial_rank=1, task_id=1)
        second_slot = layer.add_slot(initial_rank=1, task_id=2)
        open_plan = MaterializedLayerPlan(
            requested_action="open_new_slot",
            action="open_new_slot",
            selected_slot=None,
            target_rank=None,
            rank_delta=0,
            create_new_slot=True,
            new_slot_rank=1,
            candidate_slots=[first_slot, second_slot],
        )

        runtime = layer.apply_structure_change(open_plan, task_embedding, task_id=3)

        self.assertEqual(len(layer.slot_metadata), 3)
        self.assertIn(2, runtime["active_slot_candidates"])
        self.assertEqual(len(runtime["active_slot_candidates"]), 3)

    def test_route_uses_sparse_distribution_for_candidate_pool(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=4,
            router_topk=2,
            router_temperature=1.0,
            task_embedding_dim=8,
        )
        for task_id in range(3):
            layer.add_slot(initial_rank=1, task_id=task_id + 1)
        tokens = torch.randn(4, 5, 8)

        route_state = layer.route(
            tokens,
            planner_cfg={"active_slot_candidates": [0, 1, 2], "deterministic": False, "shared_only": False},
            task_state=None,
        )

        self.assertEqual(route_state["routing_distribution"].shape, (4, 3))
        self.assertEqual(route_state["routing_weights"].shape[-1], 2)
        self.assertIn("topk_indices", route_state)

    def test_chu_merge_rate_zero_keeps_merge_candidate_for_inference(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
        )
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        layer.capture_pre_task_snapshot()
        chu = ConsolidationHomeostasisUnit(
            {
                "merge_rate": 0.0,
                "usage_high_threshold": 0.1,
                "usage_low_threshold": 0.0,
                "stability_threshold": 0.0,
                "redundancy_threshold": 1.0,
                "freeze_on_keep": False,
            }
        )

        report = chu.consolidate_layer(layer, {"consolidate_flag": True}, {slot_id: 1.0})

        self.assertEqual(report.merged_slots, 0)
        self.assertEqual(report.kept_slots, 1)
        self.assertTrue(layer.slot_metadata[slot_id].retained_for_inference)

    def test_losses_follow_target_formulations(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=2,
            router_temperature=1.0,
        )
        layer.add_slot(initial_rank=1, task_id=1)
        layer.add_slot(initial_rank=1, task_id=1)
        bank = layer.point_banks["q_proj"]
        with torch.no_grad():
            bank.slot_a[0][:1].fill_(1.0)
            bank.slot_a[1][:1].fill_(1.0)
        model = _DummyModel(layer)
        orth = slot_orthogonality(model)
        self.assertGreater(float(orth.item()), 0.0)

        low_rank = rank_penalty(model, torch.device("cpu"))
        layer.expand_rank(1, 3)
        high_rank = rank_penalty(model, torch.device("cpu"))
        self.assertGreater(float(high_rank.item()), float(low_rank.item()))

        grow_on = growth_penalty({0: {"created_new_slot": True}}, torch.device("cpu"))
        grow_off = growth_penalty({0: {"created_new_slot": False}}, torch.device("cpu"))
        self.assertGreater(float(grow_on.item()), float(grow_off.item()))

        balanced = routing_balance_loss(
            {0: {"routing_distribution": torch.tensor([[0.5, 0.5], [0.5, 0.5]])}},
            torch.device("cpu"),
        )
        collapsed = routing_balance_loss(
            {0: {"routing_distribution": torch.tensor([[1.0, 0.0], [1.0, 0.0]])}},
            torch.device("cpu"),
        )
        self.assertLess(float(balanced.item()), 1e-6)
        self.assertGreater(float(collapsed.item()), float(balanced.item()))

    def test_eval_planning_not_shortcut(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = _build_test_config(str(repo_root / "outputs" / "test_tmp" / "unit_eval"))
        model = NHLoRAModel(config)
        layer = model.layers["1"]
        first_slot = layer.add_slot(initial_rank=1, task_id=1)
        second_slot = layer.add_slot(initial_rank=1, task_id=2)
        layer.slot_metadata[first_slot].retained_for_inference = True
        layer.slot_metadata[first_slot].usage_ema = 0.9
        layer.slot_metadata[second_slot].retained_for_inference = True
        layer.slot_metadata[second_slot].usage_ema = 0.2
        layer.last_shared_gate = 0.35
        layer.last_structural_action = "expand_rank_existing_slot"
        profile = model.build_inference_profile()
        self.assertEqual(profile[1]["active_slot_candidates"], [first_slot, second_slot])
        self.assertIsNone(profile[1]["selected_slot"])
        self.assertFalse(profile[1]["deterministic"])
        self.assertAlmostEqual(profile[1]["shared_gate"], 0.35, places=6)

        layer.last_structural_action = "reuse_shared"
        profile = model.build_inference_profile()
        self.assertEqual(profile[1]["active_slot_candidates"], [])
        self.assertIsNone(profile[1]["selected_slot"])
        self.assertTrue(profile[1]["deterministic"])
        self.assertTrue(profile[1]["shared_only"])
        self.assertEqual(profile[1]["rank_cfg"], {first_slot: 1, second_slot: 1})

        layer.last_structural_action = "freeze_old_strong_retention"
        profile = model.build_inference_profile()
        self.assertEqual(profile[1]["active_slot_candidates"], [])
        self.assertIsNone(profile[1]["selected_slot"])
        self.assertTrue(profile[1]["deterministic"])
        self.assertTrue(profile[1]["shared_only"])
        self.assertTrue(profile[1]["strong_retention"])

    def test_retention_feature_representation_modes(self):
        repo_root = Path(__file__).resolve().parents[1]
        images = torch.randn(2, 3, 32, 32)
        expected_shapes = {
            "cls": (2, 128),
            "mean_pool_tokens": (2, 128),
            "full_tokens": (2, 65, 128),
        }
        for mode, expected_shape in expected_shapes.items():
            config = _build_test_config(str(repo_root / "outputs" / "test_tmp" / f"retention_repr_{mode}"))
            config["loss"]["retention_feature_representation"] = mode
            model = NHLoRAModel(config)
            model.classifier.expand(2)

            outputs = model.forward_with_state(images, task_state=None, planner_out=None)

            self.assertEqual(tuple(outputs["layer_features"][1].shape), expected_shape)

    def test_invalid_retention_feature_representation_fails_fast(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = _build_test_config(str(repo_root / "outputs" / "test_tmp" / "retention_repr_invalid"))
        config["loss"]["retention_feature_representation"] = "bad_mode"

        with self.assertRaises(ValueError):
            NHLoRAModel(config)

    def test_feature_retention_supports_full_token_tensors(self):
        student_features = {1: torch.zeros(2, 3, 4)}
        teacher_features = {1: torch.ones(2, 3, 4)}

        loss = feature_retention(student_features, teacher_features, layers=[1], device=torch.device("cpu"))

        self.assertGreater(float(loss.item()), 0.0)

    def test_feature_retention_diff_diagnostics_reports_matching_and_missing_layers(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "feature_diff_diag_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["retention_feature_diff_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = {"task_number": 2, "current_epoch": 1}

        trainer._log_feature_retention_diff(
            context=context,
            current_features={1: torch.ones(2, 4)},
            teacher_features={1: torch.zeros(2, 4)},
            retention_layers=[1, 2],
        )

        joined_messages = "\n".join(logger.messages)
        self.assertIn("requested_layers=[1, 2]", joined_messages)
        self.assertIn("matched_layers=[1]", joined_messages)
        self.assertIn("missing_student=[2]", joined_messages)
        self.assertIn("missing_teacher=[2]", joined_messages)
        self.assertIn("mean_abs_diff=", joined_messages)

    def test_feature_retention_diff_diagnostics_warns_on_no_matched_layers(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "feature_diff_missing_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["retention_feature_diff_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)

        trainer._log_feature_retention_diff(
            context={"task_number": 2, "current_epoch": 1},
            current_features={1: torch.ones(2, 4)},
            teacher_features={1: torch.zeros(2, 4)},
            retention_layers=[2],
        )

        joined_messages = "\n".join(logger.messages)
        self.assertIn("matched_layers=[]", joined_messages)
        self.assertIn("no matched retention layers", joined_messages)

    def test_bootstrap_task1_invariants(self):
        seed_everything(5, deterministic=True)
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "bootstrap_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = trainer._prepare_task_context(benchmark.tasks[0])

        self.assertIsNone(context["teacher_model"])
        self.assertEqual(context["raw_planner"], {})
        self.assertAlmostEqual(float(context["task_state"].similarity.item()), 0.0, places=6)
        self.assertTrue(all(plan["created_new_slot"] for plan in context["applied_plans"].values()))
        for layer in trainer.model.layers.values():
            for slot_id in layer.live_slot_ids():
                self.assertTrue(layer.slot_snapshots[slot_id])
                self.assertIn(slot_id, layer.estimate_slot_stability())

        train_dataset, _ = benchmark.build_task_datasets(0)
        loader = trainer._build_eval_loader(train_dataset)
        batch = next(iter(loader))
        images, labels = trainer._prepare_images_labels(batch)
        outputs = trainer.model.forward_with_state(images, context["task_state"], context["applied_plans"])
        outputs["images"] = images
        losses = trainer._compute_loss(context, outputs, labels)
        self.assertAlmostEqual(float(losses["kd"].item()), 0.0, places=7)
        self.assertAlmostEqual(float(losses["feat"].item()), 0.0, places=7)
        self.assertAlmostEqual(float(losses["grow"].item()), 0.0, places=7)

    def test_normalize_loss_dict_keeps_bootstrap_losses_as_zero(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "loss_normalize_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)

        normalized = trainer._normalize_loss_dict({"total": 1.5, "cls": 1.4, "orth": 0.1, "rank": 0.2, "route": 0.3})

        self.assertEqual(set(normalized.keys()), {"total", "cls", "kd", "feat", "orth", "rank", "grow", "route"})
        self.assertEqual(normalized["kd"], 0.0)
        self.assertEqual(normalized["feat"], 0.0)
        self.assertEqual(normalized["grow"], 0.0)

    def test_old_classifier_row_gradient_masking(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_mask_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(4)
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight)

        trainer._mask_old_classifier_gradients({"old_num_classes": 2})

        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad[:2], torch.zeros_like(trainer.model.classifier.weight.grad[:2])))
        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad[2:], torch.ones_like(trainer.model.classifier.weight.grad[2:])))

    def test_task1_classifier_gradient_masking_is_noop(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_mask_noop_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(2)
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight)

        trainer._mask_old_classifier_gradients({"old_num_classes": 0})

        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad, torch.ones_like(trainer.model.classifier.weight.grad)))

    def test_old_classifier_gradient_masking_can_be_disabled(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_mask_disabled_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["freeze_old_classifier_weights"] = False
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(4)
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight)

        trainer._mask_old_classifier_gradients({"old_num_classes": 2})

        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad, torch.ones_like(trainer.model.classifier.weight.grad)))

    def test_classifier_lr_scale_default_keeps_classifier_in_base_group(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_lr_default_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(2)

        optimizer = trainer._build_optimizer()
        classifier_param_ids = {id(parameter) for parameter in trainer.model.classifier.parameters()}
        classifier_groups = [
            group
            for group in optimizer.param_groups
            if any(id(parameter) in classifier_param_ids for parameter in group["params"])
        ]

        self.assertEqual(len(classifier_groups), 1)
        self.assertAlmostEqual(float(classifier_groups[0]["lr"]), float(config["training"]["lr"]), places=12)
        self.assertGreater(len(classifier_groups[0]["params"]), len(classifier_param_ids))

    def test_classifier_lr_scale_nondefault_uses_separate_param_group(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_lr_scaled_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["classifier_lr_scale"] = 0.25
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(2)

        optimizer = trainer._build_optimizer()
        classifier_param_ids = {id(parameter) for parameter in trainer.model.classifier.parameters()}
        classifier_groups = [
            group
            for group in optimizer.param_groups
            if any(id(parameter) in classifier_param_ids for parameter in group["params"])
        ]

        self.assertEqual(len(classifier_groups), 1)
        self.assertAlmostEqual(float(classifier_groups[0]["lr"]), float(config["training"]["lr"]) * 0.25, places=12)
        self.assertTrue(all(id(parameter) in classifier_param_ids for parameter in classifier_groups[0]["params"]))

    def test_freeze_all_classifier_epochs_masks_all_rows(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_freeze_all_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["freeze_old_classifier_weights"] = False
        config["training"]["freeze_all_classifier_epochs"] = 2
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(4)
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight)

        trainer._mask_old_classifier_gradients({"old_num_classes": 2, "current_epoch": 1})

        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad, torch.zeros_like(trainer.model.classifier.weight.grad)))

    def test_freeze_new_classifier_epochs_masks_only_new_rows(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_freeze_new_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["freeze_old_classifier_weights"] = False
        config["training"]["freeze_new_classifier_epochs"] = 2
        config["training"]["classifier_lr_scale"] = 0.5
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(4)
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight)

        trainer._mask_old_classifier_gradients({"old_num_classes": 2, "current_epoch": 1})

        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad[:2], torch.ones_like(trainer.model.classifier.weight.grad[:2])))
        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad[2:], torch.zeros_like(trainer.model.classifier.weight.grad[2:])))

    def test_freeze_new_classifier_epochs_expires_after_configured_epochs(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "classifier_freeze_new_expired_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["freeze_old_classifier_weights"] = False
        config["training"]["freeze_new_classifier_epochs"] = 1
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(4)
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight)

        trainer._mask_old_classifier_gradients({"old_num_classes": 2, "current_epoch": 2})

        self.assertTrue(torch.allclose(trainer.model.classifier.weight.grad, torch.ones_like(trainer.model.classifier.weight.grad)))

    def test_teacher_profile_is_built_from_teacher_model(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "teacher_profile_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        layer = trainer.model.layers["1"]
        first_slot = layer.add_slot(initial_rank=1, task_id=1)
        second_slot = layer.add_slot(initial_rank=1, task_id=2)
        layer.last_structural_action = "expand_rank_existing_slot"
        trainer.inference_profile = {1: {"active_slot_candidates": []}}

        _, teacher_profile = trainer._build_teacher_payload()

        self.assertEqual(teacher_profile[1]["active_slot_candidates"], [first_slot, second_slot])

    def test_retention_debug_logging_can_be_disabled(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "retention_debug_disabled_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["retention_debug_logging"] = False
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = {"strong_retention_layers": set()}

        self.assertEqual(trainer._active_retention_layers(context), [1])
        trainer._update_retention_debug(
            context=context,
            outputs={"logits": torch.ones(2, 2)},
            teacher_outputs={"logits": torch.ones(2, 2)},
            teacher_num_classes=2,
            retention_layers=[1],
            kd_term=torch.tensor(0.5),
            feat_term=torch.tensor(0.25),
        )
        self.assertNotIn("last_retention_debug", context)

    def test_retention_debug_includes_old_logit_differences(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "retention_head_debug_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = {"old_num_classes": 2}

        trainer._update_retention_debug(
            context=context,
            outputs={"logits": torch.tensor([[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]])},
            teacher_outputs={"logits": torch.tensor([[1.5, 2.0], [2.0, 6.0]])},
            teacher_num_classes=2,
            retention_layers=[1],
            kd_term=torch.tensor(0.5),
            feat_term=torch.tensor(0.25),
        )

        self.assertIn("old_logits_mean_abs_diff", context["last_retention_debug"])
        self.assertIn("old_logits_max_abs_diff", context["last_retention_debug"])
        self.assertGreater(context["last_retention_debug"]["old_logits_max_abs_diff"], 0.0)

    def test_routing_debug_logging_summarizes_epoch(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "routing_debug_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["routing_debug_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = {"task_number": 2, "current_epoch": 1}

        trainer._update_routing_debug(
            context,
            {
                1: {
                    "candidate_slots": [0, 1],
                    "shared_only": False,
                    "routing_weights": torch.tensor([[0.7, 0.3], [0.6, 0.4]]),
                    "routing_distribution": torch.tensor([[0.7, 0.3], [0.6, 0.4]]),
                }
            },
        )
        trainer._log_routing_debug(context)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("[Routing][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("avg_candidate_count=2.00", joined_messages)
        self.assertIn("avg_topk=2.00", joined_messages)

    def test_stage5_slot_lifecycle_summary_exposes_slot_ids_without_mutating_inputs(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "stage5_slot_lifecycle_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        layer = trainer.model.layers["1"]
        first_slot = layer.add_slot(initial_rank=1, task_id=1)
        second_slot = layer.add_slot(initial_rank=1, task_id=2)
        layer.slot_metadata[first_slot].retained_for_inference = False
        layer.slot_metadata[first_slot].usage_ema = 0.1
        layer.slot_metadata[second_slot].usage_ema = 0.7
        layer.freeze_slot(second_slot)
        config_like = {
            "active_slot_candidates": [second_slot],
            "selected_slot": second_slot,
            "shared_only": False,
            "fallback_action": None,
        }
        before_config = deepcopy(config_like)
        before_state = deepcopy(layer.export_structure_state())

        summary = trainer._slot_lifecycle_summary(1, config_like=config_like)

        self.assertEqual(summary["live_slot_ids"], [first_slot, second_slot])
        self.assertEqual(summary["retained_slot_ids"], [second_slot])
        self.assertEqual(summary["candidate_slot_ids"], [second_slot])
        self.assertEqual(summary["selected_slot_id"], second_slot)
        self.assertEqual(summary["frozen_slot_ids"], [second_slot])
        self.assertEqual(summary["fallback_reason"], "none")
        self.assertEqual(config_like, before_config)
        self.assertEqual(layer.export_structure_state(), before_state)

    def test_stage5_profile_summary_preserves_retained_slots_when_shared_only(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "stage5_profile_summary_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        layer = trainer.model.layers["1"]
        first_slot = layer.add_slot(initial_rank=1, task_id=1)
        second_slot = layer.add_slot(initial_rank=1, task_id=2)
        layer.slot_metadata[first_slot].retained_for_inference = False
        layer.slot_metadata[second_slot].retained_for_inference = True
        layer.last_structural_action = "expand_rank_existing_slot"

        profile = trainer.model.build_inference_profile()
        summary = trainer._slot_lifecycle_summary(1, config_like=profile[1])

        self.assertEqual(summary["live_slot_ids"], [first_slot, second_slot])
        self.assertEqual(summary["retained_slot_ids"], [second_slot])
        self.assertEqual(summary["candidate_slot_ids"], [second_slot])
        self.assertFalse(profile[1]["shared_only"])

        layer.last_structural_action = "reuse_shared"
        shared_only_profile = trainer.model.build_inference_profile()
        shared_only_summary = trainer._slot_lifecycle_summary(1, config_like=shared_only_profile[1])

        self.assertEqual(shared_only_summary["retained_slot_ids"], [second_slot])
        self.assertEqual(shared_only_summary["candidate_slot_ids"], [])
        self.assertTrue(shared_only_profile[1]["shared_only"])

        layer.slot_metadata[second_slot].retained_for_inference = False
        empty_profile = trainer.model.build_inference_profile()
        empty_summary = trainer._slot_lifecycle_summary(1, config_like=empty_profile[1])

        self.assertEqual(empty_summary["retained_slot_ids"], [])
        self.assertEqual(empty_summary["candidate_slot_ids"], [])
        self.assertTrue(empty_profile[1]["shared_only"])

    def test_stage5_route_comparison_flags_empty_applied_nonempty_profile(self):
        flags = NHLoRATrainer._route_comparison_flags(
            applied_candidates=[],
            train_candidates=[],
            eval_applied_candidates=[],
            profile_candidates=[0],
            eval_profile_candidates=[0],
        )

        self.assertTrue(flags["applied_empty_profile_nonempty"])
        self.assertFalse(flags["train_eval_applied_mismatch"])
        self.assertFalse(flags["eval_profile_mismatch"])

    def test_stage5_plan_debug_does_not_mutate_plan_or_profile_state(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "stage5_no_mutation_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["routing_debug_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        layer = trainer.model.layers["1"]
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        materialized_plan = MaterializedLayerPlan(
            requested_action="expand_rank_existing_slot",
            action="expand_rank_existing_slot",
            selected_slot=slot_id,
            target_rank=2,
            rank_delta=1,
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=[slot_id],
            shared_only=False,
        )
        applied_plan = {
            "action": "expand_rank_existing_slot",
            "requested_action": "expand_rank_existing_slot",
            "active_slot_candidates": [slot_id],
            "selected_slot": slot_id,
            "rank_cfg": {slot_id: 1},
            "shared_only": False,
            "deterministic": True,
            "created_new_slot": False,
            "fallback_action": None,
        }
        profile = trainer.model.build_inference_profile()
        before_materialized = deepcopy(materialized_plan)
        before_applied = deepcopy(applied_plan)
        before_profile = deepcopy(profile)
        before_state = deepcopy(layer.export_structure_state())
        context = {
            "task_number": 2,
            "raw_planner": {},
            "materialized_plans": {1: materialized_plan},
            "applied_plans": {1: applied_plan},
        }

        trainer._log_stage5_plan_debug(context)
        trainer._log_stage5_lifecycle_for_profile(context, "Profile", profile)

        self.assertEqual(materialized_plan, before_materialized)
        self.assertEqual(applied_plan, before_applied)
        self.assertEqual(profile, before_profile)
        self.assertEqual(layer.export_structure_state(), before_state)
        self.assertIn("[MaterializedPlan][Task 2][Layer 1]", "\n".join(logger.messages))

    def test_stage5_train_eval_comparison_uses_same_input_and_restores_rng(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "stage5_mode_compare_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["routing_debug_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(2)
        layer = trainer.model.layers["1"]
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        applied_plan = {
            "active_slot_candidates": [slot_id],
            "selected_slot": slot_id,
            "rank_cfg": {slot_id: 1},
            "shared_only": False,
            "deterministic": True,
        }
        materialized_plan = MaterializedLayerPlan(
            requested_action="expand_rank_existing_slot",
            action="expand_rank_existing_slot",
            selected_slot=slot_id,
            target_rank=1,
            rank_delta=0,
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=[slot_id],
        )
        trainer.inference_profile = trainer.model.build_inference_profile()
        trainer.model.train()
        torch.manual_seed(1505)
        probe_images = torch.randn(2, 3, 32, 32)
        rng_before = torch.random.get_rng_state().clone()
        context = {
            "task_number": 2,
            "task_state": None,
            "applied_plans": {1: applied_plan},
            "materialized_plans": {1: materialized_plan},
            "routing_debug_probe_images": probe_images,
            "routing_debug_probe_source": "unit-same-input",
        }

        trainer._log_stage5_train_eval_comparison(context)

        self.assertTrue(trainer.model.training)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        joined_messages = "\n".join(logger.messages)
        self.assertIn("[RouteModeCompare][Task 2] input_source=unit-same-input", joined_messages)
        self.assertIn("[RouteModeCompare][Task 2][Layer 1]", joined_messages)

    def test_stage5_shared_only_inference_profile_matches_applied_plan(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "stage5_shared_only_profile_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["routing_debug_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(2)
        layer = trainer.model.layers["1"]
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        layer.slot_metadata[slot_id].retained_for_inference = True
        layer.last_structural_action = "reuse_shared"
        materialized_plan = MaterializedLayerPlan(
            requested_action="reuse_shared",
            action="reuse_shared",
            selected_slot=None,
            target_rank=None,
            rank_delta=0,
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=[],
            shared_only=True,
        )
        applied_plan = {
            "action": "reuse_shared",
            "requested_action": "reuse_shared",
            "active_slot_candidates": [],
            "selected_slot": None,
            "rank_cfg": {slot_id: 1},
            "shared_only": True,
            "deterministic": True,
            "created_new_slot": False,
            "fallback_action": None,
        }
        trainer.inference_profile = trainer.model.build_inference_profile()
        trainer.model.train()
        torch.manual_seed(1517)
        probe_images = torch.randn(2, 3, 32, 32)
        context = {
            "task_number": 3,
            "task_state": None,
            "applied_plans": {1: applied_plan},
            "materialized_plans": {1: materialized_plan},
            "routing_debug_probe_images": probe_images,
            "routing_debug_probe_source": "unit-shared-only",
        }

        trainer._log_stage5_train_eval_comparison(context)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("[RouteModeCompare][Task 3] input_source=unit-shared-only", joined_messages)
        self.assertIn("shared_only_train=True", joined_messages)
        self.assertIn("shared_only_profile=True", joined_messages)
        self.assertIn("mismatch_applied_empty_profile_nonempty=False", joined_messages)
        self.assertIn("mismatch_train_eval_applied=False", joined_messages)
        self.assertIn("mismatch_eval_profile=False", joined_messages)

    def test_stage4_zero_lora_block_parity_preserves_out_proj_semantics(self):
        torch.manual_seed(1404)
        for insertion_points in (["q_proj", "v_proj"], ["q_proj", "v_proj", "out_proj"]):
            with self.subTest(insertion_points=insertion_points):
                model = self._build_stage4_model(insertion_points)
                model.eval()
                block = model.backbone.core_model.blocks[1]
                layer = model.layers["1"]
                slot_id = layer.add_slot(initial_rank=1, task_id=1)
                tokens = torch.randn(2, 5, model.backbone.embed_dim)

                plain = model.backbone._forward_block_plain(block, tokens)
                adapted, _ = model.backbone._forward_block_with_adapter(
                    block,
                    tokens,
                    layer,
                    self._stage4_planner_cfg(slot_id),
                    task_state=None,
                )

                self.assertTrue(torch.allclose(adapted, plain, atol=1e-6))

    def test_stage4_active_adapter_changes_output_and_delta_logger(self):
        torch.manual_seed(1405)
        model = self._build_stage4_model(["q_proj", "v_proj"])
        model.eval()
        block = model.backbone.core_model.blocks[1]
        layer = model.layers["1"]
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        with torch.no_grad():
            for point_name in ("q_proj", "v_proj"):
                bank = layer.point_banks[point_name]
                bank.shared_b.fill_(0.05)
                bank.slot_b[slot_id].fill_(0.05)
        tokens = torch.randn(2, 5, model.backbone.embed_dim)
        accumulator = {}

        plain = model.backbone._forward_block_plain(block, tokens)
        adapted, _ = model.backbone._forward_block_with_adapter(
            block,
            tokens,
            layer,
            self._stage4_planner_cfg(slot_id, accumulator=accumulator),
            task_state=None,
        )

        self.assertGreater(float((adapted - plain).abs().max().item()), 1e-6)
        self.assertIn(1, accumulator)
        shared_max = max(
            point_stats["shared"]["max_abs"]
            for point_stats in accumulator[1].values()
            if "shared" in point_stats
        )
        slot_max = max(
            point_stats["slot"]["max_abs"]
            for point_stats in accumulator[1].values()
            if "slot" in point_stats
        )
        self.assertGreater(shared_max, 0.0)
        self.assertGreater(slot_max, 0.0)

    def test_stage4_ce_grad_flows_to_active_slot_and_shared_b(self):
        torch.manual_seed(1406)
        model = self._build_stage4_model(["q_proj", "v_proj"])
        model.train()
        model.classifier.expand(2)
        layer = model.layers["1"]
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 0, 1])

        outputs = model.forward_with_state(
            images,
            task_state=None,
            planner_out={1: self._stage4_planner_cfg(slot_id)},
        )
        loss = F.cross_entropy(outputs["logits"], labels)
        loss.backward()

        slot_grad_norm = 0.0
        shared_grad_norm = 0.0
        for point_name in ("q_proj", "v_proj"):
            bank = layer.point_banks[point_name]
            self.assertIsNotNone(bank.slot_b[slot_id].grad)
            self.assertIsNotNone(bank.shared_b.grad)
            slot_grad_norm += float(bank.slot_b[slot_id].grad.norm().item())
            shared_grad_norm += float(bank.shared_b.grad.norm().item())
        self.assertTrue(outputs["features"].requires_grad)
        self.assertGreater(slot_grad_norm, 0.0)
        self.assertGreater(shared_grad_norm, 0.0)

    def test_adapter_delta_debug_records_shared_and_slot_stats(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        )
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        hidden_states = torch.randn(2, 5, 8)
        accumulator = {}
        planner_cfg = {
            "shared_gate": 1.0,
            "rank_cfg": {slot_id: 1},
            "_debug_delta_stats": accumulator,
            "_debug_block_id": 7,
        }

        shared_delta = layer._shared_delta("q_proj", hidden_states, planner_cfg)
        slot_delta = layer._slot_delta(
            "q_proj",
            hidden_states,
            route_state={"candidate_slots": [slot_id]},
            planner_cfg=planner_cfg,
        )

        self.assertEqual(shared_delta.shape, hidden_states.shape)
        self.assertEqual(slot_delta.shape, hidden_states.shape)
        self.assertIn(7, accumulator)
        self.assertIn("q_proj", accumulator[7])
        self.assertEqual(accumulator[7]["q_proj"]["shared"]["calls"], 1)
        self.assertEqual(accumulator[7]["q_proj"]["slot"]["calls"], 1)
        self.assertIn("norm_sum", accumulator[7]["q_proj"]["shared"])
        self.assertIn("mean_abs_sum", accumulator[7]["q_proj"]["slot"])

    def test_shared_gate_tensor_changes_effective_shared_delta(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        )
        bank = layer.point_banks["q_proj"]
        with torch.no_grad():
            bank.shared_b.fill_(0.05)
        hidden_states = torch.randn(2, 5, 8)

        low_gate = layer._shared_delta("q_proj", hidden_states, {"shared_gate": torch.tensor([[0.2]])})
        high_gate = layer._shared_delta("q_proj", hidden_states, {"shared_gate": torch.tensor([[0.8]])})

        self.assertFalse(torch.allclose(low_gate, high_gate))
        self.assertGreater(float(high_gate.norm().item()), float(low_gate.norm().item()))

    def test_stage1_debug_helpers_are_config_gated_and_log_expected_records(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "stage1_debug_helpers_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        for flag in (
            "adapter_delta_debug_logging",
            "final_feature_diff_debug_logging",
            "classifier_drift_debug_logging",
            "grad_norm_debug_logging",
            "logit_margin_debug_logging",
        ):
            config["training"][flag] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.model.classifier.expand(4)

        context = {
            "task_number": 2,
            "current_epoch": 1,
            "old_num_classes": 2,
            "applied_plans": {1: {"shared_gate": 1.0}},
            "grad_norm_debug_accumulator": {
                group: {"sum": 0.0, "max": 0.0, "batches": 0}
                for group in ("shared", "slot", "router", "planner_policy", "planner_control", "planner", "classifier")
            },
        }
        disabled_config = _build_test_config(str(workspace_tmp))
        disabled_trainer = NHLoRATrainer(disabled_config, _ListLogger(), benchmark=benchmark)
        disabled_context = dict(context)
        disabled_context["applied_plans"] = {1: {"shared_gate": 1.0}}
        self.assertIs(disabled_trainer._plans_for_training_forward(disabled_context), disabled_context["applied_plans"])

        debug_plans = trainer._plans_for_training_forward(context)
        self.assertIn("_debug_delta_stats", debug_plans[1])
        self.assertNotIn("_debug_delta_stats", context["applied_plans"][1])

        feature_dim = trainer.model.backbone.embed_dim
        outputs = {
            "features": torch.randn(3, feature_dim),
            "logits": torch.tensor(
                [
                    [0.2, 0.1, 1.1, 1.3],
                    [0.1, 0.4, 1.2, 1.1],
                    [0.0, 0.3, 0.9, 1.0],
                ]
            ),
        }
        teacher_outputs = {"features": outputs["features"] + 0.01}
        context["old_classifier_weight_snapshot"] = trainer.model.classifier.weight[:2].detach().clone()
        with torch.no_grad():
            trainer.model.classifier.weight[:2].add_(0.01)

        trainer._log_final_feature_diff_debug(context, outputs, teacher_outputs)
        trainer._log_logit_margin_debug(context, outputs)
        trainer._log_classifier_drift_debug(context)

        layer = trainer.model.layers["1"]
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        bank = layer.point_banks["q_proj"]
        bank.shared_a.grad = torch.ones_like(bank.shared_a) * 0.01
        bank.slot_a[slot_id].grad = torch.ones_like(bank.slot_a[slot_id]) * 0.02
        layer.query_proj.weight.grad = torch.ones_like(layer.query_proj.weight) * 0.03
        trainer.model.classifier.weight.grad = torch.ones_like(trainer.model.classifier.weight) * 0.04
        planner_param = next(trainer.planner.parameters())
        planner_param.grad = torch.ones_like(planner_param) * 0.05
        control_param = next(trainer.planner.control_parameters())
        control_param.grad = torch.ones_like(control_param) * 0.06
        trainer._accumulate_grad_norm_debug(context)
        trainer._log_grad_norm_debug(context)

        context["adapter_delta_debug_accumulator"] = {
            1: {
                "q_proj": {
                    "shared": {"calls": 1, "norm_sum": 1.0, "mean_abs_sum": 0.5, "max_abs": 0.7},
                    "slot": {"calls": 1, "norm_sum": 0.0, "mean_abs_sum": 0.0, "max_abs": 0.0},
                }
            }
        }
        trainer._log_adapter_delta_debug(context)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("[FinalFeatureDiff][Task 2][Epoch 1]", joined_messages)
        self.assertIn("[LogitMargin][Task 2][Epoch 1]", joined_messages)
        self.assertIn("[ClassifierDrift][Task 2][Epoch 1]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][shared]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][slot]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][router]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][planner_policy]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][planner_control]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][planner]", joined_messages)
        self.assertIn("[GradNorm][Task 2][Epoch 1][classifier]", joined_messages)
        self.assertIn("[AdapterDelta][Task 2][Epoch 1][Layer 1][q_proj][shared]", joined_messages)
        self.assertIn("[AdapterDelta][Task 2][Epoch 1][Layer 1][q_proj][slot]", joined_messages)

    def test_seed_config_logging_tolerates_missing_optional_fields(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "seed_config_log_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"].pop("debug_eval_around_consolidation", None)
        config["training"].pop("estimate_eta", None)
        config["training"].pop("cuda_sync_timing", None)
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)

        trainer._log_seed_config(seed=3)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("Seed 3 config:", joined_messages)
        self.assertIn("benchmark=tiny_alignment", joined_messages)
        self.assertIn("retention_feature_representation=cls", joined_messages)
        self.assertIn("retention_feature_diff_logging=", joined_messages)
        self.assertIn("routing_debug_logging=", joined_messages)
        self.assertIn("planner_audit_logging=", joined_messages)
        self.assertIn("adapter_delta_debug_logging=", joined_messages)
        self.assertIn("final_feature_diff_debug_logging=", joined_messages)
        self.assertIn("classifier_drift_debug_logging=", joined_messages)
        self.assertIn("grad_norm_debug_logging=", joined_messages)
        self.assertIn("logit_margin_debug_logging=", joined_messages)
        self.assertIn("classifier_lr_scale=", joined_messages)
        self.assertIn("freeze_new_classifier_epochs=", joined_messages)
        self.assertIn("freeze_all_classifier_epochs=", joined_messages)
        self.assertIn("planner_mode=", joined_messages)
        self.assertIn("planner_control_recompute=", joined_messages)
        self.assertIn("planner_policy_trainable=", joined_messages)
        self.assertIn("planner_control_trainable=", joined_messages)
        self.assertIn("planner_use_learned_shared_gate=", joined_messages)
        self.assertIn("planner_soft_rank_training=", joined_messages)
        self.assertIn("planner_hard_rank_eval=", joined_messages)

    def test_seed_config_warns_for_full_token_retention_features(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "seed_config_full_tokens_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["loss"]["retention_feature_representation"] = "full_tokens"
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)

        trainer._log_seed_config(seed=3)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("retention_feature_representation=full_tokens", joined_messages)
        self.assertIn("can use substantially more memory", joined_messages)

    def test_planner_decision_label_matches_threshold_quadrants(self):
        self.assertEqual(
            NHLoRATrainer._planner_decision_label(
                novelty=0.2,
                conflict=0.2,
                tau_novelty=0.5,
                tau_conflict=0.5,
            ),
            "reuse_shared",
        )
        self.assertEqual(
            NHLoRATrainer._planner_decision_label(
                novelty=0.7,
                conflict=0.2,
                tau_novelty=0.5,
                tau_conflict=0.5,
            ),
            "expand_rank_existing_slot",
        )
        self.assertEqual(
            NHLoRATrainer._planner_decision_label(
                novelty=0.7,
                conflict=0.8,
                tau_novelty=0.5,
                tau_conflict=0.5,
            ),
            "open_new_slot",
        )
        self.assertEqual(
            NHLoRATrainer._planner_decision_label(
                novelty=0.2,
                conflict=0.8,
                tau_novelty=0.5,
                tau_conflict=0.5,
            ),
            "freeze_old_strong_retention",
        )

    def test_planner_parameter_distance_summary_tracks_zero_and_nonzero(self):
        planner = HorizonPlanner(
            selected_blocks=[0],
            task_embedding_dim=4,
            history_dim=6,
            hidden_dim=8,
            layer_embedding_dim=3,
            rank_min=1,
            rank_max=4,
            tau_novelty=0.5,
            tau_conflict=0.5,
        )
        init_snapshot = NHLoRATrainer._snapshot_named_parameters(planner)
        zero_distance = NHLoRATrainer._parameter_distance_summary(
            NHLoRATrainer._snapshot_named_parameters(planner),
            init_snapshot,
        )

        self.assertIsNotNone(zero_distance)
        self.assertAlmostEqual(float(zero_distance["l2"]), 0.0, places=8)
        self.assertAlmostEqual(float(zero_distance["max_abs"]), 0.0, places=8)

        with torch.no_grad():
            next(planner.parameters()).add_(0.125)

        moved_distance = NHLoRATrainer._parameter_distance_summary(
            NHLoRATrainer._snapshot_named_parameters(planner),
            init_snapshot,
        )

        self.assertGreater(float(moved_distance["l2"]), 0.0)
        self.assertGreater(float(moved_distance["max_abs"]), 0.0)

    def test_planner_audit_prepare_reports_optimizer_membership_without_mutating_inputs(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "planner_audit_prepare_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_audit_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=2)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        task_state, warmup_info = trainer._run_warmup_sensing(benchmark.tasks[1])
        raw_planner = trainer._compute_raw_planner(task_state)
        optimizer = trainer._build_optimizer()
        before_embedding = task_state.embedding.detach().clone()
        before_signals = {
            block_id: (
                float(signals.novelty.item()),
                float(signals.conflict.item()),
                float(signals.shared_gate.item()),
            )
            for block_id, signals in raw_planner.items()
        }
        context = {
            "task_number": 2,
            "task_state": task_state,
            "raw_planner": raw_planner,
            "optimizer": optimizer,
            "warmup_info": warmup_info,
        }

        trainer._log_planner_audit_prepare(context)

        self.assertTrue(torch.allclose(task_state.embedding, before_embedding))
        after_signals = {
            block_id: (
                float(signals.novelty.item()),
                float(signals.conflict.item()),
                float(signals.shared_gate.item()),
            )
            for block_id, signals in raw_planner.items()
        }
        self.assertEqual(before_signals, after_signals)
        self.assertTrue(context["planner_optimizer_summary"]["in_optimizer"])
        joined_messages = "\n".join(logger.messages)
        self.assertIn("[PlannerInputAudit][Task 2]", joined_messages)
        self.assertIn("[PlannerTrainPath][Task 2] in_optimizer=True", joined_messages)
        self.assertIn("[PlannerAudit][Task 2][Layer 1]", joined_messages)
        self.assertIn("[PlannerTrajectory][Layer 1]", joined_messages)

    def test_planner_audit_prepare_is_silent_when_disabled(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "planner_audit_disabled_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_audit_logging"] = False
        benchmark = _build_tiny_benchmark(num_tasks=2)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        task_state, warmup_info = trainer._run_warmup_sensing(benchmark.tasks[1])
        context = {
            "task_number": 2,
            "task_state": task_state,
            "raw_planner": trainer._compute_raw_planner(task_state),
            "optimizer": trainer._build_optimizer(),
            "warmup_info": warmup_info,
        }

        trainer._log_planner_audit_prepare(context)

        self.assertEqual(logger.messages, [])

    def test_planner_epoch_audit_reports_pre_step_grad_and_optimizer_steps(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "planner_epoch_audit_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_audit_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        optimizer = trainer._build_optimizer()
        for parameter in trainer.planner.parameters():
            parameter.grad = torch.zeros_like(parameter)
        first_parameter = next(trainer.planner.parameters())
        first_parameter.grad.fill_(0.25)
        context = {
            "task_number": 2,
            "current_epoch": 1,
            "optimizer": optimizer,
            "planner_optimizer_summary": trainer._planner_optimizer_membership(optimizer),
            "planner_audit_epoch_accumulator": {
                "batches": 0,
                "grad_l2_sum": 0.0,
                "grad_l2_max": 0.0,
                "grad_present_batches": 0,
                "grad_nonzero_batches": 0,
                "optimizer_steps": 0,
            },
        }

        trainer._accumulate_planner_audit_pre_step(context)
        trainer._record_planner_optimizer_step(context)
        trainer._log_planner_epoch_audit(context)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("[PlannerTrainPath][Task 2][Epoch 1]", joined_messages)
        self.assertIn("grad_present_batches=1/1", joined_messages)
        self.assertIn("grad_nonzero_batches=1/1", joined_messages)
        self.assertIn("optimizer_steps=1", joined_messages)

    def test_hybrid_optimizer_excludes_policy_branch_and_tracks_control_branch(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "hybrid_optimizer_membership_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_mode"] = "hybrid"
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)

        optimizer = trainer._build_optimizer()
        policy_summary = trainer._planner_policy_optimizer_membership(optimizer)
        control_summary = trainer._planner_control_optimizer_membership(optimizer)

        self.assertFalse(policy_summary["in_optimizer"])
        self.assertEqual(policy_summary["requires_grad_parameters"], 0)
        self.assertEqual(policy_summary["matched_parameters"], 0)
        self.assertTrue(control_summary["in_optimizer"])
        self.assertGreater(control_summary["requires_grad_parameters"], 0)
        self.assertGreater(control_summary["matched_parameters"], 0)

    def test_hybrid_planner_control_receives_gradients_and_drifts(self):
        seed_everything(11, deterministic=True)
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "hybrid_control_grad_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_mode"] = "hybrid"
        benchmark = _build_tiny_benchmark(num_tasks=2)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = trainer._prepare_task_context(benchmark.tasks[1])
        layer = trainer.model.layers["1"]
        with torch.no_grad():
            for point_name in ("q_proj", "v_proj"):
                bank = layer.point_banks[point_name]
                bank.shared_b.fill_(0.05)

        train_dataset, _ = benchmark.build_task_datasets(1)
        loader = trainer._build_train_loader(train_dataset)
        batch = next(iter(loader))
        images, labels = trainer._prepare_images_labels(batch)
        optimizer = context["optimizer"]
        optimizer.zero_grad(set_to_none=True)
        before_snapshot = trainer._snapshot_named_parameters(trainer.planner.control_branch)
        planner_out = trainer._plans_for_training_forward(context)
        self.assertIsInstance(planner_out[1]["shared_gate"], torch.Tensor)
        self.assertTrue(planner_out[1]["shared_gate"].requires_grad)

        outputs = trainer.model.forward_with_state(images, context["task_state"], planner_out)
        loss = F.cross_entropy(outputs["logits"], labels)
        loss.backward()

        control_grad_norm = sum(
            float(parameter.grad.norm().item())
            for parameter in trainer.planner.control_parameters()
            if parameter.grad is not None
        )
        policy_grads = [
            parameter.grad for parameter in trainer.planner.policy_parameters() if parameter.grad is not None
        ]
        self.assertGreater(control_grad_norm, 0.0)
        self.assertEqual(policy_grads, [])

        optimizer.step()
        after_snapshot = trainer._snapshot_named_parameters(trainer.planner.control_branch)
        drift = trainer._parameter_distance_summary(after_snapshot, before_snapshot)
        self.assertIsNotNone(drift)
        self.assertGreater(float(drift["l2"]), 0.0)

    def test_planner_control_scalar_summary_reports_percentiles_and_threshold_fractions(self):
        values = [0.1, 0.5, 1.0, 4.7, 7.2]
        summary = NHLoRATrainer._scalar_series_summary(values)

        self.assertEqual(int(summary["count"]), len(values))
        self.assertAlmostEqual(float(summary["min"]), min(values), places=6)
        self.assertAlmostEqual(float(summary["max"]), max(values), places=6)
        self.assertGreaterEqual(float(summary["p99"]), float(summary["p90"]))
        self.assertAlmostEqual(NHLoRATrainer._fraction_above(values, 2.0), 2 / 5, places=6)
        self.assertAlmostEqual(NHLoRATrainer._fraction_above(values, 4.59511985013459), 2 / 5, places=6)
        self.assertAlmostEqual(NHLoRATrainer._fraction_below(values, 0.2), 1 / 5, places=6)
        self.assertAlmostEqual(NHLoRATrainer._fraction_abs_above([-1.0, 0.5, 4.5], 2.0), 1 / 3, places=6)
        self.assertAlmostEqual(NHLoRATrainer._fraction_within([1.9, 2.0, 2.2], 2.0, 0.11), 2 / 3, places=6)
        self.assertAlmostEqual(NHLoRATrainer._mean_abs_gap_to_target([1.5, 2.0, 2.5], 2.0), 1 / 3, places=6)
        self.assertGreater(
            NHLoRATrainer._paired_series_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]),
            0.99,
        )

    def test_planner_control_contribution_debug_records_shared_and_slot_activity(self):
        layer = NHLoRALayer(
            embed_dim=8,
            selected_points=["q_proj"],
            shared_rank=2,
            slot_r_max=4,
            slot_init_rank=1,
            bootstrap_slot_rank=1,
            max_slots=2,
            router_topk=1,
            router_temperature=1.0,
            task_embedding_dim=8,
        )
        slot_id = layer.add_slot(initial_rank=1, task_id=1)
        with torch.no_grad():
            layer.point_banks["q_proj"].shared_b.fill_(0.05)
            layer.point_banks["q_proj"].slot_b[slot_id].fill_(0.03)
        hidden_states = torch.randn(2, 4, 8)
        contribution_accumulator = {}
        planner_cfg = {
            "shared_gate": torch.tensor([[0.995]], dtype=hidden_states.dtype),
            "rank_cfg": {slot_id: 1},
            "_debug_block_id": 3,
            "_planner_control_contribution_accumulator": contribution_accumulator,
        }

        _ = layer._shared_delta("q_proj", hidden_states, planner_cfg)
        _ = layer._slot_delta(
            "q_proj",
            hidden_states,
            route_state={"candidate_slots": [slot_id]},
            planner_cfg=planner_cfg,
        )

        block_stats = contribution_accumulator[3]
        self.assertGreater(block_stats["shared_pre_beta_norm_count"], 0)
        self.assertGreater(block_stats["shared_post_beta_norm_count"], 0)
        self.assertGreater(block_stats["slot_norm_count"], 0)
        self.assertGreater(block_stats["slot_structurally_available_calls"], 0)
        self.assertGreater(block_stats["slot_nontrivial_calls"], 0)
        self.assertGreater(block_stats["beta_gt_099_with_structural_slot_calls"], 0)

    def test_hybrid_planner_control_epoch_logs_stage12_residual_stability_signals(self):
        seed_everything(17, deterministic=True)
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "hybrid_control_stage12_logging_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_mode"] = "hybrid"
        config["training"]["planner_audit_logging"] = True
        benchmark = _build_tiny_benchmark(num_tasks=2)
        logger = _ListLogger()
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        context = trainer._prepare_task_context(benchmark.tasks[1])
        context["current_epoch"] = 1
        context["planner_control_epoch_accumulator"] = trainer._new_planner_control_epoch_accumulator()
        context["planner_control_contribution_epoch_accumulator"] = {}
        layer = trainer.model.layers["1"]
        with torch.no_grad():
            for point_name in ("q_proj", "v_proj"):
                layer.point_banks[point_name].shared_b.fill_(0.05)
        train_dataset, _ = benchmark.build_task_datasets(1)
        loader = trainer._build_train_loader(train_dataset)
        batch = next(iter(loader))
        images, labels = trainer._prepare_images_labels(batch)
        optimizer = context["optimizer"]

        optimizer.zero_grad(set_to_none=True)
        planner_out = trainer._plans_for_training_forward(context)
        outputs = trainer.model.forward_with_state(images, context["task_state"], planner_out)
        loss = F.cross_entropy(outputs["logits"], labels)
        loss.backward()

        trainer._accumulate_planner_control_pre_step(context)
        trainer._record_planner_control_optimizer_step(context)
        trainer._log_planner_control_epoch(context)

        joined_messages = "\n".join(logger.messages)
        self.assertIn("[PlannerControlAnchor][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("frac_anchor_gt_099=", joined_messages)
        self.assertIn("[PlannerControlDelta][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("beta_anchor_gap_mean_abs=", joined_messages)
        self.assertIn("[PlannerResidualRaw][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("frac_abs_gt_6=", joined_messages)
        self.assertIn("bound_derivative_mean=", joined_messages)
        self.assertIn("[PlannerControlLogits][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("frac_gt_logit099=", joined_messages)
        self.assertIn("[PlannerControlValues][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("frac_gt_099=", joined_messages)
        self.assertIn("[PlannerControlGradients][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("logit_grad_effectively_zero_fraction=", joined_messages)
        self.assertIn("[PlannerResidualGradients][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("bridge_grad_lost_fraction=", joined_messages)
        self.assertIn("[PlannerControlHead][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("delta_bias_share_mean=", joined_messages)
        self.assertIn("[PlannerResidualCap][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("cap_state=", joined_messages)
        self.assertIn("[PlannerResidualSummary][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("[PlannerContribution][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("[PlannerControlInputs][Task 2][Epoch 1][Layer 1]", joined_messages)
        self.assertIn("normalized_representation_norm_mean=", joined_messages)
        self.assertIn("[PlannerControlInputCompare][Task 2][Epoch 1]", joined_messages)

    def test_hybrid_mode_rejects_soft_rank_training_for_stage8a(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "hybrid_soft_rank_guardrail_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        config["training"]["planner_mode"] = "hybrid"
        config["training"]["planner_soft_rank_training"] = True
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)

        with self.assertRaises(ValueError):
            NHLoRATrainer(config, logger, benchmark=benchmark)

    def test_forgetting_and_parameter_growth_delta_helpers(self):
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "forgetting_unit"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = _build_test_config(str(workspace_tmp))
        benchmark = _build_tiny_benchmark(num_tasks=1)
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.accuracy_matrix = [[0.8], [0.7, 0.6]]
        trainer.task_metrics = [{"parameter_growth": 128}]

        self.assertAlmostEqual(trainer._compute_forgetting([0.65, 0.55, 0.5]), 0.10, places=6)

        with patch.object(trainer, "_estimate_parameter_growth", return_value=160), patch.object(
            trainer, "_total_active_rank", return_value=12
        ):
            metrics = trainer._summarize_task_metrics(
                context={"task_number": 2},
                eval_metrics={"avg_acc": 0.5, "per_task_acc": [0.65, 0.55]},
                chu_report={
                    "opened_slots": 1,
                    "pruned_slots": 0,
                    "merged_slots": 0,
                    "frozen_slots": 0,
                    "kept_slots": 1,
                },
                epoch_history=[
                    {
                        "loss_total": 1.2,
                        "loss_cls": 1.0,
                        "loss_kd": 0.0,
                        "loss_feat": 0.0,
                        "loss_orth": 0.1,
                        "loss_rank": 0.05,
                        "loss_grow": 0.0,
                        "loss_route": 0.05,
                    }
                ],
                training_time=4.0,
                task_wall_time=5.0,
                inference_overhead=1.1,
            )

        self.assertEqual(metrics["parameter_growth_delta"], 32)
        self.assertAlmostEqual(metrics["forgetting"], 0.15, places=6)
        self.assertEqual(metrics["task_wall_time"], 5.0)

    def test_logger_can_disable_file_handler_for_tee_mode(self):
        repo_root = Path(__file__).resolve().parents[1]
        log_path = repo_root / "outputs" / "test_tmp" / "logger_tee" / "tee.log"
        if log_path.exists():
            log_path.unlink()
        with patch.dict(os.environ, {"NH_LORA_DISABLE_FILE_LOG": "1"}):
            logger = configure_logger(log_file=log_path, level=logging.WARNING)
        self.assertFalse(any(isinstance(handler, logging.FileHandler) for handler in logger.handlers))
        self.assertFalse(log_path.exists())


if __name__ == "__main__":
    unittest.main()
