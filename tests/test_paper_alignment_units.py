from __future__ import annotations

import logging
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition
from src.datasets.transforms import build_cifar_test_transform
from src.engine.trainer import NHLoRATrainer
from src.models.lora import NHLoRALayer
from src.models.losses import growth_penalty, rank_penalty, routing_balance_loss, slot_orthogonality
from src.models.nh_lora import NHLoRAModel
from src.models.planner import HorizonPlanner, MaterializedLayerPlan, PlannerSignals, materialize_action
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
        },
    }


class _DummyModel(nn.Module):
    def __init__(self, layer: NHLoRALayer):
        super().__init__()
        self.layers = nn.ModuleDict({"0": layer})


class PaperAlignmentUnitTests(unittest.TestCase):
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
        layer.slot_metadata[first_slot].retained_for_inference = False
        layer.slot_metadata[first_slot].usage_ema = 0.9
        layer.slot_metadata[second_slot].retained_for_inference = True
        layer.slot_metadata[second_slot].usage_ema = 0.2
        layer.last_shared_gate = 0.35
        layer.last_structural_action = "expand_rank_existing_slot"
        profile = model.build_inference_profile()
        self.assertEqual(profile[1]["active_slot_candidates"], [second_slot])
        self.assertAlmostEqual(profile[1]["shared_gate"], 0.35, places=6)

        layer.last_structural_action = "reuse_shared"
        profile = model.build_inference_profile()
        self.assertEqual(profile[1]["active_slot_candidates"], [])
        self.assertTrue(profile[1]["shared_only"])

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
