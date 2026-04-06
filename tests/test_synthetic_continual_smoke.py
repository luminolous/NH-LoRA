from __future__ import annotations

import logging
import unittest
from pathlib import Path

import numpy as np

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition
from src.datasets.transforms import build_cifar_test_transform
from src.engine.trainer import NHLoRATrainer
from src.utils.logging_utils import configure_logger
from src.utils.seeding import seed_everything


def make_image_bytes(label: int, sample_id: int, size: int = 32) -> bytes:
    base = np.zeros((size, size, 3), dtype=np.uint8)
    base[..., 0] = (label * 40 + sample_id * 3) % 255
    base[..., 1] = (label * 70 + sample_id * 5) % 255
    base[..., 2] = (label * 90 + sample_id * 7) % 255
    return base.tobytes()


def make_records(class_ids, split: str, samples_per_class: int):
    records = []
    for class_id in class_ids:
        for sample_id in range(samples_per_class):
            records.append(
                SampleRecord(
                    path=None,
                    label=class_id,
                    split=split,
                    image_bytes=make_image_bytes(class_id, sample_id),
                    metadata={"size": (32, 32)},
                )
            )
    return records


def build_synthetic_benchmark():
    train_transform = build_cifar_test_transform(32)
    test_transform = build_cifar_test_transform(32)
    tasks = [
        TaskDefinition(
            task_id=0,
            class_ids=[0, 1],
            train_records=make_records([0, 1], "train", 6),
            test_records=make_records([0, 1], "test", 4),
            metadata={"name": "task1"},
        ),
        TaskDefinition(
            task_id=1,
            class_ids=[2, 3],
            train_records=make_records([2, 3], "train", 6),
            test_records=make_records([2, 3], "test", 4),
            metadata={"name": "task2"},
        ),
    ]
    return ContinualBenchmark(
        name="synthetic_smoke",
        num_classes=4,
        tasks=tasks,
        train_transform=train_transform,
        test_transform=test_transform,
    )


def build_test_config(output_root: str):
    return {
        "experiment": {
            "name": "nh_lora_smoke",
            "project_name": "nh_lora_smoke",
            "output_root": output_root,
            "log_dir": str(Path(output_root) / "logs"),
            "metrics_dir": str(Path(output_root) / "metrics"),
            "summaries_dir": str(Path(output_root) / "summaries"),
            "checkpoints_dir": str(Path(output_root) / "checkpoints"),
        },
        "runtime": {
            "device": "cpu",
            "num_workers": 0,
            "deterministic": True,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "benchmark": {
            "name": "synthetic_smoke",
            "dataset_name": "synthetic_smoke",
            "data_root": output_root,
            "num_tasks": 2,
            "classes_per_task": 2,
            "image_size": 32,
        },
        "model": {
            "name": "nh_lora",
            "backbone_name": "toy_vit_tiny",
            "backbone_source": "internal",
            "freeze_backbone": True,
            "selected_blocks": [1, 2, 3],
            "insertion_points": ["q_proj", "v_proj"],
        },
        "nh_lora": {
            "shared_rank": 4,
            "shared_lr_scale": 0.5,
            "slot_r_max": 8,
            "slot_init_rank": 2,
            "bootstrap_slot_rank": 2,
            "max_slots_per_block": 4,
            "router_topk": 2,
            "router_temperature": 1.0,
            "use_rank_mask": True,
        },
        "warmup": {
            "enabled": True,
            "num_batches": 2,
            "task_embedding_dim": 64,
            "gradient_sketch_dim": 16,
        },
        "planner": {
            "type": "mlp_per_layer",
            "hidden_dim": 64,
            "layer_embedding_dim": 16,
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
        },
        "loss": {
            "lambda_kd": 0.5,
            "lambda_feat": 0.5,
            "lambda_orth": 0.05,
            "lambda_rank": 1e-4,
            "lambda_grow": 0.01,
            "lambda_route": 0.01,
            "kd_temperature": 2.0,
            "retention_layers": [3],
        },
    }


class SyntheticContinualSmokeTest(unittest.TestCase):
    def test_two_task_end_to_end_smoke(self):
        seed_everything(7, deterministic=True)
        benchmark = build_synthetic_benchmark()
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "synthetic_smoke_runtime"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = build_test_config(str(workspace_tmp))
        logger = configure_logger(level=logging.WARNING)
        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        metrics = trainer.train(seed=7)

        self.assertEqual(metrics["benchmark"], "synthetic_smoke")
        self.assertIn("final_last_task_accuracy", metrics)
        self.assertIn("final_forgetting", metrics)
        self.assertIn("training_time_total", metrics)
        self.assertIn("seed_wall_time_total", metrics)
        self.assertGreaterEqual(metrics["seed_wall_time_total"], metrics["training_time_total"])
        self.assertEqual(len(trainer.history_bank.entries), 2)
        self.assertTrue(trainer.last_train_state["bootstrap_used"])
        self.assertTrue(trainer.last_train_state["teacher_used_on_task2"])
        self.assertTrue(trainer.last_train_state["planner_used_on_task2"])
        self.assertTrue(trainer.last_train_state["materialize_used_on_task2"])
        self.assertTrue(trainer.last_train_state["raw_planner_separated"])
        self.assertTrue(trainer.last_train_state["task2_materialized_has_candidates"])
        self.assertTrue(trainer.last_train_state["task2_history_attention_used"])
        self.assertTrue(trainer.last_train_state["router_seen"])
        self.assertTrue(trainer.last_train_state["warmup_imprinting_used"])
        self.assertTrue(trainer.last_train_state["classifier_imprinting_used"])
        self.assertEqual(trainer.last_train_state["classifier_sizes"], [2, 4])
        self.assertEqual(trainer.last_train_state["warmup_head_class_counts"], [2, 2])
        self.assertEqual(trainer.last_train_state["chu_calls_per_task"], [1, 1])
        self.assertEqual(trainer.last_train_state["history_sizes"], [1, 2])
        self.assertGreater(trainer.last_train_state["history_summary_dim"], 0)

        first_task_state = trainer.last_train_state["task_states"][0]
        second_task_state = trainer.last_train_state["task_states"][1]
        self.assertFalse(first_task_state["has_history"])
        self.assertAlmostEqual(first_task_state["similarity_mean"], 0.0, places=6)
        self.assertTrue(second_task_state["has_history"])
        self.assertNotAlmostEqual(second_task_state["similarity_mean"], 0.0, places=6)
        self.assertTrue(all(action in {
            "reuse_shared",
            "expand_rank_existing_slot",
            "open_new_slot",
            "freeze_old_strong_retention",
        } for action in trainer.last_train_state["task2_actions"].values()))

        self.assertEqual(len(metrics["task_metrics"]), 2)
        first_task_metrics = metrics["task_metrics"][0]
        self.assertEqual(first_task_metrics["forgetting"], 0.0)
        self.assertIn("training_time", first_task_metrics)
        self.assertIn("task_wall_time", first_task_metrics)
        self.assertGreaterEqual(first_task_metrics["task_wall_time"], first_task_metrics["training_time"])
        self.assertTrue(first_task_metrics["epoch_history"])
        first_epoch = first_task_metrics["epoch_history"][0]
        for key in (
            "loss_total",
            "loss_cls",
            "loss_kd",
            "loss_feat",
            "loss_orth",
            "loss_rank",
            "loss_grow",
            "loss_route",
        ):
            self.assertIn(key, first_epoch)
            self.assertIsInstance(first_epoch[key], float)
        self.assertEqual(first_epoch["loss_kd"], 0.0)
        self.assertEqual(first_epoch["loss_feat"], 0.0)
        self.assertEqual(first_epoch["loss_grow"], 0.0)

        for block_id, layer in trainer.model.layers.items():
            self.assertGreaterEqual(len(layer.slot_metadata), 1)
            inference_profile = trainer.inference_profile[int(block_id)]
            self.assertLessEqual(len(inference_profile["active_slot_candidates"]), layer.router_topk)
            if layer.last_structural_action in {"reuse_shared", "freeze_old_strong_retention"}:
                self.assertEqual(inference_profile["active_slot_candidates"], [])
            else:
                for slot_id in inference_profile["active_slot_candidates"]:
                    self.assertTrue(layer.slot_metadata[slot_id].retained_for_inference)


if __name__ == "__main__":
    unittest.main()
