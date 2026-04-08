from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.utils.config import load_config
from src.utils.metrics import write_summary


class ConfigAndSummarySmokeTest(unittest.TestCase):
    def test_config_merge_and_summary_writer(self):
        repo_root = Path(__file__).resolve().parents[1]
        config = load_config(repo_root / "configs" / "cifar100.yaml")
        self.assertEqual(config["benchmark"]["name"], "cifar100")
        self.assertEqual(config["model"]["backbone_name"], "vit_base_patch16_224_in21k")
        self.assertEqual(config["training"]["classifier_lr_scale"], 1.0)
        self.assertEqual(config["training"]["freeze_new_classifier_epochs"], 0)
        self.assertEqual(config["training"]["freeze_all_classifier_epochs"], 0)
        self.assertFalse(config["training"]["planner_audit_logging"])
        self.assertEqual(config["training"]["planner_mode"], "legacy")
        self.assertEqual(config["training"]["planner_control_recompute"], "per_batch")
        self.assertFalse(config["training"]["planner_policy_trainable"])
        self.assertTrue(config["training"]["planner_control_trainable"])
        self.assertTrue(config["training"]["planner_use_learned_shared_gate"])
        self.assertFalse(config["training"]["planner_soft_rank_training"])
        self.assertAlmostEqual(config["training"]["planner_soft_rank_temperature"], 0.5, places=6)
        self.assertTrue(config["training"]["planner_hard_rank_eval"])

        workspace_tmp = repo_root / "outputs" / "test_tmp" / "config_summary_runtime"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        output_file = workspace_tmp / "summary.json"
        write_summary(
            [
                {
                    "benchmark": "demo",
                    "seed": 1,
                    "final_avg_acc": 0.5,
                    "final_last_task_accuracy": 0.6,
                    "training_time_total": 10.0,
                    "opened_slots": 2,
                    "seed_wall_time_total": 12.0,
                    "kept_slots": 1,
                },
                {
                    "benchmark": "demo",
                    "seed": 2,
                    "final_avg_acc": 0.7,
                    "final_last_task_accuracy": 0.8,
                    "training_time_total": 14.0,
                    "opened_slots": 4,
                    "seed_wall_time_total": 16.0,
                },
            ],
            output_file,
            extras={"benchmark": "demo"},
        )
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["benchmark"], "demo")
        self.assertAlmostEqual(payload["final_avg_acc_mean"], 0.6, places=5)
        self.assertAlmostEqual(payload["final_last_task_accuracy_mean"], 0.7, places=5)
        self.assertAlmostEqual(payload["training_time_total_mean"], 12.0, places=5)
        self.assertIn("opened_slots_std", payload)
        self.assertNotIn("seed_mean", payload)
        self.assertNotIn("seed_wall_time_total_mean", payload)
        self.assertNotIn("kept_slots_mean", payload)


if __name__ == "__main__":
    unittest.main()
