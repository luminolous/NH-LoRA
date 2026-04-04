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

        workspace_tmp = repo_root / "outputs" / "test_tmp" / "config_summary"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        output_file = workspace_tmp / "summary.json"
        write_summary(
            [
                {"benchmark": "demo", "seed": 1, "final_avg_acc": 0.5, "opened_slots": 2},
                {"benchmark": "demo", "seed": 2, "final_avg_acc": 0.7, "opened_slots": 4},
            ],
            output_file,
            extras={"benchmark": "demo"},
        )
        payload = json.loads(output_file.read_text(encoding="utf-8"))
        self.assertEqual(payload["benchmark"], "demo")
        self.assertAlmostEqual(payload["final_avg_acc_mean"], 0.6, places=5)
        self.assertIn("opened_slots_std", payload)


if __name__ == "__main__":
    unittest.main()
