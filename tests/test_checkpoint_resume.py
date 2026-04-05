from __future__ import annotations

import logging
import unittest
from pathlib import Path

from src.engine.trainer import NHLoRATrainer
from src.utils.checkpoint import load_model_artifact
from src.utils.logging_utils import configure_logger
from src.utils.seeding import seed_everything
from tests.test_synthetic_continual_smoke import build_synthetic_benchmark, build_test_config


class FinalModelArtifactTests(unittest.TestCase):
    def test_only_one_final_model_artifact_is_saved(self):
        seed_everything(11, deterministic=True)
        benchmark = build_synthetic_benchmark()
        benchmark.tasks = benchmark.tasks[:1]
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "final_model_artifact"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = build_test_config(str(workspace_tmp))
        config["experiment"]["save_checkpoints"] = True
        logger = configure_logger(level=logging.WARNING)

        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        metrics = trainer.train(seed=11)

        checkpoint_dir = workspace_tmp / "checkpoints" / "synthetic_smoke"
        final_artifact = checkpoint_dir / "synthetic_smoke_seed11_final.pt"
        self.assertTrue(final_artifact.exists())
        self.assertEqual(list(checkpoint_dir.glob("*.pt")), [final_artifact])
        self.assertEqual(list(checkpoint_dir.glob("latest.pt")), [])
        self.assertEqual(list(checkpoint_dir.glob("task*_epoch*.pt")), [])

        payload = load_model_artifact(final_artifact, map_location="cpu")
        self.assertEqual(payload["benchmark"], "synthetic_smoke")
        self.assertEqual(payload["seed"], 11)
        self.assertIn("model_state", payload)
        self.assertIn("model_structure_state", payload)
        self.assertIn("inference_profile", payload)
        self.assertIn("final_metrics", payload)
        self.assertNotIn("optimizer_state", payload)
        self.assertNotIn("scheduler_state", payload)
        self.assertNotIn("history_bank_state", payload)
        self.assertEqual(payload["final_metrics"]["benchmark"], metrics["benchmark"])
        self.assertEqual(trainer.last_train_state["last_model_artifact_path"], str(final_artifact))


if __name__ == "__main__":
    unittest.main()
