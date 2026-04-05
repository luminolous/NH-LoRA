from __future__ import annotations

import logging
import unittest
from pathlib import Path

from src.engine.trainer import NHLoRATrainer
from src.utils.logging_utils import configure_logger
from src.utils.sampler import StatefulIndexSampler
from src.utils.seeding import seed_everything
from tests.test_synthetic_continual_smoke import build_synthetic_benchmark, build_test_config


class CheckpointResumeTests(unittest.TestCase):
    def test_checkpoint_save_load_and_mid_epoch_resume(self):
        seed_everything(11, deterministic=True)
        benchmark = build_synthetic_benchmark()
        benchmark.tasks = benchmark.tasks[:1]
        repo_root = Path(__file__).resolve().parents[1]
        workspace_tmp = repo_root / "outputs" / "test_tmp" / "checkpoint_resume"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        config = build_test_config(str(workspace_tmp))
        logger = configure_logger(level=logging.WARNING)

        trainer = NHLoRATrainer(config, logger, benchmark=benchmark)
        trainer.training_state["seed"] = 11
        context = trainer._prepare_task_context(benchmark.tasks[0])
        train_dataset, _ = benchmark.build_task_datasets(0)
        sampler = StatefulIndexSampler(len(train_dataset), shuffle=True, seed=11)
        sampler.mark_consumed(8)
        trainer.current_task_context = context
        trainer.training_state["current_task_id"] = 0
        trainer.training_state["epoch"] = 0
        trainer.training_state["global_step"] = 2
        trainer.training_state["step_in_epoch"] = 2
        trainer.training_state["sampler_state"] = sampler.state_dict()

        checkpoint_path = trainer.save_checkpoint(output_path=workspace_tmp / "resume_test.pt")

        resumed = NHLoRATrainer(config, logger, benchmark=benchmark)
        resumed.load_checkpoint(checkpoint_path)

        self.assertEqual(resumed.training_state["current_task_id"], 0)
        self.assertEqual(resumed.training_state["epoch"], 0)
        self.assertEqual(resumed.training_state["global_step"], 2)
        self.assertEqual(resumed.training_state["step_in_epoch"], 2)
        self.assertIsNotNone(resumed.training_state["sampler_state"])
        self.assertIsNotNone(resumed.current_task_context)
        self.assertEqual(resumed.current_task_context["task_number"], 1)
        self.assertEqual(len(resumed.history_bank.entries), 0)
        self.assertIsNotNone(resumed.inference_profile)

        resumed.train(seed=11)

        self.assertEqual(resumed.training_state["current_task_id"], 1)
        self.assertEqual(resumed.training_state["global_step"], 3)
        self.assertIsNone(resumed.current_task_context)
        self.assertEqual(len(resumed.history_bank.entries), 1)
        self.assertTrue(Path(resumed.last_train_state["last_checkpoint_path"]).exists())


if __name__ == "__main__":
    unittest.main()
