from __future__ import annotations

import argparse
from pathlib import Path

from src.engine.trainer import NHLoRATrainer
from src.utils.config import load_config, save_config_snapshot
from src.utils.logging_utils import configure_logger
from src.utils.seeding import seed_everything
from src.utils.io import ensure_output_dirs, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Train NH-LoRA on a continual learning benchmark.")
    parser.add_argument("--config", required=True, help="Path to the benchmark YAML config.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for the run.")
    parser.add_argument("--benchmark", required=True, help="Benchmark name override for output naming.")
    parser.add_argument("--output-root", default="outputs", help="Root output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config["experiment"]["output_root"] = args.output_root
    config["benchmark"]["name"] = args.benchmark
    output_dirs = ensure_output_dirs(config, benchmark_name=args.benchmark)
    log_file = output_dirs["logs"] / f"{args.benchmark}_seed{args.seed}.log"
    logger = configure_logger(log_file=log_file)
    seed_everything(args.seed, deterministic=bool(config["runtime"].get("deterministic", False)))
    trainer = NHLoRATrainer(config, logger)
    metrics = trainer.train(seed=args.seed)
    metrics["seed"] = args.seed
    metrics_path = output_dirs["benchmark_metrics"] / f"seed_{args.seed}.json"
    write_json(metrics, metrics_path)
    save_config_snapshot(config, output_dirs["logs"] / f"{args.benchmark}_seed{args.seed}_config.yaml")
    logger.info("Finished NH-LoRA run for %s seed %d", args.benchmark, args.seed)


if __name__ == "__main__":
    main()

