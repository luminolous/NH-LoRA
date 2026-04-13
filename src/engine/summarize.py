from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import load_config
from src.utils.io import ensure_dir, resolve_experiment_paths
from src.utils.metrics import write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize NH-LoRA multi-seed results.")
    parser.add_argument("--metrics-dir", help="Directory containing per-seed JSON metrics.")
    parser.add_argument("--output-file", help="Summary JSON output path.")
    parser.add_argument("--config", help="Optional config path for config-driven summary resolution.")
    parser.add_argument("--benchmark", help="Benchmark name override when using --config.")
    parser.add_argument("--output-root", default=None, help="Optional root output directory override.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.config:
        config = load_config(args.config)
        resolve_experiment_paths(config, output_root_override=args.output_root)
        benchmark_name = str(args.benchmark or config["benchmark"].get("name") or config["benchmark"]["dataset_name"])
        metrics_dir = ensure_dir(Path(config["experiment"]["metrics_dir"]) / benchmark_name)
        output_file = Path(config["experiment"]["summaries_dir"]) / f"{benchmark_name}_summary.json"
    else:
        if not args.metrics_dir or not args.output_file:
            raise ValueError("Either provide --config or both --metrics-dir and --output-file.")
        metrics_dir = Path(args.metrics_dir)
        output_file = Path(args.output_file)

    records = []
    for metric_file in sorted(metrics_dir.glob("seed_*.json")):
        records.append(json.loads(metric_file.read_text(encoding="utf-8")))
    benchmark_name = records[0]["benchmark"] if records else metrics_dir.name
    extras = {"benchmark": benchmark_name, "config_name": benchmark_name}
    write_summary(records, output_file, extras=extras)


if __name__ == "__main__":
    main()
