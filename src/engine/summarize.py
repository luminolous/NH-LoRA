from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.metrics import write_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize NH-LoRA multi-seed results.")
    parser.add_argument("--metrics-dir", required=True, help="Directory containing per-seed JSON metrics.")
    parser.add_argument("--output-file", required=True, help="Summary JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    records = []
    for metric_file in sorted(metrics_dir.glob("seed_*.json")):
        records.append(json.loads(metric_file.read_text(encoding="utf-8")))
    benchmark_name = records[0]["benchmark"] if records else metrics_dir.name
    extras = {"benchmark": benchmark_name, "config_name": benchmark_name}
    write_summary(records, args.output_file, extras=extras)


if __name__ == "__main__":
    main()
