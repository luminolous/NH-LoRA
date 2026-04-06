from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.utils.io import write_json


# Only aggregate whitelisted final metrics to avoid averaging numeric fields
# that are bookkeeping-oriented or otherwise not meaningful at summary level.
SUMMARY_NUMERIC_WHITELIST = [
    "final_avg_acc",
    "final_last_task_accuracy",
    "final_forgetting",
    "parameter_growth",
    "total_active_rank",
    "training_time_total",
    "opened_slots",
    "pruned_slots",
    "merged_slots",
    "frozen_slots",
    "kept_slots",
]


def compute_mean_std(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("No metric records were provided for summarization.")

    summary: Dict[str, Any] = {"num_seeds": len(records), "per_seed": records}
    for key in SUMMARY_NUMERIC_WHITELIST:
        if not all(key in record and isinstance(record[key], (int, float)) for record in records):
            continue
        values = [float(record[key]) for record in records]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[f"{key}_mean"] = mean
        summary[f"{key}_std"] = math.sqrt(variance)
    return summary


def write_summary(records: List[Dict[str, Any]], output_path: str | Path, extras: Dict[str, Any] | None = None) -> None:
    summary = compute_mean_std(records)
    if extras:
        summary.update(extras)
    write_json(summary, output_path)
