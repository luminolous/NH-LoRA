from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.datasets.imagefolder_benchmark import build_imagefolder_class_incremental_benchmark


def build_custom_benchmark(config: Dict[str, object]):
    benchmark = build_imagefolder_class_incremental_benchmark(
        "custom",
        config,
        require_matching_split_classes=True,
    )
    benchmark_cfg = config["benchmark"]
    expected_num_tasks = int(benchmark_cfg["num_tasks"])
    resolved_num_tasks = len(benchmark.tasks)
    if resolved_num_tasks != expected_num_tasks:
        num_classes = int(benchmark.num_classes)
        classes_per_task = benchmark_cfg.get("classes_per_task")
        raise ValueError(
            "Custom benchmark task split mismatch: "
            f"dataset exposes {num_classes} classes and classes_per_task={classes_per_task}, "
            f"which resolves to {resolved_num_tasks} tasks, but benchmark.num_tasks={expected_num_tasks}."
        )

    class_to_idx = dict(benchmark.metadata.get("class_to_idx", {}))
    idx_to_class = {int(index): str(name) for index, name in benchmark.metadata.get("idx_to_class", {}).items()}
    if not idx_to_class and class_to_idx:
        idx_to_class = {int(index): str(name) for name, index in class_to_idx.items()}
    class_names = [idx_to_class[index] for index in sorted(idx_to_class)]
    benchmark.metadata.update(
        {
            "dataset_name": str(benchmark_cfg.get("dataset_name", "custom")),
            "scenario": str(benchmark_cfg.get("scenario", "class_incremental")),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "class_names": class_names,
            "num_classes": int(benchmark.num_classes),
            "data_root": str(Path(str(benchmark_cfg["data_root"])).resolve()),
            "image_size": int(benchmark_cfg["image_size"]),
        }
    )
    return benchmark
