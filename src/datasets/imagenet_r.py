from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition, split_class_order
from src.datasets.transforms import build_imagenet_test_transform, build_imagenet_train_transform


def _collect_imagefolder_records(root: Path, split: str, class_to_idx: Dict[str, int] | None = None):
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"ImageNet-R expects a '{split}' directory under {root}.")

    class_names = sorted(path.name for path in split_root.iterdir() if path.is_dir())
    if class_to_idx is None:
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    records: List[SampleRecord] = []
    for class_name in class_names:
        label = class_to_idx[class_name]
        class_dir = split_root / class_name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                records.append(SampleRecord(path=str(file_path), label=label, split=split))
    return records, class_to_idx


def build_imagenet_r_benchmark(config: Dict[str, object]) -> ContinualBenchmark:
    benchmark_cfg = config["benchmark"]
    root = Path(str(benchmark_cfg["data_root"]))
    train_records, class_to_idx = _collect_imagefolder_records(root, "train")
    test_records, _ = _collect_imagefolder_records(root, "test", class_to_idx=class_to_idx)
    num_classes = len(class_to_idx)
    class_splits = split_class_order(
        num_classes=num_classes,
        num_tasks=int(benchmark_cfg["num_tasks"]),
        classes_per_task=benchmark_cfg.get("classes_per_task"),
    )

    tasks: List[TaskDefinition] = []
    for task_id, class_ids in enumerate(class_splits):
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                class_ids=list(class_ids),
                train_records=[record for record in train_records if record.label in class_ids],
                test_records=[record for record in test_records if record.label in class_ids],
                metadata={"benchmark": "imagenet_r", "class_to_idx": class_to_idx},
            )
        )

    image_size = int(benchmark_cfg["image_size"])
    return ContinualBenchmark(
        name="imagenet_r",
        num_classes=num_classes,
        tasks=tasks,
        train_transform=build_imagenet_train_transform(image_size),
        test_transform=build_imagenet_test_transform(image_size),
    )

