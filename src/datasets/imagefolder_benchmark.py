from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition, split_class_order
from src.datasets.transforms import build_imagenet_test_transform, build_imagenet_train_transform


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _collect_split_class_names(split_root: Path) -> List[str]:
    return sorted(path.name for path in split_root.iterdir() if path.is_dir())


def collect_imagefolder_records(
    root: Path,
    split: str,
    class_to_idx: Dict[str, int] | None = None,
    expected_class_names: Sequence[str] | None = None,
):
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"ImageFolder benchmark expects a '{split}' directory under {root}.")

    class_names = _collect_split_class_names(split_root)
    if expected_class_names is not None:
        expected_names = sorted(str(name) for name in expected_class_names)
        if class_names != expected_names:
            missing_names = [name for name in expected_names if name not in class_names]
            extra_names = [name for name in class_names if name not in expected_names]
            raise ValueError(
                f"ImageFolder benchmark split '{split}' under {root} has mismatched class folders. "
                f"Missing={missing_names} Extra={extra_names}."
            )
    if class_to_idx is None:
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    records: List[SampleRecord] = []
    for class_name in class_names:
        if class_name not in class_to_idx:
            continue
        label = class_to_idx[class_name]
        class_dir = split_root / class_name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in _IMAGE_EXTENSIONS:
                records.append(SampleRecord(path=str(file_path), label=label, split=split))
    return records, class_to_idx, class_names


def build_imagefolder_class_incremental_benchmark(
    benchmark_name: str,
    config: Dict[str, object],
    *,
    expected_num_classes: int | None = None,
    require_matching_split_classes: bool = False,
) -> ContinualBenchmark:
    benchmark_cfg = config["benchmark"]
    root = Path(str(benchmark_cfg["data_root"]))
    train_records, class_to_idx, train_class_names = collect_imagefolder_records(root, "train")
    test_records, _, test_class_names = collect_imagefolder_records(
        root,
        "test",
        class_to_idx=class_to_idx,
        expected_class_names=train_class_names if require_matching_split_classes else None,
    )
    num_classes = len(class_to_idx)
    if expected_num_classes is not None and num_classes != expected_num_classes:
        raise ValueError(
            f"{benchmark_name} expects exactly {expected_num_classes} classes under {root}, "
            f"but found {num_classes}."
        )
    class_splits = split_class_order(
        num_classes=num_classes,
        num_tasks=int(benchmark_cfg["num_tasks"]),
        classes_per_task=benchmark_cfg.get("classes_per_task"),
    )
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    tasks: List[TaskDefinition] = []
    for task_id, class_ids in enumerate(class_splits):
        task_train_records = [record for record in train_records if record.label in class_ids]
        task_test_records = [record for record in test_records if record.label in class_ids]
        task_class_names = [idx_to_class[int(class_id)] for class_id in class_ids]
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                class_ids=list(class_ids),
                train_records=task_train_records,
                test_records=task_test_records,
                metadata={
                    "benchmark": benchmark_name,
                    "class_to_idx": dict(class_to_idx),
                    "class_names": task_class_names,
                    "class_count": len(task_class_names),
                    "train_sample_count": len(task_train_records),
                    "test_sample_count": len(task_test_records),
                },
            )
        )

    image_size = int(benchmark_cfg["image_size"])
    return ContinualBenchmark(
        name=benchmark_name,
        num_classes=num_classes,
        tasks=tasks,
        train_transform=build_imagenet_train_transform(image_size),
        test_transform=build_imagenet_test_transform(image_size),
        metadata={
            "benchmark": benchmark_name,
            "dataset_type": "imagefolder",
            "class_to_idx": dict(class_to_idx),
            "idx_to_class": dict(idx_to_class),
            "class_names": [idx_to_class[int(class_id)] for class_id in range(num_classes)],
            "num_classes": num_classes,
            "image_size": image_size,
            "class_name_count": num_classes,
            "split_class_names": {
                "train": train_class_names,
                "test": test_class_names,
            },
            "expected_num_classes": expected_num_classes,
            "require_matching_split_classes": require_matching_split_classes,
            "data_root": str(root),
        },
    )
