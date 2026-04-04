from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition, split_class_order
from src.datasets.transforms import build_cifar_test_transform, build_cifar_train_transform


def _read_cifar_pickle(file_path: Path) -> Dict[bytes, object]:
    with file_path.open("rb") as handle:
        return pickle.load(handle, encoding="bytes")


def _load_split_records(file_path: Path, split: str) -> List[SampleRecord]:
    payload = _read_cifar_pickle(file_path)
    images = np.asarray(payload[b"data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = np.asarray(payload[b"fine_labels"])
    records: List[SampleRecord] = []
    for image, label in zip(images, labels):
        records.append(
            SampleRecord(
                path=None,
                label=int(label),
                split=split,
                image_bytes=image.tobytes(),
                metadata={"size": (32, 32)},
            )
        )
    return records


def build_cifar100_benchmark(config: Dict[str, object]) -> ContinualBenchmark:
    benchmark_cfg = config["benchmark"]
    root = Path(str(benchmark_cfg["data_root"]))
    dataset_root = root / "cifar-100-python" if (root / "cifar-100-python").exists() else root
    train_file = dataset_root / "train"
    test_file = dataset_root / "test"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "CIFAR-100 expects the extracted cifar-100-python directory with train/test pickle files."
        )

    train_records = _load_split_records(train_file, split="train")
    test_records = _load_split_records(test_file, split="test")
    num_classes = 100
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
                metadata={"benchmark": "cifar100"},
            )
        )

    image_size = int(benchmark_cfg["image_size"])
    return ContinualBenchmark(
        name="cifar100",
        num_classes=num_classes,
        tasks=tasks,
        train_transform=build_cifar_train_transform(image_size),
        test_transform=build_cifar_test_transform(image_size),
    )

