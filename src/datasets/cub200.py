from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition, split_class_order
from src.datasets.transforms import build_imagenet_test_transform, build_imagenet_train_transform


def _read_mapping(file_path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for line in file_path.read_text(encoding="utf-8").splitlines():
        idx, value = line.split(" ", 1)
        mapping[int(idx)] = value.strip()
    return mapping


def build_cub200_benchmark(config: Dict[str, object]) -> ContinualBenchmark:
    benchmark_cfg = config["benchmark"]
    root = Path(str(benchmark_cfg["data_root"]))
    images_root = root / "images"
    required = [
        root / "images.txt",
        root / "image_class_labels.txt",
        root / "train_test_split.txt",
        images_root,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "CUB-200-2011 expects the official metadata files and images directory. Missing: "
            + ", ".join(missing)
        )

    image_map = _read_mapping(root / "images.txt")
    class_map = {
        int(idx): int(label) - 1
        for idx, label in (
            line.split(" ", 1)
            for line in (root / "image_class_labels.txt").read_text(encoding="utf-8").splitlines()
        )
    }
    split_map = {
        int(idx): "train" if int(is_train) == 1 else "test"
        for idx, is_train in (
            line.split(" ", 1)
            for line in (root / "train_test_split.txt").read_text(encoding="utf-8").splitlines()
        )
    }

    train_records: List[SampleRecord] = []
    test_records: List[SampleRecord] = []
    for image_id, relative_path in image_map.items():
        record = SampleRecord(
            path=str(images_root / relative_path),
            label=class_map[image_id],
            split=split_map[image_id],
        )
        if record.split == "train":
            train_records.append(record)
        else:
            test_records.append(record)

    num_classes = 200
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
                metadata={"benchmark": "cub200"},
            )
        )

    image_size = int(benchmark_cfg["image_size"])
    return ContinualBenchmark(
        name="cub200",
        num_classes=num_classes,
        tasks=tasks,
        train_transform=build_imagenet_train_transform(image_size),
        test_transform=build_imagenet_test_transform(image_size),
    )

