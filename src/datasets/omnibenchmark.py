from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.datasets.base import ContinualBenchmark, SampleRecord, TaskDefinition
from src.datasets.transforms import build_imagenet_test_transform, build_imagenet_train_transform


def _discover_realms(root: Path) -> List[Path]:
    explicit_root = root / "realms"
    if explicit_root.exists():
        return sorted(path for path in explicit_root.iterdir() if path.is_dir())
    return sorted(path for path in root.iterdir() if path.is_dir())


def _collect_realm_records(realm_root: Path, class_offset: int) -> Tuple[List[SampleRecord], List[SampleRecord], int]:
    train_root = realm_root / "train"
    test_root = realm_root / "test"
    if not train_root.exists() or not test_root.exists():
        raise FileNotFoundError(
            "OmniBenchmark expects each realm to contain train/ and test/ directories with class folders."
        )

    class_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    class_to_idx = {name: class_offset + idx for idx, name in enumerate(class_names)}
    train_records: List[SampleRecord] = []
    test_records: List[SampleRecord] = []
    for split, split_root, bucket in [
        ("train", train_root, train_records),
        ("test", test_root, test_records),
    ]:
        for class_name in class_names:
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue
            label = class_to_idx[class_name]
            for file_path in sorted(class_dir.rglob("*")):
                if file_path.is_file() and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    bucket.append(SampleRecord(path=str(file_path), label=label, split=split))
    return train_records, test_records, len(class_names)


def build_omnibenchmark_benchmark(config: Dict[str, object]) -> ContinualBenchmark:
    benchmark_cfg = config["benchmark"]
    root = Path(str(benchmark_cfg["data_root"]))
    realms = _discover_realms(root)
    if not realms:
        raise FileNotFoundError(
            "OmniBenchmark expects realm directories, either directly under the root or under root/realms."
        )

    tasks: List[TaskDefinition] = []
    total_classes = 0
    for task_id, realm_root in enumerate(realms):
        train_records, test_records, num_realm_classes = _collect_realm_records(realm_root, total_classes)
        class_ids = list(range(total_classes, total_classes + num_realm_classes))
        tasks.append(
            TaskDefinition(
                task_id=task_id,
                class_ids=class_ids,
                train_records=train_records,
                test_records=test_records,
                metadata={"benchmark": "omnibenchmark", "realm": realm_root.name},
            )
        )
        total_classes += num_realm_classes

    image_size = int(benchmark_cfg["image_size"])
    return ContinualBenchmark(
        name="omnibenchmark",
        num_classes=total_classes,
        tasks=tasks,
        train_transform=build_imagenet_train_transform(image_size),
        test_transform=build_imagenet_test_transform(image_size),
        metadata={"realm_wise": True},
    )

