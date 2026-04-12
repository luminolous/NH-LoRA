from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SampleRecord:
    path: str | None
    label: int
    split: str
    image_bytes: bytes | None = None
    metadata: Dict[str, object] = field(default_factory=dict)


class RecordDataset(Dataset):
    def __init__(self, records: Sequence[SampleRecord], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        if record.image_bytes is not None:
            image = Image.frombytes("RGB", record.metadata["size"], record.image_bytes)
        else:
            if record.path is None:
                raise ValueError("A record must provide either a file path or in-memory bytes.")
            image = Image.open(record.path).convert("RGB")
        tensor = self.transform(image)
        return index, tensor, record.label


@dataclass
class TaskDefinition:
    task_id: int
    class_ids: List[int]
    train_records: List[SampleRecord]
    test_records: List[SampleRecord]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class ContinualBenchmark:
    name: str
    num_classes: int
    tasks: List[TaskDefinition]
    train_transform: object
    test_transform: object
    metadata: Dict[str, object] = field(default_factory=dict)

    def build_task_datasets(self, task_id: int):
        task = self.tasks[task_id]
        return (
            RecordDataset(task.train_records, self.train_transform),
            RecordDataset(task.test_records, self.test_transform),
        )

    def seen_classes_up_to(self, task_id: int) -> List[int]:
        seen: List[int] = []
        for task in self.tasks[: task_id + 1]:
            seen.extend(task.class_ids)
        return sorted(set(seen))


def split_class_order(num_classes: int, num_tasks: int, classes_per_task: int | None = None) -> List[List[int]]:
    if classes_per_task is not None:
        splits: List[List[int]] = []
        class_ids = list(range(num_classes))
        for start in range(0, len(class_ids), classes_per_task):
            splits.append(class_ids[start : start + classes_per_task])
        return splits
    if num_tasks <= 0:
        raise ValueError("num_tasks must be positive.")
    class_ids = list(range(num_classes))
    chunk_size = max(1, len(class_ids) // num_tasks)
    splits = []
    cursor = 0
    for task_idx in range(num_tasks):
        end = len(class_ids) if task_idx == num_tasks - 1 else min(len(class_ids), cursor + chunk_size)
        splits.append(class_ids[cursor:end])
        cursor = end
    return [chunk for chunk in splits if chunk]

