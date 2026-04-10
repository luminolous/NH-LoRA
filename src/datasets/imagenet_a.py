from __future__ import annotations

from typing import Dict

from src.datasets.imagefolder_benchmark import build_imagefolder_class_incremental_benchmark


def build_imagenet_a_benchmark(config: Dict[str, object]):
    return build_imagefolder_class_incremental_benchmark(
        "imagenet_a",
        config,
        expected_num_classes=200,
        require_matching_split_classes=True,
    )
