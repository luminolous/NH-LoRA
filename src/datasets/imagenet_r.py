from __future__ import annotations

from typing import Dict

from src.datasets.imagefolder_benchmark import build_imagefolder_class_incremental_benchmark


def build_imagenet_r_benchmark(config: Dict[str, object]):
    return build_imagefolder_class_incremental_benchmark("imagenet_r", config)
