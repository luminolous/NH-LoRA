from __future__ import annotations

from typing import Any, Callable, Dict

from src.datasets.cifar100 import build_cifar100_benchmark
from src.datasets.cub200 import build_cub200_benchmark
from src.datasets.imagenet_r import build_imagenet_r_benchmark
from src.datasets.omnibenchmark import build_omnibenchmark_benchmark


DATASET_REGISTRY: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "cifar100": build_cifar100_benchmark,
    "cub200": build_cub200_benchmark,
    "imagenet_r": build_imagenet_r_benchmark,
    "omnibenchmark": build_omnibenchmark_benchmark,
}


def build_benchmark(config: Dict[str, Any]):
    dataset_name = str(config["benchmark"]["dataset_name"]).lower()
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Unsupported benchmark dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name](config)
