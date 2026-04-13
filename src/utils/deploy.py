from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch
import yaml
from PIL import Image, UnidentifiedImageError

from src.datasets.base import ContinualBenchmark
from src.datasets.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from src.models.nh_lora import NHLoRAModel
from src.utils.checkpoint import load_model_artifact, save_model_artifact
from src.utils.io import ensure_dir, write_json


DEPLOY_ARTIFACT_VERSION = "1.0"
REQUIRED_DEPLOY_FILES = (
    "model_final.pt",
    "deploy_manifest.json",
    "class_to_idx.json",
    "idx_to_class.json",
    "inference_config.yaml",
    "preprocess_config.json",
)
REQUIRED_MANIFEST_KEYS = (
    "artifact_version",
    "benchmark",
    "dataset_name",
    "seed",
    "num_classes",
    "class_names",
    "image_size",
    "selected_blocks",
    "insertion_points",
    "backbone_name",
    "checkpoint_file",
    "inference_profile_available",
    "exported_at",
)
IMAGE_EXTENSIONS = (".bmp", ".jpeg", ".jpg", ".png")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sorted_class_names(idx_to_class: Dict[int, str]) -> list[str]:
    return [str(idx_to_class[index]) for index in sorted(idx_to_class)]


def _resolved_benchmark_metadata(
    benchmark: ContinualBenchmark,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = deepcopy(benchmark.metadata if isinstance(benchmark.metadata, dict) else {})
    class_to_idx = {str(name): int(index) for name, index in metadata.get("class_to_idx", {}).items()}
    idx_to_class = {
        int(index): str(name)
        for index, name in metadata.get("idx_to_class", {}).items()
    }
    if not idx_to_class and class_to_idx:
        idx_to_class = {int(index): str(name) for name, index in class_to_idx.items()}
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        class_names = _sorted_class_names(idx_to_class)
    metadata.update(
        {
            "benchmark": benchmark.name,
            "dataset_name": str(config["benchmark"].get("dataset_name", benchmark.name)),
            "scenario": str(config["benchmark"].get("scenario", "class_incremental")),
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "class_names": [str(name) for name in class_names],
            "num_classes": int(metadata.get("num_classes", benchmark.num_classes)),
            "data_root": str(config["benchmark"].get("data_root", metadata.get("data_root", ""))),
            "image_size": int(config["benchmark"]["image_size"]),
        }
    )
    return metadata


def build_preprocess_config(config: Dict[str, Any]) -> Dict[str, Any]:
    image_size = int(config["benchmark"]["image_size"])
    return {
        "transform_family": "imagenet_eval",
        "image_size": image_size,
        "resize_size": int(image_size * 256 / 224),
        "crop_size": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "color_mode": "RGB",
        "supported_extensions": list(IMAGE_EXTENSIONS),
    }


def build_inference_transform(preprocess_config: Dict[str, Any]) -> Compose:
    return Compose(
        [
            Resize(int(preprocess_config["resize_size"])),
            CenterCrop(int(preprocess_config["crop_size"])),
            ToTensor(),
            Normalize(
                mean=tuple(float(value) for value in preprocess_config["mean"]),
                std=tuple(float(value) for value in preprocess_config["std"]),
            ),
        ]
    )


def _build_inference_config(config: Dict[str, Any], benchmark: ContinualBenchmark) -> Dict[str, Any]:
    snapshot = deepcopy(config)
    snapshot["benchmark"]["name"] = benchmark.name
    snapshot["benchmark"]["dataset_name"] = str(snapshot["benchmark"].get("dataset_name", benchmark.name))
    return snapshot


def _build_deploy_manifest(
    config: Dict[str, Any],
    benchmark_metadata: Dict[str, Any],
    seed: int,
    artifact_payload: Dict[str, Any],
) -> Dict[str, Any]:
    model_cfg = config["model"]
    return {
        "artifact_version": DEPLOY_ARTIFACT_VERSION,
        "benchmark": str(benchmark_metadata["benchmark"]),
        "dataset_name": str(benchmark_metadata["dataset_name"]),
        "seed": int(seed),
        "num_classes": int(benchmark_metadata["num_classes"]),
        "class_names": [str(name) for name in benchmark_metadata["class_names"]],
        "image_size": int(benchmark_metadata["image_size"]),
        "selected_blocks": [int(block_id) for block_id in model_cfg["selected_blocks"]],
        "insertion_points": [str(point) for point in model_cfg["insertion_points"]],
        "backbone_name": str(model_cfg["backbone_name"]),
        "checkpoint_file": "model_final.pt",
        "inference_profile_available": bool(artifact_payload.get("inference_profile") is not None),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }


def _deploy_readme_text(manifest: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "# NH-LoRA Deploy Bundle",
            "",
            f"Benchmark: {manifest['benchmark']}",
            f"Dataset: {manifest['dataset_name']}",
            f"Artifact version: {manifest['artifact_version']}",
            "",
            "Use the prediction CLI for local inference:",
            "",
            "```bash",
            "python -m src.engine.predict \\",
            "  --artifact-dir . \\",
            "  --image path/to/image.jpg \\",
            "  --topk 5",
            "```",
            "",
            "The deploy bundle stores the final structured checkpoint, class mappings,",
            "resolved benchmark metadata, and the preprocessing contract used for inference.",
            "",
        ]
    )


def export_deploy_bundle(
    *,
    config: Dict[str, Any],
    benchmark: ContinualBenchmark,
    seed: int,
    artifact_payload: Dict[str, Any],
    output_dirs: Dict[str, Path],
) -> Path:
    benchmark_deploy_dir = output_dirs.get("benchmark_deploy")
    if benchmark_deploy_dir is None:
        raise ValueError("Deploy export requested but experiment.deploy_dir is not configured.")

    seed_dir = ensure_dir(Path(benchmark_deploy_dir) / f"seed_{int(seed)}")
    benchmark_metadata = _resolved_benchmark_metadata(benchmark, config)
    preprocess_config = build_preprocess_config(config)
    inference_config = _build_inference_config(config, benchmark)
    manifest = _build_deploy_manifest(config, benchmark_metadata, seed, artifact_payload)

    save_model_artifact(artifact_payload, seed_dir / "model_final.pt")
    write_json(benchmark_metadata["class_to_idx"], seed_dir / "class_to_idx.json")
    write_json(benchmark_metadata["idx_to_class"], seed_dir / "idx_to_class.json")
    write_json(manifest, seed_dir / "deploy_manifest.json")
    write_json(preprocess_config, seed_dir / "preprocess_config.json")
    write_json(benchmark_metadata, seed_dir / "benchmark_metadata.json")
    (seed_dir / "inference_config.yaml").write_text(
        yaml.safe_dump(inference_config, sort_keys=False),
        encoding="utf-8",
    )
    (seed_dir / "README_DEPLOY.md").write_text(_deploy_readme_text(manifest), encoding="utf-8")
    return seed_dir


def validate_artifact_dir(artifact_dir: str | Path) -> Path:
    path = Path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Artifact path is not a directory: {path}")
    missing_files = [filename for filename in REQUIRED_DEPLOY_FILES if not (path / filename).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Deploy artifact directory is missing required files: {missing_files} under {path}."
        )
    return path


def normalize_idx_to_class(idx_to_class: Dict[str, Any] | Dict[int, Any]) -> Dict[int, str]:
    normalized: Dict[int, str] = {}
    for key, value in idx_to_class.items():
        try:
            normalized[int(key)] = str(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid idx_to_class key '{key}'. Expected integer-like keys.") from exc
    return normalized


def validate_class_mappings(
    class_to_idx: Dict[str, Any],
    idx_to_class: Dict[int, str],
    *,
    num_classes: int,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    normalized_class_to_idx = {str(name): int(index) for name, index in class_to_idx.items()}
    if len(normalized_class_to_idx) != num_classes or len(idx_to_class) != num_classes:
        raise ValueError(
            "Class mapping size mismatch: "
            f"class_to_idx={len(normalized_class_to_idx)} idx_to_class={len(idx_to_class)} num_classes={num_classes}."
        )
    rebuilt_idx_to_class = {int(index): str(name) for name, index in normalized_class_to_idx.items()}
    if rebuilt_idx_to_class != idx_to_class:
        raise ValueError("class_to_idx and idx_to_class are inconsistent.")
    return normalized_class_to_idx, idx_to_class


def load_deploy_artifact(
    artifact_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    path = validate_artifact_dir(artifact_dir)
    manifest = json.loads((path / "deploy_manifest.json").read_text(encoding="utf-8"))
    missing_manifest_keys = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing_manifest_keys:
        raise ValueError(f"Deploy manifest is missing required keys: {missing_manifest_keys}.")
    class_to_idx_raw = json.loads((path / "class_to_idx.json").read_text(encoding="utf-8"))
    idx_to_class_raw = json.loads((path / "idx_to_class.json").read_text(encoding="utf-8"))
    inference_config = yaml.safe_load((path / "inference_config.yaml").read_text(encoding="utf-8")) or {}
    preprocess_config = json.loads((path / "preprocess_config.json").read_text(encoding="utf-8"))
    benchmark_metadata = {}
    benchmark_metadata_path = path / "benchmark_metadata.json"
    if benchmark_metadata_path.exists():
        benchmark_metadata = json.loads(benchmark_metadata_path.read_text(encoding="utf-8"))

    num_classes = int(manifest["num_classes"])
    class_to_idx, idx_to_class = validate_class_mappings(
        class_to_idx_raw,
        normalize_idx_to_class(idx_to_class_raw),
        num_classes=num_classes,
    )
    artifact_payload = load_model_artifact(path / "model_final.pt", map_location=map_location)
    return {
        "artifact_dir": path,
        "artifact_payload": artifact_payload,
        "manifest": manifest,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "inference_config": inference_config,
        "preprocess_config": preprocess_config,
        "benchmark_metadata": benchmark_metadata,
    }


def resolve_prediction_device(requested_device: str) -> Tuple[torch.device, str | None]:
    requested = str(requested_device).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu"), "CUDA requested but unavailable; falling back to CPU."
    return torch.device(requested), None


def load_prediction_image(image_path: str | Path) -> Image.Image:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")
    try:
        image = Image.open(path)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Unable to open image file '{path}': {exc}") from exc


def restore_model_for_prediction(
    inference_config: Dict[str, Any],
    artifact_payload: Dict[str, Any],
    *,
    device: torch.device,
) -> Tuple[NHLoRAModel, Dict[int, Dict[str, Any]], bool]:
    if "model_structure_state" not in artifact_payload:
        raise ValueError("Deploy artifact is missing 'model_structure_state' in model_final.pt.")
    if "model_state" not in artifact_payload:
        raise ValueError("Deploy artifact is missing 'model_state' in model_final.pt.")

    model = NHLoRAModel(inference_config).to(device)
    model.load_structure_state(artifact_payload["model_structure_state"])
    model.load_state_dict(artifact_payload["model_state"])
    inference_profile = artifact_payload.get("inference_profile")
    used_fallback = False
    if inference_profile is None:
        inference_profile = model.build_inference_profile()
        used_fallback = True
    model.eval()
    return model, inference_profile, used_fallback


def clamp_topk(topk: int, num_classes: int) -> Tuple[int, str | None]:
    requested_topk = int(topk)
    if requested_topk <= 0:
        raise ValueError("topk must be positive.")
    if requested_topk > num_classes:
        return num_classes, f"Requested topk={requested_topk} exceeds num_classes={num_classes}; using topk={num_classes}."
    return requested_topk, None
