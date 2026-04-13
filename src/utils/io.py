from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _rebase_output_path(
    path_value: str | Path,
    current_root: str | Path,
    new_root: str | Path,
) -> str:
    path = Path(path_value)
    current_root_path = Path(current_root)
    new_root_path = Path(new_root)
    try:
        relative_path = path.relative_to(current_root_path)
    except ValueError:
        return str(path)
    return str(new_root_path / relative_path)


def resolve_experiment_paths(
    config: Dict[str, Any],
    output_root_override: str | Path | None = None,
) -> Dict[str, Any]:
    experiment = config.setdefault("experiment", {})
    current_root = experiment.get("output_root", "outputs")
    if output_root_override is None:
        return config

    new_root = str(output_root_override)
    if str(current_root) == new_root:
        experiment["output_root"] = new_root
        return config

    path_keys = [
        "output_root",
        "log_dir",
        "metrics_dir",
        "summaries_dir",
        "checkpoints_dir",
        "deploy_dir",
    ]
    for key in path_keys:
        if key not in experiment:
            continue
        if key == "output_root":
            experiment[key] = new_root
            continue
        experiment[key] = _rebase_output_path(experiment[key], current_root, new_root)
    return config


def ensure_output_dirs(config: Dict[str, Any], benchmark_name: str | None = None) -> Dict[str, Path]:
    experiment = config["experiment"]
    dirs = {
        "output_root": ensure_dir(experiment["output_root"]),
        "logs": ensure_dir(experiment["log_dir"]),
        "metrics": ensure_dir(experiment["metrics_dir"]),
        "summaries": ensure_dir(experiment["summaries_dir"]),
        "checkpoints": ensure_dir(experiment["checkpoints_dir"]),
    }
    if "deploy_dir" in experiment:
        dirs["deploy"] = ensure_dir(experiment["deploy_dir"])
    if benchmark_name:
        dirs["benchmark_metrics"] = ensure_dir(dirs["metrics"] / benchmark_name)
        dirs["benchmark_checkpoints"] = ensure_dir(dirs["checkpoints"] / benchmark_name)
        if "deploy" in dirs:
            dirs["benchmark_deploy"] = ensure_dir(dirs["deploy"] / benchmark_name)
    return dirs


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_serializable(asdict(value))
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write_json(data: Any, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_to_serializable(data), indent=2),
        encoding="utf-8",
    )
