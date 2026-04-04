from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_dirs(config: Dict[str, Any], benchmark_name: str | None = None) -> Dict[str, Path]:
    experiment = config["experiment"]
    dirs = {
        "output_root": ensure_dir(experiment["output_root"]),
        "logs": ensure_dir(experiment["log_dir"]),
        "metrics": ensure_dir(experiment["metrics_dir"]),
        "summaries": ensure_dir(experiment["summaries_dir"]),
        "checkpoints": ensure_dir(experiment["checkpoints_dir"]),
    }
    if benchmark_name:
        dirs["benchmark_metrics"] = ensure_dir(dirs["metrics"] / benchmark_name)
        dirs["benchmark_checkpoints"] = ensure_dir(dirs["checkpoints"] / benchmark_name)
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

