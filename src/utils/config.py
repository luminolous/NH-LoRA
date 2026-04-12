from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_base_config(config_path: Path, raw_config: Dict[str, Any]) -> Path | None:
    base_ref = raw_config.get("_base_")
    if base_ref is None:
        candidate = config_path.parent / "base.yaml"
        if candidate.exists() and candidate.resolve() != config_path.resolve():
            return candidate
        return None
    return (config_path.parent / str(base_ref)).resolve()


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_path = _resolve_base_config(config_path, raw_config)
    raw_config.pop("_base_", None)
    if base_path is None:
        config = raw_config
    else:
        base_config = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        config = deep_merge_dict(base_config, raw_config)
    config.setdefault("meta", {})
    config["meta"]["config_path"] = str(config_path)
    return config


def save_config_snapshot(config: Dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

