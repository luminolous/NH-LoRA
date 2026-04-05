from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_model_artifact(payload: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_model_artifact(artifact_path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(artifact_path), map_location=map_location, weights_only=False)
