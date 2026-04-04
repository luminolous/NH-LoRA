from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn
from torch.nn import functional as F


class IncrementalCosineClassifier(nn.Module):
    def __init__(self, feature_dim: int, tau: float = 16.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.tau = tau
        self.weight = nn.Parameter(torch.empty(0, feature_dim))

    @property
    def num_classes(self) -> int:
        return self.weight.size(0)

    def expand(self, num_new_classes: int) -> List[int]:
        if num_new_classes <= 0:
            return []
        device = self.weight.device if self.weight.numel() else None
        new_weights = torch.randn(num_new_classes, self.feature_dim, device=device) * 0.02
        self.weight = nn.Parameter(torch.cat([self.weight.detach(), new_weights], dim=0))
        start = self.num_classes - num_new_classes
        return list(range(start, self.num_classes))

    def imprint(self, features: torch.Tensor, labels: torch.Tensor, class_ids: Iterable[int]) -> None:
        with torch.no_grad():
            for class_id in class_ids:
                mask = labels == class_id
                if mask.any():
                    prototype = F.normalize(features[mask].mean(dim=0, keepdim=True), dim=-1)
                    self.weight[class_id : class_id + 1].copy_(prototype)

    def imprint_from_prototypes(self, prototypes: dict[int, torch.Tensor], class_ids: Iterable[int]) -> None:
        with torch.no_grad():
            for class_id in class_ids:
                if class_id not in prototypes:
                    continue
                prototype = F.normalize(prototypes[class_id].view(1, -1), dim=-1)
                self.weight[class_id : class_id + 1].copy_(prototype)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 0:
            raise RuntimeError("Classifier head has not been expanded yet.")
        return self.tau * (F.normalize(features, dim=-1) @ F.normalize(self.weight, dim=-1).t())
