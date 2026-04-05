from __future__ import annotations

from typing import Dict, Iterator, List

import torch
from torch.utils.data import Sampler


class StatefulIndexSampler(Sampler[int]):
    def __init__(
        self,
        num_samples: int,
        shuffle: bool,
        seed: int = 0,
        permutation: List[int] | None = None,
        position: int = 0,
    ):
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.permutation = permutation
        self.position = position

    def _build_permutation(self) -> List[int]:
        if self.permutation is not None and len(self.permutation) == self.num_samples:
            return self.permutation
        if not self.shuffle:
            self.permutation = list(range(self.num_samples))
            return self.permutation
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.permutation = torch.randperm(self.num_samples, generator=generator).tolist()
        return self.permutation

    def __iter__(self) -> Iterator[int]:
        permutation = self._build_permutation()
        for index in permutation[self.position :]:
            yield index

    def __len__(self) -> int:
        return max(self.num_samples - self.position, 0)

    def mark_consumed(self, batch_size: int) -> None:
        self.position = min(self.position + batch_size, self.num_samples)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.permutation = None
        self.position = 0

    def state_dict(self) -> Dict[str, object]:
        return {
            "num_samples": self.num_samples,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "permutation": self._build_permutation(),
            "position": self.position,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.num_samples = int(state["num_samples"])
        self.shuffle = bool(state["shuffle"])
        self.seed = int(state["seed"])
        self.permutation = list(state["permutation"])
        self.position = int(state["position"])
