from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import nn


@dataclass
class TaskState:
    embedding: torch.Tensor
    feature_mean: torch.Tensor
    feature_var: torch.Tensor
    gradient_sketch: torch.Tensor
    similarity: torch.Tensor
    entropy: torch.Tensor


@dataclass
class HistoryEntry:
    task_id: int
    task_embedding: torch.Tensor
    feature_norm: float
    entropy: float
    usage: float
    active_rank: float
    summary_vector: torch.Tensor


class TaskStateEncoder(nn.Module):
    def __init__(self, feature_dim: int, grad_dim: int = 16, embedding_dim: int = 128):
        super().__init__()
        self.feature_proj = nn.Sequential(nn.Linear(feature_dim * 2, embedding_dim), nn.GELU())
        self.grad_proj = nn.Sequential(nn.Linear(grad_dim, embedding_dim), nn.GELU())
        self.scalar_proj = nn.Sequential(nn.Linear(2, embedding_dim), nn.GELU())
        self.final_proj = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.grad_dim = grad_dim
        self.embedding_dim = embedding_dim

    def forward(self, feature_mean, feature_var, gradient_sketch, similarity, entropy) -> TaskState:
        feature_stack = torch.cat([feature_mean, feature_var], dim=-1)
        scalar_stack = torch.cat([similarity, entropy], dim=-1)
        embedding = self.final_proj(
            torch.cat(
                [
                    self.feature_proj(feature_stack),
                    self.grad_proj(gradient_sketch),
                    self.scalar_proj(scalar_stack),
                ],
                dim=-1,
            )
        )
        return TaskState(
            embedding=embedding,
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=similarity,
            entropy=entropy,
        )


class HistoryBank:
    def __init__(self):
        self.entries: List[HistoryEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def append(self, entry: HistoryEntry) -> None:
        self.entries.append(entry)

    def aggregate(self) -> torch.Tensor | None:
        if not self.entries:
            return None
        return torch.stack([entry.summary_vector for entry in self.entries], dim=0).mean(dim=0)

    def mean_similarity(self, current_embedding: torch.Tensor) -> torch.Tensor:
        if not self.entries:
            return current_embedding.new_zeros(current_embedding.size(0), 1)
        anchors = torch.stack([entry.task_embedding.squeeze(0) for entry in self.entries], dim=0)
        anchors = torch.nn.functional.normalize(anchors, dim=-1)
        current = torch.nn.functional.normalize(current_embedding, dim=-1)
        return (current @ anchors.t()).mean(dim=-1, keepdim=True)


def build_history_entry(task_id: int, task_state: TaskState, usage: float, active_rank: float) -> HistoryEntry:
    summary_vector = torch.cat(
        [
            task_state.embedding.detach(),
            task_state.feature_mean.norm(dim=-1, keepdim=True).detach(),
            task_state.entropy.detach(),
            task_state.embedding.new_tensor([[usage, active_rank]]),
        ],
        dim=-1,
    )
    return HistoryEntry(
        task_id=task_id,
        task_embedding=task_state.embedding.detach(),
        feature_norm=float(task_state.feature_mean.norm().item()),
        entropy=float(task_state.entropy.mean().item()),
        usage=usage,
        active_rank=active_rank,
        summary_vector=summary_vector,
    )
