from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TaskState:
    embedding: torch.Tensor
    raw_task_vector: torch.Tensor
    feature_mean: torch.Tensor
    feature_var: torch.Tensor
    gradient_sketch: torch.Tensor
    similarity: torch.Tensor
    entropy: torch.Tensor
    similarity_anchor: torch.Tensor
    summary_vector: torch.Tensor
    class_prototypes: Dict[int, torch.Tensor] = field(default_factory=dict)
    warmup_logits: torch.Tensor | None = None


@dataclass
class HistoryEntry:
    task_id: int
    pooled_feature_mean: torch.Tensor
    pooled_feature_var: torch.Tensor
    gradient_sketch: torch.Tensor
    usage_summary: torch.Tensor
    active_rank_summary: torch.Tensor
    entropy_summary: torch.Tensor
    similarity_anchor: torch.Tensor
    task_embedding: torch.Tensor
    summary_vector: torch.Tensor


class TaskStateEncoder(nn.Module):
    def __init__(self, feature_dim: int, grad_dim: int = 16, embedding_dim: int = 128, pool_dim: int | None = None):
        super().__init__()
        self.pool_dim = pool_dim or feature_dim
        raw_dim = self.pool_dim * 2 + grad_dim + 2
        self.input_norm = nn.LayerNorm(raw_dim)
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.grad_dim = grad_dim
        self.embedding_dim = embedding_dim

    def build_raw_task_vector(
        self,
        feature_mean: torch.Tensor,
        feature_var: torch.Tensor,
        gradient_sketch: torch.Tensor,
        similarity: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        pooled_mean = pool_vector(feature_mean, self.pool_dim)
        pooled_var = pool_vector(feature_var, self.pool_dim)
        return torch.cat([pooled_mean, pooled_var, gradient_sketch, similarity, entropy], dim=-1)

    def forward(
        self,
        feature_mean,
        feature_var,
        gradient_sketch,
        similarity,
        entropy,
        similarity_anchor: torch.Tensor | None = None,
        summary_vector: torch.Tensor | None = None,
        class_prototypes: Dict[int, torch.Tensor] | None = None,
        warmup_logits: torch.Tensor | None = None,
    ) -> TaskState:
        raw_task_vector = self.build_raw_task_vector(
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=similarity,
            entropy=entropy,
        )
        embedding = self.encoder(self.input_norm(raw_task_vector))
        return TaskState(
            embedding=embedding,
            raw_task_vector=raw_task_vector,
            feature_mean=feature_mean,
            feature_var=feature_var,
            gradient_sketch=gradient_sketch,
            similarity=similarity,
            entropy=entropy,
            similarity_anchor=similarity_anchor if similarity_anchor is not None else F.normalize(feature_mean, dim=-1),
            summary_vector=summary_vector if summary_vector is not None else torch.cat([raw_task_vector], dim=-1),
            class_prototypes=class_prototypes or {},
            warmup_logits=warmup_logits,
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
        return torch.cat([entry.summary_vector for entry in self.entries], dim=0)

    def mean_similarity(self, current_summary_vector: torch.Tensor, current_anchor: torch.Tensor) -> torch.Tensor:
        if not self.entries:
            return current_summary_vector.new_zeros(current_summary_vector.size(0), 1)
        summary_bank = torch.stack([entry.summary_vector.squeeze(0) for entry in self.entries], dim=0)
        anchor_bank = torch.stack([entry.similarity_anchor.squeeze(0) for entry in self.entries], dim=0)
        summary_bank = F.normalize(summary_bank, dim=-1)
        anchor_bank = F.normalize(anchor_bank, dim=-1)
        current_summary = F.normalize(current_summary_vector, dim=-1)
        current_anchor = F.normalize(current_anchor, dim=-1)
        summary_scores = current_summary @ summary_bank.t()
        anchor_scores = current_anchor @ anchor_bank.t()
        combined_scores = 0.5 * (summary_scores + anchor_scores)
        return combined_scores.max(dim=-1, keepdim=True).values

    def state_dict(self) -> Dict[str, object]:
        serialized_entries = []
        for entry in self.entries:
            serialized_entries.append(
                {
                    "task_id": entry.task_id,
                    "pooled_feature_mean": entry.pooled_feature_mean,
                    "pooled_feature_var": entry.pooled_feature_var,
                    "gradient_sketch": entry.gradient_sketch,
                    "usage_summary": entry.usage_summary,
                    "active_rank_summary": entry.active_rank_summary,
                    "entropy_summary": entry.entropy_summary,
                    "similarity_anchor": entry.similarity_anchor,
                    "task_embedding": entry.task_embedding,
                    "summary_vector": entry.summary_vector,
                }
            )
        return {"entries": serialized_entries}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.entries = []
        for payload in state.get("entries", []):
            self.entries.append(
                HistoryEntry(
                    task_id=int(payload["task_id"]),
                    pooled_feature_mean=payload["pooled_feature_mean"],
                    pooled_feature_var=payload["pooled_feature_var"],
                    gradient_sketch=payload["gradient_sketch"],
                    usage_summary=payload["usage_summary"],
                    active_rank_summary=payload["active_rank_summary"],
                    entropy_summary=payload["entropy_summary"],
                    similarity_anchor=payload["similarity_anchor"],
                    task_embedding=payload["task_embedding"],
                    summary_vector=payload["summary_vector"],
                )
            )


def pool_vector(vector: torch.Tensor, output_dim: int) -> torch.Tensor:
    flattened = vector.reshape(vector.size(0), -1)
    chunks = torch.chunk(flattened, output_dim, dim=-1)
    pooled = []
    for chunk in chunks:
        pooled.append(chunk.mean(dim=-1, keepdim=True))
    if len(pooled) < output_dim:
        pooled.extend([flattened.new_zeros(flattened.size(0), 1) for _ in range(output_dim - len(pooled))])
    return torch.cat(pooled[:output_dim], dim=-1)


def build_history_summary_vector(
    pooled_feature_mean: torch.Tensor,
    pooled_feature_var: torch.Tensor,
    gradient_sketch: torch.Tensor,
    usage_summary: torch.Tensor,
    active_rank_summary: torch.Tensor,
    entropy_summary: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            pooled_feature_mean,
            pooled_feature_var,
            gradient_sketch,
            usage_summary,
            active_rank_summary,
            entropy_summary,
        ],
        dim=-1,
    )


def build_history_entry(
    task_id: int,
    task_state: TaskState,
    usage_summary: torch.Tensor,
    active_rank_summary: torch.Tensor,
    pooled_feature_mean: torch.Tensor,
    pooled_feature_var: torch.Tensor,
) -> HistoryEntry:
    return HistoryEntry(
        task_id=task_id,
        pooled_feature_mean=pooled_feature_mean.detach(),
        pooled_feature_var=pooled_feature_var.detach(),
        gradient_sketch=task_state.gradient_sketch.detach(),
        usage_summary=usage_summary.detach(),
        active_rank_summary=active_rank_summary.detach(),
        entropy_summary=task_state.entropy.detach(),
        similarity_anchor=task_state.similarity_anchor.detach(),
        task_embedding=task_state.embedding.detach(),
        summary_vector=build_history_summary_vector(
            pooled_feature_mean=pooled_feature_mean.detach(),
            pooled_feature_var=pooled_feature_var.detach(),
            gradient_sketch=task_state.gradient_sketch.detach(),
            usage_summary=usage_summary.detach(),
            active_rank_summary=active_rank_summary.detach(),
            entropy_summary=task_state.entropy.detach(),
        ),
    )


def serialize_task_state(task_state: TaskState) -> Dict[str, object]:
    return {
        "embedding": task_state.embedding,
        "raw_task_vector": task_state.raw_task_vector,
        "feature_mean": task_state.feature_mean,
        "feature_var": task_state.feature_var,
        "gradient_sketch": task_state.gradient_sketch,
        "similarity": task_state.similarity,
        "entropy": task_state.entropy,
        "similarity_anchor": task_state.similarity_anchor,
        "summary_vector": task_state.summary_vector,
        "class_prototypes": task_state.class_prototypes,
        "warmup_logits": task_state.warmup_logits,
    }


def deserialize_task_state(payload: Dict[str, object]) -> TaskState:
    return TaskState(
        embedding=payload["embedding"],
        raw_task_vector=payload["raw_task_vector"],
        feature_mean=payload["feature_mean"],
        feature_var=payload["feature_var"],
        gradient_sketch=payload["gradient_sketch"],
        similarity=payload["similarity"],
        entropy=payload["entropy"],
        similarity_anchor=payload["similarity_anchor"],
        summary_vector=payload["summary_vector"],
        class_prototypes=payload.get("class_prototypes", {}),
        warmup_logits=payload.get("warmup_logits"),
    )


def detach_task_state(task_state: TaskState) -> TaskState:
    return TaskState(
        embedding=task_state.embedding.detach(),
        raw_task_vector=task_state.raw_task_vector.detach(),
        feature_mean=task_state.feature_mean.detach(),
        feature_var=task_state.feature_var.detach(),
        gradient_sketch=task_state.gradient_sketch.detach(),
        similarity=task_state.similarity.detach(),
        entropy=task_state.entropy.detach(),
        similarity_anchor=task_state.similarity_anchor.detach(),
        summary_vector=task_state.summary_vector.detach(),
        class_prototypes={class_id: prototype.detach() for class_id, prototype in task_state.class_prototypes.items()},
        warmup_logits=task_state.warmup_logits.detach() if task_state.warmup_logits is not None else None,
    )
