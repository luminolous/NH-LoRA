from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class PlannerSignals:
    novelty: torch.Tensor
    conflict: torch.Tensor
    rank_score: torch.Tensor
    rank_budget: int
    consolidate: torch.Tensor
    shared_gate: torch.Tensor
    history_attention: torch.Tensor | None = None
    history_context: torch.Tensor | None = None
    planner_representation: torch.Tensor | None = None


def select_most_compatible_slot(slot_bank, task_embedding: torch.Tensor) -> int | None:
    live_slots = slot_bank.live_slot_ids()
    if not live_slots:
        return None
    normalized_task = torch.nn.functional.normalize(task_embedding.squeeze(0), dim=-1)
    keys = torch.stack([slot_bank.slot_keys[slot_id] for slot_id in live_slots], dim=0)
    keys = torch.nn.functional.normalize(keys, dim=-1)
    similarity = keys @ normalized_task
    return live_slots[int(torch.argmax(similarity).item())]


def materialize_action(
    action: str,
    signals: PlannerSignals,
    slot_bank,
    task_embedding: torch.Tensor,
    task_id: int,
    max_slots_per_block: int,
    tau_consolidate: float = 0.5,
) -> Dict[str, object]:
    live_slots = slot_bank.live_slot_ids()
    rank_cfg = {slot_id: slot_bank.slot_metadata[slot_id].rank for slot_id in live_slots}
    created_new = False
    fallback_action = None
    selected_slot = None
    if action == "reuse_shared":
        active_slot_candidates = live_slots
    elif action == "expand_rank_existing_slot":
        selected_slot = select_most_compatible_slot(slot_bank, task_embedding)
        if selected_slot is None:
            selected_slot = slot_bank.add_slot(signals.rank_budget, task_id=task_id)
            created_new = True
            fallback_action = "open_new_slot"
        expanded_rank = slot_bank.slot_metadata[selected_slot].rank + signals.rank_budget
        slot_bank.expand_rank(selected_slot, expanded_rank)
        rank_cfg[selected_slot] = slot_bank.slot_metadata[selected_slot].rank
        active_slot_candidates = [selected_slot] + [slot_id for slot_id in live_slots if slot_id != selected_slot]
    elif action == "open_new_slot":
        if len(slot_bank.slot_metadata) < max_slots_per_block:
            selected_slot = slot_bank.add_slot(signals.rank_budget, task_id=task_id)
            created_new = True
        else:
            fallback_action = "expand_rank_existing_slot"
            selected_slot = select_most_compatible_slot(slot_bank, task_embedding)
            if selected_slot is None:
                raise RuntimeError("Planner requested slot growth but no slot could be selected.")
            expanded_rank = slot_bank.slot_metadata[selected_slot].rank + signals.rank_budget
            slot_bank.expand_rank(selected_slot, expanded_rank)
        rank_cfg[selected_slot] = slot_bank.slot_metadata[selected_slot].rank
        current_live = slot_bank.live_slot_ids()
        active_slot_candidates = [selected_slot] + [slot_id for slot_id in current_live if slot_id != selected_slot]
    elif action == "freeze_old_strong_retention":
        active_slot_candidates = live_slots
    else:
        raise ValueError(f"Unsupported planner action: {action}")
    return {
        "action": action,
        "active_slot_candidates": active_slot_candidates,
        "selected_slot": selected_slot,
        "rank_cfg": rank_cfg,
        "shared_gate": float(signals.shared_gate.item()),
        "consolidate_flag": bool(signals.consolidate.item() >= tau_consolidate),
        "deterministic": len(active_slot_candidates) <= 1,
        "created_new_slot": created_new,
        "fallback_action": fallback_action,
        "strong_retention": action == "freeze_old_strong_retention",
    }


class HorizonPlanner(nn.Module):
    def __init__(self, selected_blocks: List[int], task_embedding_dim: int, history_dim: int, hidden_dim: int, layer_embedding_dim: int, rank_min: int, rank_max: int, tau_novelty: float, tau_conflict: float):
        super().__init__()
        self.selected_blocks = list(selected_blocks)
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.tau_novelty = tau_novelty
        self.tau_conflict = tau_conflict
        self.layer_embeddings = nn.Embedding(max(selected_blocks) + 1, layer_embedding_dim)
        self.history_query = nn.Linear(task_embedding_dim, task_embedding_dim)
        self.history_key = nn.Linear(history_dim, task_embedding_dim)
        self.history_value = nn.Linear(history_dim, task_embedding_dim)
        input_dim = task_embedding_dim * 3 + layer_embedding_dim
        self.trunks = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        for block_id in self.selected_blocks:
            self.trunks[str(block_id)] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )
            self.output_heads[str(block_id)] = nn.Linear(hidden_dim, 5)

    def forward(self, block_id: int, task_embedding: torch.Tensor, history_summary: torch.Tensor | None) -> PlannerSignals:
        history_context, history_attention = self.aggregate_history(task_embedding, history_summary)
        layer_embed = self.layer_embeddings(torch.tensor([block_id], device=task_embedding.device)).expand(task_embedding.size(0), -1)
        planner_inputs = torch.cat([task_embedding, history_context, task_embedding * history_context, layer_embed], dim=-1)
        planner_representation = self.trunks[str(block_id)](planner_inputs)
        outputs = self.output_heads[str(block_id)](planner_representation)
        novelty = torch.sigmoid(outputs[:, 0:1])
        conflict = torch.sigmoid(outputs[:, 1:2])
        rank_value = torch.sigmoid(outputs[:, 2:3])
        consolidate = torch.sigmoid(outputs[:, 3:4])
        shared_gate = torch.sigmoid(outputs[:, 4:5])
        rank_budget = int(round(self.rank_min + float(rank_value.item()) * (self.rank_max - self.rank_min)))
        return PlannerSignals(
            novelty=novelty,
            conflict=conflict,
            rank_score=rank_value,
            rank_budget=max(self.rank_min, min(self.rank_max, rank_budget)),
            consolidate=consolidate,
            shared_gate=shared_gate,
            history_attention=history_attention,
            history_context=history_context,
            planner_representation=planner_representation,
        )

    def aggregate_history(self, task_embedding: torch.Tensor, history_summary: torch.Tensor | None):
        if history_summary is None or history_summary.numel() == 0:
            zeros = task_embedding.new_zeros(task_embedding.size(0), task_embedding.size(-1))
            return zeros, None
        if history_summary.dim() == 1:
            history_summary = history_summary.unsqueeze(0)
        query = self.history_query(task_embedding)
        keys = self.history_key(history_summary)
        values = self.history_value(history_summary)
        scale = query.size(-1) ** 0.5
        attention_logits = query @ keys.transpose(0, 1) / scale
        attention = F.softmax(attention_logits, dim=-1)
        context = attention @ values
        return context, attention

    def decide_action(self, signals: PlannerSignals) -> str:
        novelty = float(signals.novelty.item())
        conflict = float(signals.conflict.item())
        if novelty < self.tau_novelty and conflict < self.tau_conflict:
            return "reuse_shared"
        if novelty >= self.tau_novelty and conflict < self.tau_conflict:
            return "expand_rank_existing_slot"
        if novelty >= self.tau_novelty and conflict >= self.tau_conflict:
            return "open_new_slot"
        return "freeze_old_strong_retention"
