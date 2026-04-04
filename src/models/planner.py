from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn


@dataclass
class PlannerSignals:
    novelty: torch.Tensor
    conflict: torch.Tensor
    rank_budget: int
    consolidate: torch.Tensor
    shared_gate: torch.Tensor


def select_most_compatible_slot(slot_bank, task_embedding: torch.Tensor) -> int | None:
    live_slots = slot_bank.live_slot_ids()
    if not live_slots:
        return None
    normalized_task = torch.nn.functional.normalize(task_embedding.squeeze(0), dim=-1)
    keys = torch.stack([slot_bank.slot_keys[slot_id] for slot_id in live_slots], dim=0)
    keys = torch.nn.functional.normalize(keys, dim=-1)
    similarity = keys @ normalized_task
    return live_slots[int(torch.argmax(similarity).item())]


def materialize_action(action: str, signals: PlannerSignals, slot_bank, task_embedding: torch.Tensor, task_id: int, max_slots_per_block: int) -> Dict[str, object]:
    live_slots = slot_bank.live_slot_ids()
    rank_cfg = {slot_id: slot_bank.slot_metadata[slot_id].rank for slot_id in live_slots}
    created_new = False
    if action == "reuse_shared":
        active_slots = live_slots[:1]
    elif action == "expand_rank_existing_slot":
        slot_id = select_most_compatible_slot(slot_bank, task_embedding)
        if slot_id is None:
            slot_id = slot_bank.add_slot(slot_bank.slot_init_rank, task_id=task_id)
            created_new = True
        slot_bank.expand_rank(slot_id, signals.rank_budget)
        rank_cfg[slot_id] = slot_bank.slot_metadata[slot_id].rank
        active_slots = [slot_id]
    elif action == "open_new_slot":
        if len(slot_bank.slot_metadata) < max_slots_per_block:
            slot_id = slot_bank.add_slot(signals.rank_budget, task_id=task_id)
            created_new = True
        else:
            slot_id = select_most_compatible_slot(slot_bank, task_embedding)
            if slot_id is None:
                raise RuntimeError("Planner requested slot growth but no slot could be selected.")
            slot_bank.expand_rank(slot_id, signals.rank_budget)
        rank_cfg[slot_id] = slot_bank.slot_metadata[slot_id].rank
        active_slots = [slot_id]
    elif action == "freeze_old_strong_retention":
        active_slots = live_slots[:1]
    else:
        raise ValueError(f"Unsupported planner action: {action}")
    return {
        "action": action,
        "active_slots": active_slots,
        "rank_cfg": rank_cfg,
        "shared_gate": float(signals.shared_gate.item()),
        "consolidate_flag": bool(signals.consolidate.item() >= 0.5),
        "deterministic": len(active_slots) == 1,
        "created_new_slot": created_new,
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
        input_dim = task_embedding_dim + history_dim + layer_embedding_dim
        self.heads = nn.ModuleDict()
        for block_id in self.selected_blocks:
            self.heads[str(block_id)] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 5),
            )

    def forward(self, block_id: int, task_embedding: torch.Tensor, history_summary: torch.Tensor | None) -> PlannerSignals:
        if history_summary is None:
            history_summary = task_embedding.new_zeros(task_embedding.size(0), task_embedding.size(-1) + 4)
        layer_embed = self.layer_embeddings(torch.tensor([block_id], device=task_embedding.device))
        inputs = torch.cat([task_embedding, history_summary, layer_embed], dim=-1)
        outputs = self.heads[str(block_id)](inputs)
        novelty = torch.sigmoid(outputs[:, 0:1])
        conflict = torch.sigmoid(outputs[:, 1:2])
        rank_value = torch.sigmoid(outputs[:, 2:3])
        consolidate = torch.sigmoid(outputs[:, 3:4])
        shared_gate = torch.sigmoid(outputs[:, 4:5])
        rank_budget = int(round(self.rank_min + float(rank_value.item()) * (self.rank_max - self.rank_min)))
        return PlannerSignals(
            novelty=novelty,
            conflict=conflict,
            rank_budget=max(self.rank_min, min(self.rank_max, rank_budget)),
            consolidate=consolidate,
            shared_gate=shared_gate,
        )

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
