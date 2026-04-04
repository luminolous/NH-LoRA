from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)


def feature_retention(current_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(current_features, teacher_features)


def slot_orthogonality(model) -> torch.Tensor:
    penalties = []
    device = next(model.parameters()).device
    for layer in model.layers.values():
        live_slots = layer.live_slot_ids()
        if len(live_slots) <= 1:
            continue
        keys = torch.stack([layer.slot_keys[slot_id] for slot_id in live_slots], dim=0)
        keys = F.normalize(keys, dim=-1)
        gram = keys @ keys.t()
        penalties.append((gram - torch.eye(gram.size(0), device=gram.device)).pow(2).mean())
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def rank_penalty(model) -> torch.Tensor:
    penalties = []
    device = next(model.parameters()).device
    for layer in model.layers.values():
        for slot_id in layer.live_slot_ids():
            penalties.append(
                torch.tensor(layer.slot_metadata[slot_id].rank / layer.slot_r_max, device=device)
            )
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def growth_penalty(planner_out: Dict[int, Dict[str, object]], device: torch.device) -> torch.Tensor:
    created = 0.0
    active_rank = 0.0
    denom = max(len(planner_out), 1)
    for layer_cfg in planner_out.values():
        created += 1.0 if layer_cfg.get("created_new_slot", False) else 0.0
        active_rank += float(sum(layer_cfg.get("rank_cfg", {}).values()))
    return torch.tensor((created + active_rank / max(denom, 1)) / max(denom, 1), device=device)


def routing_balance_loss(route_info: Dict[int, Dict[str, object]], device: torch.device) -> torch.Tensor:
    losses = []
    for layer_state in route_info.values():
        weights = layer_state.get("routing_weights")
        if weights is None or weights.numel() == 0:
            continue
        losses.append((weights.mean(dim=0) - weights.mean()).pow(2).mean())
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()

