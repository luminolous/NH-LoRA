from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)


def feature_retention(
    current_features: Dict[int, torch.Tensor],
    teacher_features: Dict[int, torch.Tensor],
    layers: list[int],
    device: torch.device,
) -> torch.Tensor:
    losses = []
    for layer_id in layers:
        if layer_id not in current_features or layer_id not in teacher_features:
            continue
        losses.append(F.mse_loss(current_features[layer_id], teacher_features[layer_id]))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def slot_orthogonality(model) -> torch.Tensor:
    penalties = []
    device = next(model.parameters()).device
    for layer in model.layers.values():
        live_slots = layer.live_slot_ids()
        if len(live_slots) <= 1:
            continue
        signatures = torch.stack([layer.slot_update_signature(slot_id) for slot_id in live_slots], dim=0)
        signatures = F.normalize(signatures, dim=-1)
        gram = signatures @ signatures.t()
        penalties.append((gram - torch.eye(gram.size(0), device=gram.device)).pow(2).mean())
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def rank_penalty(model, raw_planner: Dict[int, object] | None, device: torch.device) -> torch.Tensor:
    penalties = []
    if raw_planner:
        for signals in raw_planner.values():
            penalties.append(signals.rank_score.mean())
    for layer in model.layers.values():
        for slot_id in layer.live_slot_ids():
            rank_fraction = layer.slot_metadata[slot_id].rank / layer.slot_r_max
            penalties.append(torch.tensor(rank_fraction, device=device))
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def growth_penalty(raw_planner: Dict[int, object] | None, device: torch.device) -> torch.Tensor:
    if not raw_planner:
        return torch.zeros((), device=device)
    penalties = []
    for signals in raw_planner.values():
        penalties.append((signals.novelty * signals.conflict).mean())
    return torch.stack(penalties).mean() if penalties else torch.zeros((), device=device)


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
