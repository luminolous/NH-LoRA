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
        for point_name in layer.selected_points:
            factors = []
            for slot_id in live_slots:
                factor = layer.active_slot_a(slot_id, point_name)
                factors.append(F.normalize(factor, dim=0))
            for left_index in range(len(factors)):
                for right_index in range(left_index + 1, len(factors)):
                    overlap = factors[left_index] @ factors[right_index].transpose(0, 1)
                    penalties.append(overlap.pow(2).mean())
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def rank_penalty(model, device: torch.device) -> torch.Tensor:
    penalties = []
    for layer in model.layers.values():
        for slot_id in layer.live_slot_ids():
            mask = layer.active_rank_mask(slot_id, device=device)
            penalties.append(mask.abs().sum())
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def growth_penalty(applied_plans: Dict[int, Dict[str, object]] | None, device: torch.device) -> torch.Tensor:
    if not applied_plans:
        return torch.zeros((), device=device)
    penalties = []
    for plan in applied_plans.values():
        penalties.append(torch.tensor(float(bool(plan.get("created_new_slot", False))), device=device))
    if not penalties:
        return torch.zeros((), device=device)
    return torch.stack(penalties).mean()


def routing_balance_loss(route_info: Dict[int, Dict[str, object]], device: torch.device) -> torch.Tensor:
    losses = []
    for layer_state in route_info.values():
        distribution = layer_state.get("routing_distribution")
        if distribution is None or distribution.numel() == 0:
            continue
        mean_distribution = distribution.mean(dim=0)
        mean_distribution = mean_distribution / mean_distribution.sum().clamp_min(1e-8)
        uniform = torch.full_like(mean_distribution, 1.0 / max(mean_distribution.numel(), 1))
        losses.append(
            torch.sum(
                mean_distribution * (
                    mean_distribution.clamp_min(1e-8).log() - uniform.clamp_min(1e-8).log()
                )
            )
        )
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()
