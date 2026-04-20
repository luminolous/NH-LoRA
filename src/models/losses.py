from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F

from src.models.orthogonality import gram_error, pairwise_overlap_stats


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


def _row_overlap_penalty(left_rows: torch.Tensor, right_rows: torch.Tensor) -> torch.Tensor:
    if left_rows.numel() == 0 or right_rows.numel() == 0:
        return left_rows.new_zeros(())
    normalized_left = F.normalize(left_rows, dim=1)
    normalized_right = F.normalize(right_rows, dim=1)
    overlap = normalized_left @ normalized_right.transpose(0, 1)
    return overlap.pow(2).mean()


def orthogonality_components(model) -> Dict[str, torch.Tensor | float]:
    slot_slot_penalties = []
    slot_shared_penalties = []
    shared_gram_errors = []
    slot_slot_overlaps = []
    slot_shared_overlaps = []
    device = next(model.parameters()).device
    for layer in model.layers.values():
        live_slots = layer.live_slot_ids()
        active_points = layer.orthogonality_active_points() or list(layer.selected_points)
        for point_name in active_points:
            bank = layer.point_banks[point_name]
            if bank.uses_fixed_shared_a():
                shared_gram_errors.append(gram_error(bank.shared_a.detach()))
            if len(live_slots) > 1:
                for left_index in range(len(live_slots)):
                    left_rows = layer.active_slot_a(live_slots[left_index], point_name)
                    for right_index in range(left_index + 1, len(live_slots)):
                        right_rows = layer.active_slot_a(live_slots[right_index], point_name)
                        slot_slot_penalties.append(_row_overlap_penalty(left_rows, right_rows))
                        slot_slot_overlaps.append(
                            pairwise_overlap_stats(left_rows.detach(), right_rows.detach())["max_abs_cos"]
                        )
            shared_rows = bank.shared_a
            for slot_id in live_slots:
                slot_rows = layer.active_slot_a(slot_id, point_name)
                slot_shared_penalties.append(_row_overlap_penalty(slot_rows, shared_rows))
                slot_shared_overlaps.append(
                    pairwise_overlap_stats(slot_rows.detach(), shared_rows.detach())["max_abs_cos"]
                )
    slot_slot = (
        torch.stack(slot_slot_penalties).mean()
        if slot_slot_penalties
        else torch.zeros((), device=device)
    )
    slot_shared = (
        torch.stack(slot_shared_penalties).mean()
        if slot_shared_penalties
        else torch.zeros((), device=device)
    )
    mean_shared_gram_error = (
        float(torch.stack(shared_gram_errors).mean().item())
        if shared_gram_errors
        else 0.0
    )
    mean_slot_slot_overlap = (
        float(sum(slot_slot_overlaps) / len(slot_slot_overlaps))
        if slot_slot_overlaps
        else 0.0
    )
    mean_slot_shared_overlap = (
        float(sum(slot_shared_overlaps) / len(slot_shared_overlaps))
        if slot_shared_overlaps
        else 0.0
    )
    return {
        "slot_slot": slot_slot,
        "slot_shared": slot_shared,
        "mean_shared_gram_error": mean_shared_gram_error,
        "mean_slot_slot_overlap": mean_slot_slot_overlap,
        "mean_slot_shared_overlap": mean_slot_shared_overlap,
    }


def slot_orthogonality(model) -> torch.Tensor:
    components = orthogonality_components(model)
    return components["slot_slot"] + components["slot_shared"]


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
