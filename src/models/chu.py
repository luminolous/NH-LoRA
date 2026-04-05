from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch.nn import functional as F


@dataclass
class CHUDecision:
    should_merge: bool
    should_prune: bool
    should_keep_or_freeze: bool


@dataclass
class CHUReport:
    merged_slots: int = 0
    pruned_slots: int = 0
    kept_slots: int = 0
    frozen_slots: int = 0


class ConsolidationHomeostasisUnit:
    def __init__(self, config: Dict[str, float]):
        self.merge_rate = float(config["merge_rate"])
        self.usage_high_threshold = float(config["usage_high_threshold"])
        self.usage_low_threshold = float(config["usage_low_threshold"])
        self.stability_threshold = float(config["stability_threshold"])
        self.redundancy_threshold = float(config["redundancy_threshold"])
        self.freeze_on_keep = bool(config.get("freeze_on_keep", True))

    def estimate_redundancy(self, layer) -> Dict[int, float]:
        live_slots = layer.live_slot_ids()
        if len(live_slots) <= 1:
            shared_signature = F.normalize(layer.shared_update_signature().detach().view(1, -1), dim=-1)
            return {
                slot_id: float(
                    F.cosine_similarity(
                        layer.slot_update_signature(slot_id).detach().view(1, -1),
                        shared_signature,
                        dim=-1,
                    ).item()
                )
                for slot_id in live_slots
            }
        slot_signatures = torch.stack([layer.slot_update_signature(slot_id).detach() for slot_id in live_slots], dim=0)
        slot_signatures = F.normalize(slot_signatures, dim=-1)
        shared_signature = F.normalize(layer.shared_update_signature().detach().view(1, -1), dim=-1)
        similarity = slot_signatures @ slot_signatures.t()
        redundancy = {}
        for row, slot_id in enumerate(live_slots):
            row_values = similarity[row]
            other_values = torch.cat([row_values[:row], row_values[row + 1 :]], dim=0)
            shared_similarity = F.cosine_similarity(slot_signatures[row : row + 1], shared_signature, dim=-1)
            candidate_values = other_values
            if shared_similarity.numel():
                candidate_values = torch.cat([candidate_values, shared_similarity], dim=0)
            redundancy[slot_id] = float(candidate_values.max().item()) if candidate_values.numel() else 0.0
        return redundancy

    def decide_slot(
        self,
        usage: float,
        stability: float,
        redundancy: float,
        consolidate_flag: bool,
    ) -> CHUDecision:
        should_merge = consolidate_flag and usage >= self.usage_high_threshold and stability >= self.stability_threshold
        should_prune = usage <= self.usage_low_threshold and redundancy >= self.redundancy_threshold
        should_keep_or_freeze = not should_merge and not should_prune
        return CHUDecision(
            should_merge=should_merge,
            should_prune=should_prune,
            should_keep_or_freeze=should_keep_or_freeze,
        )

    def consolidate_layer(self, layer, planner_cfg: Dict[str, object], usage_stats: Dict[int, float]) -> CHUReport:
        report = CHUReport()
        stability = layer.estimate_slot_stability()
        redundancy = self.estimate_redundancy(layer)
        consolidate_flag = bool(planner_cfg.get("consolidate_flag", False))

        for slot_id in list(layer.live_slot_ids()):
            usage = float(usage_stats.get(slot_id, 0.0))
            stable = float(stability.get(slot_id, 0.0))
            redundant = float(redundancy.get(slot_id, 0.0))
            decision = self.decide_slot(
                usage=usage,
                stability=stable,
                redundancy=redundant,
                consolidate_flag=consolidate_flag,
            )

            if decision.should_merge:
                for bank in layer.point_banks.values():
                    bank.merge_slot_into_shared(slot_id, self.merge_rate, rank=layer.slot_metadata[slot_id].rank)
                layer.freeze_slot(slot_id)
                layer.slot_metadata[slot_id].retained_for_inference = False
                report.merged_slots += 1
                report.frozen_slots += 1
                continue

            if decision.should_prune:
                layer.prune_slot(slot_id)
                report.pruned_slots += 1
                continue

            if decision.should_keep_or_freeze:
                layer.slot_metadata[slot_id].retained_for_inference = True
                if self.freeze_on_keep:
                    layer.freeze_slot(slot_id)
                    report.frozen_slots += 1
                report.kept_slots += 1

        return report
