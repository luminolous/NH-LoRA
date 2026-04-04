from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


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
            return {slot_id: 0.0 for slot_id in live_slots}
        keys = torch.stack([layer.slot_keys[slot_id].detach() for slot_id in live_slots], dim=0)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        similarity = keys @ keys.t()
        redundancy = {}
        for row, slot_id in enumerate(live_slots):
            row_values = similarity[row]
            other_values = torch.cat([row_values[:row], row_values[row + 1 :]], dim=0)
            redundancy[slot_id] = float(other_values.max().item()) if other_values.numel() else 0.0
        return redundancy

    def consolidate_layer(self, layer, planner_cfg: Dict[str, object], usage_stats: Dict[int, float]) -> CHUReport:
        report = CHUReport()
        stability = layer.estimate_slot_stability()
        redundancy = self.estimate_redundancy(layer)
        consolidate_flag = bool(planner_cfg.get("consolidate_flag", False))
        for slot_id in list(layer.live_slot_ids()):
            usage = usage_stats.get(slot_id, 0.0)
            stable = stability.get(slot_id, 0.0)
            redundant = redundancy.get(slot_id, 0.0)
            if consolidate_flag and usage >= self.usage_high_threshold and stable >= self.stability_threshold:
                for bank in layer.point_banks.values():
                    bank.merge_slot_into_shared(slot_id, self.merge_rate)
                layer.freeze_slot(slot_id)
                report.merged_slots += 1
                report.frozen_slots += 1
            elif usage <= self.usage_low_threshold and redundant >= self.redundancy_threshold:
                layer.prune_slot(slot_id)
                report.pruned_slots += 1
            else:
                if self.freeze_on_keep:
                    layer.freeze_slot(slot_id)
                    report.frozen_slots += 1
                report.kept_slots += 1
        return report

