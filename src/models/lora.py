from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F


def sanitize_key(name: str) -> str:
    return name.replace(".", "_")


@dataclass
class SlotMetadata:
    rank: int
    frozen: bool = False
    pruned: bool = False
    opened_at_task: int = 0
    last_usage: float = 0.0


class ProjectionBank(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, shared_rank: int, slot_r_max: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shared_rank = shared_rank
        self.slot_r_max = slot_r_max
        self.shared_a = nn.Parameter(torch.empty(shared_rank, input_dim))
        self.shared_b = nn.Parameter(torch.empty(output_dim, shared_rank))
        nn.init.kaiming_uniform_(self.shared_a, a=5**0.5)
        nn.init.zeros_(self.shared_b)
        self.slot_a = nn.ParameterList()
        self.slot_b = nn.ParameterList()

    def add_slot(self) -> int:
        slot_a = nn.Parameter(torch.empty(self.slot_r_max, self.input_dim))
        slot_b = nn.Parameter(torch.empty(self.output_dim, self.slot_r_max))
        nn.init.kaiming_uniform_(slot_a, a=5**0.5)
        nn.init.zeros_(slot_b)
        self.slot_a.append(slot_a)
        self.slot_b.append(slot_b)
        return len(self.slot_a) - 1

    def shared_delta(self, hidden_states: torch.Tensor, shared_gate: float) -> torch.Tensor:
        delta = F.linear(F.linear(hidden_states, self.shared_a), self.shared_b)
        return delta * float(shared_gate)

    def slot_delta(self, hidden_states: torch.Tensor, slot_id: int, rank: int) -> torch.Tensor:
        low_rank = F.linear(hidden_states, self.slot_a[slot_id])
        mask = hidden_states.new_zeros(low_rank.shape[-1])
        mask[:rank] = 1.0
        low_rank = low_rank * mask
        return F.linear(low_rank, self.slot_b[slot_id])

    def shared_update_matrix(self) -> torch.Tensor:
        return self.shared_b @ self.shared_a

    def slot_update_matrix(self, slot_id: int, rank: int) -> torch.Tensor:
        return self.slot_b[slot_id][:, :rank] @ self.slot_a[slot_id][:rank, :]

    def merge_slot_into_shared(self, slot_id: int, merge_rate: float, rank: int | None = None) -> None:
        with torch.no_grad():
            slot_rank = min(rank or self.slot_r_max, self.slot_a[slot_id].size(0))
            merged_update = self.shared_update_matrix() + merge_rate * self.slot_update_matrix(slot_id, slot_rank)
            u, s, vh = torch.linalg.svd(merged_update, full_matrices=False)
            target_rank = min(self.shared_rank, s.numel())
            shared_a = torch.zeros_like(self.shared_a)
            shared_b = torch.zeros_like(self.shared_b)
            if target_rank > 0:
                sqrt_s = torch.sqrt(s[:target_rank])
                shared_b[:, :target_rank] = u[:, :target_rank] * sqrt_s.unsqueeze(0)
                shared_a[:target_rank, :] = sqrt_s.unsqueeze(1) * vh[:target_rank, :]
            self.shared_a.copy_(shared_a)
            self.shared_b.copy_(shared_b)


class NHLoRALayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        selected_points: List[str],
        shared_rank: int,
        slot_r_max: int,
        slot_init_rank: int,
        bootstrap_slot_rank: int,
        max_slots: int,
        router_topk: int,
        router_temperature: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.selected_points = list(selected_points)
        self.shared_rank = shared_rank
        self.slot_r_max = slot_r_max
        self.slot_init_rank = slot_init_rank
        self.bootstrap_slot_rank = bootstrap_slot_rank
        self.max_slots = max_slots
        self.router_topk = router_topk
        self.router_temperature = router_temperature
        self.router_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, self.router_dim)

        output_dims = {
            "q_proj": embed_dim,
            "k_proj": embed_dim,
            "v_proj": embed_dim,
            "out_proj": embed_dim,
            "mlp_fc1": embed_dim * 4,
            "mlp_fc2": embed_dim,
        }
        input_dims = {
            "q_proj": embed_dim,
            "k_proj": embed_dim,
            "v_proj": embed_dim,
            "out_proj": embed_dim,
            "mlp_fc1": embed_dim,
            "mlp_fc2": embed_dim * 4,
        }
        self.point_banks = nn.ModuleDict()
        for point_name in self.selected_points:
            self.point_banks[sanitize_key(point_name)] = ProjectionBank(
                input_dim=input_dims[point_name],
                output_dim=output_dims[point_name],
                shared_rank=shared_rank,
                slot_r_max=slot_r_max,
            )

        self.slot_keys = nn.ParameterList()
        self.slot_metadata: List[SlotMetadata] = []
        self.slot_snapshots: List[Dict[str, torch.Tensor]] = []
        self.bootstrap_initialized = False

    def live_slot_ids(self) -> List[int]:
        return [slot_id for slot_id, meta in enumerate(self.slot_metadata) if not meta.pruned]

    def add_slot(self, initial_rank: int, task_id: int) -> int:
        if len(self.slot_metadata) >= self.max_slots:
            raise RuntimeError("Maximum number of slots reached for this layer.")
        for bank in self.point_banks.values():
            bank.add_slot()
        key = nn.Parameter(torch.randn(self.router_dim) * 0.02)
        self.slot_keys.append(key)
        self.slot_metadata.append(SlotMetadata(rank=min(initial_rank, self.slot_r_max), opened_at_task=task_id))
        self.slot_snapshots.append({})
        return len(self.slot_metadata) - 1

    def ensure_bootstrap_slot(self, task_id: int) -> int:
        if self.bootstrap_initialized:
            return 0
        slot_id = self.add_slot(self.bootstrap_slot_rank, task_id=task_id)
        self.bootstrap_initialized = True
        return slot_id

    def expand_rank(self, slot_id: int, new_rank: int) -> None:
        self.slot_metadata[slot_id].rank = min(max(new_rank, 1), self.slot_r_max)

    def freeze_slot(self, slot_id: int) -> None:
        self.slot_metadata[slot_id].frozen = True
        for bank in self.point_banks.values():
            bank.slot_a[slot_id].requires_grad = False
            bank.slot_b[slot_id].requires_grad = False
        self.slot_keys[slot_id].requires_grad = False

    def prune_slot(self, slot_id: int) -> None:
        self.slot_metadata[slot_id].pruned = True
        self.freeze_slot(slot_id)

    def route(self, normalized_tokens: torch.Tensor, planner_cfg: Dict[str, object], task_state) -> Dict[str, object]:
        pooled = normalized_tokens[:, 0]
        live_slots = self.live_slot_ids()
        requested = planner_cfg.get("active_slot_candidates", planner_cfg.get("active_slots", live_slots))
        candidate_slots = [slot_id for slot_id in requested if slot_id in live_slots]
        if not candidate_slots:
            candidate_slots = live_slots[:1]
        if not candidate_slots:
            return {
                "candidate_slots": [],
                "selected_slots": [],
                "routing_weights": normalized_tokens.new_zeros(normalized_tokens.size(0), 0),
            }
        if len(candidate_slots) == 1 and bool(planner_cfg.get("deterministic", False)):
            weights = normalized_tokens.new_ones(normalized_tokens.size(0), 1)
            return {
                "candidate_slots": candidate_slots,
                "selected_slots": list(candidate_slots),
                "routing_weights": weights,
                "usage_vector": weights.mean(dim=0),
            }
        query = F.normalize(self.query_proj(pooled), dim=-1)
        keys = torch.stack([self.slot_keys[slot_id] for slot_id in candidate_slots], dim=0)
        keys = F.normalize(keys, dim=-1)
        scores = query @ keys.t()
        topk = min(self.router_topk, scores.size(1))
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
        weights = torch.softmax(topk_scores / self.router_temperature, dim=-1)
        usage_vector = normalized_tokens.new_zeros(len(candidate_slots))
        selected_slots = []
        for row, indices in enumerate(topk_indices):
            row_selected = []
            for col, candidate_index in enumerate(indices.tolist()):
                usage_vector[candidate_index] += weights[row, col].detach()
                row_selected.append(candidate_slots[candidate_index])
            if row == 0:
                selected_slots = row_selected
        usage_vector = usage_vector / max(scores.size(0), 1)
        return {
            "candidate_slots": candidate_slots,
            "selected_slots": selected_slots,
            "routing_weights": weights,
            "topk_indices": topk_indices,
            "usage_vector": usage_vector,
        }

    def _point_output_dim(self, point_name: str) -> int:
        return self.embed_dim * 4 if point_name == "mlp_fc1" else self.embed_dim

    def _shared_delta(self, point_name: str, hidden_states: torch.Tensor, planner_cfg: Dict[str, object]) -> torch.Tensor:
        if point_name not in self.selected_points:
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        bank = self.point_banks[sanitize_key(point_name)]
        return bank.shared_delta(hidden_states, shared_gate=float(planner_cfg.get("shared_gate", 1.0)))

    def _slot_delta(self, point_name: str, hidden_states: torch.Tensor, route_state: Dict[str, object], planner_cfg: Dict[str, object]) -> torch.Tensor:
        if point_name not in self.selected_points:
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        candidate_slots = route_state.get("candidate_slots", route_state.get("active_slots", []))
        if not candidate_slots:
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        bank = self.point_banks[sanitize_key(point_name)]
        rank_cfg = planner_cfg.get("rank_cfg", {})
        weights = route_state.get("routing_weights")
        topk_indices = route_state.get("topk_indices")
        delta = hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        if weights is None or topk_indices is None:
            slot_id = candidate_slots[0]
            rank = int(rank_cfg.get(slot_id, self.slot_metadata[slot_id].rank))
            return bank.slot_delta(hidden_states, slot_id, rank)
        for batch_index in range(hidden_states.size(0)):
            batch_hidden = hidden_states[batch_index : batch_index + 1]
            for col, candidate_index in enumerate(topk_indices[batch_index].tolist()):
                slot_id = candidate_slots[candidate_index]
                rank = int(rank_cfg.get(slot_id, self.slot_metadata[slot_id].rank))
                slot_delta = bank.slot_delta(batch_hidden, slot_id, rank)
                delta[batch_index : batch_index + 1] += slot_delta * weights[batch_index, col]
                self.slot_metadata[slot_id].last_usage = float(weights[batch_index, col].detach().item())
        return delta

    def apply_to_qkv(self, hidden_states: torch.Tensor, base_qkv: torch.Tensor, route_state: Dict[str, object], planner_cfg: Dict[str, object]) -> torch.Tensor:
        batch_size, seq_len, total_dim = base_qkv.shape
        qkv = base_qkv.reshape(batch_size, seq_len, 3, self.embed_dim)
        if "q_proj" in self.selected_points:
            qkv[:, :, 0, :] += self._shared_delta("q_proj", hidden_states, planner_cfg)
            qkv[:, :, 0, :] += self._slot_delta("q_proj", hidden_states, route_state, planner_cfg)
        if "k_proj" in self.selected_points:
            qkv[:, :, 1, :] += self._shared_delta("k_proj", hidden_states, planner_cfg)
            qkv[:, :, 1, :] += self._slot_delta("k_proj", hidden_states, route_state, planner_cfg)
        if "v_proj" in self.selected_points:
            qkv[:, :, 2, :] += self._shared_delta("v_proj", hidden_states, planner_cfg)
            qkv[:, :, 2, :] += self._slot_delta("v_proj", hidden_states, route_state, planner_cfg)
        return qkv.reshape(batch_size, seq_len, total_dim)

    def apply_to_projection(self, point_name: str, hidden_states: torch.Tensor, route_state: Dict[str, object], planner_cfg: Dict[str, object], base_output: torch.Tensor | None = None) -> torch.Tensor:
        if base_output is None:
            base_output = hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        if point_name not in self.selected_points:
            return base_output
        return base_output + self._shared_delta(point_name, hidden_states, planner_cfg) + self._slot_delta(point_name, hidden_states, route_state, planner_cfg)

    def capture_pre_task_snapshot(self) -> None:
        for slot_id in self.live_slot_ids():
            snapshot = {}
            for point_name, bank in self.point_banks.items():
                snapshot[f"{point_name}.a"] = bank.slot_a[slot_id].detach().clone()
                snapshot[f"{point_name}.b"] = bank.slot_b[slot_id].detach().clone()
            self.slot_snapshots[slot_id] = snapshot

    def estimate_slot_stability(self) -> Dict[int, float]:
        stability = {}
        for slot_id in self.live_slot_ids():
            snapshot = self.slot_snapshots[slot_id]
            if not snapshot:
                stability[slot_id] = 1.0
                continue
            delta_norm = 0.0
            current_norm = 0.0
            for point_name, bank in self.point_banks.items():
                current_a = bank.slot_a[slot_id].detach()
                current_b = bank.slot_b[slot_id].detach()
                prev_a = snapshot[f"{point_name}.a"]
                prev_b = snapshot[f"{point_name}.b"]
                delta_norm += float((current_a - prev_a).norm().item() + (current_b - prev_b).norm().item())
                current_norm += float(current_a.norm().item() + current_b.norm().item())
            stability[slot_id] = float(torch.exp(torch.tensor(-delta_norm / max(current_norm, 1e-6))).item())
        return stability

    def slot_update_signature(self, slot_id: int) -> torch.Tensor:
        fragments = []
        rank = self.slot_metadata[slot_id].rank
        for bank in self.point_banks.values():
            fragments.append(bank.slot_update_matrix(slot_id, rank).reshape(-1))
        return torch.cat(fragments, dim=0)

    def shared_update_signature(self) -> torch.Tensor:
        fragments = []
        for bank in self.point_banks.values():
            fragments.append(bank.shared_update_matrix().reshape(-1))
        return torch.cat(fragments, dim=0)
