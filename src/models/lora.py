from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn
from torch.nn import functional as F

from src.models.orthogonality import (
    GRAM_FAST_PATH_TOLERANCE,
    build_row_basis,
    fixed_basis_merge_update,
    gram_error,
    orthogonal_rows_against_basis,
    pairwise_overlap_stats,
    row_orthonormal_matrix,
)
from src.models.planner import MaterializedLayerPlan


def sanitize_key(name: str) -> str:
    return name.replace(".", "_")


@dataclass
class SlotMetadata:
    rank: int
    frozen: bool = False
    pruned: bool = False
    opened_at_task: int = 0
    last_usage: float = 0.0
    retained_for_inference: bool = True
    cumulative_usage: float = 0.0
    usage_ema: float = 0.0


class ProjectionBank(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        shared_rank: int,
        slot_r_max: int,
        *,
        point_name: str,
        orthogonality_cfg: Dict[str, Any] | None = None,
    ):
        super().__init__()
        self.point_name = str(point_name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shared_rank = shared_rank
        self.slot_r_max = slot_r_max
        self.orthogonality_cfg = dict(orthogonality_cfg or {})
        shared_init_warning = None
        if self.uses_fixed_shared_a():
            shared_init = row_orthonormal_matrix(
                shared_rank,
                input_dim,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            shared_a = nn.Parameter(shared_init.rows, requires_grad=False)
            shared_init_warning = shared_init.orth_warning
        else:
            shared_a = nn.Parameter(torch.empty(shared_rank, input_dim))
            nn.init.kaiming_uniform_(shared_a, a=5**0.5)
        self.shared_a = shared_a
        self.shared_b = nn.Parameter(torch.zeros(output_dim, shared_rank))
        self.shared_init_warning = shared_init_warning
        self.slot_a = nn.ParameterList()
        self.slot_b = nn.ParameterList()

    def current_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        return self.shared_a.device, self.shared_a.dtype

    def orthogonality_state(self) -> Dict[str, Any]:
        enabled = bool(self.orthogonality_cfg.get("enabled", False))
        if not enabled:
            return {
                "point_name": self.point_name,
                "enabled": False,
                "shared_enabled": False,
                "shared_factor": "legacy",
                "shared_mode": "legacy",
                "shared_fixed": False,
                "slots_enabled": False,
                "against_live_slots": False,
                "against_shared": False,
                "on_open_new_slot": False,
                "on_expand_rank": False,
                "basis_source": "legacy",
            }
        return {
            "point_name": self.point_name,
            "enabled": enabled,
            "shared_enabled": bool(self.orthogonality_cfg.get("shared_enabled", False)),
            "shared_factor": str(self.orthogonality_cfg.get("shared_factor", "legacy")),
            "shared_mode": str(self.orthogonality_cfg.get("shared_mode", "legacy")),
            "shared_fixed": bool(self.uses_fixed_shared_a()),
            "slots_enabled": bool(self.orthogonality_cfg.get("slots_enabled", False)),
            "against_live_slots": bool(self.orthogonality_cfg.get("against_live_slots", False)),
            "against_shared": bool(self.orthogonality_cfg.get("against_shared", False)),
            "on_open_new_slot": bool(self.orthogonality_cfg.get("on_open_new_slot", False)),
            "on_expand_rank": bool(self.orthogonality_cfg.get("on_expand_rank", False)),
            "basis_source": str(self.orthogonality_cfg.get("basis_source", "active_rank_only")),
        }

    def uses_fixed_shared_a(self) -> bool:
        return (
            bool(self.orthogonality_cfg.get("enabled", False))
            and bool(self.orthogonality_cfg.get("shared_enabled", False))
            and str(self.orthogonality_cfg.get("shared_factor", "")).strip().lower() == "a"
            and str(self.orthogonality_cfg.get("shared_mode", "")).strip().lower() == "one_side_fixed"
        )

    def slots_use_structural_orthogonality(self) -> bool:
        return bool(self.orthogonality_cfg.get("enabled", False)) and bool(
            self.orthogonality_cfg.get("slots_enabled", False)
        )

    def shared_a_gram_error(self) -> float:
        return float(gram_error(self.shared_a.detach()).item())

    def add_slot(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> int:
        current_device, current_dtype = self.current_device_dtype()
        slot_a = nn.Parameter(
            torch.empty(
                self.slot_r_max,
                self.input_dim,
                device=device or current_device,
                dtype=dtype or current_dtype,
            )
        )
        slot_b = nn.Parameter(
            torch.empty(
                self.output_dim,
                self.slot_r_max,
                device=device or current_device,
                dtype=dtype or current_dtype,
            )
        )
        nn.init.kaiming_uniform_(slot_a, a=5**0.5)
        nn.init.zeros_(slot_b)
        self.slot_a.append(slot_a)
        self.slot_b.append(slot_b)
        return len(self.slot_a) - 1

    def clear_slots(self) -> None:
        self.slot_a = nn.ParameterList()
        self.slot_b = nn.ParameterList()

    def shared_delta(self, hidden_states: torch.Tensor, shared_gate: float | torch.Tensor) -> torch.Tensor:
        delta = F.linear(F.linear(hidden_states, self.shared_a), self.shared_b)
        if isinstance(shared_gate, torch.Tensor):
            gate = shared_gate.to(device=delta.device, dtype=delta.dtype)
            while gate.dim() < delta.dim():
                gate = gate.unsqueeze(-1)
            return delta * gate
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

    def merge_slot_into_shared(self, slot_id: int, merge_rate: float, rank: int | None = None) -> Dict[str, Any]:
        with torch.no_grad():
            slot_rank = min(rank or self.slot_r_max, self.slot_a[slot_id].size(0))
            merged_update = self.shared_update_matrix() + merge_rate * self.slot_update_matrix(slot_id, slot_rank)
            if self.uses_fixed_shared_a():
                shared_b, merge_result = fixed_basis_merge_update(
                    self.shared_a.detach(),
                    merged_update,
                    gram_fast_path_tolerance=GRAM_FAST_PATH_TOLERANCE,
                )
                self.shared_b.copy_(shared_b.to(device=self.shared_b.device, dtype=self.shared_b.dtype))
                return {
                    "point_name": self.point_name,
                    "solver": merge_result.solver,
                    "gram_error": merge_result.gram_error,
                    "reconstruction_error": merge_result.reconstruction_error,
                    "shared_a_preserved": merge_result.shared_a_preserved,
                }

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
            return {
                "point_name": self.point_name,
                "solver": "svd_overwrite",
                "gram_error": float(gram_error(self.shared_a.detach()).item()),
                "reconstruction_error": 0.0,
                "shared_a_preserved": False,
            }


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
        task_embedding_dim: int | None = None,
        orthogonality_cfg: Dict[str, Any] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.selected_points = list(selected_points)
        self.orthogonality_cfg = deepcopy(orthogonality_cfg or {"enabled": False, "points": {}})
        self.shared_rank = shared_rank
        self.slot_r_max = slot_r_max
        self.slot_init_rank = slot_init_rank
        self.bootstrap_slot_rank = bootstrap_slot_rank
        self.max_slots = max_slots
        self.router_topk = router_topk
        self.router_temperature = router_temperature
        self.router_dim = task_embedding_dim or embed_dim
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
            point_orthogonality_cfg = dict(self.orthogonality_cfg.get("points", {}).get(point_name, {}))
            self.point_banks[sanitize_key(point_name)] = ProjectionBank(
                input_dim=input_dims[point_name],
                output_dim=output_dims[point_name],
                shared_rank=shared_rank,
                slot_r_max=slot_r_max,
                point_name=point_name,
                orthogonality_cfg=point_orthogonality_cfg,
            )

        self.slot_keys = nn.ParameterList()
        self.slot_metadata: List[SlotMetadata] = []
        self.slot_snapshots: List[Dict[str, torch.Tensor]] = []
        self.bootstrap_initialized = False
        self.last_shared_gate = 1.0
        self.last_structural_action = "reuse_shared"
        self.last_consolidate_flag = False

    def _current_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        return self.query_proj.weight.device, self.query_proj.weight.dtype

    def export_structure_state(self) -> Dict[str, object]:
        return {
            "slot_metadata": [
                {
                    "rank": meta.rank,
                    "frozen": meta.frozen,
                    "pruned": meta.pruned,
                    "opened_at_task": meta.opened_at_task,
                    "last_usage": meta.last_usage,
                    "retained_for_inference": meta.retained_for_inference,
                    "cumulative_usage": meta.cumulative_usage,
                    "usage_ema": meta.usage_ema,
                }
                for meta in self.slot_metadata
            ],
            "bootstrap_initialized": self.bootstrap_initialized,
            "last_shared_gate": self.last_shared_gate,
            "last_structural_action": self.last_structural_action,
            "last_consolidate_flag": self.last_consolidate_flag,
            "orthogonality_state": self._resolved_orthogonality_state(),
        }

    def load_structure_state(self, state: Dict[str, object]) -> None:
        layer_device, layer_dtype = self._current_device_dtype()
        stored_orthogonality_state = state.get("orthogonality_state")
        current_orthogonality_state = self._resolved_orthogonality_state()
        normalized_stored_state = (
            self._legacy_orthogonality_state()
            if stored_orthogonality_state is None
            else stored_orthogonality_state
        )
        if normalized_stored_state != current_orthogonality_state:
            raise ValueError(
                "Orthogonality checkpoint/config mismatch for NH-LoRA layer. "
                f"checkpoint_state={normalized_stored_state} current_config={current_orthogonality_state}"
            )
        self.slot_keys = nn.ParameterList()
        self.slot_metadata = []
        self.slot_snapshots = []
        for bank in self.point_banks.values():
            bank.clear_slots()
        for payload in state.get("slot_metadata", []):
            for bank in self.point_banks.values():
                bank.add_slot(device=layer_device, dtype=layer_dtype)
            self.slot_keys.append(
                nn.Parameter(torch.zeros(self.router_dim, device=layer_device, dtype=layer_dtype))
            )
            self.slot_metadata.append(
                SlotMetadata(
                    rank=int(payload["rank"]),
                    frozen=bool(payload["frozen"]),
                    pruned=bool(payload["pruned"]),
                    opened_at_task=int(payload["opened_at_task"]),
                    last_usage=float(payload["last_usage"]),
                    retained_for_inference=bool(payload["retained_for_inference"]),
                    cumulative_usage=float(payload["cumulative_usage"]),
                    usage_ema=float(payload["usage_ema"]),
                )
            )
            self.slot_snapshots.append({})
        for slot_id, meta in enumerate(self.slot_metadata):
            if meta.frozen or meta.pruned:
                self.freeze_slot(slot_id)
        self.bootstrap_initialized = bool(state.get("bootstrap_initialized", False))
        self.last_shared_gate = float(state.get("last_shared_gate", 1.0))
        self.last_structural_action = str(state.get("last_structural_action", "reuse_shared"))
        self.last_consolidate_flag = bool(state.get("last_consolidate_flag", False))

    def _legacy_orthogonality_state(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "active_points": [],
            "points": {
                point_name: {
                    "point_name": point_name,
                    "enabled": False,
                    "shared_enabled": False,
                    "shared_factor": "legacy",
                    "shared_mode": "legacy",
                    "shared_fixed": False,
                    "slots_enabled": False,
                    "against_live_slots": False,
                    "against_shared": False,
                    "on_open_new_slot": False,
                    "on_expand_rank": False,
                    "basis_source": "legacy",
                }
                for point_name in self.selected_points
            },
        }

    def _resolved_orthogonality_state(self) -> Dict[str, Any]:
        point_states = {
            point_name: self.point_banks[sanitize_key(point_name)].orthogonality_state()
            for point_name in self.selected_points
        }
        active_points = [
            point_name
            for point_name, payload in point_states.items()
            if bool(payload.get("enabled", False))
        ]
        return {
            "enabled": bool(active_points),
            "active_points": active_points,
            "points": point_states,
        }

    def live_slot_ids(self) -> List[int]:
        return [slot_id for slot_id, meta in enumerate(self.slot_metadata) if not meta.pruned]

    def orthogonality_active_points(self) -> List[str]:
        return [
            point_name
            for point_name in self.selected_points
            if bool(self.point_banks[sanitize_key(point_name)].orthogonality_state().get("enabled", False))
        ]

    def _point_bank(self, point_name: str) -> ProjectionBank:
        return self.point_banks[sanitize_key(point_name)]

    def _point_orthogonality_cfg(self, point_name: str) -> Dict[str, Any]:
        return dict(self.orthogonality_cfg.get("points", {}).get(point_name, {}))

    def _point_orthogonality_enabled(self, point_name: str) -> bool:
        return bool(self._point_bank(point_name).orthogonality_state().get("enabled", False))

    def _slot_structural_orth_enabled(self, point_name: str, *, on_open: bool) -> bool:
        if not self._point_orthogonality_enabled(point_name):
            return False
        point_cfg = self._point_orthogonality_cfg(point_name)
        if not bool(point_cfg.get("slots_enabled", False)):
            return False
        if on_open:
            return bool(point_cfg.get("on_open_new_slot", False))
        return bool(point_cfg.get("on_expand_rank", False))

    def _shared_basis_rows(self, point_name: str) -> torch.Tensor:
        bank = self._point_bank(point_name)
        if not bool(self._point_orthogonality_cfg(point_name).get("against_shared", False)):
            return bank.shared_a.new_zeros((0, bank.input_dim))
        return build_row_basis(
            [bank.shared_a.detach()],
            input_dim=bank.input_dim,
            device=bank.shared_a.device,
            dtype=bank.shared_a.dtype,
        )

    def _live_slot_basis_rows(self, point_name: str, *, exclude_slot_id: int | None = None) -> torch.Tensor:
        bank = self._point_bank(point_name)
        point_cfg = self._point_orthogonality_cfg(point_name)
        if not bool(point_cfg.get("against_live_slots", False)):
            return bank.shared_a.new_zeros((0, bank.input_dim))
        rows = []
        for slot_id in self.live_slot_ids():
            if exclude_slot_id is not None and int(slot_id) == int(exclude_slot_id):
                continue
            rows.append(self.active_slot_a(slot_id, point_name).detach())
        return build_row_basis(rows, input_dim=bank.input_dim, device=bank.shared_a.device, dtype=bank.shared_a.dtype)

    def _available_orthogonal_rank(
        self,
        point_name: str,
        *,
        include_slot_rows: torch.Tensor | None = None,
        exclude_slot_id: int | None = None,
    ) -> Dict[str, Any]:
        bank = self._point_bank(point_name)
        row_groups = [self._live_slot_basis_rows(point_name, exclude_slot_id=exclude_slot_id)]
        if include_slot_rows is not None and include_slot_rows.numel() > 0:
            row_groups.append(include_slot_rows.detach())
        if bool(self._point_orthogonality_cfg(point_name).get("against_shared", False)):
            row_groups.append(self._shared_basis_rows(point_name))
        basis_rows = build_row_basis(
            row_groups,
            input_dim=bank.input_dim,
            device=bank.shared_a.device,
            dtype=bank.shared_a.dtype,
        )
        return {
            "basis_rows": basis_rows,
            "basis_rank": int(basis_rows.size(0)),
            "available_rank": max(int(bank.input_dim - basis_rows.size(0)), 0),
        }

    def _requested_actual_added_rank(
        self,
        *,
        requested_added_rank: int,
        include_self_rows_by_point: Dict[str, torch.Tensor] | None = None,
        exclude_slot_id: int | None = None,
        on_open: bool,
    ) -> Dict[str, Any]:
        requested = max(min(int(requested_added_rank), self.slot_r_max), 0)
        if requested <= 0:
            return {
                "requested_added_rank": 0,
                "actual_added_rank": 0,
                "orth_exhausted": False,
                "orth_warning": None,
                "point_capacity": {},
                "targeted_points": [],
            }
        targeted_points = [
            point_name
            for point_name in self.selected_points
            if self._slot_structural_orth_enabled(point_name, on_open=on_open)
        ]
        point_capacity: Dict[str, Dict[str, Any]] = {}
        if not targeted_points:
            return {
                "requested_added_rank": requested,
                "actual_added_rank": requested,
                "orth_exhausted": False,
                "orth_warning": None,
                "point_capacity": point_capacity,
                "targeted_points": targeted_points,
            }
        min_available = requested
        bottlenecks = []
        for point_name in targeted_points:
            slot_rows = None if include_self_rows_by_point is None else include_self_rows_by_point.get(point_name)
            capacity = self._available_orthogonal_rank(
                point_name,
                include_slot_rows=slot_rows,
                exclude_slot_id=exclude_slot_id,
            )
            point_capacity[point_name] = capacity
            available_rank = int(capacity["available_rank"])
            if available_rank < min_available:
                min_available = available_rank
            if available_rank < requested:
                bottlenecks.append(f"{point_name}:{available_rank}")
        actual = max(min_available, 0)
        orth_exhausted = actual < requested
        orth_warning = None
        if orth_exhausted:
            orth_warning = (
                f"Orthogonal complement exhausted for requested_added_rank={requested}; "
                f"actual_added_rank={actual}; bottlenecks={bottlenecks}."
            )
        return {
            "requested_added_rank": requested,
            "actual_added_rank": actual,
            "orth_exhausted": orth_exhausted,
            "orth_warning": orth_warning,
            "point_capacity": point_capacity,
            "targeted_points": targeted_points,
        }

    def _initialize_slot_point_rows(
        self,
        *,
        slot_id: int,
        point_name: str,
        start_rank: int,
        added_rank: int,
        point_capacity: Dict[str, Any],
    ) -> Dict[str, Any]:
        bank = self._point_bank(point_name)
        slot_rows = bank.slot_a[slot_id]
        if added_rank > 0:
            init_result = orthogonal_rows_against_basis(
                requested_rows=added_rank,
                input_dim=bank.input_dim,
                basis_rows=point_capacity["basis_rows"],
                device=slot_rows.device,
                dtype=slot_rows.dtype,
            )
            slot_rows.data[start_rank : start_rank + added_rank].copy_(init_result.rows[:added_rank])
        else:
            init_result = orthogonal_rows_against_basis(
                requested_rows=0,
                input_dim=bank.input_dim,
                basis_rows=point_capacity["basis_rows"],
                device=slot_rows.device,
                dtype=slot_rows.dtype,
            )
        slot_slot_stats = pairwise_overlap_stats(
            slot_rows[start_rank : start_rank + added_rank].detach(),
            self._live_slot_basis_rows(point_name, exclude_slot_id=slot_id),
        )
        slot_shared_stats = pairwise_overlap_stats(
            slot_rows[start_rank : start_rank + added_rank].detach(),
            self._shared_basis_rows(point_name),
        )
        return {
            "point_name": point_name,
            "requested_added_rank": int(init_result.requested_added_rank),
            "actual_added_rank": int(added_rank),
            "orth_exhausted": bool(init_result.orth_exhausted or added_rank < init_result.requested_added_rank),
            "orth_warning": init_result.orth_warning,
            "basis_rank": int(point_capacity["basis_rank"]),
            "available_rank": int(point_capacity["available_rank"]),
            "slot_slot_max_overlap": float(slot_slot_stats["max_abs_cos"]),
            "slot_shared_max_overlap": float(slot_shared_stats["max_abs_cos"]),
        }

    def _rollback_uncommitted_slot(self, slot_id: int) -> None:
        for bank in self.point_banks.values():
            bank.slot_a = nn.ParameterList(list(bank.slot_a)[:-1])
            bank.slot_b = nn.ParameterList(list(bank.slot_b)[:-1])
        self.slot_keys = nn.ParameterList(list(self.slot_keys)[:-1])

    def _add_slot_with_diagnostics(self, initial_rank: int, task_id: int) -> tuple[int | None, Dict[str, Any]]:
        if len(self.slot_metadata) >= self.max_slots:
            raise RuntimeError("Maximum number of slots reached for this layer.")
        layer_device, layer_dtype = self._current_device_dtype()
        for bank in self.point_banks.values():
            bank.add_slot(device=layer_device, dtype=layer_dtype)
        key = nn.Parameter(torch.randn(self.router_dim, device=layer_device, dtype=layer_dtype) * 0.02)
        self.slot_keys.append(key)
        slot_id = len(self.slot_keys) - 1
        rank_info = self._requested_actual_added_rank(requested_added_rank=initial_rank, on_open=True)
        actual_rank = int(rank_info["actual_added_rank"])
        point_events: Dict[str, Dict[str, Any]] = {}
        if actual_rank <= 0 and rank_info["targeted_points"]:
            for point_name in rank_info["targeted_points"]:
                capacity = rank_info["point_capacity"][point_name]
                point_events[point_name] = {
                    "point_name": point_name,
                    "requested_added_rank": int(rank_info["requested_added_rank"]),
                    "actual_added_rank": 0,
                    "orth_exhausted": True,
                    "orth_warning": rank_info["orth_warning"],
                    "basis_rank": int(capacity["basis_rank"]),
                    "available_rank": int(capacity["available_rank"]),
                    "slot_slot_max_overlap": 0.0,
                    "slot_shared_max_overlap": 0.0,
                }
            self._rollback_uncommitted_slot(slot_id)
            return None, {
                "requested_added_rank": int(rank_info["requested_added_rank"]),
                "actual_added_rank": 0,
                "orth_exhausted": True,
                "orth_warning": rank_info["orth_warning"],
                "point_events": point_events,
            }
        for point_name in self.selected_points:
            if self._slot_structural_orth_enabled(point_name, on_open=True):
                point_events[point_name] = self._initialize_slot_point_rows(
                    slot_id=slot_id,
                    point_name=point_name,
                    start_rank=0,
                    added_rank=actual_rank,
                    point_capacity=rank_info["point_capacity"][point_name],
                )
        self.slot_metadata.append(
            SlotMetadata(
                rank=actual_rank if rank_info["targeted_points"] else min(initial_rank, self.slot_r_max),
                opened_at_task=task_id,
                retained_for_inference=True,
            )
        )
        self.slot_snapshots.append({})
        return len(self.slot_metadata) - 1, {
            "requested_added_rank": int(rank_info["requested_added_rank"]),
            "actual_added_rank": int(self.slot_metadata[-1].rank),
            "orth_exhausted": bool(rank_info["orth_exhausted"]),
            "orth_warning": rank_info["orth_warning"],
            "point_events": point_events,
        }

    def add_slot(self, initial_rank: int, task_id: int) -> int:
        slot_id, diagnostics = self._add_slot_with_diagnostics(initial_rank, task_id)
        if slot_id is None:
            raise RuntimeError(
                diagnostics.get("orth_warning") or "Unable to allocate an orthogonal slot for the requested rank."
            )
        return slot_id

    def initialize_slot_key(self, slot_id: int, task_embedding: torch.Tensor) -> None:
        with torch.no_grad():
            normalized = F.normalize(
                task_embedding.squeeze(0).to(
                    device=self.slot_keys[slot_id].device,
                    dtype=self.slot_keys[slot_id].dtype,
                ),
                dim=-1,
            )
            self.slot_keys[slot_id].copy_(normalized)

    def ensure_bootstrap_slot(self, task_id: int, task_embedding: torch.Tensor | None = None) -> int:
        if self.bootstrap_initialized:
            return 0
        slot_id = self.add_slot(self.bootstrap_slot_rank, task_id=task_id)
        if task_embedding is not None:
            self.initialize_slot_key(slot_id, task_embedding)
        self.bootstrap_initialized = True
        return slot_id

    def expand_rank(self, slot_id: int, new_rank: int) -> Dict[str, Any]:
        old_rank = int(self.slot_metadata[slot_id].rank)
        clamped_new_rank = min(max(int(new_rank), 1), self.slot_r_max)
        requested_added_rank = max(clamped_new_rank - old_rank, 0)
        if requested_added_rank <= 0:
            self.slot_metadata[slot_id].rank = clamped_new_rank
            return {
                "old_rank": old_rank,
                "requested_added_rank": 0,
                "actual_added_rank": 0,
                "orth_exhausted": False,
                "orth_warning": None,
                "point_events": {},
            }

        include_self_rows = {
            point_name: self.active_slot_a(slot_id, point_name).detach()
            for point_name in self.selected_points
            if self._slot_structural_orth_enabled(point_name, on_open=False)
        }
        rank_info = self._requested_actual_added_rank(
            requested_added_rank=requested_added_rank,
            include_self_rows_by_point=include_self_rows,
            exclude_slot_id=slot_id,
            on_open=False,
        )
        actual_added_rank = int(rank_info["actual_added_rank"])
        point_events: Dict[str, Dict[str, Any]] = {}
        for point_name in self.selected_points:
            bank = self._point_bank(point_name)
            if self._slot_structural_orth_enabled(point_name, on_open=False):
                point_events[point_name] = self._initialize_slot_point_rows(
                    slot_id=slot_id,
                    point_name=point_name,
                    start_rank=old_rank,
                    added_rank=actual_added_rank,
                    point_capacity=rank_info["point_capacity"][point_name],
                )
                if actual_added_rank < requested_added_rank:
                    bank.slot_a[slot_id].data[old_rank:clamped_new_rank].zero_()
            bank.slot_b[slot_id].data[:, old_rank:clamped_new_rank].zero_()
        self.slot_metadata[slot_id].rank = old_rank + actual_added_rank
        return {
            "old_rank": old_rank,
            "requested_added_rank": int(rank_info["requested_added_rank"]),
            "actual_added_rank": actual_added_rank,
            "orth_exhausted": bool(rank_info["orth_exhausted"]),
            "orth_warning": rank_info["orth_warning"],
            "point_events": point_events,
        }

    def freeze_slot(self, slot_id: int) -> None:
        self.slot_metadata[slot_id].frozen = True
        for bank in self.point_banks.values():
            bank.slot_a[slot_id].requires_grad = False
            bank.slot_b[slot_id].requires_grad = False
        self.slot_keys[slot_id].requires_grad = False

    def unfreeze_slot(self, slot_id: int) -> None:
        if self.slot_metadata[slot_id].pruned:
            return
        self.slot_metadata[slot_id].frozen = False
        for bank in self.point_banks.values():
            bank.slot_a[slot_id].requires_grad = True
            bank.slot_b[slot_id].requires_grad = True
        self.slot_keys[slot_id].requires_grad = True

    def prune_slot(self, slot_id: int) -> None:
        self.slot_metadata[slot_id].pruned = True
        self.slot_metadata[slot_id].retained_for_inference = False
        self.freeze_slot(slot_id)

    def apply_structure_change(
        self,
        plan: MaterializedLayerPlan,
        task_embedding: torch.Tensor,
        task_id: int,
    ) -> Dict[str, object]:
        selected_slot = plan.selected_slot
        created_new_slot = False
        effective_action = plan.action
        orth_requested_added_rank = 0
        orth_actual_added_rank = 0
        orth_exhausted = False
        orth_warning = None
        orth_point_events: Dict[str, Dict[str, Any]] = {}
        previous_rank = None

        if plan.create_new_slot:
            selected_slot, slot_open_summary = self._add_slot_with_diagnostics(
                plan.new_slot_rank or self.slot_init_rank,
                task_id=task_id,
            )
            orth_requested_added_rank = int(slot_open_summary["requested_added_rank"])
            orth_actual_added_rank = int(slot_open_summary["actual_added_rank"])
            orth_exhausted = bool(slot_open_summary["orth_exhausted"])
            orth_warning = slot_open_summary.get("orth_warning")
            orth_point_events = dict(slot_open_summary.get("point_events", {}))
            if selected_slot is not None:
                self.initialize_slot_key(selected_slot, task_embedding)
                self.slot_metadata[selected_slot].retained_for_inference = True
                created_new_slot = orth_actual_added_rank > 0

        if effective_action == "expand_rank_existing_slot" and selected_slot is not None:
            self.unfreeze_slot(selected_slot)
            if plan.target_rank is not None:
                previous_rank = int(self.slot_metadata[selected_slot].rank)
                expand_summary = self.expand_rank(selected_slot, plan.target_rank)
                orth_requested_added_rank = int(expand_summary["requested_added_rank"])
                orth_actual_added_rank = int(expand_summary["actual_added_rank"])
                orth_exhausted = bool(expand_summary["orth_exhausted"])
                orth_warning = expand_summary.get("orth_warning")
                orth_point_events = dict(expand_summary.get("point_events", {}))
            self.slot_metadata[selected_slot].retained_for_inference = True

        if effective_action == "freeze_old_strong_retention":
            for slot_id in self.live_slot_ids():
                self.freeze_slot(slot_id)
                self.slot_metadata[slot_id].retained_for_inference = True

        self.last_shared_gate = plan.shared_gate
        self.last_structural_action = effective_action
        self.last_consolidate_flag = plan.consolidate_flag

        candidate_slots: List[int] = []
        if not plan.shared_only:
            for slot_id in plan.candidate_slots:
                if slot_id in self.live_slot_ids() and slot_id not in candidate_slots:
                    candidate_slots.append(slot_id)
            if selected_slot is not None and selected_slot in self.live_slot_ids() and selected_slot not in candidate_slots:
                candidate_slots.append(selected_slot)

        rank_cfg = {slot_id: self.slot_metadata[slot_id].rank for slot_id in self.live_slot_ids()}
        runtime_shared_only = bool(plan.shared_only or (not candidate_slots and selected_slot is None))
        return {
            "action": effective_action,
            "requested_action": plan.requested_action,
            "active_slot_candidates": candidate_slots,
            "selected_slot": selected_slot,
            "rank_cfg": rank_cfg,
            "shared_gate": plan.shared_gate,
            "consolidate_flag": plan.consolidate_flag,
            "deterministic": len(candidate_slots) <= 1,
            "created_new_slot": created_new_slot,
            "fallback_action": plan.fallback_action,
            "strong_retention": plan.strong_retention,
            "shared_only": runtime_shared_only,
            "compatibility_scores": plan.compatibility_scores,
            "requested_added_rank": orth_requested_added_rank,
            "actual_added_rank": orth_actual_added_rank,
            "orth_exhausted": orth_exhausted,
            "orth_warning": orth_warning,
            "orth_point_events": orth_point_events,
            "old_rank": previous_rank,
            "new_rank": rank_cfg.get(selected_slot) if selected_slot is not None else None,
        }

    def route(self, normalized_tokens: torch.Tensor, planner_cfg: Dict[str, object], task_state) -> Dict[str, object]:
        del task_state
        pooled = normalized_tokens[:, 0]
        live_slots = self.live_slot_ids()

        if bool(planner_cfg.get("shared_only", False)):
            return {
                "candidate_slots": [],
                "selected_slots": [],
                "routing_weights": normalized_tokens.new_zeros(normalized_tokens.size(0), 0),
                "routing_distribution": normalized_tokens.new_zeros(normalized_tokens.size(0), 0),
            }

        requested = planner_cfg.get("active_slot_candidates")
        if requested is None:
            candidate_slots = live_slots
        else:
            candidate_slots = [slot_id for slot_id in requested if slot_id in live_slots]

        if not candidate_slots:
            return {
                "candidate_slots": [],
                "selected_slots": [],
                "routing_weights": normalized_tokens.new_zeros(normalized_tokens.size(0), 0),
                "routing_distribution": normalized_tokens.new_zeros(normalized_tokens.size(0), 0),
            }

        if len(candidate_slots) == 1 and bool(planner_cfg.get("deterministic", False)):
            weights = normalized_tokens.new_ones(normalized_tokens.size(0), 1)
            distribution = normalized_tokens.new_ones(normalized_tokens.size(0), 1)
            return {
                "candidate_slots": candidate_slots,
                "selected_slots": list(candidate_slots),
                "routing_weights": weights,
                "routing_distribution": distribution,
                "usage_vector": weights.mean(dim=0),
            }

        query = F.normalize(self.query_proj(pooled), dim=-1)
        keys = torch.stack([self.slot_keys[slot_id] for slot_id in candidate_slots], dim=0)
        keys = F.normalize(keys, dim=-1)
        scores = query @ keys.t()
        topk = min(self.router_topk, scores.size(1))
        topk_scores, topk_indices = torch.topk(scores, k=topk, dim=1)
        weights = torch.softmax(topk_scores / self.router_temperature, dim=-1)
        distribution = normalized_tokens.new_zeros(scores.size(0), len(candidate_slots))
        distribution.scatter_(1, topk_indices, weights)
        usage_vector = distribution.mean(dim=0)
        selected_slots = [candidate_slots[index] for index in topk_indices[0].tolist()]
        return {
            "candidate_slots": candidate_slots,
            "selected_slots": selected_slots,
            "routing_weights": weights,
            "routing_distribution": distribution,
            "topk_indices": topk_indices,
            "usage_vector": usage_vector,
        }

    def _point_output_dim(self, point_name: str) -> int:
        return self.embed_dim * 4 if point_name == "mlp_fc1" else self.embed_dim

    def _record_delta_debug(
        self,
        kind: str,
        point_name: str,
        delta: torch.Tensor,
        planner_cfg: Dict[str, object],
    ) -> None:
        accumulator = planner_cfg.get("_debug_delta_stats")
        if not isinstance(accumulator, dict):
            return
        block_id = int(planner_cfg.get("_debug_block_id", -1))
        block_entry = accumulator.setdefault(block_id, {})
        point_entry = block_entry.setdefault(point_name, {})
        stats = point_entry.setdefault(
            kind,
            {
                "calls": 0,
                "norm_sum": 0.0,
                "mean_abs_sum": 0.0,
                "max_abs": 0.0,
            },
        )
        detached = delta.detach()
        stats["calls"] += 1
        stats["norm_sum"] += float(detached.norm().item())
        stats["mean_abs_sum"] += float(detached.abs().mean().item()) if detached.numel() else 0.0
        stats["max_abs"] = max(float(stats["max_abs"]), float(detached.abs().max().item()) if detached.numel() else 0.0)

    def _record_control_contribution_debug(
        self,
        planner_cfg: Dict[str, object],
        *,
        shared_pre_beta_norm: float | None = None,
        shared_post_beta_norm: float | None = None,
        slot_norm: float | None = None,
        slot_structurally_available: bool | None = None,
        slot_nontrivial: bool | None = None,
        beta_value: float | None = None,
    ) -> None:
        accumulator = planner_cfg.get("_planner_control_contribution_accumulator")
        if not isinstance(accumulator, dict):
            return
        block_id = int(planner_cfg.get("_debug_block_id", -1))
        block_entry = accumulator.setdefault(
            block_id,
            {
                "shared_pre_beta_norm_sum": 0.0,
                "shared_pre_beta_norm_count": 0,
                "shared_post_beta_norm_sum": 0.0,
                "shared_post_beta_norm_count": 0,
                "slot_norm_sum": 0.0,
                "slot_norm_count": 0,
                "slot_structurally_available_calls": 0,
                "slot_nontrivial_calls": 0,
                "slot_zero_contribution_calls": 0,
                "beta_gt_099_with_structural_slot_calls": 0,
                "beta_gt_0999_with_structural_slot_calls": 0,
            },
        )
        if shared_pre_beta_norm is not None:
            block_entry["shared_pre_beta_norm_sum"] += float(shared_pre_beta_norm)
            block_entry["shared_pre_beta_norm_count"] += 1
        if shared_post_beta_norm is not None:
            block_entry["shared_post_beta_norm_sum"] += float(shared_post_beta_norm)
            block_entry["shared_post_beta_norm_count"] += 1
        if slot_norm is not None:
            block_entry["slot_norm_sum"] += float(slot_norm)
            block_entry["slot_norm_count"] += 1
        if slot_structurally_available:
            block_entry["slot_structurally_available_calls"] += 1
            if slot_nontrivial:
                block_entry["slot_nontrivial_calls"] += 1
            else:
                block_entry["slot_zero_contribution_calls"] += 1
            if beta_value is not None and float(beta_value) > 0.99:
                block_entry["beta_gt_099_with_structural_slot_calls"] += 1
            if beta_value is not None and float(beta_value) > 0.999:
                block_entry["beta_gt_0999_with_structural_slot_calls"] += 1

    def _shared_delta(self, point_name: str, hidden_states: torch.Tensor, planner_cfg: Dict[str, object]) -> torch.Tensor:
        if point_name not in self.selected_points:
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        bank = self.point_banks[sanitize_key(point_name)]
        shared_gate = planner_cfg.get("shared_gate", 1.0)
        delta = bank.shared_delta(hidden_states, shared_gate=shared_gate)
        self._record_delta_debug("shared", point_name, delta, planner_cfg)
        beta_value = float(shared_gate.detach().mean().item()) if isinstance(shared_gate, torch.Tensor) else float(shared_gate)
        with torch.no_grad():
            detached_hidden = hidden_states.detach()
            pre_beta_delta = F.linear(F.linear(detached_hidden, bank.shared_a.detach()), bank.shared_b.detach())
        self._record_control_contribution_debug(
            planner_cfg,
            shared_pre_beta_norm=float(pre_beta_delta.norm().item()),
            shared_post_beta_norm=float(delta.detach().norm().item()),
            beta_value=beta_value,
        )
        return delta

    def _slot_delta(self, point_name: str, hidden_states: torch.Tensor, route_state: Dict[str, object], planner_cfg: Dict[str, object]) -> torch.Tensor:
        if point_name not in self.selected_points:
            return hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        candidate_slots = route_state.get("candidate_slots", [])
        shared_gate = planner_cfg.get("shared_gate", 1.0)
        beta_value = float(shared_gate.detach().mean().item()) if isinstance(shared_gate, torch.Tensor) else float(shared_gate)
        if not candidate_slots:
            delta = hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
            self._record_delta_debug("slot", point_name, delta, planner_cfg)
            self._record_control_contribution_debug(
                planner_cfg,
                slot_norm=0.0,
                slot_structurally_available=False,
                slot_nontrivial=False,
                beta_value=beta_value,
            )
            return delta
        bank = self.point_banks[sanitize_key(point_name)]
        rank_cfg = planner_cfg.get("rank_cfg", {})
        weights = route_state.get("routing_weights")
        topk_indices = route_state.get("topk_indices")
        delta = hidden_states.new_zeros(*hidden_states.shape[:-1], self._point_output_dim(point_name))
        if weights is None or topk_indices is None:
            slot_id = candidate_slots[0]
            rank = int(rank_cfg.get(slot_id, self.slot_metadata[slot_id].rank))
            delta = bank.slot_delta(hidden_states, slot_id, rank)
            self._record_delta_debug("slot", point_name, delta, planner_cfg)
            slot_norm = float(delta.detach().norm().item())
            self._record_control_contribution_debug(
                planner_cfg,
                slot_norm=slot_norm,
                slot_structurally_available=True,
                slot_nontrivial=slot_norm > 1e-12,
                beta_value=beta_value,
            )
            return delta
        for batch_index in range(hidden_states.size(0)):
            batch_hidden = hidden_states[batch_index : batch_index + 1]
            for col, candidate_index in enumerate(topk_indices[batch_index].tolist()):
                slot_id = candidate_slots[candidate_index]
                rank = int(rank_cfg.get(slot_id, self.slot_metadata[slot_id].rank))
                slot_delta = bank.slot_delta(batch_hidden, slot_id, rank)
                delta[batch_index : batch_index + 1] += slot_delta * weights[batch_index, col]
                self.slot_metadata[slot_id].last_usage = float(weights[batch_index, col].detach().item())
        self._record_delta_debug("slot", point_name, delta, planner_cfg)
        slot_norm = float(delta.detach().norm().item())
        self._record_control_contribution_debug(
            planner_cfg,
            slot_norm=slot_norm,
            slot_structurally_available=True,
            slot_nontrivial=slot_norm > 1e-12,
            beta_value=beta_value,
        )
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
                current_norm += float(prev_a.norm().item() + prev_b.norm().item())
            stability[slot_id] = float(torch.exp(torch.tensor(-delta_norm / max(current_norm, 1e-6))).item())
        return stability

    def update_usage_statistics(self, usage_stats: Dict[int, float], ema_decay: float = 0.5) -> None:
        for slot_id in self.live_slot_ids():
            usage = float(usage_stats.get(slot_id, 0.0))
            metadata = self.slot_metadata[slot_id]
            metadata.cumulative_usage += usage
            metadata.usage_ema = ema_decay * metadata.usage_ema + (1.0 - ema_decay) * usage
            metadata.last_usage = usage

    def active_rank_mask(self, slot_id: int, device: torch.device | None = None) -> torch.Tensor:
        mask = torch.zeros(self.slot_r_max, device=device or self.slot_keys[slot_id].device)
        mask[: self.slot_metadata[slot_id].rank] = 1.0
        return mask

    def active_slot_a(self, slot_id: int, point_name: str) -> torch.Tensor:
        bank = self.point_banks[sanitize_key(point_name)]
        rank = self.slot_metadata[slot_id].rank
        return bank.slot_a[slot_id][:rank]

    def shared_orthogonality_summaries(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        for point_name in self.orthogonality_active_points():
            bank = self._point_bank(point_name)
            summaries[point_name] = {
                "point_name": point_name,
                "fixed": bool(bank.uses_fixed_shared_a()),
                "shared_a_gram_error": float(bank.shared_a_gram_error()),
                "orth_warning": bank.shared_init_warning,
            }
        return summaries

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
