from __future__ import annotations

from dataclasses import dataclass, field
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
    planner_input: torch.Tensor | None = None
    planner_representation: torch.Tensor | None = None


@dataclass
class PlannerControlOutputs:
    shared_gate: torch.Tensor
    shared_gate_logit: torch.Tensor
    delta_raw: torch.Tensor
    delta_logit: torch.Tensor | None = None
    anchor_beta: torch.Tensor | None = None
    anchor_logit: torch.Tensor | None = None
    delta_from_representation: torch.Tensor | None = None
    delta_bias: torch.Tensor | None = None
    control_head_weight_norm: float | None = None
    control_head_bias_norm: float | None = None
    history_attention: torch.Tensor | None = None
    history_context: torch.Tensor | None = None
    planner_input: torch.Tensor | None = None
    planner_representation: torch.Tensor | None = None
    normalized_planner_representation: torch.Tensor | None = None


@dataclass
class MaterializedLayerPlan:
    requested_action: str
    action: str
    selected_slot: int | None
    target_rank: int | None
    rank_delta: int
    create_new_slot: bool
    new_slot_rank: int | None
    candidate_slots: List[int] = field(default_factory=list)
    compatibility_scores: Dict[int, float] = field(default_factory=dict)
    shared_gate: float = 1.0
    consolidate_flag: bool = False
    strong_retention: bool = False
    shared_only: bool = False
    fallback_action: str | None = None


def compute_slot_compatibility(slot_bank, task_embedding: torch.Tensor) -> Dict[int, float]:
    live_slots = slot_bank.live_slot_ids()
    if not live_slots:
        return {}
    normalized_task = F.normalize(task_embedding.squeeze(0), dim=-1)
    compatibility = {}
    for slot_id in live_slots:
        score = F.cosine_similarity(
            normalized_task.view(1, -1),
            F.normalize(slot_bank.slot_keys[slot_id].view(1, -1), dim=-1),
            dim=-1,
        )
        compatibility[slot_id] = float(score.item())
    return compatibility


def select_compatible_slot_pool(slot_bank, task_embedding: torch.Tensor, pool_size: int | None = None) -> List[int]:
    compatibility_scores = compute_slot_compatibility(slot_bank, task_embedding)
    if not compatibility_scores:
        return []
    requested_pool = pool_size if pool_size is not None else max(int(getattr(slot_bank, "router_topk", 1)), 3)
    effective_pool_size = min(max(int(requested_pool), 1), len(compatibility_scores))
    return [
        slot_id
        for slot_id, _ in sorted(
            compatibility_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:effective_pool_size]
    ]


def select_most_compatible_slot(slot_bank, task_embedding: torch.Tensor) -> int | None:
    compatibility_scores = compute_slot_compatibility(slot_bank, task_embedding)
    if not compatibility_scores:
        return None
    return max(compatibility_scores.items(), key=lambda item: item[1])[0]


def materialize_action(
    action: str,
    signals: PlannerSignals,
    slot_bank,
    task_embedding: torch.Tensor,
    task_id: int,
    max_slots_per_block: int,
    tau_consolidate: float = 0.5,
    router_candidate_pool: int | None = None,
) -> MaterializedLayerPlan:
    del task_id
    live_slots = slot_bank.live_slot_ids()
    compatibility_scores = compute_slot_compatibility(slot_bank, task_embedding)
    candidate_pool = select_compatible_slot_pool(slot_bank, task_embedding, pool_size=router_candidate_pool)
    shared_gate = float(signals.shared_gate.item())
    consolidate_flag = bool(signals.consolidate.item() >= tau_consolidate)
    rank_budget = max(1, signals.rank_budget)

    if action == "reuse_shared":
        return MaterializedLayerPlan(
            requested_action=action,
            action="reuse_shared",
            selected_slot=None,
            target_rank=None,
            rank_delta=0,
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=[],
            compatibility_scores=compatibility_scores,
            shared_gate=shared_gate,
            consolidate_flag=consolidate_flag,
            strong_retention=False,
            shared_only=True,
        )

    if action == "freeze_old_strong_retention":
        return MaterializedLayerPlan(
            requested_action=action,
            action="freeze_old_strong_retention",
            selected_slot=None,
            target_rank=None,
            rank_delta=0,
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=[],
            compatibility_scores=compatibility_scores,
            shared_gate=shared_gate,
            consolidate_flag=consolidate_flag,
            strong_retention=True,
            shared_only=True,
        )

    if action == "expand_rank_existing_slot":
        selected_slot = select_most_compatible_slot(slot_bank, task_embedding)
        if selected_slot is None:
            return MaterializedLayerPlan(
                requested_action=action,
                action="open_new_slot",
                selected_slot=None,
                target_rank=None,
                rank_delta=0,
                create_new_slot=True,
                new_slot_rank=rank_budget,
                candidate_slots=candidate_pool,
                compatibility_scores=compatibility_scores,
                shared_gate=shared_gate,
                consolidate_flag=consolidate_flag,
                strong_retention=False,
                shared_only=False,
                fallback_action="open_new_slot",
            )
        current_rank = slot_bank.slot_metadata[selected_slot].rank
        target_rank = min(current_rank + rank_budget, slot_bank.slot_r_max)
        return MaterializedLayerPlan(
            requested_action=action,
            action="expand_rank_existing_slot",
            selected_slot=selected_slot,
            target_rank=target_rank,
            rank_delta=max(target_rank - current_rank, 0),
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=candidate_pool,
            compatibility_scores=compatibility_scores,
            shared_gate=shared_gate,
            consolidate_flag=consolidate_flag,
            strong_retention=False,
            shared_only=False,
        )

    if action == "open_new_slot":
        if len(live_slots) < max_slots_per_block:
            return MaterializedLayerPlan(
                requested_action=action,
                action="open_new_slot",
                selected_slot=None,
                target_rank=None,
                rank_delta=0,
                create_new_slot=True,
                new_slot_rank=rank_budget,
                candidate_slots=candidate_pool,
                compatibility_scores=compatibility_scores,
                shared_gate=shared_gate,
                consolidate_flag=consolidate_flag,
                strong_retention=False,
                shared_only=False,
            )
        selected_slot = select_most_compatible_slot(slot_bank, task_embedding)
        if selected_slot is None:
            raise RuntimeError("Planner requested slot growth but no compatible slot could be selected.")
        current_rank = slot_bank.slot_metadata[selected_slot].rank
        target_rank = min(current_rank + rank_budget, slot_bank.slot_r_max)
        return MaterializedLayerPlan(
            requested_action=action,
            action="expand_rank_existing_slot",
            selected_slot=selected_slot,
            target_rank=target_rank,
            rank_delta=max(target_rank - current_rank, 0),
            create_new_slot=False,
            new_slot_rank=None,
            candidate_slots=candidate_pool,
            compatibility_scores=compatibility_scores,
            shared_gate=shared_gate,
            consolidate_flag=consolidate_flag,
            strong_retention=False,
            shared_only=False,
            fallback_action="expand_rank_existing_slot",
        )

    raise ValueError(f"Unsupported planner action: {action}")


class _PlannerBranchBase(nn.Module):
    def __init__(
        self,
        selected_blocks: List[int],
        task_embedding_dim: int,
        history_dim: int,
        hidden_dim: int,
        layer_embedding_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.selected_blocks = list(selected_blocks)
        self.layer_embeddings = nn.Embedding(max(selected_blocks) + 1, layer_embedding_dim)
        self.history_query = nn.Linear(task_embedding_dim, task_embedding_dim)
        self.history_key = nn.Linear(history_dim, task_embedding_dim)
        self.history_value = nn.Linear(history_dim, task_embedding_dim)
        input_dim = task_embedding_dim * 4 + layer_embedding_dim
        self.trunks = nn.ModuleDict()
        self.output_heads = nn.ModuleDict()
        for block_id in self.selected_blocks:
            self.trunks[str(block_id)] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
            )
            self.output_heads[str(block_id)] = nn.Linear(hidden_dim, output_dim)

    def aggregate_history(self, task_embedding: torch.Tensor, history_summary: torch.Tensor | None):
        if history_summary is None or history_summary.numel() == 0:
            zeros = task_embedding.new_zeros(task_embedding.size(0), task_embedding.size(-1))
            return zeros, None
        if history_summary.dim() == 1:
            history_summary = history_summary.unsqueeze(0)
        query = self.history_query(task_embedding)
        keys = self.history_key(history_summary)
        values = self.history_value(history_summary)
        attention_logits = query @ keys.transpose(0, 1) / (query.size(-1) ** 0.5)
        attention = F.softmax(attention_logits, dim=-1)
        history_context = attention @ values
        return history_context, attention

    def compose_layer_input(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_context: torch.Tensor,
    ) -> torch.Tensor:
        layer_embedding = self.layer_embeddings(
            torch.tensor([block_id], device=task_embedding.device)
        ).expand(task_embedding.size(0), -1)
        return torch.cat(
            [
                task_embedding,
                history_context,
                torch.abs(task_embedding - history_context),
                task_embedding * history_context,
                layer_embedding,
            ],
            dim=-1,
        )

    def forward_branch(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_summary: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        history_context, history_attention = self.aggregate_history(task_embedding, history_summary)
        planner_input = self.compose_layer_input(block_id, task_embedding, history_context)
        planner_representation = self.trunks[str(block_id)](planner_input)
        outputs = self.output_heads[str(block_id)](planner_representation)
        return outputs, history_context, history_attention, planner_input, planner_representation


class PlannerPolicyBranch(_PlannerBranchBase):
    def __init__(
        self,
        selected_blocks: List[int],
        task_embedding_dim: int,
        history_dim: int,
        hidden_dim: int,
        layer_embedding_dim: int,
        rank_min: int,
        rank_max: int,
    ):
        super().__init__(
            selected_blocks=selected_blocks,
            task_embedding_dim=task_embedding_dim,
            history_dim=history_dim,
            hidden_dim=hidden_dim,
            layer_embedding_dim=layer_embedding_dim,
            output_dim=5,
        )
        self.rank_min = rank_min
        self.rank_max = rank_max

    def forward_policy(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_summary: torch.Tensor | None,
    ) -> PlannerSignals:
        outputs, history_context, history_attention, planner_input, planner_representation = self.forward_branch(
            block_id,
            task_embedding,
            history_summary,
        )
        novelty = torch.sigmoid(outputs[:, 0:1])
        conflict = torch.sigmoid(outputs[:, 1:2])
        rank_score = torch.sigmoid(outputs[:, 2:3])
        consolidate = torch.sigmoid(outputs[:, 3:4])
        shared_gate = torch.sigmoid(outputs[:, 4:5])
        rank_budget = int(round(self.rank_min + float(rank_score.item()) * (self.rank_max - self.rank_min)))
        return PlannerSignals(
            novelty=novelty,
            conflict=conflict,
            rank_score=rank_score,
            rank_budget=max(self.rank_min, min(self.rank_max, rank_budget)),
            consolidate=consolidate,
            shared_gate=shared_gate,
            history_attention=history_attention,
            history_context=history_context,
            planner_input=planner_input,
            planner_representation=planner_representation,
        )


class PlannerControlBranch(_PlannerBranchBase):
    def __init__(
        self,
        selected_blocks: List[int],
        task_embedding_dim: int,
        history_dim: int,
        hidden_dim: int,
        layer_embedding_dim: int,
    ):
        super().__init__(
            selected_blocks=selected_blocks,
            task_embedding_dim=task_embedding_dim,
            history_dim=history_dim,
            hidden_dim=hidden_dim,
            layer_embedding_dim=layer_embedding_dim,
            output_dim=1,
        )
        self._zero_init_output_heads()

    def _zero_init_output_heads(self) -> None:
        with torch.no_grad():
            for output_head in self.output_heads.values():
                output_head.weight.zero_()
                if output_head.bias is not None:
                    output_head.bias.zero_()

    @staticmethod
    def _normalize_control_representation(planner_representation: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(planner_representation.pow(2).mean(dim=-1, keepdim=True))
        return planner_representation / torch.clamp(rms, min=1e-6)

    def forward_control(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_summary: torch.Tensor | None,
    ) -> PlannerControlOutputs:
        outputs, history_context, history_attention, planner_input, planner_representation = self.forward_branch(
            block_id,
            task_embedding,
            history_summary,
        )
        del outputs
        output_head = self.output_heads[str(block_id)]
        normalized_planner_representation = self._normalize_control_representation(planner_representation)
        delta_from_representation = F.linear(normalized_planner_representation, output_head.weight, bias=None)
        if output_head.bias is None:
            delta_bias = torch.zeros_like(delta_from_representation)
            bias_norm = 0.0
        else:
            delta_bias = output_head.bias.view(1, -1).expand_as(delta_from_representation)
            bias_norm = float(output_head.bias.detach().norm().item())
        shared_gate_logit = delta_from_representation + delta_bias
        return PlannerControlOutputs(
            shared_gate=torch.sigmoid(shared_gate_logit),
            shared_gate_logit=shared_gate_logit,
            delta_raw=shared_gate_logit,
            delta_from_representation=delta_from_representation,
            delta_bias=delta_bias,
            control_head_weight_norm=float(output_head.weight.detach().norm().item()),
            control_head_bias_norm=bias_norm,
            history_attention=history_attention,
            history_context=history_context,
            planner_input=planner_input,
            planner_representation=planner_representation,
            normalized_planner_representation=normalized_planner_representation,
        )


class HorizonPlanner(nn.Module):
    def __init__(
        self,
        selected_blocks: List[int],
        task_embedding_dim: int,
        history_dim: int,
        hidden_dim: int,
        layer_embedding_dim: int,
        rank_min: int,
        rank_max: int,
        tau_novelty: float,
        tau_conflict: float,
    ):
        super().__init__()
        self.selected_blocks = list(selected_blocks)
        self.rank_min = rank_min
        self.rank_max = rank_max
        self.tau_novelty = tau_novelty
        self.tau_conflict = tau_conflict
        self.policy_branch = PlannerPolicyBranch(
            selected_blocks=selected_blocks,
            task_embedding_dim=task_embedding_dim,
            history_dim=history_dim,
            hidden_dim=hidden_dim,
            layer_embedding_dim=layer_embedding_dim,
            rank_min=rank_min,
            rank_max=rank_max,
        )
        self.control_branch = PlannerControlBranch(
            selected_blocks=selected_blocks,
            task_embedding_dim=task_embedding_dim,
            history_dim=history_dim,
            hidden_dim=hidden_dim,
            layer_embedding_dim=layer_embedding_dim,
        )
        # Keep legacy attribute names available for tests and existing diagnostics.
        self.layer_embeddings = self.policy_branch.layer_embeddings
        self.history_query = self.policy_branch.history_query
        self.history_key = self.policy_branch.history_key
        self.history_value = self.policy_branch.history_value
        self.trunks = self.policy_branch.trunks
        self.output_heads = self.policy_branch.output_heads

    def aggregate_history(self, task_embedding: torch.Tensor, history_summary: torch.Tensor | None):
        return self.policy_branch.aggregate_history(task_embedding, history_summary)

    def compose_layer_input(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_context: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy_branch.compose_layer_input(block_id, task_embedding, history_context)

    def forward_policy(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_summary: torch.Tensor | None,
    ) -> PlannerSignals:
        return self.policy_branch.forward_policy(block_id, task_embedding, history_summary)

    def forward_control(
        self,
        block_id: int,
        task_embedding: torch.Tensor,
        history_summary: torch.Tensor | None,
    ) -> PlannerControlOutputs:
        return self.control_branch.forward_control(block_id, task_embedding, history_summary)

    def policy_parameters(self):
        return self.policy_branch.parameters()

    def control_parameters(self):
        return self.control_branch.parameters()

    def freeze_policy_branch(self) -> None:
        for parameter in self.policy_branch.parameters():
            parameter.requires_grad = False

    def unfreeze_policy_branch(self) -> None:
        for parameter in self.policy_branch.parameters():
            parameter.requires_grad = True

    def freeze_control_branch(self) -> None:
        for parameter in self.control_branch.parameters():
            parameter.requires_grad = False

    def unfreeze_control_branch(self) -> None:
        for parameter in self.control_branch.parameters():
            parameter.requires_grad = True

    def forward(self, block_id: int, task_embedding: torch.Tensor, history_summary: torch.Tensor | None) -> PlannerSignals:
        return self.forward_policy(block_id, task_embedding, history_summary)

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
