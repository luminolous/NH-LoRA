from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn

from src.backbones.vision_transformer import FrozenVisionTransformerBackbone
from src.models.classifier import IncrementalCosineClassifier
from src.models.lora import NHLoRALayer
from src.models.planner import MaterializedLayerPlan
from src.models.task_state import TaskState


class NHLoRAModel(nn.Module):
    def __init__(self, config: Dict[str, object]):
        super().__init__()
        model_cfg = config["model"]
        benchmark_cfg = config["benchmark"]
        nh_cfg = config["nh_lora"]
        self.backbone = FrozenVisionTransformerBackbone.build(model_cfg, benchmark_cfg)
        self.selected_blocks = [int(block_id) for block_id in model_cfg["selected_blocks"]]
        self.insertion_points = [str(point) for point in model_cfg["insertion_points"]]
        self.layers = nn.ModuleDict()
        for block_id in self.selected_blocks:
            self.layers[str(block_id)] = NHLoRALayer(
                embed_dim=self.backbone.embed_dim,
                selected_points=self.insertion_points,
                shared_rank=int(nh_cfg["shared_rank"]),
                slot_r_max=int(nh_cfg["slot_r_max"]),
                slot_init_rank=int(nh_cfg["slot_init_rank"]),
                bootstrap_slot_rank=int(nh_cfg["bootstrap_slot_rank"]),
                max_slots=int(nh_cfg["max_slots_per_block"]),
                router_topk=int(nh_cfg["router_topk"]),
                router_temperature=float(nh_cfg["router_temperature"]),
                task_embedding_dim=int(config["warmup"]["task_embedding_dim"]),
            )
        self.classifier = IncrementalCosineClassifier(
            feature_dim=self.backbone.embed_dim,
            tau=float(config["classifier"]["tau_cls"]),
        )

    def nh_layers_by_block(self):
        return {int(block_id): layer for block_id, layer in self.layers.items()}

    def export_structure_state(self) -> Dict[str, object]:
        return {
            "layers": {
                str(block_id): self.layers[str(block_id)].export_structure_state()
                for block_id in self.selected_blocks
            },
            "classifier_num_classes": self.classifier.num_classes,
        }

    def load_structure_state(self, state: Dict[str, object]) -> None:
        for block_id in self.selected_blocks:
            layer_state = state.get("layers", {}).get(str(block_id), {})
            self.layers[str(block_id)].load_structure_state(layer_state)
        classifier_classes = int(state.get("classifier_num_classes", 0))
        self.classifier = IncrementalCosineClassifier(
            feature_dim=self.backbone.embed_dim,
            tau=self.classifier.tau,
        )
        self.classifier.expand(classifier_classes)
        self.classifier.to(next(self.parameters()).device)

    def build_bootstrap_plan(self, task_embedding: torch.Tensor) -> Dict[int, MaterializedLayerPlan]:
        del task_embedding
        plans = {}
        for block_id in self.selected_blocks:
            plans[block_id] = MaterializedLayerPlan(
                requested_action="reuse_shared",
                action="open_new_slot",
                selected_slot=None,
                target_rank=None,
                rank_delta=0,
                create_new_slot=True,
                new_slot_rank=self.layers[str(block_id)].bootstrap_slot_rank,
                candidate_slots=[],
                compatibility_scores={},
                shared_gate=1.0,
                consolidate_flag=False,
                strong_retention=False,
                shared_only=False,
            )
        return plans

    def apply_structure_changes(
        self,
        materialized_plans: Dict[int, MaterializedLayerPlan],
        task_embedding: torch.Tensor,
        task_id: int,
    ) -> Dict[int, Dict[str, object]]:
        applied = {}
        for block_id, plan in materialized_plans.items():
            applied[block_id] = self.layers[str(block_id)].apply_structure_change(plan, task_embedding, task_id=task_id)
        return applied

    def build_inference_profile(self) -> Dict[int, Dict[str, object]]:
        profile = {}
        for block_id in self.selected_blocks:
            layer = self.layers[str(block_id)]
            live_slots = [
                slot_id
                for slot_id in layer.live_slot_ids()
                if layer.slot_metadata[slot_id].retained_for_inference
            ]
            live_slots = sorted(
                live_slots,
                key=lambda slot_id: (
                    layer.slot_metadata[slot_id].usage_ema,
                    layer.slot_metadata[slot_id].cumulative_usage,
                ),
                reverse=True,
            )[: layer.router_topk]
            if layer.last_structural_action in {"reuse_shared", "freeze_old_strong_retention"}:
                live_slots = []
            profile[block_id] = {
                "action": "inference_profile",
                "requested_action": layer.last_structural_action,
                "active_slot_candidates": live_slots,
                "selected_slot": live_slots[0] if live_slots else None,
                "rank_cfg": {slot_id: layer.slot_metadata[slot_id].rank for slot_id in layer.live_slot_ids()},
                "shared_gate": layer.last_shared_gate,
                "consolidate_flag": layer.last_consolidate_flag,
                "deterministic": len(live_slots) <= 1,
                "created_new_slot": False,
                "fallback_action": None,
                "strong_retention": layer.last_structural_action == "freeze_old_strong_retention",
                "shared_only": len(live_slots) == 0,
                "compatibility_scores": {},
            }
        return profile

    def capture_pre_task_snapshots(self) -> None:
        for layer in self.layers.values():
            layer.capture_pre_task_snapshot()

    def expand_classifier(self, num_new_classes: int) -> List[int]:
        return self.classifier.expand(num_new_classes)

    def encode(self, images: torch.Tensor, task_state: TaskState | None = None, planner_out=None):
        return self.backbone.forward_features(images, self.nh_layers_by_block() if planner_out is not None else None, task_state, planner_out)

    def forward_with_state(self, images: torch.Tensor, task_state: TaskState | None, planner_out):
        features_dict = self.encode(images, task_state=task_state, planner_out=planner_out)
        layer_features = {
            int(block_id): tokens[:, 0]
            for block_id, tokens in features_dict["block_outputs"].items()
        }
        logits = self.classifier(features_dict["features"])
        return {
            "logits": logits,
            "features": features_dict["features"],
            "route_info": features_dict["route_info"],
            "layer_features": layer_features,
        }

    def forward(self, images: torch.Tensor, task_state: TaskState | None, planner_out):
        state = self.forward_with_state(images, task_state, planner_out)
        return state["logits"], state["features"], state["route_info"]

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode(images)["features"]

    def make_teacher_snapshot(self) -> "NHLoRAModel":
        teacher = deepcopy(self)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher
