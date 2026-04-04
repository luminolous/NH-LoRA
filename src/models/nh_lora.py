from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import torch
from torch import nn

from src.backbones.vision_transformer import FrozenVisionTransformerBackbone
from src.models.classifier import IncrementalCosineClassifier
from src.models.lora import NHLoRALayer
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
            )
        self.classifier = IncrementalCosineClassifier(
            feature_dim=self.backbone.embed_dim,
            tau=float(config["classifier"]["tau_cls"]),
        )

    def nh_layers_by_block(self):
        return {int(block_id): layer for block_id, layer in self.layers.items()}

    def initialize_bootstrap_structure(self, task_id: int):
        planner_out = {}
        for block_id in self.selected_blocks:
            layer = self.layers[str(block_id)]
            slot_id = layer.ensure_bootstrap_slot(task_id)
            planner_out[block_id] = {
                "action": "reuse_shared",
                "active_slots": [slot_id],
                "rank_cfg": {slot_id: layer.slot_metadata[slot_id].rank},
                "shared_gate": 1.0,
                "consolidate_flag": False,
                "deterministic": True,
                "created_new_slot": False,
            }
        return planner_out

    def capture_pre_task_snapshots(self) -> None:
        for layer in self.layers.values():
            layer.capture_pre_task_snapshot()

    def expand_classifier(self, num_new_classes: int) -> List[int]:
        return self.classifier.expand(num_new_classes)

    def forward(self, images: torch.Tensor, task_state: TaskState | None, planner_out):
        features_dict = self.backbone.forward_features(images, self.nh_layers_by_block(), task_state, planner_out)
        logits = self.classifier(features_dict["features"])
        return logits, features_dict["features"], features_dict["route_info"]

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(images)["features"]

    def make_teacher_snapshot(self) -> "NHLoRAModel":
        teacher = deepcopy(self)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
        return teacher
