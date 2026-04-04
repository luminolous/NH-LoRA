from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VisionTransformerSpec:
    image_size: int
    patch_size: int
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: float


def build_vit_spec(backbone_name: str) -> VisionTransformerSpec:
    if backbone_name == "vit_base_patch16_224_in21k":
        return VisionTransformerSpec(224, 16, 768, 12, 12, 4.0)
    if backbone_name == "toy_vit_tiny":
        return VisionTransformerSpec(32, 4, 128, 4, 4, 2.0)
    raise ValueError(f"Unsupported backbone name: {backbone_name}")


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(images)
        return tokens.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio)


class InternalVisionTransformer(nn.Module):
    def __init__(self, spec: VisionTransformerSpec):
        super().__init__()
        self.patch_embed = PatchEmbed(spec.image_size, spec.patch_size, spec.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, spec.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, spec.embed_dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList(
            [TransformerBlock(spec.embed_dim, spec.num_heads, spec.mlp_ratio) for _ in range(spec.depth)]
        )
        self.norm = nn.LayerNorm(spec.embed_dim)
        self.embed_dim = spec.embed_dim
        self.depth = spec.depth
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class FrozenVisionTransformerBackbone(nn.Module):
    def __init__(self, core_model: nn.Module, spec: VisionTransformerSpec, freeze: bool = True):
        super().__init__()
        self.core_model = core_model
        self.spec = spec
        self.embed_dim = spec.embed_dim
        self.depth = spec.depth
        if freeze:
            self.freeze_parameters()

    @classmethod
    def build(cls, model_config: Dict[str, object], benchmark_config: Dict[str, object]):
        backbone_name = str(model_config["backbone_name"])
        spec = build_vit_spec(backbone_name)
        source = str(model_config.get("backbone_source", "internal"))
        if source == "timm":
            try:
                import timm  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "The timm backbone source was requested but timm is not installed. "
                    "Use backbone_source=internal for local smoke tests or install timm on the SSH machine."
                ) from exc
            model = timm.create_model(backbone_name, pretrained=True)
        else:
            model = InternalVisionTransformer(spec)
        return cls(model, spec, freeze=bool(model_config.get("freeze_backbone", True)))

    def freeze_parameters(self) -> None:
        for parameter in self.core_model.parameters():
            parameter.requires_grad = False

    def prepare_tokens(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.core_model.patch_embed(images)
        cls_token = self.core_model.cls_token.expand(images.size(0), -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        tokens = tokens + self.core_model.pos_embed[:, : tokens.size(1)]
        return self.core_model.pos_drop(tokens)

    def _forward_block_plain(self, block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        normed = block.norm1(tokens)
        qkv = F.linear(normed, block.attn.qkv.weight, block.attn.qkv.bias)
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = torch.softmax((query @ key.transpose(-2, -1)) * block.attn.scale, dim=-1)
        attended = attention @ value
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)
        tokens = tokens + F.linear(attended, block.attn.proj.weight, block.attn.proj.bias)
        mlp_input = block.norm2(tokens)
        hidden = block.mlp.act(F.linear(mlp_input, block.mlp.fc1.weight, block.mlp.fc1.bias))
        hidden = F.linear(hidden, block.mlp.fc2.weight, block.mlp.fc2.bias)
        return tokens + hidden

    def _forward_block_with_adapter(self, block, tokens, nh_layer, planner_cfg, task_state):
        normed = block.norm1(tokens)
        route_state = nh_layer.route(normed, planner_cfg, task_state)
        qkv = F.linear(normed, block.attn.qkv.weight, block.attn.qkv.bias)
        qkv = nh_layer.apply_to_qkv(normed, qkv, route_state, planner_cfg)
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = torch.softmax((query @ key.transpose(-2, -1)) * block.attn.scale, dim=-1)
        attended = attention @ value
        attended = attended.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attended = nh_layer.apply_to_projection("out_proj", attended, route_state, planner_cfg)
        attn_output = F.linear(attended, block.attn.proj.weight, block.attn.proj.bias)
        tokens = tokens + attn_output
        mlp_input = block.norm2(tokens)
        fc1_output = F.linear(mlp_input, block.mlp.fc1.weight, block.mlp.fc1.bias)
        fc1_output = nh_layer.apply_to_projection("mlp_fc1", mlp_input, route_state, planner_cfg, base_output=fc1_output)
        hidden = block.mlp.act(fc1_output)
        fc2_output = F.linear(hidden, block.mlp.fc2.weight, block.mlp.fc2.bias)
        fc2_output = nh_layer.apply_to_projection("mlp_fc2", hidden, route_state, planner_cfg, base_output=fc2_output)
        return tokens + fc2_output, route_state

    def forward_features(self, images: torch.Tensor, nh_layers=None, task_state=None, planner_out=None):
        tokens = self.prepare_tokens(images)
        route_info = {}
        block_outputs = {}
        for block_idx, block in enumerate(self.core_model.blocks):
            if nh_layers is not None and block_idx in nh_layers and planner_out is not None:
                tokens, layer_route = self._forward_block_with_adapter(
                    block,
                    tokens,
                    nh_layers[block_idx],
                    planner_out.get(block_idx, {}),
                    task_state,
                )
                route_info[block_idx] = layer_route
            else:
                tokens = self._forward_block_plain(block, tokens)
            block_outputs[block_idx] = tokens
        tokens = self.core_model.norm(tokens)
        return {
            "features": tokens[:, 0],
            "tokens": tokens,
            "block_outputs": block_outputs,
            "route_info": route_info,
        }
