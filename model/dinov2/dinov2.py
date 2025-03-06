import os
import torch
import torch.nn as nn

descriptor_size = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

descriptor_map = {
    "dinov2_vits14": "vit_small",
    "dinov2_vitb14": "vit_base",
    "dinov2_vitl14": "vit_large",
    "dinov2_vitg14": "vit_giant2",
    "gigapose_dinov2": "vit_large",
}

from enum import Enum
from typing import Union


class Weights(Enum):
    LVD142M = "LVD142M"

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


def _make_dinov2_model_name(arch_name: str, patch_size: int, num_register_tokens: int = 0) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    registers_suffix = f"_reg{num_register_tokens}" if num_register_tokens else ""
    return f"dinov2_{compact_arch_name}{patch_size}{registers_suffix}"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from . import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model


class DINOV2(nn.Module):
    def __init__(self, cfg, freeze=False):
        super().__init__()
        self.cfg = cfg
        self.blocks_to_take = [blocks[-1] for blocks in cfg.interaction_indexes]

        self.dinov2 = _make_dinov2_model(arch_name=descriptor_map[cfg.vit_type], pretrained=cfg.pretrained)
        
        self.num_features = self.dinov2.num_features
        self.patch_size = self.dinov2.patch_size

    def forward(self, x):
        bs, _, h, w = x.shape

        x = self.dinov2.prepare_tokens_with_masks(x)

        outputs = []
        for i, blk in enumerate(self.dinov2.blocks):
            x = blk(x)
            if i in self.blocks_to_take:
                outputs.append(x)

        # class_tokens = [out[:, 0] for out in outputs]
        outputs = [
            out[:, 1:].permute(0,2,1).reshape(bs, self.num_features, h//self.patch_size, w//self.patch_size) for out in outputs
        ]

        return outputs


