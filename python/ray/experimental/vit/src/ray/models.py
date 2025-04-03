from functools import partial

import torch
import torch.nn as nn
from open_clip import get_model_config
from open_clip.model import _build_text_tower, _build_vision_tower
from open_clip.transformer import (
    ResidualAttentionBlock,
    text_global_pool,
    VisionTransformer,
    TextTransformer,
    Transformer,
)
from torch import optim
from torch.distributed._tensor import Replicate
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.nn import functional as F

from typing import Iterator, Tuple, Union, Callable


class VisionEncoder(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()

        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        assert self.model_config is not None, f"Unsupported {model_name}!"

        self.visual: VisionTransformer = _build_vision_tower(
            self.model_config["embed_dim"], self.model_config["vision_cfg"]
        )

    def forward(self, images: torch.Tensor, normalize: bool = True):
        features = self.visual(images)
        return F.normalize(features, dim=-1) if normalize else features


class TextEncoder(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()

        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        assert self.model_config is not None, f"Unsupported {model_name}!"

        text: TextTransformer = _build_text_tower(
            self.model_config["embed_dim"], self.model_config["text_cfg"]
        )
        self.transformer = text.transformer
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.text_pool_type = text.pool_type
        self.register_buffer("attn_mask", text.attn_mask, persistent=False)

    def forward(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x


def parallelize_transformer(
    transformer: Transformer, tp_mesh: DeviceMesh
) -> Transformer:
    for _, transformer_block in enumerate(transformer.resblocks):
        layer_tp_plan = {
            "mlp.c_fc": ColwiseParallel(),
            "mlp.c_proj": RowwiseParallel(),
        }

        # Custom parallelization plan for the model
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_tp_plan,
        )
    return transformer


def parallelize_tp(
    model: Union[TextEncoder, VisionEncoder],
    tp_mesh: DeviceMesh,
    text: bool,
    vision: bool,
) -> Union[TextEncoder, VisionEncoder]:
    """
    Imitate the example in
    https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/fsdp_tp_example.py
    """
    if text:
        model = parallelize_module(
            model,
            tp_mesh,
            {
                "token_embedding": RowwiseParallel(
                    input_layouts=Replicate(),
                ),
            },
        )
        assert isinstance(model, TextEncoder)
        parallelize_transformer(model.transformer, tp_mesh)

    if vision:
        assert isinstance(model, VisionEncoder)
        parallelize_transformer(model.visual.transformer, tp_mesh)

    model.to("cuda")
    return model


def parallelize_dp(
    model: Union[TextEncoder, VisionEncoder], dp_mesh: DeviceMesh
) -> Union[TextEncoder, VisionEncoder]:
    model.to(torch.cuda.current_device())
    model = FSDP(
        model,
        device_mesh=dp_mesh,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=get_clip_wrap_policy(),
    )
    return model


def parallelize_2d(
    model: Union[TextEncoder, VisionEncoder],
    dp_size: int,
    tp_size: int,
    text: bool,
    vision: bool,
) -> Tuple[Union[TextEncoder, VisionEncoder], DeviceMesh]:
    assert dp_size * tp_size > 1, "DP or TP must be greater than 1!"

    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("dp", "tp"),
    )
    if dp_size == 1:
        model = parallelize_tp(model, device_mesh["tp"], text, vision)
    elif tp_size == 1:
        model = parallelize_dp(model, device_mesh["dp"])
    else:
        model = parallelize_tp(model, device_mesh["tp"], text, vision)
        model = parallelize_dp(model, device_mesh["dp"])

    return model, device_mesh


# def init_tp_params(transformer):
#     proj_std = (transformer.width**-0.5) * ((2 * transformer.layers) ** -0.5)
#     fc_std = (2 * transformer.width) ** -0.5
#     for block in transformer.resblocks:
#         nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
#         nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
#         nn.init.zeros_(block.mlp.c_fc.bias)


def get_clip_wrap_policy() -> Callable[[nn.Module, bool, int], bool]:
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ResidualAttentionBlock},
    )


def init_optimizer(
    named_parameters: Iterator[Tuple[str, nn.Parameter]],
) -> optim.Optimizer:
    params = [p for _, p in named_parameters if p.requires_grad]

    return optim.AdamW(params, lr=1e-3, betas=(0.9, 0.98), eps=1e-06, foreach=True)
