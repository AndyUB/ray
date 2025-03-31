# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector

# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )


logger = logging.getLogger(__name__)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


LLAMA_DEBUG = ModelArgs(
    dim=1024,  # 1/2
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=256,
    ffn_dim_multiplier=1.5,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)


LLAMA_1B = ModelArgs(
    dim=2048,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=256,
    ffn_dim_multiplier=1.5,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)


LLAMA_3B = ModelArgs(
    dim=3072,
    n_layers=28,
    n_heads=24,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)

LLAMA_8B = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    vocab_size=128256,
    multiple_of=1024,
    ffn_dim_multiplier=1.3,
    norm_eps=1e-5,
    rope_theta=500000,
    max_batch_size=32,
    max_seq_len=2048,
)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        # self.wq = ColumnParallelLinear(
        self.wq = torch.nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.wk = ColumnParallelLinear(
        self.wk = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.wv = ColumnParallelLinear(
        self.wv = torch.nn.Linear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.wo = RowParallelLinear(
        self.wo = torch.nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            # input_is_parallel=True,
            # init_method=lambda x: x,
        )

        # [NOTE] Disable KV cache during training.
        # self.cache_k = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()
        # self.cache_v = torch.zeros(
        #     (
        #         args.max_batch_size,
        #         args.max_seq_len,
        #         self.n_local_kv_heads,
        #         self.head_dim,
        #     )
        # ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # [NOTE] Disable KV cache during training.

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)

        # self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # self.w1 = ColumnParallelLinear(
        self.w1 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )
        # self.w2 = RowParallelLinear(
        self.w2 = torch.nn.Linear(
            hidden_dim,
            dim,
            bias=False,
            # input_is_parallel=True,
            # init_method=lambda x: x,
        )
        # self.w3 = ColumnParallelLinear(
        self.w3 = torch.nn.Linear(
            dim,
            hidden_dim,
            bias=False,
            # gather_output=False,
            # init_method=lambda x: x,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        # buckets = [
        #     "VocabParallelEmbedding",
        #     [
        #         "Attention",
        #         "FeedForward",
        #         "RMSNorm * 2",
        #     ],
        #     "ColumnParallelLinear",
        # ]

        def log_size(layer, indent=0):
            num_params = sum(p.numel() for p in layer.parameters())
            size_mib = num_params * 4 / (1024 * 1024)
            indent_str = "  " * indent
            logger.info(
                f"{indent_str}{layer.__class__.__name__}: {round(size_mib)} MiB"
            )
            if size_mib < 25:
                return
            for _, child in layer.named_children():
                log_size(child, indent + 1)

        # self.tok_embeddings = VocabParallelEmbedding(
        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size,
            params.dim,
            # init_method=lambda x: x,
        )
        log_size(self.tok_embeddings)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        log_size(self.layers[0])

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        log_size(self.norm)
        # self.output = ColumnParallelLinear(
        self.output = torch.nn.Linear(
            params.dim,
            params.vocab_size,
            bias=False,
            # init_method=lambda x: x,
        )
        log_size(self.output)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class BucketParameter(nn.Module):
    def __init__(
        self,
        layers: List[nn.Module],
        pre_hook=None,
        post_hook=None,
        hook_layers: Optional[List[nn.Module]] = None,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.pre_hook = pre_hook
        self.post_hook = post_hook
        self.hook_layers = (
            torch.nn.ModuleList(hook_layers) if hook_layers is not None else None
        )

        self.input = None
        self.target = None
        if hook_layers is None:
            self.named_params = list(self.layers.named_parameters())
        else:
            self.named_params = list(
                chain(
                    self.layers.named_parameters(),
                    self.hook_layers.named_parameters(),
                )
            )
        self.criterion = torch.nn.CrossEntropyLoss()
        params = [param for _, param in self.named_params]
        self.optimizer = torch.optim.AdamW(params, lr=1e-6)

        self.init_weights()

    def init_weights(self):
        for _, param in self.named_params:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_transformer(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        assert len(self.layers) == 1
        transformer = self.layers[0]
        h = x + transformer.attention(
            transformer.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + transformer.feed_forward(transformer.ffn_norm(h))
        return out

    def backward(
        self,
        loss: Optional[torch.Tensor] = None,
        pred: Optional[torch.Tensor] = None,
        grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loss is not None:
            assert pred is None
            loss.backward()
        elif pred is not None:
            assert grad is not None
            pred.backward(grad)

        # [NOTE] If param.grad is None, we need to set a flag.
        grads_cat = parameters_to_vector(
            # [param.grad for _, param in self.named_params if param.grad is not None]
            [param.grad for _, param in self.named_params]
        )
        return grads_cat

    def update(self, grads_cat: torch.Tensor, grads_passed: bool) -> None:
        if grads_passed:
            offset = 0
            for _, param in self.named_params:
                # if param.grad is None:
                #     continue
                size = param.data.numel()
                grad = grads_cat[offset : offset + size].reshape(param.data.shape)
                param.grad = grad
                offset += size
            del grads_cat

        self.optimizer.step()
        self.optimizer.zero_grad()

    def copy(self, grads_cat: torch.Tensor, grads_passed: bool) -> None:
        raise NotImplementedError
        if grads_passed:
            offset = 0
            for _, param in self.named_params:
                # if param.grad is None:
                #     continue
                size = param.data.numel()
                grad = grads_cat[offset : offset + size].reshape(param.data.shape)
                param.grad = grad
                offset += size

    def step(self) -> None:
        raise NotImplementedError
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _log_versions(self, prompt: str, grad_is_none: bool):
        logger.info(prompt)
        version_to_count = defaultdict(int)
        version_to_names = defaultdict(list)
        version_to_count_grad = defaultdict(int)
        for name, param in self.named_params:
            if grad_is_none:
                assert param.grad is None
            else:
                assert param.grad is not None
            version_to_count[param._version] += 1
            version_to_names[param._version].append(name)
            if param.grad is not None:
                version_to_count_grad[param.grad._version] += 1
        logger.info(f"version_to_count: {version_to_count}")
        logger.info(f"version_to_names: {version_to_names}")
        logger.info(f"version_to_count_grad: {version_to_count_grad}")


class Shard(torch.nn.Module):
    # Simulate FSDP sharding.

    def __init__(
        self,
        model: BucketParameter,
        sharded_param: torch.Tensor,
        model_metadata: List[Tuple[torch.Size, int]],
    ) -> None:
        super().__init__()

        self.model = model
        self.pre_hook = model.pre_hook
        self.post_hook = model.post_hook
        self.sharded_param = torch.nn.Parameter(sharded_param, requires_grad=True)
        self.model_metadata = model_metadata
        self.optimizer = torch.optim.AdamW([self.sharded_param], lr=1e-6)

    def set_flat_param(self, flat_param: torch.Tensor) -> None:
        _set_flat_param(self.model, flat_param, self.model_metadata)

    def get_flat_grad(self) -> torch.Tensor:
        flat_grad = parameters_to_vector(
            [param.grad for param in self.model.parameters()]
        )
        return flat_grad

    def free_peer_shards(self) -> None:
        _free_peer_shards(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward(x)

    def forward_transformer(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        return self.model.forward_transformer(x, start_pos, freqs_cis, mask)

    def update(self, grad: torch.Tensor, grad_passed: bool) -> None:
        if grad_passed:
            self.sharded_param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()


def shard_model(model: torch.nn.Module, num_shards: int) -> List[Shard]:
    param = parameters_to_vector(model.parameters())
    padding = (num_shards - param.numel() % num_shards) % num_shards
    if padding > 0:
        param = torch.cat(
            [param, torch.zeros(padding, dtype=param.dtype, device=param.device)]
        )
    sharded_param_size = param.numel() // num_shards
    sharded_params = [
        param[i : i + sharded_param_size].reshape(-1)
        for i in range(0, param.numel(), sharded_param_size)
    ]
    model_metadata = [(param.shape, param.numel()) for param in model.parameters()]
    _free_peer_shards(model)
    shards = [
        Shard(model, sharded_param, model_metadata) for sharded_param in sharded_params
    ]
    return shards


def _set_flat_param(
    model: torch.nn.Module,
    flat_param: torch.Tensor,
    model_metadata: List[Tuple[torch.Size, int]],
) -> None:
    offset = 0
    for param, (shape, numel) in zip(model.parameters(), model_metadata):
        param.data = flat_param[offset : offset + numel].reshape(shape)
        offset += numel


def _free_peer_shards(model: torch.nn.Module) -> None:
    def get_first_param():
        for param in model.parameters():
            return param
        raise ValueError("Expected parameters")

    # Assume all parameters have the same dtype and device.
    first_param = get_first_param()
    dtype = first_param.dtype
    device = first_param.device
    empty_tensor = torch.empty(0, dtype=dtype, device=device)
    for param in model.parameters():
        param.data = empty_tensor
        param.grad = None


class TransformerBP(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.bparams: List[BucketParameter] = []
        # buckets = [
        #     "VocabParallelEmbedding",
        #     [
        #         "Attention",
        #         "FeedForward",
        #         "RMSNorm * 2",
        #     ],
        #     "ColumnParallelLinear",
        # ]

        def log_size(layer, indent=0):
            num_params = sum(p.numel() for p in layer.parameters())
            size_mib = num_params * 4 / (1024 * 1024)
            indent_str = "  " * indent
            logger.info(
                f"{indent_str}{layer.__class__.__name__}: {round(size_mib)} MiB"
            )
            if size_mib < 25:
                return
            for _, child in layer.named_children():
                log_size(child, indent + 1)

        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size,
            params.dim,
        )
        log_size(self.tok_embeddings)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        log_size(self.layers[0])

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        log_size(self.norm)
        self.output = torch.nn.Linear(
            params.dim,
            params.vocab_size,
            bias=False,
            # init_method=lambda x: x,
        )
        log_size(self.output)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

        self.bparams.append(
            BucketParameter([self.tok_embeddings], post_hook=self.post_embeddings)
        )
        for layer in self.layers:
            # [TODO] Separate attention and feedforward layers.
            self.bparams.append(BucketParameter([layer]))
        self.bparams.append(
            BucketParameter(
                [self.output], pre_hook=self.pre_output, hook_layers=[self.norm]
            )
        )

    def post_embeddings(self, tokens: torch.Tensor, h: torch.Tensor):
        start_pos = 0
        self.freqs_cis = self.freqs_cis.to(h.device)
        _bsz, seqlen = tokens.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        return freqs_cis, mask

    def pre_output(self, h: torch.Tensor):
        h = self.norm(h)
        return h

    def forward(self, tokens: torch.Tensor):
        # [NOTE] This is used for torch DDP.
        bp = self.bparams[0]
        h = bp.forward(tokens)
        freqs_cis, mask = bp.post_hook(tokens, h)

        for bp in self.bparams[1:-1]:
            h = bp.forward_transformer(h, 0, freqs_cis, mask)

        bp = self.bparams[-1]
        h = bp.pre_hook(h)
        output = bp.forward(h)

        return output


def log_size(layer, indent=0):
    num_params = sum(p.numel() for p in layer.parameters())
    size_mib = num_params * 4 / (1024 * 1024)
    indent_str = "  " * indent
    logger.info(f"{indent_str}{layer.__class__.__name__}: {round(size_mib)} MiB")
    if size_mib < 25:
        return
    for _, child in layer.named_children():
        log_size(child, indent + 1)


class BucketParameterBase(nn.Module):
    pass


class BucketParameterFirst(BucketParameterBase):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(
            params.vocab_size,
            params.dim,
        )
        log_size(self.tok_embeddings)
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    def post_embeddings(self, tokens: torch.Tensor, h: torch.Tensor):
        start_pos = 0
        self.freqs_cis = self.freqs_cis.to(h.device)
        _bsz, seqlen = tokens.shape
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)
        return freqs_cis, mask

    def forward(self, x: torch.Tensor):
        return self.tok_embeddings(x)


class BucketParameterTransformerBlock(BucketParameterBase):
    def __init__(self, params: ModelArgs, layer_id: int):
        super().__init__()
        self.layer = TransformerBlock(layer_id, params)
        if layer_id == 0:
            log_size(self.layer)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        transformer = self.layer
        h = x + transformer.attention(
            transformer.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + transformer.feed_forward(transformer.ffn_norm(h))
        return out


class BucketParameterLast(BucketParameterBase):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        log_size(self.norm)
        self.output = torch.nn.Linear(
            params.dim,
            params.vocab_size,
            bias=False,
        )
        log_size(self.output)

    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        return self.output(x)


class TransformerWrapped(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.bparams = []
        self.bparams.append(BucketParameterFirst(params))
        for layer_id in range(params.n_layers):
            self.bparams.append(BucketParameterTransformerBlock(params, layer_id))
        self.bparams.append(BucketParameterLast(params))
        self.bparams = torch.nn.ModuleList(self.bparams)

    def forward(self, tokens: torch.Tensor):
        bp = self.bparams[0]
        h = bp.forward(tokens)
        freqs_cis, mask = bp.post_embeddings(tokens, h)

        for bp in self.bparams[1:-1]:
            h = bp.forward(h, 0, freqs_cis, mask)

        bp = self.bparams[-1]
        output = bp.forward(h)

        return output
