from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of a large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 1024


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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
    # See https://blog.eleuther.ai/rotary-embeddings/
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        assert embedding_dim % num_heads == 0, (embedding_dim, num_heads)
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # The query, key, value and output projections are all EMBEDDING_DIM x EMBEDDING_DIM
        # matrices. For clarity, EMBEDDING_DIM is written as NUM_HEADS x HEAD_DIM where
        # conceptually the operation is performed independently for different heads.
        self.wq = nn.Linear(
            embedding_dim,
            num_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            embedding_dim,
            num_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            embedding_dim,
            self.num_heads * self.head_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            self.num_heads * self.head_dim,
            embedding_dim,
            bias=False,
        )

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        """
        Inputs
         - x: N x L x D

        Output: N x L x D

        Dimensions:
         - N, L, D, Nh, Dh := batch_size, sequence_length, embedding_dim, num_heads, head_dim
        """
        batch_size, seq_len, _ = x.shape

        xq = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        queries, keys, values = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)

        output = (
            F.scaled_dot_product_attention(queries, keys, values, is_causal=True) # N x Nh x L x Dh
            .transpose(1, 2) # N x L x Nh x Dh
            .contiguous() # N x L x Nh x Dh
            .view(batch_size, seq_len, -1) # N x L x D
        )

        return self.wo(output) # N x L x D


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
          x: N x

        Output:

        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(embedding_dim=args.dim, num_heads=args.n_heads)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = torch.nn.Embedding(params.vocab_size, params.dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _batch_size, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = h.to(layer.parameters().__next__().device)
            h = layer(h, freqs_cis)
        h = h.to(self.norm.parameters().__next__().device)
        h = self.norm(h)

        hl = h[:, -1, :]
        # hl = h

        hl = hl.to(self.output.parameters().__next__().device)
        output = self.output(hl)
        return output.float()
