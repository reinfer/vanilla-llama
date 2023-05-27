from dataclasses import dataclass
from typing import Tuple

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


class Rotary(torch.nn.Module):
    def __init__(
        self, dim: int, max_seq_len: int, base: float = 10000.0, device: torch.device | None = None
    ):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, device=device).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = freqs[..., None].repeat(1, 1, 2).flatten(1)

        flip_imag = torch.tensor([1, -1])
        self.register_buffer("flip_imag", flip_imag[None, None, None, None, :], persistent=False)
        self.flip_imag: Tensor

        self.register_buffer("cos_cached", emb.cos()[None, :, None, :], persistent=False)
        self.cos_cached: Tensor
        self.register_buffer("sin_cached", emb.sin()[None, :, None, :], persistent=False)
        self.sin_cached: Tensor

    def forward(self, x):
        _, seq_len, _, _ = x.shape
        cos, sin = self.cos_cached[:, :seq_len].to(x.device), self.sin_cached[:, :seq_len].to(x.device)
        shifted = (x.view(*x.shape[:-1], -1, 2) * self.flip_imag.to(x.device)).flip(-1).reshape(x.shape)
        return (x.float() * cos + shifted.float() * sin).type_as(x)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


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
         - freqs_cis: L x (D / 2)

        Output: N x L x D

        Dimensions:
         - N, L, D, Nh, Dh := batch_size, sequence_length, embedding_dim, num_heads, head_dim
        """
        batch_size, seq_len, _ = x.shape

        q = self.wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # N x L x Nh x Dh
        k = self.wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # N x L x Nh x Dh
        v = self.wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)  # N x L x Nh x Dh

        q, k = freqs_cis(q), freqs_cis(k)  # N x L x Nh x Dh
        # q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # N x L x Nh x Dh
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # N x Nh x L x Dh

        output = (
            F.scaled_dot_product_attention(q, k, v, is_causal=True)  # N x Nh x L x Dh
            .transpose(1, 2)  # N x L x Nh x Dh
            .contiguous()  # N x L x Nh x Dh
            .view(batch_size, seq_len, -1)  # N x L x D
        )

        return self.wo(output)  # N x L x D


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, multiple_of: int) -> None:
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(embedding_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.w3 = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: N x L x D
        Output: N x L x D

        Dimensions:
         - N, L, D := batch_size, sequence_length, embedding_dim
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self, embedding_dim: int, num_heads: int, multiple_of: int, norm_eps: float
    ) -> None:
        super().__init__()

        self.attention = Attention(embedding_dim=embedding_dim, num_heads=num_heads)
        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim, hidden_dim=4 * embedding_dim, multiple_of=multiple_of
        )
        self.attention_norm = RMSNorm(embedding_dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(self, x: Tensor, freqs_cis: Tensor) -> Tensor:
        out = x + self.attention(self.attention_norm(x), freqs_cis)
        out = out + self.feed_forward(self.ffn_norm(out))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        self.layers = nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(
                TransformerBlock(
                    embedding_dim=params.dim,
                    num_heads=params.n_heads,
                    multiple_of=params.multiple_of,
                    norm_eps=params.norm_eps,
                )
            )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # self.freqs_cis = precompute_freqs_cis(
        #     self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        # )
        self.rotary = Rotary(
            dim=self.params.dim // self.params.n_heads, max_seq_len=self.params.max_seq_len
        )
        self.forwardc = None

    @torch.no_grad()
    def forward2(self, tokens: Tensor, start_pos: int) -> Tensor:
        _, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        self.rotary = self.rotary.to(h.device)
        # self.freqs_cis = self.freqs_cis.to(h.device)
        # freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        for layer in self.layers:
            h = h.to(layer.parameters().__next__().device)
            h = layer(h, self.rotary)
        h = h.to(self.norm.parameters().__next__().device)
        h = self.norm(h)

        hl = h[:, -1, :]
        # hl = h

        hl = hl.to(self.output.parameters().__next__().device)
        output = self.output(hl)
        return output.float()

    # @torch.inference_mode()
    def forward(self, tokens: Tensor, start_pos: int) -> Tensor:
        # if self.forwardc is None:
        #     self.forwardc = torch.compile(self.forward2)
        return self.forward2(tokens, start_pos)
