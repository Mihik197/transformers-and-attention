import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32  # no. of heads for the query
    n_kv_heads: Optional[int] = None  # no. of heads for the kv
    vocab_size: int = -1  # will be set when tokenizer is loaded
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (batch, seq_length, dim) * (batch, seq_length, 1) -> (batch, seq_length, dim)
        # rsqrt(x) = 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (dim) * (batch, seq_len, dim) -> (batch, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class EncoderBlock(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalization before self-attention
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # normalization before feed-forward block
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (batch, seq_len, dim) + (batch, seq_len, dim) -> (batch, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(x))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (batch, seq_length)
        batch_size, seq_len = tokens.shape
        assert seq_len == -1, "Only one token can be processed at a time. This can only be used for inference"

        # (batch, seq_length) -> (batch, seq_length, dim)
        h = self.tok_embeddings(tokens)

        # retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]

        # apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    
def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # embedding dimension must be even
    assert head_dim % 2 == 0, "Dimension must be even"

    # theta parameter
    # theta_i = 1000 ^ (-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # construct "m" parameter (positions)
    # (seq_len)
    m = torch.arange(seq_len, device=device)

    # multiply each theta with each position using outer product
    # (seq_len) outer product (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()

    # compute the complex number in polar form: c = R * exp(i * m * theta), where r = 1, angle = m * theta
    # (seq_len, head_dim / 2) -> (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch, seq_len, h, head_dim) -> (batch, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (batch, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) -> (batch, seq_len, h, head_dim / 2)
    x_rotated = x_complex * freqs_complex
    # (batch, seq_len, h, head_dim / 2) -> (batch, seq_len, h, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (batch, seq_len, h, head_dim / 2, 2) -> (batch, seq_len, h, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)
