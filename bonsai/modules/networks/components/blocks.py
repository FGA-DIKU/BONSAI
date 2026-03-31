import torch
import torch.nn as nn
from flash_attn.modules.mlp import Mlp
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.modules.mha import FlashSelfAttention
from flash_attn.modules.mlp import FusedMLP


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, bias, max_seqlen, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, bias=bias)
        self.mha = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            bias=bias,
            max_seqlen=max_seqlen,
            causal=causal,
        )
        self.resid_dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_size, bias=bias)
        self.mlp = FusedMLP(hidden_size, bias1=bias, bias2=bias)

    def forward(self, x, cu_seqlens=None):
        # LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add
        x = x + self.resid_dropout(self.mha(self.ln1(x), cu_seqlens=cu_seqlens))
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, bias, max_seqlen, causal):
        self.num_heads = num_heads
        assert (hidden_size % self.num_heads) == 0, (
            f"Hidden size {hidden_size} must be divisible by num_heads {self.num_heads} "
        )
        self.head_dim = hidden_size // self.num_heads

        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)

        self.rotary_embedding = RotaryEmbedding(dim=self.head_dim)
        self.rotary_embedding._update_cos_sin_cache(max_seqlen)

        self.self_attn = FlashSelfAttention(causal=causal)

        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, cu_seqlens=None):
        bs, seqlen, hidden_dim = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(
            bs, seqlen, 3, self.num_heads, self.head_dim
        )  # (bs, seqlen, dim) -> (bs, seqlen, 3, nh, hd)

        qkv = self.rotary_embedding(qkv)

        y = self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=self.max_seqlen)

        y = self.out_proj(y)
        return y
