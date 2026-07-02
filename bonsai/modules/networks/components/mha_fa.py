import torch.nn as nn
from flash_attn.layers.rotary import RotaryEmbedding
from flash_attn.modules.mha import FlashSelfAttention


class FlashMultiHeadAttention(nn.Module):
    def __init__(
        self, hidden_size, num_heads, attention_dropout, bias, max_seqlen, causal
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"Hidden size {hidden_size} must be divisible by num_heads {num_heads} "
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seqlen = max_seqlen

        self.Wqkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.rotary_embedding = RotaryEmbedding(dim=self.head_dim)
        self.self_attn = FlashSelfAttention(
            causal=causal, attention_dropout=attention_dropout
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, cu_seqlens=None, **kwargs):
        bs, seqlen, hidden_dim = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(
            bs, seqlen, 3, self.num_heads, self.head_dim
        )  # (bs, seqlen, dim) -> (bs, seqlen, 3, nh, hd)

        qkv = self.rotary_embedding(qkv, max_seqlen=self.max_seqlen)

        y = self.self_attn(qkv, cu_seqlens=cu_seqlens, max_seqlen=self.max_seqlen)
        y = y.reshape(bs, seqlen, -1)
        y = self.out_proj(y)
        return y
