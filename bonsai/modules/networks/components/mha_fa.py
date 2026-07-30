import torch.nn as nn
from flash_attn.modules.mha import FlashSelfAttention

from bonsai.modules.networks.components.rope_fa import FlashRotaryEmbedding


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
        self.rotary_embedding = FlashRotaryEmbedding(dim=self.head_dim)
        self.self_attn = FlashSelfAttention(
            causal=causal, attention_dropout=attention_dropout
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(
        self,
        x,
        cu_seqlens=None,
        max_seqlen=None,
        **kwargs,
    ):
        if cu_seqlens is not None:
            # Unpadded / variable-length path.
            #
            # x shape:
            # (total_valid_tokens, hidden_size)
            total_tokens, hidden_dim = x.shape

            qkv = self.Wqkv(x)
            qkv = qkv.view(
                total_tokens,
                3,
                self.num_heads,
                self.head_dim,
            )

            # Prefer the actual longest sequence in this batch.
            if max_seqlen is None:
                max_seqlen = self.max_seqlen

            qkv = self.rotary_embedding(
                qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            y = self.self_attn(
                qkv,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            y = y.reshape(total_tokens, hidden_dim)
            return self.out_proj(y)

        # Existing padded path.
        batch_size, seqlen, hidden_dim = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(
            batch_size,
            seqlen,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = self.rotary_embedding(
            qkv,
            cu_seqlens=None,
            max_seqlen=seqlen,
        )

        y = self.self_attn(qkv)

        y = y.reshape(batch_size, seqlen, hidden_dim)
        return self.out_proj(y)
