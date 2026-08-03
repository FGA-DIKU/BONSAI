import torch.nn as nn

from bonsai.modules.networks.components.mha import MultiHeadAttention
from bonsai.modules.networks.components.mlp import Mlp

attn_types = {"sdpa": MultiHeadAttention}
try:
    from bonsai.modules.networks.components.mha_fa import FlashMultiHeadAttention

    attn_types["flash"] = FlashMultiHeadAttention
except Exception:
    pass


class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        dropout,
        attention_dropout,
        bias,
        max_seqlen,
        causal,
        attn_type,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, bias=bias)
        self.mha = attn_types[attn_type](
            hidden_size=hidden_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            bias=bias,
            max_seqlen=max_seqlen,
            causal=causal,
        )
        self.resid_dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_size, bias=bias)
        self.mlp = Mlp(hidden_size, bias1=bias, bias2=bias)

    def forward(self, x, attn_mask=None, cu_seqlens=None, max_seqlen=None):
        # LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add
        x = x + self.resid_dropout(
            self.mha(
                self.ln1(x),
                attn_mask=attn_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        )

        x = x + self.resid_dropout(self.mlp(self.ln2(x)))

        return x
