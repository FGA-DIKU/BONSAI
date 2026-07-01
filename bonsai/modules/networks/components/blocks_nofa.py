import torch.nn as nn
import torch.nn.functional as F

from bonsai.modules.networks.components.rope import RotaryPositionalEmbeddings


class SDPA(nn.Module):
    def __init__(self, causal, dropout):
        super().__init__()
        self.causal = causal
        self.dropout = dropout

    def forward(self, q, k, v, attn_mask=None):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and attn_mask is None,
        )


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, bias, max_seqlen, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, bias=bias)
        self.mha = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            max_seqlen=max_seqlen,
            causal=causal,
        )
        self.resid_dropout = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(hidden_size, bias=bias)
        self.mlp = Mlp(hidden_size, bias1=bias, bias2=bias)

    def forward(self, x, attn_mask=None, **kwargs):
        # LN -> MHA -> Dropout -> Add -> LN -> MLP -> Dropout -> Add
        x = x + self.resid_dropout(self.mha(self.ln1(x), attn_mask=attn_mask))
        x = x + self.resid_dropout(self.mlp(self.ln2(x)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, bias, max_seqlen, causal):
        super().__init__()
        assert hidden_size % num_heads == 0, (
            f"Hidden size {hidden_size} must be divisible by num_heads {num_heads} "
        )
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seqlen = max_seqlen

        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.rotary_embedding = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=max_seqlen
        )
        self.self_attn = SDPA(causal=causal, dropout=dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x, attn_mask=None):
        bs, seqlen, hidden_dim = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(
            bs, seqlen, 3, self.num_heads, self.head_dim
        )  # (bs, seqlen, dim) -> (bs, seqlen, 3, nh, hd)

        q, k, v = qkv.unbind(2)  # each (bs, seqlen, nh, hd)
        q = self.rotary_embedding(q)  # torchtune RoPE wants (bs, s, nh, hd)
        k = self.rotary_embedding(k)

        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (bs, nh, seqlen, hd)

        y = self.self_attn(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(bs, seqlen, -1)
        y = self.out_proj(y)
        return y


class Mlp(nn.Module):
    def __init__(self, hidden_size, bias1, bias2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size, bias=bias1)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size, bias=bias2)
        self.activation = nn.GELU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y
