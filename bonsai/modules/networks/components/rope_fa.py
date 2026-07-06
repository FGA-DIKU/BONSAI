"""Rotary embedding supporting both padded and packed-varlen qkv

Padded mode (cu_seqlens=None):
            qkv (batch, seqlen, 3, nheads, headdim), cu_seqlens=None
            -> parent RotaryEmbedding.forward

Varlen mode (cu_seqlens is not None):
            qkv (total_tokens, 3, nheads, headdim) + cu_seqlens (batch+1,)
            -> apply_rotary_emb (the only public varlen-aware entry point)
"""

import torch
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb


class FlashRotaryEmbedding(RotaryEmbedding):
    def forward(
        self, qkv: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
    ) -> torch.Tensor:
        if cu_seqlens is None:
            assert qkv.dim() == 5, (
                f"padded qkv must be (batch, seqlen, 3, nheads, headdim), got {tuple(qkv.shape)}"
            )
            return super().forward(qkv, max_seqlen=max_seqlen)

        assert qkv.dim() == 4, (
            f"varlen qkv must be (total_tokens, 3, nheads, headdim), got {tuple(qkv.shape)}"
        )
        assert max_seqlen is not None, "varlen mode requires max_seqlen"
        self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)

        q = apply_rotary_emb(
            qkv[:, 0],
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        k = apply_rotary_emb(
            qkv[:, 1],
            self._cos_cached,
            self._sin_cached,
            interleaved=self.interleaved,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return torch.stack([q, k, qkv[:, 2]], dim=1)
