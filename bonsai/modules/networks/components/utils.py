"""
Attention functionality for ModernBERT.
"""

import torch


def make_attention_causal(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Converts a standard attention mask into a causal attention mask.

    This function takes an attention mask and adds causality constraints to prevent
    tokens from attending to future positions. It handles both boolean and float masks.

    Args:
        attn_mask (torch.Tensor): Input attention mask of shape [B, H, L, L] or [..., L, L],
            where B is batch size, H is number of heads, and L is sequence length.
            For float masks: 0 for attended positions, -inf for masked positions.
            For bool masks: False for attended positions, True for masked positions.

    Returns:
        torch.Tensor: Causal attention mask with the same shape and dtype as input.
            The resulting mask combines the input mask with causal constraints:
            - Float masks: Positions above diagonal are set to -inf
            - Bool masks: Positions above diagonal are set to True

    Example:
        >>> # Input mask [1, 1, 3, 3] allowing all attention
        >>> mask = torch.zeros(1, 1, 3, 3)
        >>> print(mask[0, 0])
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
        >>>
        >>> # Output causal mask
        >>> causal = _make_attention_causal(mask)
        >>> print(causal[0, 0])
        tensor([[  0., -inf, -inf],
                [  0.,   0., -inf],
                [  0.,   0.,   0.]])
    """
    # Get sequence length from last dimension
    L = attn_mask.size(-1)
    device, dtype = attn_mask.device, attn_mask.dtype

    # Create causal mask: -inf (float) or True (bool) for future positions
    neg_inf = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((L, L), neg_inf, device=device, dtype=dtype), diagonal=1
    )
    # Expand dimensions to match input mask shape for broadcasting
    for _ in range(attn_mask.ndim - 2):
        causal = causal.unsqueeze(0)
    # Combine input mask with causal constraints
    return attn_mask + causal
