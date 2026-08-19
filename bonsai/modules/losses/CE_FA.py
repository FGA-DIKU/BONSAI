from flash_attn.losses.cross_entropy import (
    CrossEntropyLoss as CrossEntropyLossFlash,
)
from torch import Tensor


class CE_FA(CrossEntropyLossFlash):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.view(-1, input.size(-1)), target.view(-1))
