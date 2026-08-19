from torch import Tensor, nn


class CE(nn.CrossEntropyLoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.view(-1, input.size(-1)), target.view(-1))
