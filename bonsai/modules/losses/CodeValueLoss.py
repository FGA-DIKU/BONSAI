from torch import nn


class CodeValueLoss(nn.Module):
    """Code CE + weighted value MSE; empty value batches contribute 0 value loss."""

    def __init__(self, code_loss_fn: nn.Module, value_loss_weight: float = 1.0):
        super().__init__()
        self.code_loss_fn = code_loss_fn
        self.value_loss_fn = nn.MSELoss()
        self.value_loss_weight = value_loss_weight

    def forward(self, logits, val_logits, labels, val_labels):
        code_loss = self.code_loss_fn(logits, labels)
        # MLM may mask no numeric tokens in a batch → empty tensors → NaN mean MSE
        if val_logits.numel() == 0:
            value_loss = code_loss.new_zeros(())
        else:
            value_loss = self.value_loss_fn(val_logits, val_labels)
        loss = code_loss + self.value_loss_weight * value_loss
        return loss
