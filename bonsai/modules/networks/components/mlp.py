import torch.nn as nn


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
