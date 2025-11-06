import torch.nn as nn
from torch import Tensor

class PositionFeedForward(nn.Module):
    def __init__(
        self,
        d_model : int = 512,
        d_ff : int = 2048,
        dropout : float = 0.1
    )->None:
        super(PositionFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x : Tensor)->Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
