import torch.nn as nn
import torch
import math
from torch import Tensor


class PositionalEncoder(nn.Module):
    def __init__(
        self,
        d_model : int = 512,
        dropout : float = 0.1,
        max_len : int = 5000
    )->None:
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        # 偶正 奇余
        pe[:, 0, 0::2] = torch.sin(position * div_term)  
        pe[:, 0, 1::2] = torch.cos(position * div_term)  
        self.register_buffer('pe', pe)

    def forward(self, x : Tensor)-> Tensor:
        # 取前seq_len位置的编码
        # x 的形状(seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x) 