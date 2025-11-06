import torch.nn as nn
from torch import Tensor
from .MultiHeadAttention import MultiHeadAttention
from .PositionFeedForward import PositionFeedForward


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model : int = 512,
        num_heads : int = 8,
        d_ff : int = 2048,
        dropout: float = 0.1
    )->None:
        super(EncoderLayer, self).__init__()
        self.multi_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = PositionFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src : Tensor, src_mask : Tensor = None)->Tensor:
        # 多头注意力+ 残差连接 + 层归一化
        # 前馈网络 + 残差连接 + 层归一化
        src_shortcut = self.multi_attn(src, src, src, src_mask)
        src = src + self.dropout1(src_shortcut)

        src = self.norm1(src)

        src_shortcut = self.ffn(src)
        src = src + self.dropout2(src_shortcut)

        src = self.norm2(src)
        return src
