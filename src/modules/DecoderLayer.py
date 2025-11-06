import torch.nn as nn
from torch import Tensor
from .MultiHeadAttention import MultiHeadAttention
from .PositionFeedForward import PositionFeedForward

class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model : int = 512,
        num_heads : int = 8,
        d_ff : int = 2048,
        dropout : float = 0.1 
    )->None:
        super(DecoderLayer, self).__init__()

        self.mask_att = MultiHeadAttention(d_model, num_heads, dropout)

        self.cross_att = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = PositionFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        target : Tensor,
        encoder_input : Tensor = None,
        src_mask : Tensor = None,
        target_mask : Tensor = None
    )->Tensor:
        target_shortcut = self.mask_att(target, target, target, target_mask)
        # 掩码注意力机制
        target = target + self.dropout1(target_shortcut)
        target = self.norm1(target)
       
        # 编码解码器注意力机制
        target_shortcut = self.cross_att(target, encoder_input, encoder_input, src_mask)
        target = target + self.dropout2(target_shortcut)
        target = self.norm2(target)

        # 前馈网络
        target_shortcut = self.ffn(target)
        target = target + self.dropout3(target_shortcut)
        target = self.norm3(target)

        return target
