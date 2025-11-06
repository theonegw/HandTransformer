import torch
import torch.nn as nn
from torch import Tensor
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model : int = 512,
        num_heads : int = 8,
        dropout : float = 0.1
    )->None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头的维度
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    def scaled_dot_product_attention(
        self,
        q : Tensor,
        k : Tensor,
        v : Tensor,
        mask : Tensor = None,
        dropout : nn.Dropout = None
    )->tuple[Tensor, Tensor]:
        att_score = torch.matmul(q, k.transpose(-2, -1) / math.sqrt(self.d_k)) 
        if mask is not None:
            # 形状为(batch, num_heads, seq_len, d_k)
            att_score = att_score.masked_fill(mask == 0, -1e9)
        
        att_weight = F.softmax(att_score, dim = -1)

        if dropout is not None:
            att_weight = dropout(att_weight)
        
        output = torch.matmul(att_weight, v)
        return output, att_weight

    def forward(self, q : Tensor, k : Tensor, v : Tensor, mask : Tensor = None)->Tensor:
        batch_size = q.size(1)
        
        # (seq_len, batch, d_model) -> (seq_len, batch, d_model)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 拆分多头 (seq_len, batch, d_model)-> (batch, num_head, seq_len , d_k)
        q = q.view(-1, batch_size, self.num_heads, self.d_k).transpose(0,1).transpose(2,1)
        k = k.view(-1, batch_size, self.num_heads, self.d_k).transpose(0,1).transpose(2,1)
        v = v.view(-1, batch_size, self.num_heads, self.d_k).transpose(0,1).transpose(2,1)

        # ()
        att_output, _ = self.scaled_dot_product_attention(q, k, v, mask)

        # 合并头
        output = att_output.transpose(1, 2).contiguous().reshape(batch_size, -1, self.num_heads * self.d_k)
        output = output.transpose(0, 1)
        
        return self.w_o(output)
