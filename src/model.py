import torch.nn as nn
import torch
from torch import Tensor
from .modules import *
import math


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        src_vocab : int,
        d_model : int = 512,
        num_heads : int = 8,
        num_layer : int = 6,
        d_ff : int = 2048,
        dropout : float = 0.1
    )->None:
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab, d_model)
        self.position_encoder = PositionalEncoder(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layer)
        ])
        self.d_model = d_model
    
    def forward(self, src : Tensor, src_mask : Tensor = None)->Tensor:
        # 词嵌入 + 位置编码
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.position_encoder(src)

        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        target_vocab : int,
        d_model : int = 512,
        num_heads : int = 8,
        num_layer : int = 6,
        d_ff : int = 2048,
        dropout : float = 0.1
    )->None:
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab, d_model)
        self.position_decoder = PositionalEncoder(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layer)]
        )
        self.d_model = d_model
    
    def forward(
        self,
        target : Tensor,
        encoder_input,
        src_mask : Tensor = None,
        target_mask : Tensor = None
    )->Tensor:
        target = self.embedding(target) * math.sqrt(self.d_model)
        target = self.position_decoder(target)

        for layer in self.layers:
            target = layer(target, encoder_input, src_mask, target_mask)
        
        return target

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab : int,
        target_vocab : int, 
        d_model : int = 512,
        num_layer: int = 6,
        num_heads : int = 8,
        d_ff : int = 2048,
        dropout : float = 0.1
    )->None:
        super(Transformer, self).__init__()
        # self.src_embedding = nn.Embedding(src_vocab, d_model)

        self.encoder = TransformerEncoder(
            src_vocab = src_vocab,
            d_model = d_model,
            num_heads = num_heads,
            d_ff = d_ff,
            num_layer = num_layer,
            dropout = dropout
        )

        self.decoder = TransformerDecoder(
            target_vocab = target_vocab,
            d_model = d_model,
            num_heads = num_heads,
            d_ff = d_ff,
            num_layer = num_layer,
            dropout = dropout
        )
        self.fc_out = nn.Linear(d_model, target_vocab)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_subsequent_mask(self, seq_len)->Tensor:
        att_size = (1, seq_len, seq_len)
        subsequent_mask = torch.triu(torch.ones(att_size), diagonal=1).type(torch.uint8)
        return subsequent_mask == 0
    
    def get_src_mask(self, src : Tensor, pad_idx : int = 0)->Tensor:
        # src (src_len, batch) -> (batch, 1, 1, src_len)
        return (src != pad_idx).transpose(0, 1).unsqueeze(1).unsqueeze(2)

    def forward(
            self, 
            src : Tensor, 
            target : Tensor, 
            src_mask : Tensor = None, 
            target_mask : Tensor = None,
            pad_idx = 0
        )->Tensor:
        seq_len = src.size(0)
        target_len = target.size(0)

        if src_mask is None:
            # src_mask = self.get_subsequent_mask(seq_len)
            src_mask = self.get_src_mask(src, pad_idx).to(src.device)
        if target_mask is None:
            # target_mask = self.get_src_mask(src, pad_idx)
            # 3.1 目标序列的 [PAD] 掩码
            # (target_len, batch) -> (batch, 1, 1, target_len)
            target_pad_mask = self.get_src_mask(target, pad_idx).to(target.device) 
            
            # 3.2 目标序列的 "未来" 掩码
            # (target_len) -> (1, target_len, target_len)
            target_subsequent_mask = self.get_subsequent_mask(target_len).to(target.device)

            # 3.3 合并两个掩码
            # (batch, 1, 1, target_len) & (1, target_len, target_len) 
            # -> 广播为 (batch, 1, target_len, target_len)
            target_mask = target_pad_mask & target_subsequent_mask
            # target_pad_mask = self.get_src_mask(target, pad_idx).to(target.device)
        encoder = self.encoder(src, src_mask)
        decoder = self.decoder(target, encoder, src_mask, target_mask)

        fc_out = self.fc_out(decoder)

        return fc_out