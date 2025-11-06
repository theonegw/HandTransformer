import torch
import torch.nn as nn
import argparse
import os
from typing import List

# 从 src 模块导入
from src import *
from transformers import PreTrainedTokenizer

def greedy_decode(
    model: nn.Module, 
    src: torch.Tensor, 
    src_mask: torch.Tensor, 
    bos_idx: int, 
    eos_idx: int, 
    max_len: int, 
    device: torch.device
) -> torch.Tensor:
    """
    使用贪心解码 (Greedy Decoding) 生成序列。
    
    Args:
        model: 已加载的 Transformer 模型。
        src: 源序列张量 (S, 1)。
        src_mask: 源序列掩码 (1, 1, 1, S)。
        bos_idx: [BOS] token ID。
        eos_idx: [EOS] token ID。
        max_len: 生成的最大长度。
        device: 'cuda' 或 'cpu'。

    Returns:
        torch.Tensor: 生成的 token ID 序列 (T, 1)。
    """
    model.eval()
    
    # 1. Encoder 只运行一次
    with torch.no_grad():
        memory = model.encoder(src, src_mask) # (S, 1, D)

    # 2. Decoder 从 [BOS] 开始
    tgt_input = torch.tensor([[bos_idx]], dtype=torch.long, device=device) # (1, 1)

    for _ in range(max_len):
        # 3. 创建 Decoder 的掩码
        tgt_len = tgt_input.shape[0]
        tgt_subsequent_mask = model.get_subsequent_mask(tgt_len).to(device)
        
        # 4. Decoder 前向传播
        with torch.no_grad():
            decoder_output = model.decoder(
                tgt_input, 
                memory, 
                src_mask, 
                tgt_subsequent_mask
            )
            output_logits = model.fc_out(decoder_output) # (T, 1, V)

        # 5. 取最后一个 token 的 logits，并使用 argmax
        last_token_logits = output_logits[-1, 0, :] # (V)
        next_token_id = last_token_logits.argmax()
        
        # 6. 将新 token 拼接到输入序列中
        next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        tgt_input = torch.cat((tgt_input, next_token_tensor), dim=0)
        
        # 7. 如果是 [EOS] 则停止
        if next_token_id.item() == eos_idx:
            break
            
    return tgt_input

def predict_summary(
    model: nn.Module, 
    tokenizer: PreTrainedTokenizer, 
    sentence: str, 
    device: torch.device, 
    args: argparse.Namespace
) -> str:
    """
    对输入的句子生成摘要。
    
    Args:
        model: 已加载的 Transformer 模型。
        tokenizer: 已加载的 tokenizer。
        sentence:
        device: 'cuda' 或 'cpu'。
        args: 包含超参数的对象。

    Returns:
        str: 生成的摘要文本。
    """
    model.eval()

    # 1. 准备输入
    src_tokens = tokenizer.encode(sentence, max_length=args.max_seq_len, truncation=True)
    src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1).to(device) # (S, 1)
    
    # 2. 创建掩码
    pad_idx = tokenizer.pad_token_id
    src_mask = model.get_src_mask(src_tensor, pad_idx).to(device) # (1, 1, 1, S)
    
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id

    # 3. 执行贪心解码
    generated_ids = greedy_decode(
        model, 
        src_tensor, 
        src_mask, 
        bos_idx, 
        eos_idx, 
        args.max_gen_len, 
        device
    )
    
    # 4. 解码为文本
    output_ids = generated_ids.squeeze().tolist()
    if output_ids and output_ids[0] == bos_idx:
        output_ids = output_ids[1:]
    if output_ids and output_ids[-1] == eos_idx:
        output_ids = output_ids[:-1]
        
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary

def define_predict_parser() -> argparse.ArgumentParser:
    """
    定义 predict.py 的命令行参数。
    """
    parser = argparse.ArgumentParser(description="Transformer 推理脚本")
    
    # *** 新增 ***：用于传入句子的参数
    parser.add_argument("--sentence", type=str, required=True, help="需要进行摘要的源句子。")
    
    # 加载训练时的模型配置
    parser.add_argument("--d_model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # 推理特定参数
    parser.add_argument("--model_path", type=str, default="models/hand_transformer.pt", help="已保存模型的路径")
    parser.add_argument("--max_seq_len", type=int, default=256, help="输入的最大截断长度")
    parser.add_argument("--max_gen_len", type=int, default=50, help="生成摘要的最大长度")
    
    return parser

def main(args: argparse.Namespace) -> None:
    """
    主推理流程 (非交互式)。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载 Tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    # 2. 加载模型架构
    model = Transformer(
        src_vocab = vocab_size,
        target_vocab = vocab_size,
        d_model = args.d_model,
        num_layer = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        dropout = args.dropout
    ).to(device)
    
    # 3. 加载训练好的权重
    if not os.path.exists(args.model_path):
        print(f"模型文件未找到: {args.model_path}")
        return
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("模型权重加载成功。")

    # 4. *** 修改 ***：直接处理传入的句子
    print(f"\n[源句子]:\n{args.sentence}")
    
    summary = predict_summary(model, tokenizer, args.sentence, device, args)
    
    print("\n[生成的摘要]:")
    print(summary)
            
if __name__ == "__main__":
    parser = define_predict_parser()
    args = parser.parse_args()
    main(args)