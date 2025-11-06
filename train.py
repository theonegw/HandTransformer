import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizer
from tqdm import tqdm
import argparse
import os
from typing import Tuple
from src import *

def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module, 
    scheduler: optim.lr_scheduler._LRScheduler, 
    device: torch.device, 
    pad_idx: int, 
    clip_value: float
) -> float:
    """
    执行一个训练 epoch。
    
    Args:
        model: Transformer 模型。
        dataloader: 训练数据加载器。
        optimizer: 优化器 (AdamW)。
        criterion: 损失函数 (CrossEntropyLoss)。
        scheduler: 学习率调度器。
        device: 'cuda' 或 'cpu'。
        pad_idx: padding token 的 ID (用于损失忽略)。
        clip_value: 梯度裁剪的值。

    Returns:
        float: 这个 epoch 的平均训练损失。
    """
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for src, tgt in pbar:
        # src: [Src_Seq_Len, Batch_Size]
        # tgt: [Tgt_Seq_Len, Batch_Size] (已包含 [BOS] 和 [EOS])
        
        src = src.to(device)
        tgt = tgt.to(device)

        # 准备 decoder 的输入和标签 (Teacher Forcing)
        # 输入: [BOS], "w1", "w2", ... "wn"
        tgt_input = tgt[:-1, :]
        
        # 标签: "w1", "w2", ... "wn", [EOS]
        # (Seq_Len, Batch) -> (Seq_Len * Batch)
        tgt_output = tgt[1:, :].reshape(-1)

        # 1. 前向传播
        optimizer.zero_grad()
        
        # 你的模型内部会自动创建掩码
        # output: [Tgt_Seq_Len-1, Batch, Vocab_Size]
        output = model(src, tgt_input, pad_idx=pad_idx)
        
        # (Seq_Len, Batch, Vocab) -> (Seq_Len * Batch, Vocab)
        output_flat = output.reshape(-1, output.shape[-1])

        # 2. 计算损失 (忽略 pad_idx)
        loss = criterion(output_flat, tgt_output)
        
        # 3. 反向传播和优化
        loss.backward()
        
        # 梯度裁剪 (作业进阶要求)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        # 更新学习率
        if scheduler:
            scheduler.step()
            
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
    return epoch_loss / len(dataloader)

def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device, 
    pad_idx: int
) -> float:
    """
    在验证集上评估模型。
    
    Args:
        model: Transformer 模型。
        dataloader: 验证数据加载器。
        criterion: 损失函数。
        device: 'cuda' 或 'cpu'。
        pad_idx: padding token 的 ID。

    Returns:
        float: 这个 epoch 的平均验证损失。
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation Epoch", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :].reshape(-1)

            output = model(src, tgt_input, pad_idx=pad_idx)
            
            output_flat = output.reshape(-1, output.shape[-1])
            loss = criterion(output_flat, tgt_output)
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    return epoch_loss / len(dataloader)

def define_arg_parser() -> argparse.ArgumentParser:
    """
    定义脚本的命令行参数。
    """
    parser = argparse.ArgumentParser(description="从零实现 Transformer 训练脚本")
    
    parser.add_argument("--d_model", type=int, default=128, help="Embedding dimension ")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads ")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension ")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers ")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size ")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate ")
    
    # 其他训练参数
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility ")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value ")
    parser.add_argument("--model_save_path", type=str, default="models/hand_transformer.pt", help="Path to save the model ")
    
    # 数据集相关
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length for truncation")
    parser.add_argument("--train_subset_size", type=int, default=10000, help="Subset size for training ")
    parser.add_argument("--val_subset_size", type=int, default=500, help="Subset size for validation ")

    return parser

def main(args: argparse.Namespace) -> None:
    """
    主训练流程。
    """
    # 0. 准备 (设置随机种子和设备)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    os.makedirs("models", exist_ok=True)

    # 1. 数据准备
    tokenizer = get_tokenizer()
    train_loader, val_loader, pad_idx, vocab_size = get_dataloaders(args, tokenizer)
    
    if train_loader is None:
        print("数据加载失败，退出训练。")
        return
        
    print(f"词汇表大小: {vocab_size}")
    print(f"Padding ID: {pad_idx}")

    # 2. 初始化模型 (使用作业建议的超参)
    model = Transformer(
        src_vocab = vocab_size,
        target_vocab = vocab_size,
        d_model = args.d_model,
        num_layer = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        dropout = args.dropout
    ).to(device)

    # 统计参数 
    count_parameters(model)

    # 3. 定义损失函数和优化器
    # 忽略 padding token 的损失
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # 使用 AdamW 
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 使用学习率调度器 
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1) # 10% warmup
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 4. 训练循环
    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    print("--- 开始训练 ---")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, 
            device, pad_idx, args.clip_grad
        )
        
        val_loss = evaluate(
            model, val_loader, criterion, device, pad_idx
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1:02}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f}")

        # 保存最佳模型 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.model_save_path)
            print(f"验证损失提升，已保存最佳模型至: {args.model_save_path}")
            
    print("--- 训练完成 ---")

    # 5. 保存最终模型和训练曲线
    plot_loss_curves(train_losses, val_losses, save_path="results/training_curve.png")

if __name__ == "__main__":
    parser = define_arg_parser()
    args = parser.parse_args()
    main(args)