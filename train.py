import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, PreTrainedTokenizer
from tqdm import tqdm
import argparse
import os
from typing import Tuple, List

# ----------------- 导入 -----------------
import evaluate as hf_evaluate # 使用别名，避免函数冲突
# -------------------------------------------

# 从 src 模块导入你的模型和我们刚创建的工具
from src.model import Transformer
from src.data_loader import get_tokenizer, get_dataloaders
from src.utils import set_seed, plot_loss_curves, count_parameters, plot_metric_curves


# -----------------------------------------------------------------
# 复制: 从 test.py 复制 greedy_decode 函数
# -----------------------------------------------------------------
@torch.no_grad() # 确保在推理时没有梯度
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
    (此函数与 test.py 中的版本相同)
    
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
    model.eval() # 确保模型处于评估模式
    
    # 1. Encoder 只运行一次
    memory = model.encoder(src, src_mask) # (S, 1, D)

    # 2. Decoder 从 [BOS] 开始
    tgt_input = torch.tensor([[bos_idx]], dtype=torch.long, device=device) # (1, 1)

    for _ in range(max_len):
        # 3. 创建 Decoder 的掩码
        tgt_len = tgt_input.shape[0]
        tgt_subsequent_mask = model.get_subsequent_mask(tgt_len).to(device)
        
        # 4. Decoder 前向传播
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
# -----------------------------------------------------------------
# 复制结束
# -----------------------------------------------------------------


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
    执行一个训练 epoch。 (此函数保持不变)
    """
    model.train()
    epoch_loss = 0
    
    pbar = tqdm(dataloader, desc="Training Epoch", leave=False)
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :].reshape(-1)

        optimizer.zero_grad()
        output = model(src, tgt_input, pad_idx=pad_idx)
        output_flat = output.reshape(-1, output.shape[-1])
        loss = criterion(output_flat, tgt_output)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
    return epoch_loss / len(dataloader)

# -----------------------------------------------------------------
# 函数修改: evaluate (返回 ROUGE-1 和 ROUGE-2)
# -----------------------------------------------------------------
def evaluate(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module,
    tokenizer: PreTrainedTokenizer, 
    device: torch.device, 
    pad_idx: int,
    args: argparse.Namespace
) -> Tuple[float, float, float]: # <-- 修改返回类型
    """
    在验证集上评估模型，同时计算损失、ROUGE-1 和 ROUGE-2。
    
    Returns:
        Tuple[float, float, float]: (avg_loss, rouge1_score, rouge2_score)。
    """
    model.eval()
    epoch_loss = 0
    
    # 1. 初始化 ROUGE 评价指标
    rouge_metric = hf_evaluate.load("rouge")
    
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id

    generated_summaries = []
    reference_summaries = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation Epoch", leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            # --- 步骤 1: 计算损失 (和以前一样) ---
            tgt_input = tgt[:-1, :]
            tgt_output = tgt[1:, :].reshape(-1)
            output = model(src, tgt_input, pad_idx=pad_idx)
            output_flat = output.reshape(-1, output.shape[-1])
            loss = criterion(output_flat, tgt_output)
            epoch_loss += loss.item()

            # --- 步骤 2: 生成摘要用于 ROUGE 计算 ---
            batch_size = src.shape[1]
            for i in range(batch_size):
                src_item = src[:, i:i+1]
                tgt_item = tgt[:, i:i+1]
                src_mask = model.get_src_mask(src_item, pad_idx).to(device)

                generated_ids = greedy_decode(
                    model, src_item, src_mask, bos_idx, eos_idx, 
                    args.max_gen_len, device
                )
                
                pred_ids = generated_ids.squeeze().tolist()
                if pred_ids and pred_ids[0] == bos_idx: pred_ids = pred_ids[1:]
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                
                ref_ids = tgt_item.squeeze().tolist()
                if ref_ids and ref_ids[0] == bos_idx: ref_ids = ref_ids[1:]
                if ref_ids and ref_ids[-1] == eos_idx: ref_ids = ref_ids[:-1]
                ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

                generated_summaries.append(pred_text)
                reference_summaries.append(ref_text)

    # --- 步骤 3: 计算 ROUGE (同时获取 R-1 和 R-2) ---
    rouge_metric.add_batch(predictions=generated_summaries, references=reference_summaries)
    final_rouge_scores = rouge_metric.compute()
    
    rouge1_score = final_rouge_scores['rouge1'] 
    rouge2_score = final_rouge_scores['rouge2'] 
    
    avg_loss = epoch_loss / len(dataloader)
    
    return avg_loss, rouge1_score, rouge2_score # <-- 修改返回
# -----------------------------------------------------------------
# 修改结束
# -----------------------------------------------------------------


def define_arg_parser() -> argparse.ArgumentParser:
    """
    定义脚本的命令行参数。(此函数保持不变)
    """
    parser = argparse.ArgumentParser(description="从零实现 Transformer 训练脚本")
    
    parser.add_argument("--d_model", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--d_ff", type=int, default=512, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--model_save_path", type=str, default="models/hand_transformer.pt", help="Path to save the model")
    
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length for truncation")
    parser.add_argument("--train_subset_size", type=int, default=10000, help="Subset size for training")
    parser.add_argument("--val_subset_size", type=int, default=500, help="Subset size for validation")
    
    parser.add_argument("--max_gen_len", type=int, default=50, help="验证时生成摘要的最大长度")
    
    return parser

def main(args: argparse.Namespace) -> None:
    """
    主训练流程。
    """
    # 0. 准备
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
        
    print(f"词汇表大小 (含特殊 token): {vocab_size}")
    print(f"Padding ID: {pad_idx}")

    # 2. 初始化模型
    model = Transformer(
        src_vocab = vocab_size,
        target_vocab = vocab_size,
        d_model = args.d_model,
        num_layer = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        dropout = args.dropout
    ).to(device)

    count_parameters(model)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    num_training_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(num_training_steps * 0.1)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # 4. 训练循环 (修改)
    train_losses, val_losses = [], []
    val_rouge1_scores = [] # <-- 新增 ROUGE-1 列表
    val_rouge2_scores = [] # <-- 新增 ROUGE-2 列表
    
    best_val_rouge1 = float('-inf') # <-- 决策: 仍然基于 ROUGE-1 保存

    print("--- 开始训练 ---")
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, 
            device, pad_idx, args.clip_grad
        )
        
        # <-- 修改: 现在返回三个值
        val_loss, val_rouge1, val_rouge2 = evaluate(
            model, val_loader, criterion, tokenizer, 
            device, pad_idx, args
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rouge1_scores.append(val_rouge1) # <-- 新增
        val_rouge2_scores.append(val_rouge2) # <-- 新增
        
        # <-- 修改: 打印所有指标
        print(f"Epoch {epoch+1:02}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val ROUGE-1: {val_rouge1:.4f} | " # <-- 修改
              f"Val ROUGE-2: {val_rouge2:.4f}")  # <-- 新增

        # <-- 修改: 仍然基于 ROUGE-1 保存
        if val_rouge1 > best_val_rouge1:
            best_val_rouge1 = val_rouge1
            torch.save(model.state_dict(), args.model_save_path)
            print(f"验证 ROUGE-1 提升，已保存最佳模型至: {args.model_save_path}")
            
    print("--- 训练完成 ---")

    # 5. 保存最终模型和训练曲线
    plot_loss_curves(train_losses, val_losses, save_path="results/training_loss_curve.png")
    
    # <-- 新增: 绘制 ROUGE-1 曲线
    plot_metric_curves(val_rouge1_scores, "ROUGE-1", save_path="results/validation_rouge1_curve.png")
    # <-- 新增: 绘制 ROUGE-2 曲线
    plot_metric_curves(val_rouge2_scores, "ROUGE-2", save_path="results/validation_rouge2_curve.png")

if __name__ == "__main__":
    parser = define_arg_parser()
    args = parser.parse_args()
    main(args)