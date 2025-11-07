import random
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from typing import List

def set_seed(seed: int) -> None:
    """
    设置全局随机种子以确保实验可复现性。
    
    Args:
        seed (int): 用于所有库的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保CUDNN的确定性，这可能会牺牲一些性能
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def plot_loss_curves(
    train_losses: List[float], 
    val_losses: List[float], 
    save_path: str = "results/training_curves.png"
) -> None:
    """
    绘制训练和验证损失曲线并将其保存到文件。
    
    Args:
        train_losses (List[float]): 每个 epoch 的训练损失列表。
        val_losses (List[float]): 每个 epoch 的验证损失列表。
        save_path (str): 图像文件的保存路径。
    """
    # 确保 results 目录存在 (符合作业要求)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss Curves")
    plt.savefig(save_path)
    plt.close()
    print(f"训练曲线图已保存至: {save_path}")

def count_parameters(model: nn.Module) -> int:
    """
    统计模型中可训练参数的总数。
    
    Args:
        model (nn.Module): 要检查的 PyTorch 模型。

    Returns:
        int: 可训练参数的总数。
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总可训练参数量: {total_params:,}")
    return total_params

def plot_metric_curves(
    metric_scores: List[float], 
    metric_name: str, 
    save_path: str = "results/metric_curves.png"
) -> None:
    """
    绘制一个评价指标 (如 ROUGE) 在
    每个 epoch 上的变化曲线并保存。
    
    Args:
        metric_scores (List[float]): 每个 epoch 的指标分数列表。
        metric_name (str): 指标的名称 (例如 "ROUGE-2")。
        save_path (str): 图像文件的保存路径。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(metric_scores, label=f"{metric_name} Score")
    plt.xlabel("Epoch")
    plt.ylabel(f"{metric_name} Score")
    plt.legend()
    plt.title(f"Validation {metric_name} Curve")
    plt.savefig(save_path)
    plt.close()
    print(f"{metric_name} 曲线图已保存至: {save_path}")