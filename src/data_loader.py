import torch
import os
import argparse
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import Tuple, Dict, List

def get_tokenizer(model_name: str = "bert-base-cased") -> PreTrainedTokenizer:
    """
    加载预训练的 Tokenizer 并添加自定义的特殊 token。
    
    Args:
        model_name (str): Hugging Face Hub 上的 tokenizer 模型名称。

    Returns:
        PreTrainedTokenizer: 配置好特殊 token 的 tokenizer 实例。
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    special_tokens = {
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "pad_token": "[PAD]",
        "unk_token": "[UNK]"
    }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

class CollateFn:
    """
    自定义的 CollateFn 类，用于 DataLoader。
    它处理 padding 并将数据格式化为 (Seq_Len, Batch_Size)，
    以匹配你的 'batch_first=False' 模型设计。
    """
    def __init__(self, pad_idx: int, bos_idx: int, eos_idx: int):
        """
        初始化 CollateFn。
        
        Args:
            pad_idx (int): [PAD] token 的 ID。
            bos_idx (int): [BOS] token 的 ID。
            eos_idx (int): [EOS] token 的 ID。
        """
        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __call__(self, batch: List[Dict[str, List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理一个 batch 的数据。
        
        Args:
            batch (List[Dict[str, List[int]]]): 
                一个列表，其中每个元素是 {"src_ids": [...], "tgt_ids": [...]}。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                (src_padded, tgt_padded)，形状均为 (Seq_Len, Batch_Size)。
        """
        src_list, tgt_list = [], []

        for item in batch:
            # 源序列 (article)
            src_tensor = torch.tensor(item['src_ids'], dtype=torch.long)
            
            # 目标序列 (highlights)，手动添加 [BOS] 和 [EOS]
            tgt_tensor = torch.tensor(
                [self.bos_idx] + item['tgt_ids'] + [self.eos_idx], 
                dtype=torch.long
            )
            
            src_list.append(src_tensor)
            tgt_list.append(tgt_tensor)

        # pad_sequence 默认 batch_first=False，完美匹配你的模型
        src_padded = pad_sequence(src_list, batch_first=False, padding_value=self.pad_idx)
        tgt_padded = pad_sequence(tgt_list, batch_first=False, padding_value=self.pad_idx)

        return src_padded, tgt_padded

def get_dataloaders(
    args: argparse.Namespace, 
    tokenizer: PreTrainedTokenizer
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    加载、预处理 CNN/DailyMail 数据集，并创建 DataLoaders。
    
    Args:
        args (argparse.Namespace): 包含所有超参数的对象 (如 batch_size, max_seq_len 等)。
        tokenizer (PreTrainedTokenizer): 已初始化的 tokenizer。

    Returns:
        Tuple[DataLoader, DataLoader, int, int]:
            (train_loader, val_loader, pad_idx, vocab_size)。
            如果失败则返回 (None, None, None, None)。
    """
    # 1. 加载 CNN/DailyMail (基于你的 test.ipynb)
    try:
        dataset = load_dataset("cnn_dailymail", "3.0.0")
    except Exception as e:
        print(f"从网络加载 cnn_dailymail 失败: {e}.")
        # 尝试使用你在 test.ipynb 中发现的本地缓存
        cache_dir = "/home/gongwan/.cache/huggingface/datasets"
        print(f"尝试从本地缓存加载: {cache_dir}")
        if os.path.exists(cache_dir):
            try:
                dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir=cache_dir)
            except Exception as e_cache:
                print(f"从缓存加载失败: {e_cache}. 请确保数据集已完全下载。")
                return None, None, None, None
        else:
            print(f"缓存目录 {cache_dir} 不存在。请在网络连接良好的环境中运行一次。")
            return None, None, None, None
            
    # 2. 截取小子集 (按作业要求)
    dataset['train'] = dataset['train'].select(range(args.train_subset_size))
    dataset['validation'] = dataset['validation'].select(range(args.val_subset_size))

    # 3. 定义预处理函数 (仅分词，不 padding)
    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """对一个 batch 的 'article' 和 'highlights' 进行分词。"""
        inputs = tokenizer(
            examples["article"], 
            max_length=args.max_seq_len, 
            truncation=True, 
            padding=False
        )
        
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["highlights"], 
                max_length=args.max_seq_len, 
                truncation=True, 
                padding=False
            )
            
        return {"src_ids": inputs.input_ids, "tgt_ids": labels.input_ids}

    # 4. 应用预处理 (使用多进程)
    print("正在对数据集进行分词...")
    processed_datasets = dataset.map(
        preprocess_function, 
        batched=True, 
        num_proc=os.cpu_count(),
        remove_columns=["article", "highlights", "id"]
    )

    # 5. 创建 CollateFn 实例
    pad_idx = tokenizer.pad_token_id
    bos_idx = tokenizer.bos_token_id
    eos_idx = tokenizer.eos_token_id
    vocab_size = len(tokenizer) 
    
    collate_fn = CollateFn(
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx
    )

    # 6. 创建 DataLoader
    train_loader = DataLoader(
        processed_datasets["train"], 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        processed_datasets["validation"], 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print("DataLoader 创建完毕。")
    return train_loader, val_loader, pad_idx, vocab_size