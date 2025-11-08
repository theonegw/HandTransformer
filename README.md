## 🌟简介
这是一份手动实现的 Transfomer 模型，其中包含 multi-head self-attention、position-wise FFN、残差网络+LayerNorm和位置编码。本次项目是在数据集`CNN/DailyMail`进行实验。
## Project structure
```
handtransformer
├── results
│   ├── training_curve.png
│   ├── training_loss_curve.png
│   ├── validation_rouge1_curve.png
│   └── validation_rouge2_curve.png
├── scripts
│   ├── test.sh                     # 测试脚本
│   └── train.sh                    # 训练脚本
├── src
│   ├── __init__.py
│   ├── modules                     # 模型的各个部件
│   │   ├── __init__.py
│   │   ├── DecoderLayer.py         # 解码器
│   │   ├── EncoderLayer.py         # 编码器
│   │   ├── MultiHeadAttention.py   # 多头注意力
│   │   ├── PositionalEncoder.py    # 位置编码
│   │   ├── PositionFeedForward.py  # FFN
│   ├── data_loader.py              # 数据下载
│   ├── model.py                    # 模型定义
│   └── utils.py                    # 损失函数和其他辅助函数
├── README.md       
├── requirements.txt                # 依赖库
├── test.py                         # 测试函数
└── train.py                        # 训练函数
```
## 🛠️Installation
### Prerequires
+ Linux
+ python=3.9
+ pytoch
+ CUDA=12.0
### Required Libraries
+ datasets
+ transformers
+ matplotlib
+ numpy<2.0
+ evaluate
+ rouge_score

### Environment Setup
克隆项目
```bash
git clone https://github.com/theonegw/HandTransformer
```
**Setp1:** 下载[miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)。

或者使用命令
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Setp2:** 构建conda环境同时激活。
```bash
conda create -n handtransformer python=3.9  
conda activate handtransformer
```
**Setp3:** 下载 [pytorch](https://pytorch.org/get-started/previous-versions/) 版本
```bash
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
**Setp4:** 下载环境所需库
```bash
pip install -r requirements.txt
```
## 🏋️Training
### 数据集下载
使用的数据集为`CNN/DailyMail`，通过使用datasets库下载的：
```python
dataset = load_dataset("cnn_dailymail", "3.0.0")
```
训练时会自动下载，如果下载失败可以自己从网上下载。
### 训练
运行训练脚本 `scripts/train.sh`
```bash
sh scripts/train.sh
```

或者直接使用
```bash
CUDA_VISIBLE_DEVICES=7 python train.py \
    --d_model 128 \
    --num_heads 8 \
    --num_layers 2 \
    --d_ff 512 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --epochs 20 \
    --seed 42 \
    --train_subset_size 10000 \
    --val_subset_size 500 \
    --model_save_path "models/hand_transformer.pt"
```
+ `CUDA_VISIBLE_DEVICES`： 本地使用的显卡编号
+ `d_model`：·
+ `num_heads`：注意力的头数
+ `num_layers`：encoder-decoder的层数
+ `d_ff`：
+ `learning_rate`：学习率
+ `epochs`：迭代次数
+ `seed`：随机种子设置，用来复现效果
+ `train_subset_size`：每次训练使用的数据大小
+ `val_subset_size`：每次测试所使用的数据大小
+ `model_save_path`：模型保存地址

## 📺test

### 测试
运行测试脚本 `scripts/test.sh`
```bash
sh scripts/test.sh
```
或者直接使用
```bash
python test.py \
    --sentence "Weather forecasts predict heavy rain and strong winds moving in from the west, expected to arrive by tomorrow morning." \
    --model_path "models/hand_transformer.pt" 
```
+ `sentence`：你要进行摘要的句子
+ `model_path`：加载训练好的模型
