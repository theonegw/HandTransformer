## 🌟简介
这是一份手动实现的 Transfomer 模型，其中包含 multi-head self-attention、position-wise FFN、残差网络+LayerNorm和位置编码。

## 🛠️Installation
### Prerequires
+ Linux
+ python=3.9
+ pytoch
+ CUDA

### Environment Setup
克隆环境
```bash
git clone https://github.com/theonegw/HandTransformer
```
**Setp1:** 下载miniconda。

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
### 数据集构建
数据集的树形结构如下（数据集的地址为 `data/train`）
```
```
### 训练
运行训练脚本 `scripts/train.sh`
```bash
sh scripts/train.sh
```

## 📺test
### 准备检查点

### 测试
运行测试脚本 `scripts/test.sh`
```bash
sh scripts/test.sh
```

