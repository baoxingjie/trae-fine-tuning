# Qwen大模型微调与评测项目

本项目使用MindSpeed框架对Qwen大模型进行微调，并对训练结果进行全面评测。

## 项目结构

```
大模型训练/
├── README.md                 # 项目说明文档
├── requirements.txt          # 依赖包列表
├── config/                   # 配置文件目录
│   ├── training_config.yaml  # 训练配置
│   └── eval_config.yaml      # 评测配置
├── data/                     # 数据目录
│   ├── download_data.py      # 数据下载脚本
│   └── preprocess.py         # 数据预处理脚本
├── scripts/                  # 脚本目录
│   ├── train.py             # 训练脚本
│   ├── evaluate.py          # 评测脚本
│   └── inference.py         # 推理脚本
├── models/                   # 模型保存目录
└── results/                  # 结果保存目录
    ├── logs/                # 训练日志
    ├── checkpoints/         # 模型检查点
    └── evaluation/          # 评测结果
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- MindSpeed
- Transformers
- Datasets
- CUDA 11.8+ (GPU训练)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

```bash
python data/download_data.py
```

### 3. 开始训练

```bash
python scripts/train.py --config config/training_config.yaml
```

### 4. 模型评测

```bash
python scripts/evaluate.py --config config/eval_config.yaml
```

## 数据集说明

本项目使用以下HuggingFace上的常用数据集：

- **alpaca_gpt4_data**: 高质量指令微调数据集
- **OpenAssistant/oasst1**: 多轮对话数据集
- **tatsu-lab/alpaca**: Stanford Alpaca数据集
- **yahma/alpaca-cleaned**: 清洗后的Alpaca数据集

## 评测指标

- **BLEU**: 文本生成质量评估
- **ROUGE**: 摘要生成质量评估
- **Perplexity**: 语言模型困惑度
- **Human Evaluation**: 人工评估（可选）

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 根据硬件配置调整batch_size和学习率
3. 定期保存检查点以防训练中断
4. 监控训练过程中的loss变化

## 许可证

MIT License