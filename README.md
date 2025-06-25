# 大模型微调项目

本项目基于Transformers和PEFT库对Qwen大模型进行微调，支持LoRA高效微调，并提供完整的模型对比测试功能。适合在个人PC上进行大模型微调实验。

## 主要特性

✨ **高效微调**: 基于LoRA技术，显著降低训练成本和时间
🚀 **多模型支持**: 支持Qwen2系列多种规模模型
📊 **智能对比**: 内置模型对比工具，直观评估微调效果
🔧 **灵活配置**: 丰富的配置选项，适应不同硬件环境
📈 **实时监控**: TensorBoard集成，实时查看训练进度
🌐 **多数据集**: 支持多种指令微调数据集格式
💾 **自动保存**: 智能检查点管理，防止训练意外中断
🎯 **开箱即用**: 预配置的训练参数，快速开始实验

## 项目结构

```
大模型训练/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── config/                      # 配置文件目录
│   ├── training_config.yaml     # 训练配置
│   └── eval_config.yaml         # 评测配置
├── data/                        # 数据目录
│   ├── download_data.py         # 数据下载脚本
│   ├── preprocess.py            # 数据预处理脚本
│   └── cache/                   # 数据缓存目录
├── scripts/                     # 脚本目录
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评测脚本
│   ├── inference.py            # 推理脚本
│   └── model_comparison.py      # 模型对比脚本
├── docs/                        # 文档目录
│   └── model_comparison_guide.md # 模型对比指南
├── test_questions.json          # 测试问题集
├── models/                      # 模型保存目录
└── results/                     # 结果保存目录
    ├── logs/                   # 训练日志
    ├── checkpoints/            # 模型检查点
    └── evaluation/             # 评测结果
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

### 2. 数据准备

#### 下载数据集
```bash
python data/download_data.py
```

#### 数据预处理
```bash
python data/preprocess.py \
    --dataset_name chinese_alpaca \
    --tokenizer_name Qwen/Qwen2-0.5B \
    --max_length 1024 \
    --output_dir ./data/processed
```

### 3. 模型微调

#### 使用默认配置训练
```bash
python scripts/train.py --config config/training_config.yaml
```

#### 自定义训练参数
```bash
python scripts/train.py \
    --config config/training_config.yaml \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --output_dir ./results/my_experiment \
    --num_train_epochs 3 \
    --learning_rate 0.0002
```

### 4. 模型评测

```bash
python scripts/evaluate.py --config config/eval_config.yaml
```

### 5. 模型对比测试

#### 交互式对比（推荐）
```bash
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --interactive
```

#### 批量对比测试
```bash
python scripts/model_comparison.py \
    --base-model "Qwen/Qwen2-0.5B" \
    --finetuned-model "./results/checkpoints" \
    --questions-file "test_questions.json" \
    --output "comparison_results.json"
```

## 支持的模型

- **Qwen2-0.5B**: 轻量级模型，适合快速实验和资源受限环境
- **Qwen2-7B**: 标准规模模型，平衡性能和资源消耗
- **Qwen2-14B**: 大规模模型，追求最佳性能

## 数据集说明

本项目支持多种指令微调数据集：

### 主要数据集
- **chinese_alpaca**: 中文指令微调数据集（默认使用）
- **alpaca_gpt4_data**: 高质量英文指令微调数据集
- **OpenAssistant/oasst1**: 多轮对话数据集
- **tatsu-lab/alpaca**: Stanford Alpaca数据集
- **yahma/alpaca-cleaned**: 清洗后的Alpaca数据集

### 数据格式
支持的数据格式包括：
- **Alpaca格式**: instruction + input + output
- **对话格式**: 多轮对话数据
- **自定义格式**: 通过配置文件自定义数据处理逻辑

数据预处理会自动将不同格式的数据转换为统一的训练格式，并进行分词、截断等处理。

## 评测指标

- **BLEU**: 文本生成质量评估
- **ROUGE**: 摘要生成质量评估
- **Perplexity**: 语言模型困惑度
- **Human Evaluation**: 人工评估（可选）

## 配置说明

### LoRA微调配置
本项目使用LoRA（Low-Rank Adaptation）进行高效微调：
- **LoRA秩 (r)**: 8（小模型适用）
- **Alpha值**: 16
- **目标模块**: 包含所有注意力层和MLP层
- **Dropout**: 0.1（防止过拟合）

### 训练优化（个人PC适配）
- **混合精度**: 使用bfloat16提升训练效率，减少显存占用
- **梯度累积**: 4步累积，在有限显存下保持有效批次大小
- **优化器**: AdamW融合版本，充分利用现代GPU性能
- **学习率调度**: Cosine退火策略，稳定收敛
- **内存优化**: 针对个人PC显存限制进行优化
- **数据加载**: Windows环境下的多进程优化

## 注意事项

### 硬件要求

#### 推荐配置（个人PC）
- **显卡**: RTX A5000 16GB / RTX 4080 16GB / RTX 4090 24GB
- **处理器**: Intel i7-12700K 或 AMD Ryzen 7 5800X 及以上
- **内存**: 32GB DDR4-3200 或 DDR5-4800
- **存储**: 1TB NVMe SSD（PCIe 4.0）
- **电源**: 750W 80+ Gold认证

#### 最低配置
- **显卡**: RTX 3060 12GB / RTX 4060 Ti 16GB
- **处理器**: Intel i5-10400 或 AMD Ryzen 5 3600
- **内存**: 16GB DDR4
- **存储**: 500GB NVMe SSD
- **电源**: 650W 80+ Bronze认证

### 训练建议（个人PC环境）
1. **显存管理**: 根据显卡型号调整批次大小
   - RTX 4090 24GB: `per_device_train_batch_size=8`
   - RTX A5000 16GB: `per_device_train_batch_size=4`
   - RTX 3060 12GB: `per_device_train_batch_size=2`
2. **检查点保存**: 每300步自动保存，防止意外断电
3. **温度监控**: 使用MSI Afterburner等工具监控GPU温度
4. **系统优化**: 关闭不必要的后台程序，释放系统资源
5. **电源管理**: 确保电源功率充足，避免训练时断电
6. **存储优化**: 使用NVMe SSD提升数据加载速度

### 常见问题（个人PC环境）
1. **显存不足**: 
   - 减小批次大小 `per_device_train_batch_size`
   - 启用4bit量化 `use_4bit: true`
   - 使用更小的模型（如Qwen2-0.5B）
2. **训练速度慢**: 
   - 检查GPU利用率，确保达到90%+
   - 升级到更快的NVMe SSD
   - 增加系统内存减少swap使用
3. **系统卡顿**: 
   - 降低 `dataloader_num_workers` 到0
   - 关闭浏览器等占用内存的程序
4. **温度过高**: 
   - 清理显卡散热器灰尘
   - 调整机箱风扇转速
   - 降低训练强度
5. **Windows兼容性**: 
   - 使用WSL2或虚拟环境
   - 确保CUDA版本匹配
   - 关闭Windows Defender实时保护（训练期间）

## 个人PC环境特别说明

### 🖥️ 系统环境推荐
- **操作系统**: Windows 11 22H2 或 Ubuntu 22.04 LTS
- **Python版本**: 3.9-3.11（推荐3.10）
- **CUDA版本**: 11.8 或 12.1
- **驱动版本**: 最新的Game Ready或Studio驱动

### ⚡ 性能优化技巧
1. **BIOS设置**: 启用XMP/DOCP内存超频
2. **电源计划**: 设置为"高性能"模式
3. **虚拟内存**: 设置为物理内存的1.5-2倍
4. **温度控制**: 保持GPU温度在80°C以下
5. **后台程序**: 训练时关闭Chrome、游戏等占用资源的程序

### 💡 成本优化建议
- **云服务对比**: 个人PC训练成本通常比云服务低60-80%
- **电费考虑**: RTX A5000满载约300W，24小时训练约消耗7.2度电
- **硬件投资**: 一次性投资，可重复使用，适合长期研究

### 🔧 故障排除
如遇到问题，请按以下顺序检查：
1. 显卡驱动是否为最新版本
2. CUDA和PyTorch版本是否匹配
3. 显存是否充足（使用`nvidia-smi`查看）
4. 系统内存是否充足
5. 硬盘空间是否充足

## 许可证

MIT License