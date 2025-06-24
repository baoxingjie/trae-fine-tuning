#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用示例脚本
展示如何使用本项目进行Qwen大模型微调和评测
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from scripts.train import QwenTrainer
from scripts.evaluate import ModelEvaluator
from scripts.inference import QwenInference
from data.download_data import DatasetDownloader
from data.preprocess import DataPreprocessor


def example_1_basic_training():
    """
    示例1: 基础训练流程
    """
    print("\n=== 示例1: 基础训练流程 ===")
    
    # 1. 下载数据
    print("\n步骤1: 下载数据")
    downloader = DatasetDownloader()
    downloader.download_dataset("alpaca_gpt4_data")
    
    # 2. 预处理数据
    print("\n步骤2: 预处理数据")
    preprocessor = DataPreprocessor()
    # 这里需要实际的数据集路径
    # processed_dataset = preprocessor.preprocess_dataset(dataset)
    
    # 3. 训练模型
    print("\n步骤3: 训练模型")
    trainer = QwenTrainer("config/training_config.yaml")
    # trainer.run()  # 实际训练
    
    print("基础训练流程示例完成")


def example_2_custom_config():
    """
    示例2: 自定义配置训练
    """
    print("\n=== 示例2: 自定义配置训练 ===")
    
    # 创建自定义配置
    custom_config = {
        'model': {
            'model_name_or_path': 'Qwen/Qwen-7B-Chat',
            'trust_remote_code': True,
            'torch_dtype': 'bfloat16'
        },
        'training': {
            'num_train_epochs': 2,
            'per_device_train_batch_size': 2,
            'learning_rate': 1e-5,
            'output_dir': './results/custom_checkpoints'
        },
        'lora': {
            'use_lora': True,
            'r': 8,
            'lora_alpha': 16
        }
    }
    
    print("自定义配置创建完成")
    print(f"训练轮数: {custom_config['training']['num_train_epochs']}")
    print(f"批次大小: {custom_config['training']['per_device_train_batch_size']}")
    print(f"学习率: {custom_config['training']['learning_rate']}")
    print(f"LoRA rank: {custom_config['lora']['r']}")


def example_3_batch_inference():
    """
    示例3: 批量推理
    """
    print("\n=== 示例3: 批量推理 ===")
    
    # 模拟已训练好的模型路径
    model_path = "./results/checkpoints"
    
    if not Path(model_path).exists():
        print(f"模型路径不存在: {model_path}")
        print("请先完成模型训练")
        return
    
    # 初始化推理器
    # inference = QwenInference(model_path)
    
    # 批量推理示例
    test_inputs = [
        "请解释什么是机器学习？",
        "如何提高深度学习模型的性能？",
        "什么是Transformer架构？",
        "请介绍一下注意力机制。",
        "如何防止模型过拟合？"
    ]
    
    print(f"准备进行批量推理，共 {len(test_inputs)} 个问题")
    
    # responses = inference.batch_inference(test_inputs)
    
    # 显示结果
    # for i, (question, answer) in enumerate(zip(test_inputs, responses), 1):
    #     print(f"\n问题 {i}: {question}")
    #     print(f"回答: {answer}")
    
    print("批量推理示例准备完成")


def example_4_comprehensive_evaluation():
    """
    示例4: 综合评测
    """
    print("\n=== 示例4: 综合评测 ===")
    
    # 检查模型是否存在
    model_path = "./results/checkpoints"
    if not Path(model_path).exists():
        print(f"模型路径不存在: {model_path}")
        print("请先完成模型训练")
        return
    
    # 初始化评测器
    # evaluator = ModelEvaluator("config/eval_config.yaml")
    
    # 运行评测
    # evaluator.run_evaluation()
    
    print("综合评测示例准备完成")
    print("评测将包括以下指标:")
    print("- BLEU分数")
    print("- ROUGE分数")
    print("- BERTScore")
    print("- 困惑度")
    print("- 多样性指标")


def example_5_interactive_demo():
    """
    示例5: 交互式演示
    """
    print("\n=== 示例5: 交互式演示 ===")
    
    # 模拟交互式对话
    conversation_examples = [
        {
            "user": "你好，请介绍一下自己。",
            "assistant": "你好！我是基于Qwen架构微调的AI助手，专门为了更好地理解和回应中文对话而训练。我可以帮助您解答问题、进行对话交流、提供信息等。有什么我可以帮助您的吗？"
        },
        {
            "user": "请解释一下什么是大语言模型？",
            "assistant": "大语言模型（Large Language Model, LLM）是一种基于深度学习的人工智能模型，具有以下特点：\n\n1. **规模庞大**: 通常包含数十亿到数千亿个参数\n2. **预训练**: 在大量文本数据上进行无监督学习\n3. **通用能力**: 可以处理多种自然语言处理任务\n4. **涌现能力**: 在达到一定规模后展现出意想不到的能力\n\n代表性的大语言模型包括GPT系列、BERT、T5、PaLM等。这些模型通过学习人类语言的模式和规律，能够生成连贯、有意义的文本。"
        },
        {
            "user": "如何评估大语言模型的性能？",
            "assistant": "评估大语言模型性能通常从以下几个维度进行：\n\n**自动评测指标:**\n- BLEU: 衡量生成文本与参考文本的相似度\n- ROUGE: 主要用于摘要任务的评估\n- BERTScore: 基于语义相似度的评测\n- 困惑度(Perplexity): 衡量模型对文本的预测能力\n\n**任务特定评测:**\n- 问答准确率\n- 文本分类F1分数\n- 代码生成通过率\n- 数学推理正确率\n\n**人工评估:**\n- 流畅性\n- 相关性\n- 有用性\n- 安全性\n\n本项目实现了多种自动评测指标，可以全面评估模型性能。"
        }
    ]
    
    print("交互式对话示例:")
    for i, turn in enumerate(conversation_examples, 1):
        print(f"\n--- 对话轮次 {i} ---")
        print(f"用户: {turn['user']}")
        print(f"助手: {turn['assistant']}")
    
    print("\n要启动真实的交互式对话，请运行:")
    print("python scripts/inference.py --model-path ./results/checkpoints --interactive")


def example_6_data_analysis():
    """
    示例6: 数据分析
    """
    print("\n=== 示例6: 数据分析 ===")
    
    # 模拟数据统计
    dataset_stats = {
        "alpaca_gpt4_data": {
            "samples": 52000,
            "avg_length": 256,
            "language": "English",
            "type": "Instruction-following"
        },
        "oasst1": {
            "samples": 161000,
            "avg_length": 312,
            "language": "Multilingual",
            "type": "Conversational"
        },
        "chinese_alpaca": {
            "samples": 52000,
            "avg_length": 198,
            "language": "Chinese",
            "type": "Instruction-following"
        }
    }
    
    print("数据集统计信息:")
    print(f"{'数据集':<20} {'样本数':<10} {'平均长度':<10} {'语言':<12} {'类型'}")
    print("-" * 70)
    
    for name, stats in dataset_stats.items():
        print(f"{name:<20} {stats['samples']:<10} {stats['avg_length']:<10} {stats['language']:<12} {stats['type']}")
    
    total_samples = sum(stats['samples'] for stats in dataset_stats.values())
    print(f"\n总样本数: {total_samples:,}")
    
    print("\n数据分布建议:")
    print("- 训练集: 90% (用于模型学习)")
    print("- 验证集: 5% (用于超参数调优)")
    print("- 测试集: 5% (用于最终评估)")


def example_7_monitoring_and_logging():
    """
    示例7: 监控和日志
    """
    print("\n=== 示例7: 监控和日志 ===")
    
    print("训练监控工具:")
    print("1. TensorBoard: 实时查看训练指标")
    print("   启动命令: tensorboard --logdir ./results/logs")
    print("   访问地址: http://localhost:6006")
    
    print("\n2. Weights & Biases (可选):")
    print("   - 在线实验跟踪")
    print("   - 超参数优化")
    print("   - 模型版本管理")
    
    print("\n关键监控指标:")
    monitoring_metrics = [
        "训练损失 (Training Loss)",
        "验证损失 (Validation Loss)",
        "学习率 (Learning Rate)",
        "梯度范数 (Gradient Norm)",
        "GPU内存使用率",
        "训练速度 (Samples/sec)"
    ]
    
    for i, metric in enumerate(monitoring_metrics, 1):
        print(f"   {i}. {metric}")
    
    print("\n日志文件位置:")
    print("- 训练日志: ./results/logs/training.log")
    print("- 评测日志: ./results/evaluation/evaluation.log")
    print("- 系统日志: ./results/logs/system.log")


def main():
    """
    主函数：运行所有示例
    """
    print("🚀 Qwen大模型微调项目使用示例")
    print("=" * 50)
    
    examples = [
        example_1_basic_training,
        example_2_custom_config,
        example_3_batch_inference,
        example_4_comprehensive_evaluation,
        example_5_interactive_demo,
        example_6_data_analysis,
        example_7_monitoring_and_logging
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"❌ 示例执行出错: {str(e)}")
    
    print("\n" + "=" * 50)
    print("📚 更多信息:")
    print("- 项目文档: README.md")
    print("- 配置说明: config/")
    print("- 脚本使用: scripts/")
    print("- 快速开始: python quick_start.py --help")
    
    print("\n🎯 推荐的使用流程:")
    print("1. python quick_start.py --status          # 检查项目状态")
    print("2. python quick_start.py --install         # 安装依赖")
    print("3. python quick_start.py --download        # 下载数据")
    print("4. python quick_start.py --preprocess      # 预处理数据")
    print("5. python quick_start.py --train           # 训练模型")
    print("6. python quick_start.py --evaluate        # 评测模型")
    print("7. python scripts/inference.py --interactive  # 交互式对话")
    
    print("\n或者一键运行完整流程:")
    print("python quick_start.py --full")


if __name__ == "__main__":
    main()