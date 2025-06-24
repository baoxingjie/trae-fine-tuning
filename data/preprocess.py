#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
将下载的数据集转换为训练所需的格式
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, tokenizer_name: str = "Qwen/Qwen-7B-Chat", max_length: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = max_length
        
        # 数据格式转换模板
        self.format_templates = {
            "alpaca": {
                "instruction_key": "instruction",
                "input_key": "input",
                "output_key": "output",
                "template": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            },
            "oasst": {
                "text_key": "text",
                "role_key": "role",
                "template": "{text}"
            },
            "dolly": {
                "instruction_key": "instruction",
                "context_key": "context",
                "response_key": "response",
                "template": "### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
            },
            "belle": {
                "instruction_key": "instruction",
                "output_key": "output",
                "template": "Human: {instruction}\n\nAssistant: {output}"
            },
            "firefly": {
                "input_key": "input",
                "target_key": "target",
                "template": "User: {input}\n\nAssistant: {target}"
            }
        }
    
    def detect_dataset_format(self, dataset: Dataset) -> str:
        """自动检测数据集格式"""
        features = set(dataset.features.keys())
        
        # 检查Alpaca格式
        if {"instruction", "input", "output"}.issubset(features):
            return "alpaca"
        
        # 检查OASST格式
        if {"text", "role"}.issubset(features):
            return "oasst"
        
        # 检查Dolly格式
        if {"instruction", "context", "response"}.issubset(features):
            return "dolly"
        
        # 检查BELLE格式
        if {"instruction", "output"}.issubset(features):
            return "belle"
        
        # 检查Firefly格式
        if {"input", "target"}.issubset(features):
            return "firefly"
        
        # 默认返回alpaca格式
        return "alpaca"
    
    def format_sample(self, sample: Dict[str, Any], format_type: str) -> str:
        """格式化单个样本"""
        template_config = self.format_templates[format_type]
        template = template_config["template"]
        
        if format_type == "alpaca":
            instruction = sample.get(template_config["instruction_key"], "")
            input_text = sample.get(template_config["input_key"], "")
            output = sample.get(template_config["output_key"], "")
            
            if input_text.strip():
                formatted_text = template.format(
                    instruction=instruction,
                    input=input_text,
                    output=output
                )
            else:
                # 如果没有input，使用简化模板
                simplified_template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
                formatted_text = simplified_template.format(
                    instruction=instruction,
                    output=output
                )
        
        elif format_type == "oasst":
            text = sample.get(template_config["text_key"], "")
            formatted_text = template.format(text=text)
        
        elif format_type == "dolly":
            instruction = sample.get(template_config["instruction_key"], "")
            context = sample.get(template_config["context_key"], "")
            response = sample.get(template_config["response_key"], "")
            
            formatted_text = template.format(
                instruction=instruction,
                context=context,
                response=response
            )
        
        elif format_type == "belle":
            instruction = sample.get(template_config["instruction_key"], "")
            output = sample.get(template_config["output_key"], "")
            
            formatted_text = template.format(
                instruction=instruction,
                output=output
            )
        
        elif format_type == "firefly":
            input_text = sample.get(template_config["input_key"], "")
            target = sample.get(template_config["target_key"], "")
            
            formatted_text = template.format(
                input=input_text,
                target=target
            )
        
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")
        
        return formatted_text
    
    def tokenize_function(self, examples: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        """分词函数"""
        # 分词
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_overflowing_tokens=False,
        )
        
        # 设置labels（用于计算loss）
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def preprocess_dataset(self, dataset: Dataset, format_type: Optional[str] = None) -> Dataset:
        """预处理数据集"""
        if format_type is None:
            format_type = self.detect_dataset_format(dataset)
        
        print(f"检测到数据集格式: {format_type}")
        
        # 格式化文本
        def format_function(examples):
            formatted_texts = []
            for i in range(len(examples[list(examples.keys())[0]])):
                sample = {key: examples[key][i] for key in examples.keys()}
                formatted_text = self.format_sample(sample, format_type)
                formatted_texts.append(formatted_text)
            
            return {"text": formatted_texts}
        
        # 应用格式化
        formatted_dataset = dataset.map(
            format_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="格式化数据"
        )
        
        # 分词
        tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names,
            desc="分词处理"
        )
        
        return tokenized_dataset
    
    def filter_by_length(self, dataset: Dataset, min_length: int = 10, max_length: Optional[int] = None) -> Dataset:
        """根据长度过滤数据"""
        if max_length is None:
            max_length = self.max_length
        
        def length_filter(example):
            length = len(example["input_ids"])
            return min_length <= length <= max_length
        
        filtered_dataset = dataset.filter(length_filter, desc="长度过滤")
        
        print(f"过滤前: {len(dataset)} 样本")
        print(f"过滤后: {len(filtered_dataset)} 样本")
        
        return filtered_dataset
    
    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.9, val_ratio: float = 0.05, test_ratio: float = 0.05) -> DatasetDict:
        """分割数据集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须等于1"
        
        # 首先分割出训练集和其余部分
        train_test_split = dataset.train_test_split(test_size=1-train_ratio, seed=42)
        train_dataset = train_test_split["train"]
        temp_dataset = train_test_split["test"]
        
        # 再从其余部分分割出验证集和测试集
        if val_ratio > 0 and test_ratio > 0:
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_test_split = temp_dataset.train_test_split(test_size=1-val_test_ratio, seed=42)
            val_dataset = val_test_split["train"]
            test_dataset = val_test_split["test"]
            
            return DatasetDict({
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            })
        elif val_ratio > 0:
            return DatasetDict({
                "train": train_dataset,
                "validation": temp_dataset
            })
        elif test_ratio > 0:
            return DatasetDict({
                "train": train_dataset,
                "test": temp_dataset
            })
        else:
            return DatasetDict({"train": train_dataset})
    
    def save_dataset_stats(self, dataset: DatasetDict, output_dir: Path) -> None:
        """保存数据集统计信息"""
        stats = {}
        
        for split_name, split_dataset in dataset.items():
            lengths = [len(example["input_ids"]) for example in split_dataset]
            
            stats[split_name] = {
                "num_samples": len(split_dataset),
                "avg_length": sum(lengths) / len(lengths),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "median_length": sorted(lengths)[len(lengths) // 2]
            }
        
        stats_file = output_dir / "dataset_stats.yaml"
        with open(stats_file, 'w', encoding='utf-8') as f:
            yaml.safe_dump(stats, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        print(f"数据集统计信息已保存到: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="预处理训练数据")
    parser.add_argument("--input-dir", default="./data/cache", help="输入数据目录")
    parser.add_argument("--output-dir", default="./data/processed", help="输出数据目录")
    parser.add_argument("--tokenizer", default="Qwen/Qwen-7B-Chat", help="分词器名称")
    parser.add_argument("--max-length", type=int, default=2048, help="最大序列长度")
    parser.add_argument("--min-length", type=int, default=10, help="最小序列长度")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="测试集比例")
    parser.add_argument("--datasets", nargs="+", help="要处理的数据集名称")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor(
        tokenizer_name=args.tokenizer,
        max_length=args.max_length
    )
    
    # 查找可用的数据集
    if args.datasets:
        dataset_names = args.datasets
    else:
        # 自动查找所有数据集
        dataset_names = []
        for info_file in input_dir.glob("*_info.yaml"):
            dataset_name = info_file.stem.replace("_info", "")
            dataset_names.append(dataset_name)
    
    if not dataset_names:
        print("未找到任何数据集，请先运行 download_data.py 下载数据")
        return
    
    print(f"找到数据集: {dataset_names}")
    
    # 处理每个数据集
    all_datasets = []
    
    for dataset_name in dataset_names:
        print(f"\n处理数据集: {dataset_name}")
        
        try:
            # 加载数据集 - 使用datasets库直接从缓存加载
            dataset_mapping = {
                "alpaca_gpt4_data": "vicgalle/alpaca-gpt4",
                "chinese_alpaca": "shibing624/alpaca-zh", 
                "oasst1": "OpenAssistant/oasst1"
            }
            
            if dataset_name not in dataset_mapping:
                print(f"未知数据集: {dataset_name}")
                continue
                
            huggingface_path = dataset_mapping[dataset_name]
            
            try:
                from datasets import load_dataset
                dataset = load_dataset(
                    huggingface_path,
                    cache_dir=str(input_dir),
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"加载数据集 {dataset_name} 失败: {str(e)}")
                continue
            
            # 如果是DatasetDict，合并所有分割
            if isinstance(dataset, DatasetDict):
                combined_dataset = []
                for split_name, split_dataset in dataset.items():
                    combined_dataset.extend(split_dataset)
                
                from datasets import Dataset
                dataset = Dataset.from_list(combined_dataset)
            
            # 预处理数据集
            processed_dataset = preprocessor.preprocess_dataset(dataset)
            
            # 过滤长度
            filtered_dataset = preprocessor.filter_by_length(
                processed_dataset,
                min_length=args.min_length,
                max_length=args.max_length
            )
            
            all_datasets.append(filtered_dataset)
            
        except Exception as e:
            print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
            continue
    
    if not all_datasets:
        print("没有成功处理任何数据集")
        return
    
    # 合并所有数据集
    print("\n合并所有数据集...")
    from datasets import concatenate_datasets
    combined_dataset = concatenate_datasets(all_datasets)
    
    print(f"合并后总样本数: {len(combined_dataset)}")
    
    # 分割数据集
    print("\n分割数据集...")
    final_dataset = preprocessor.split_dataset(
        combined_dataset,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # 保存处理后的数据集
    print("\n保存处理后的数据集...")
    final_dataset.save_to_disk(str(output_dir / "final_dataset"))
    
    # 保存统计信息
    preprocessor.save_dataset_stats(final_dataset, output_dir)
    
    # 打印最终统计
    print("\n=== 数据预处理完成 ===")
    for split_name, split_dataset in final_dataset.items():
        print(f"{split_name}: {len(split_dataset)} 样本")
    
    print(f"\n处理后的数据集已保存到: {output_dir / 'final_dataset'}")
    print("可以开始训练了！")


if __name__ == "__main__":
    main()