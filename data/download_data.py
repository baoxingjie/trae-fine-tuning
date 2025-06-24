#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载脚本
从HuggingFace下载常用的大模型训练数据集
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any

from datasets import load_dataset, DatasetDict
from huggingface_hub import snapshot_download
import yaml
from tqdm import tqdm


class DatasetDownloader:
    """数据集下载器"""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 常用数据集配置
        self.datasets_config = {
            "alpaca_gpt4_data": {
                "path": "vicgalle/alpaca-gpt4",
                "description": "高质量GPT-4生成的Alpaca指令数据集",
                "size": "~52K samples",
                "language": "English"
            },
            "alpaca_cleaned": {
                "path": "yahma/alpaca-cleaned",
                "description": "清洗后的Stanford Alpaca数据集",
                "size": "~52K samples",
                "language": "English"
            },
            "oasst1": {
                "path": "OpenAssistant/oasst1",
                "description": "OpenAssistant多轮对话数据集",
                "size": "~161K samples",
                "language": "Multilingual"
            },
            "dolly_15k": {
                "path": "databricks/databricks-dolly-15k",
                "description": "Databricks Dolly指令数据集",
                "size": "~15K samples",
                "language": "English"
            },
            "self_instruct": {
                "path": "yizhongw/self_instruct",
                "description": "Self-Instruct生成的指令数据集",
                "size": "~82K samples",
                "language": "English"
            },
            "chinese_alpaca": {
                "path": "shibing624/alpaca-zh",
                "description": "中文Alpaca指令数据集",
                "size": "~52K samples",
                "language": "Chinese"
            },
            "belle_train": {
                "path": "BelleGroup/train_1M_CN",
                "description": "BELLE中文指令数据集",
                "size": "~1M samples",
                "language": "Chinese"
            },
            "firefly_train": {
                "path": "YeungNLP/firefly-train-1.1M",
                "description": "Firefly中文对话数据集",
                "size": "~1.1M samples",
                "language": "Chinese"
            }
        }
    
    def list_available_datasets(self) -> None:
        """列出所有可用的数据集"""
        print("\n=== 可用数据集列表 ===")
        print(f"{'数据集名称':<20} {'大小':<15} {'语言':<12} {'描述'}")
        print("-" * 80)
        
        for name, config in self.datasets_config.items():
            print(f"{name:<20} {config['size']:<15} {config['language']:<12} {config['description']}")
        print()
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> DatasetDict:
        """下载指定数据集"""
        if dataset_name not in self.datasets_config:
            raise ValueError(f"未知数据集: {dataset_name}. 可用数据集: {list(self.datasets_config.keys())}")
        
        config = self.datasets_config[dataset_name]
        dataset_path = config["path"]
        
        print(f"\n正在下载数据集: {dataset_name}")
        print(f"HuggingFace路径: {dataset_path}")
        print(f"描述: {config['description']}")
        print(f"大小: {config['size']}")
        print(f"语言: {config['language']}")
        
        try:
            # 下载数据集
            dataset = load_dataset(
                dataset_path,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # 保存数据集信息
            info_file = self.cache_dir / f"{dataset_name}_info.yaml"
            with open(info_file, 'w', encoding='utf-8') as f:
                yaml.safe_dump({
                    'dataset_name': dataset_name,
                    'huggingface_path': dataset_path,
                    'description': config['description'],
                    'size': config['size'],
                    'language': config['language'],
                    'splits': list(dataset.keys()),
                    'features': {split: list(dataset[split].features.keys()) for split in dataset.keys()}
                }, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            print(f"[SUCCESS] 数据集 {dataset_name} 下载完成!")
            print(f"数据集分割: {list(dataset.keys())}")
            for split in dataset.keys():
                print(f"  - {split}: {len(dataset[split])} 样本")
            
            return dataset
            
        except Exception as e:
            print(f"[ERROR] 下载数据集 {dataset_name} 失败: {str(e)}")
            raise
    
    def download_multiple_datasets(self, dataset_names: List[str], force_redownload: bool = False) -> Dict[str, DatasetDict]:
        """下载多个数据集"""
        datasets = {}
        
        for dataset_name in tqdm(dataset_names, desc="下载数据集"):
            try:
                datasets[dataset_name] = self.download_dataset(dataset_name, force_redownload)
            except Exception as e:
                print(f"跳过数据集 {dataset_name}: {str(e)}")
                continue
        
        return datasets
    
    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, DatasetDict]:
        """下载所有可用数据集"""
        return self.download_multiple_datasets(list(self.datasets_config.keys()), force_redownload)
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """获取数据集信息"""
        info_file = self.cache_dir / f"{dataset_name}_info.yaml"
        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return None


def main():
    parser = argparse.ArgumentParser(description="下载HuggingFace数据集")
    parser.add_argument("--datasets", nargs="+", help="要下载的数据集名称")
    parser.add_argument("--all", action="store_true", help="下载所有数据集")
    parser.add_argument("--list", action="store_true", help="列出所有可用数据集")
    parser.add_argument("--cache-dir", default="./data/cache", help="数据集缓存目录")
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(cache_dir=args.cache_dir)
    
    if args.list:
        downloader.list_available_datasets()
        return
    
    if args.all:
        print("开始下载所有数据集...")
        datasets = downloader.download_all_datasets(force_redownload=args.force)
        print(f"\n[SUCCESS] 成功下载 {len(datasets)} 个数据集")
    
    elif args.datasets:
        print(f"开始下载指定数据集: {args.datasets}")
        datasets = downloader.download_multiple_datasets(args.datasets, force_redownload=args.force)
        print(f"\n[SUCCESS] 成功下载 {len(datasets)} 个数据集")
    
    else:
        # 默认下载推荐的数据集
        recommended_datasets = ["alpaca_gpt4_data", "oasst1", "chinese_alpaca"]
        print(f"开始下载推荐数据集: {recommended_datasets}")
        datasets = downloader.download_multiple_datasets(recommended_datasets, force_redownload=args.force)
        print(f"\n[SUCCESS] 成功下载 {len(datasets)} 个数据集")
    
    print("\n数据下载完成! 可以开始训练了。")
    print("使用以下命令开始训练:")
    print("python scripts/train.py --config config/training_config.yaml")


if __name__ == "__main__":
    main()