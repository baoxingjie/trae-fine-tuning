#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本
一键运行Qwen大模型微调和评测的完整流程
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

import yaml
from tqdm import tqdm


class QuickStart:
    """快速开始管理器"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.scripts_dir = self.project_dir / "scripts"
        self.data_dir = self.project_dir / "data"
        self.config_dir = self.project_dir / "config"
        
        # 检查项目结构
        self.check_project_structure()
    
    def check_project_structure(self):
        """检查项目结构"""
        required_dirs = ["scripts", "data", "config", "models", "results"]
        required_files = [
            "config/training_config.yaml",
            "config/eval_config.yaml",
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/inference.py",
            "data/download_data.py",
            "data/preprocess.py"
        ]
        
        print("检查项目结构...")
        
        # 检查目录
        for dir_name in required_dirs:
            dir_path = self.project_dir / dir_name
            if not dir_path.exists():
                print(f"[ERROR] 缺少目录: {dir_name}")
                return False
            else:
                print(f"[OK] 目录存在: {dir_name}")
        
        # 检查文件
        for file_path in required_files:
            full_path = self.project_dir / file_path
            if not full_path.exists():
                print(f"[ERROR] 缺少文件: {file_path}")
                return False
            else:
                print(f"[OK] 文件存在: {file_path}")
        
        print("[SUCCESS] 项目结构检查通过")
        return True
    
    def run_command(self, command: List[str], description: str, cwd: Optional[Path] = None) -> bool:
        """运行命令"""
        print(f"\n🚀 {description}")
        print(f"命令: {' '.join(command)}")
        
        try:
            # 在Windows上处理编码问题
            import locale
            system_encoding = locale.getpreferredencoding()
            
            result = subprocess.run(
                command,
                cwd=cwd or self.project_dir,
                check=True,
                capture_output=True,
                text=True,
                encoding=system_encoding,
                errors='replace'  # 替换无法解码的字符
            )
            
            if result.stdout:
                print("输出:")
                print(result.stdout)
            
            print(f"[SUCCESS] {description} 完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {description} 失败")
            print(f"错误代码: {e.returncode}")
            if e.stdout:
                print("标准输出:")
                print(e.stdout)
            if e.stderr:
                print("错误输出:")
                print(e.stderr)
            return False
        
        except Exception as e:
            print(f"[ERROR] {description} 出现异常: {str(e)}")
            return False
    
    def install_dependencies(self) -> bool:
        """安装依赖"""
        requirements_file = self.project_dir / "requirements.txt"
        if not requirements_file.exists():
            print("[ERROR] requirements.txt 不存在")
            return False
        
        return self.run_command(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            "安装Python依赖包"
        )
    
    def download_data(self, datasets: Optional[List[str]] = None) -> bool:
        """下载数据"""
        command = [sys.executable, "data/download_data.py"]
        
        if datasets:
            command.extend(["--datasets"] + datasets)
        else:
            # 使用推荐的数据集
            command.append("--datasets")
            command.extend(["alpaca_gpt4_data", "oasst1", "chinese_alpaca"])
        
        return self.run_command(command, "下载训练数据集")
    
    def preprocess_data(self) -> bool:
        """预处理数据"""
        return self.run_command(
            [sys.executable, "data/preprocess.py"],
            "预处理训练数据"
        )
    
    def train_model(self, config_file: str = "config/training_config.yaml") -> bool:
        """训练模型"""
        return self.run_command(
            [sys.executable, "scripts/train.py", "--config", config_file],
            "训练Qwen模型"
        )
    
    def evaluate_model(self, config_file: str = "config/eval_config.yaml") -> bool:
        """评测模型"""
        return self.run_command(
            [sys.executable, "scripts/evaluate.py", "--config", config_file],
            "评测模型性能"
        )
    
    def test_inference(self, model_path: str = "./results/checkpoints") -> bool:
        """测试推理"""
        test_input = "请介绍一下人工智能的发展历史。"
        
        return self.run_command(
            [sys.executable, "scripts/inference.py", "--model-path", model_path, "--input", test_input],
            "测试模型推理"
        )
    
    def run_full_pipeline(self, 
                         skip_install: bool = False,
                         skip_download: bool = False,
                         skip_preprocess: bool = False,
                         skip_train: bool = False,
                         skip_eval: bool = False,
                         datasets: Optional[List[str]] = None) -> bool:
        """运行完整流程"""
        print("\n" + "=" * 60)
        print("[START] 开始Qwen大模型微调完整流程")
        print("=" * 60)
        
        steps = [
            ("install_dependencies", "安装依赖", not skip_install),
            ("download_data", "下载数据", not skip_download),
            ("preprocess_data", "预处理数据", not skip_preprocess),
            ("train_model", "训练模型", not skip_train),
            ("evaluate_model", "评测模型", not skip_eval),
            ("test_inference", "测试推理", True)
        ]
        
        for i, (method_name, description, should_run) in enumerate(steps, 1):
            if not should_run:
                print(f"\n[SKIP] 步骤 {i}: {description} (跳过)")
                continue
            
            print(f"\n[STEP] 步骤 {i}/{len([s for s in steps if s[2]])}: {description}")
            
            if method_name == "download_data":
                success = self.download_data(datasets)
            else:
                method = getattr(self, method_name)
                success = method()
            
            if not success:
                print(f"\n[ERROR] 流程在步骤 '{description}' 失败")
                return False
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 完整流程执行成功！")
        print("=" * 60)
        
        # 显示结果位置
        print("\n📁 结果文件位置:")
        print(f"  - 训练模型: ./results/checkpoints/")
        print(f"  - 训练日志: ./results/logs/")
        print(f"  - 评测结果: ./results/evaluation/")
        
        # 显示下一步操作
        print("\n🚀 下一步操作:")
        print("  1. 查看训练日志: tensorboard --logdir ./results/logs")
        print("  2. 交互式对话: python scripts/inference.py --model-path ./results/checkpoints --interactive")
        print("  3. 查看评测报告: 打开 ./results/evaluation/evaluation_report.html")
        
        return True
    
    def create_custom_config(self, 
                           model_name: str = "Qwen/Qwen-7B-Chat",
                           epochs: int = 3,
                           batch_size: int = 4,
                           learning_rate: float = 2e-5) -> bool:
        """创建自定义配置"""
        print("\n📝 创建自定义训练配置...")
        
        # 读取默认配置
        config_file = self.config_dir / "training_config.yaml"
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 更新配置
        config['model']['model_name_or_path'] = model_name
        config['training']['num_train_epochs'] = epochs
        config['training']['per_device_train_batch_size'] = batch_size
        config['training']['learning_rate'] = learning_rate
        
        # 保存自定义配置
        custom_config_file = self.config_dir / "custom_training_config.yaml"
        with open(custom_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"[SUCCESS] 自定义配置已保存到: {custom_config_file}")
        return True
    
    def show_status(self):
        """显示项目状态"""
        print("\n📊 项目状态检查")
        print("-" * 40)
        
        # 检查数据
        data_cache = self.data_dir / "cache"
        processed_data = self.data_dir / "processed" / "final_dataset"
        
        print(f"数据缓存目录: {'[OK]' if data_cache.exists() else '[MISSING]'} {data_cache}")
        print(f"预处理数据: {'[OK]' if processed_data.exists() else '[MISSING]'} {processed_data}")
        
        # 检查模型
        model_dir = self.project_dir / "results" / "checkpoints"
        if model_dir.exists():
            checkpoints = list(model_dir.glob("checkpoint-*"))
            print(f"训练检查点: [OK] {len(checkpoints)} 个检查点")
            if checkpoints:
                latest = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
                print(f"  最新检查点: {latest.name}")
        else:
            print(f"训练检查点: [MISSING] 未找到")
        
        # 检查评测结果
        eval_dir = self.project_dir / "results" / "evaluation"
        eval_report = eval_dir / "evaluation_report.html"
        print(f"评测结果: {'[OK]' if eval_report.exists() else '[MISSING]'} {eval_report}")
        
        # 检查日志
        log_dir = self.project_dir / "results" / "logs"
        if log_dir.exists():
            log_files = list(log_dir.glob("*.log"))
            print(f"训练日志: [OK] {len(log_files)} 个日志文件")
        else:
            print(f"训练日志: [MISSING] 未找到")


def main():
    parser = argparse.ArgumentParser(description="Qwen大模型微调快速开始")
    
    # 主要操作
    parser.add_argument("--full", action="store_true", help="运行完整流程")
    parser.add_argument("--install", action="store_true", help="仅安装依赖")
    parser.add_argument("--download", action="store_true", help="仅下载数据")
    parser.add_argument("--preprocess", action="store_true", help="仅预处理数据")
    parser.add_argument("--train", action="store_true", help="仅训练模型")
    parser.add_argument("--evaluate", action="store_true", help="仅评测模型")
    parser.add_argument("--inference", action="store_true", help="仅测试推理")
    parser.add_argument("--status", action="store_true", help="显示项目状态")
    
    # 跳过选项
    parser.add_argument("--skip-install", action="store_true", help="跳过安装依赖")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载数据")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过预处理")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评测")
    
    # 配置选项
    parser.add_argument("--datasets", nargs="+", help="指定下载的数据集")
    parser.add_argument("--model", default="Qwen/Qwen-7B-Chat", help="基础模型名称")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--custom-config", action="store_true", help="创建自定义配置")
    
    args = parser.parse_args()
    
    # 初始化快速开始管理器
    quick_start = QuickStart()
    
    # 根据参数执行相应操作
    if args.status:
        quick_start.show_status()
    
    elif args.custom_config:
        quick_start.create_custom_config(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    
    elif args.full:
        quick_start.run_full_pipeline(
            skip_install=args.skip_install,
            skip_download=args.skip_download,
            skip_preprocess=args.skip_preprocess,
            skip_train=args.skip_train,
            skip_eval=args.skip_eval,
            datasets=args.datasets
        )
    
    elif args.install:
        quick_start.install_dependencies()
    
    elif args.download:
        quick_start.download_data(args.datasets)
    
    elif args.preprocess:
        quick_start.preprocess_data()
    
    elif args.train:
        quick_start.train_model()
    
    elif args.evaluate:
        quick_start.evaluate_model()
    
    elif args.inference:
        quick_start.test_inference()
    
    else:
        print("请指定要执行的操作，使用 --help 查看帮助")
        print("\n常用命令:")
        print("  python quick_start.py --full                    # 运行完整流程")
        print("  python quick_start.py --status                  # 查看项目状态")
        print("  python quick_start.py --custom-config           # 创建自定义配置")
        print("  python quick_start.py --full --skip-install     # 跳过安装依赖")


if __name__ == "__main__":
    main()