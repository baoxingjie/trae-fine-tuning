#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用MindSpeed框架微调Qwen大模型的训练脚本
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from tqdm import tqdm

# 尝试导入MindSpeed
try:
    import mindspeed
    from mindspeed import MindSpeedTrainer
    MINDSPEED_AVAILABLE = True
except ImportError:
    print("警告: MindSpeed未安装，将使用标准Transformers训练")
    MINDSPEED_AVAILABLE = False


class QwenTrainer:
    """Qwen模型训练器"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_environment()
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path(self.config['training']['output_dir']) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "training.log", encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_environment(self):
        """设置环境变量"""
        # 设置随机种子
        seed = self.config['environment']['seed']
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # 设置CUDA设备
        if torch.cuda.is_available():
            self.logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.warning("未检测到GPU，将使用CPU训练")
        
        # 设置环境变量
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # 初始化wandb（如果配置了）
        if 'wandb' in self.config['logging']['report_to']:
            wandb.init(
                project="qwen-finetune",
                name=self.config['logging']['run_name'],
                config=self.config
            )
    
    def load_tokenizer(self):
        """加载分词器"""
        self.logger.info("加载分词器...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['model_name_or_path'],
            trust_remote_code=self.config['model']['trust_remote_code'],
            padding_side="right"
        )
        
        # 设置pad_token - 对于Qwen模型使用特殊方法
        if self.tokenizer.pad_token is None:
            # 为Qwen模型添加特殊的pad token
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            # 如果还是没有，使用eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.logger.info(f"分词器加载完成，词汇表大小: {len(self.tokenizer)}")
    
    def load_model(self):
        """加载模型"""
        self.logger.info("加载模型...")
        
        # 量化配置
        quantization_config = None
        if self.config['quantization']['use_4bit']:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config['quantization']['bnb_4bit_compute_dtype']),
                bnb_4bit_use_double_quant=self.config['quantization']['bnb_4bit_use_double_quant'],
                bnb_4bit_quant_type=self.config['quantization']['bnb_4bit_quant_type']
            )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['model_name_or_path'],
            trust_remote_code=self.config['model']['trust_remote_code'],
            torch_dtype=getattr(torch, self.config['model']['torch_dtype']),
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            attn_implementation=self.config['model'].get('attn_implementation', 'eager')
        )
        
        # 手动移动到GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # 应用LoRA
        if self.config['lora']['use_lora']:
            self.logger.info("应用LoRA配置...")
            
            lora_config = LoraConfig(
                r=self.config['lora']['r'],
                lora_alpha=self.config['lora']['lora_alpha'],
                target_modules=self.config['lora']['target_modules'],
                lora_dropout=self.config['lora']['lora_dropout'],
                bias=self.config['lora']['bias'],
                task_type=TaskType.CAUSAL_LM
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        self.logger.info("模型加载完成")
    
    def load_datasets(self):
        """加载数据集"""
        self.logger.info("加载数据集...")
        
        # 加载预处理后的数据集
        dataset_path = Path("./data/processed/final_dataset")
        if not dataset_path.exists():
            raise FileNotFoundError(f"数据集不存在: {dataset_path}. 请先运行数据预处理脚本")
        
        dataset = load_from_disk(str(dataset_path))
        
        self.train_dataset = dataset['train']
        if 'validation' in dataset:
            self.eval_dataset = dataset['validation']
        
        self.logger.info(f"训练集样本数: {len(self.train_dataset)}")
        if self.eval_dataset:
            self.logger.info(f"验证集样本数: {len(self.eval_dataset)}")
    
    def setup_trainer(self):
        """设置训练器"""
        self.logger.info("设置训练器...")
        
        # 确保tokenizer有pad_token
        if self.tokenizer.pad_token is None:
            # 为Qwen模型添加特殊的pad token
            self.tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
            # 如果还是没有，使用eos_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info(f"设置pad_token: {self.tokenizer.pad_token}")
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_train_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            warmup_ratio=float(self.config['training']['warmup_ratio']),
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            
            save_strategy=self.config['training']['save_strategy'],
            save_steps=self.config['training']['save_steps'],
            eval_strategy=self.config['training']['eval_strategy'] if self.eval_dataset else "no",
            eval_steps=self.config['training']['eval_steps'] if self.eval_dataset else None,
            logging_steps=self.config['training']['logging_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            
            optim=self.config['training']['optim'],
            adam_beta1=float(self.config['training']['adam_beta1']),
            adam_beta2=float(self.config['training']['adam_beta2']),
            adam_epsilon=float(self.config['training']['adam_epsilon']),
            max_grad_norm=float(self.config['training']['max_grad_norm']),
            
            fp16=self.config['environment']['fp16'],
            bf16=self.config['environment']['bf16'],
            gradient_checkpointing=self.config['environment']['gradient_checkpointing'],
            dataloader_num_workers=self.config['environment']['dataloader_num_workers'],
            dataloader_pin_memory=self.config['environment']['dataloader_pin_memory'],
            remove_unused_columns=self.config['environment']['remove_unused_columns'],
            
            report_to=self.config['logging']['report_to'],
            logging_dir=self.config['logging']['logging_dir'],
            run_name=self.config['logging']['run_name'],
            
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False if self.eval_dataset else None
        )
        
        # 数据整理器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8
        )
        
        # 回调函数
        callbacks = []
        if self.eval_dataset:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # 选择训练器
        if MINDSPEED_AVAILABLE and self.config['mindspeed']['enable']:
            self.logger.info("使用MindSpeed训练器")
            
            # MindSpeed特定配置
            mindspeed_config = {
                'parallel_mode': self.config['mindspeed']['parallel_mode'],
                'tensor_parallel_size': self.config['mindspeed']['tensor_parallel_size'],
                'pipeline_parallel_size': self.config['mindspeed']['pipeline_parallel_size'],
                'micro_batch_size': self.config['mindspeed']['micro_batch_size']
            }
            
            self.trainer = MindSpeedTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks,
                mindspeed_config=mindspeed_config
            )
        else:
            self.logger.info("使用标准Transformers训练器")
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                callbacks=callbacks
            )
    
    def train(self):
        """开始训练"""
        self.logger.info("开始训练...")
        
        # 检查是否从检查点恢复
        resume_from_checkpoint = None
        output_dir = Path(self.config['training']['output_dir'])
        if output_dir.exists():
            checkpoints = list(output_dir.glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
                resume_from_checkpoint = str(latest_checkpoint)
                self.logger.info(f"从检查点恢复训练: {resume_from_checkpoint}")
        
        # 开始训练
        try:
            train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # 保存最终模型
            self.trainer.save_model()
            self.trainer.save_state()
            
            # 保存训练指标
            metrics = train_result.metrics
            self.trainer.log_metrics("train", metrics)
            self.trainer.save_metrics("train", metrics)
            
            self.logger.info("训练完成！")
            self.logger.info(f"最终训练损失: {metrics.get('train_loss', 'N/A')}")
            
            # 如果有验证集，进行最终评估
            if self.eval_dataset:
                self.logger.info("进行最终评估...")
                eval_result = self.trainer.evaluate()
                self.trainer.log_metrics("eval", eval_result)
                self.trainer.save_metrics("eval", eval_result)
                self.logger.info(f"最终验证损失: {eval_result.get('eval_loss', 'N/A')}")
            
        except KeyboardInterrupt:
            self.logger.info("训练被用户中断")
            self.trainer.save_model()
            self.trainer.save_state()
        
        except Exception as e:
            self.logger.error(f"训练过程中出现错误: {str(e)}")
            raise
    
    def run(self):
        """运行完整的训练流程"""
        try:
            self.load_tokenizer()
            self.load_model()
            self.load_datasets()
            self.setup_trainer()
            self.train()
            
            self.logger.info("训练流程完成！")
            
        except Exception as e:
            self.logger.error(f"训练流程失败: {str(e)}")
            raise
        
        finally:
            # 清理资源
            if 'wandb' in self.config['logging']['report_to']:
                wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="训练Qwen大模型")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--resume", action="store_true", help="从最新检查点恢复训练")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 开始训练
    trainer = QwenTrainer(args.config)
    trainer.run()


if __name__ == "__main__":
    main()