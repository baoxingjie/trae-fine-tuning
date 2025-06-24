#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型评测脚本
对微调后的Qwen模型进行全面评测
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
import yaml
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 评测指标
from rouge_score import rouge_scorer
from sacrebleu import BLEU
try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    print("警告: bert_score未安装，将跳过BERTScore评测")
    BERT_SCORE_AVAILABLE = False

try:
    from bleurt import score as bleurt_score
    BLEURT_AVAILABLE = True
except ImportError:
    print("警告: BLEURT未安装，将跳过BLEURT评测")
    BLEURT_AVAILABLE = False


class ModelEvaluator:
    """模型评测器"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_logging()
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        # 评测结果
        self.results = defaultdict(dict)
        
        # 设置输出目录
        self.output_dir = Path(self.config['evaluation_settings']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_logging(self):
        """设置日志"""
        log_file = self.output_dir / "evaluation.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        self.logger.info("加载模型和分词器...")
        
        model_path = self.config['model']['model_path']
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=self.config['model']['trust_remote_code']
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=self.config['model']['trust_remote_code'],
            torch_dtype=getattr(torch, self.config['model']['torch_dtype']),
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config['inference_config']['max_new_tokens'],
            do_sample=self.config['inference_config']['do_sample'],
            temperature=self.config['inference_config']['temperature'],
            top_p=self.config['inference_config']['top_p'],
            top_k=self.config['inference_config']['top_k'],
            repetition_penalty=self.config['inference_config']['repetition_penalty'],
            num_return_sequences=self.config['inference_config']['num_return_sequences'],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        self.logger.info("模型和分词器加载完成")
    
    def generate_response(self, prompt: str) -> str:
        """生成模型回复"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def calculate_perplexity(self, texts: List[str]) -> float:
        """计算困惑度"""
        total_loss = 0
        total_tokens = 0
        
        self.model.eval()
        
        for text in tqdm(texts, desc="计算困惑度"):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].shape[1]
                total_tokens += inputs['input_ids'].shape[1]
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BLEU分数"""
        bleu = BLEU()
        
        # 计算corpus-level BLEU
        corpus_score = bleu.corpus_score(predictions, [references])
        
        # 计算不同n-gram的BLEU
        bleu_scores = {
            'bleu': corpus_score.score,
            'bleu-1': 0,
            'bleu-2': 0,
            'bleu-3': 0,
            'bleu-4': corpus_score.score
        }
        
        # 计算各个n-gram的BLEU
        for n in [1, 2, 3]:
            bleu_n = BLEU(effective_order=n)
            score_n = bleu_n.corpus_score(predictions, [references])
            bleu_scores[f'bleu-{n}'] = score_n.score
        
        return bleu_scores
    
    def calculate_rouge_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            for metric, score in scores.items():
                rouge_scores[f'{metric}_f1'].append(score.fmeasure)
                rouge_scores[f'{metric}_precision'].append(score.precision)
                rouge_scores[f'{metric}_recall'].append(score.recall)
        
        # 计算平均分数
        avg_scores = {}
        for metric, scores in rouge_scores.items():
            avg_scores[metric] = np.mean(scores)
        
        return avg_scores
    
    def calculate_bert_score(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算BERTScore"""
        if not BERT_SCORE_AVAILABLE:
            return {}
        
        P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
        
        return {
            'bert_score_precision': P.mean().item(),
            'bert_score_recall': R.mean().item(),
            'bert_score_f1': F1.mean().item()
        }
    
    def calculate_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """计算多样性指标"""
        # Distinct-n
        def distinct_n(texts, n):
            all_ngrams = []
            total_ngrams = 0
            
            for text in texts:
                words = text.split()
                ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
                all_ngrams.extend(ngrams)
                total_ngrams += len(ngrams)
            
            if total_ngrams == 0:
                return 0
            
            unique_ngrams = len(set(all_ngrams))
            return unique_ngrams / total_ngrams
        
        diversity_scores = {}
        for n in [1, 2, 3, 4]:
            diversity_scores[f'distinct_{n}'] = distinct_n(texts, n)
        
        return diversity_scores
    
    def evaluate_dataset(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """评测单个数据集"""
        dataset_name = dataset_config['name']
        self.logger.info(f"评测数据集: {dataset_name}")
        
        try:
            # 加载数据集
            if 'subset' in dataset_config:
                dataset = load_dataset(
                    dataset_config['path'],
                    dataset_config['subset'],
                    split=dataset_config['split']
                )
            else:
                dataset = load_dataset(
                    dataset_config['path'],
                    split=dataset_config['split']
                )
            
            # 限制样本数量
            max_samples = self.config['evaluation_settings']['max_samples_per_dataset']
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            predictions = []
            references = []
            prompts = []
            
            # 生成预测
            for example in tqdm(dataset, desc=f"生成 {dataset_name} 预测"):
                # 根据数据集类型构建prompt
                if 'instruction' in example:
                    prompt = f"Instruction: {example['instruction']}\n\nResponse:"
                    if 'output' in example:
                        reference = example['output']
                    elif 'response' in example:
                        reference = example['response']
                    else:
                        reference = ""
                elif 'question' in example:
                    prompt = f"Question: {example['question']}\n\nAnswer:"
                    reference = example.get('answer', example.get('choices', [""])[0])
                else:
                    # 通用格式
                    prompt = str(example.get('input', example.get('text', "")))
                    reference = str(example.get('target', example.get('label', "")))
                
                prediction = self.generate_response(prompt)
                
                predictions.append(prediction)
                references.append(reference)
                prompts.append(prompt)
            
            # 计算指标
            results = {'dataset': dataset_name, 'num_samples': len(predictions)}
            
            # 自动评测指标
            if references and any(ref.strip() for ref in references):
                # BLEU
                bleu_scores = self.calculate_bleu_score(predictions, references)
                results.update(bleu_scores)
                
                # ROUGE
                rouge_scores = self.calculate_rouge_score(predictions, references)
                results.update(rouge_scores)
                
                # BERTScore
                bert_scores = self.calculate_bert_score(predictions, references)
                results.update(bert_scores)
            
            # 多样性指标
            diversity_scores = self.calculate_diversity_metrics(predictions)
            results.update(diversity_scores)
            
            # 困惑度
            if len(predictions) > 0:
                perplexity = self.calculate_perplexity(predictions[:100])  # 限制样本数以节省时间
                results['perplexity'] = perplexity
            
            # 保存详细结果
            if self.config['evaluation_settings']['save_predictions']:
                detailed_results = []
                for i, (prompt, pred, ref) in enumerate(zip(prompts, predictions, references)):
                    detailed_results.append({
                        'id': i,
                        'prompt': prompt,
                        'prediction': pred,
                        'reference': ref
                    })
                
                detail_file = self.output_dir / f"{dataset_name}_detailed_results.json"
                with open(detail_file, 'w', encoding='utf-8') as f:
                    json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            
            return results
            
        except Exception as e:
            self.logger.error(f"评测数据集 {dataset_name} 时出错: {str(e)}")
            return {'dataset': dataset_name, 'error': str(e)}
    
    def run_evaluation(self):
        """运行完整评测"""
        self.logger.info("开始模型评测")
        
        # 加载模型
        self.load_model_and_tokenizer()
        
        # 评测各个数据集类别
        all_results = []
        
        for category, datasets in self.config['evaluation_datasets'].items():
            self.logger.info(f"评测类别: {category}")
            
            category_results = []
            for dataset_config in datasets:
                result = self.evaluate_dataset(dataset_config)
                category_results.append(result)
                all_results.append(result)
            
            self.results[category] = category_results
        
        # 保存结果
        self.save_results(all_results)
        
        # 生成报告
        self.generate_report(all_results)
        
        self.logger.info("评测完成")
    
    def save_results(self, results: List[Dict[str, Any]]):
        """保存评测结果"""
        # JSON格式
        if self.config['reporting']['generate_json_report']:
            json_file = self.output_dir / "evaluation_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        
        # CSV格式
        if self.config['reporting']['generate_csv_report']:
            df = pd.DataFrame(results)
            csv_file = self.output_dir / "evaluation_results.csv"
            df.to_csv(csv_file, index=False, encoding='utf-8')
        
        self.logger.info(f"评测结果已保存到: {self.output_dir}")
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """生成评测报告"""
        if not self.config['reporting']['generate_html_report']:
            return
        
        # 创建HTML报告
        html_content = self.create_html_report(results)
        
        html_file = self.output_dir / "evaluation_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML报告已生成: {html_file}")
    
    def create_html_report(self, results: List[Dict[str, Any]]) -> str:
        """创建HTML报告"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>模型评测报告</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 30px; }
                .summary { background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
                .dataset { margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                .metric { background-color: #f9f9f9; padding: 10px; border-radius: 3px; }
                .metric-name { font-weight: bold; color: #333; }
                .metric-value { font-size: 1.2em; color: #007bff; }
                .error { color: #dc3545; background-color: #f8d7da; padding: 10px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Qwen模型评测报告</h1>
                <p>生成时间: {timestamp}</p>
            </div>
        """.format(timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # 添加总结
        valid_results = [r for r in results if 'error' not in r]
        html += f"""
            <div class="summary">
                <h2>评测总结</h2>
                <p>总数据集数量: {len(results)}</p>
                <p>成功评测: {len(valid_results)}</p>
                <p>失败数量: {len(results) - len(valid_results)}</p>
            </div>
        """
        
        # 添加各数据集结果
        for result in results:
            dataset_name = result['dataset']
            
            if 'error' in result:
                html += f"""
                    <div class="dataset">
                        <h3>{dataset_name}</h3>
                        <div class="error">评测失败: {result['error']}</div>
                    </div>
                """
            else:
                html += f"""
                    <div class="dataset">
                        <h3>{dataset_name}</h3>
                        <p>样本数量: {result.get('num_samples', 'N/A')}</p>
                        <div class="metrics">
                """
                
                # 添加指标
                for key, value in result.items():
                    if key not in ['dataset', 'num_samples'] and isinstance(value, (int, float)):
                        html += f"""
                            <div class="metric">
                                <div class="metric-name">{key}</div>
                                <div class="metric-value">{value:.4f}</div>
                            </div>
                        """
                
                html += "</div></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    parser = argparse.ArgumentParser(description="评测Qwen大模型")
    parser.add_argument("--config", required=True, help="评测配置文件路径")
    
    args = parser.parse_args()
    
    # 检查配置文件
    if not Path(args.config).exists():
        print(f"配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 开始评测
    evaluator = ModelEvaluator(args.config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()