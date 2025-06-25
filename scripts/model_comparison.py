# -*- coding: utf-8 -*-
"""
模型对比测试脚本
对比微调前后的模型回答效果
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from peft import PeftModel


class ModelComparison:
    """模型对比器"""
    
    def __init__(self, base_model_path: str, finetuned_model_path: str, device: str = "auto"):
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        self.device = device
        
        # 模型组件
        self.tokenizer = None
        self.base_model = None
        self.finetuned_model = None
        self.generation_config = None
        
        self.load_models()
    
    def load_models(self):
        """加载基础模型和微调模型"""
        print("正在加载模型...")
        
        # 加载分词器
        print(f"加载分词器: {self.base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        print(f"加载基础模型: {self.base_model_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "auto" else "auto"
        )
        self.base_model.eval()
        
        # 加载微调模型
        print(f"加载微调模型: {self.finetuned_model_path}")
        self.finetuned_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "auto" else "auto"
        )
        
        # 加载LoRA权重并合并
        self.finetuned_model = PeftModel.from_pretrained(
            self.finetuned_model, 
            self.finetuned_model_path
        )
        self.finetuned_model = self.finetuned_model.merge_and_unload()
        self.finetuned_model.eval()
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        print("模型加载完成！")
        
        # 显示模型信息
        if torch.cuda.is_available():
            print(f"使用设备: {next(self.base_model.parameters()).device}")
            base_params = sum(p.numel() for p in self.base_model.parameters()) / 1e9
            print(f"基础模型参数量: {base_params:.2f}B")
    
    def format_prompt(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """格式化输入提示"""
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        else:
            prompt = f"User: {user_input}\n\nAssistant:"
        
        return prompt
    
    def generate_response(self, model, prompt: str) -> tuple[str, float]:
        """生成回复并计算耗时"""
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # 移动到正确的设备
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 生成回复
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        end_time = time.time()
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip(), end_time - start_time
    
    def compare_single(self, user_input: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """对比单个问题的回答"""
        prompt = self.format_prompt(user_input, system_prompt)
        
        print(f"\n问题: {user_input}")
        print("=" * 80)
        
        # 基础模型回答
        print("\n【微调前模型回答】")
        base_response, base_time = self.generate_response(self.base_model, prompt)
        print(base_response)
        print(f"\n生成时间: {base_time:.2f}秒")
        
        # 微调模型回答
        print("\n【微调后模型回答】")
        finetuned_response, finetuned_time = self.generate_response(self.finetuned_model, prompt)
        print(finetuned_response)
        print(f"\n生成时间: {finetuned_time:.2f}秒")
        
        return {
            "question": user_input,
            "system_prompt": system_prompt,
            "base_model": {
                "response": base_response,
                "time": base_time
            },
            "finetuned_model": {
                "response": finetuned_response,
                "time": finetuned_time
            }
        }
    
    def interactive_comparison(self):
        """交互式对比模式"""
        print("\n=== 模型对比交互模式 ===")
        print("输入问题进行对比，输入 'quit' 退出")
        print("输入 'config' 修改生成参数")
        print("输入 'save' 保存对比结果")
        
        results = []
        
        while True:
            try:
                user_input = input("\n请输入问题: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'config':
                    self.update_config_interactive()
                    continue
                elif user_input.lower() == 'save':
                    self.save_results(results)
                    continue
                elif not user_input:
                    continue
                
                result = self.compare_single(user_input)
                results.append(result)
                
            except KeyboardInterrupt:
                print("\n\n对比已中断")
                break
        
        # 询问是否保存结果
        if results:
            save_choice = input("\n是否保存对比结果? (y/n): ").strip().lower()
            if save_choice == 'y':
                self.save_results(results)
    
    def batch_comparison(self, questions: List[str], output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """批量对比模式"""
        print(f"\n开始批量对比，共 {len(questions)} 个问题...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n处理第 {i}/{len(questions)} 个问题...")
            result = self.compare_single(question)
            results.append(result)
        
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def update_config_interactive(self):
        """交互式更新生成配置"""
        print("\n当前生成配置:")
        print(f"max_new_tokens: {self.generation_config.max_new_tokens}")
        print(f"temperature: {self.generation_config.temperature}")
        print(f"top_p: {self.generation_config.top_p}")
        print(f"top_k: {self.generation_config.top_k}")
        print(f"repetition_penalty: {self.generation_config.repetition_penalty}")
        
        try:
            max_tokens = input("\n新的max_new_tokens (回车跳过): ").strip()
            if max_tokens:
                self.generation_config.max_new_tokens = int(max_tokens)
            
            temperature = input("新的temperature (回车跳过): ").strip()
            if temperature:
                self.generation_config.temperature = float(temperature)
            
            top_p = input("新的top_p (回车跳过): ").strip()
            if top_p:
                self.generation_config.top_p = float(top_p)
            
            top_k = input("新的top_k (回车跳过): ").strip()
            if top_k:
                self.generation_config.top_k = int(top_k)
            
            rep_penalty = input("新的repetition_penalty (回车跳过): ").strip()
            if rep_penalty:
                self.generation_config.repetition_penalty = float(rep_penalty)
            
            print("配置已更新！")
            
        except ValueError as e:
            print(f"配置更新失败: {e}")
    
    def save_results(self, results: List[Dict[str, Any]], output_file: Optional[str] = None):
        """保存对比结果"""
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"model_comparison_{timestamp}.json"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n对比结果已保存到: {output_path}")
        print(f"共保存 {len(results)} 个对比结果")


def load_test_questions(file_path: str) -> List[str]:
    """加载测试问题"""
    questions = []
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
    
    return questions


def main():
    parser = argparse.ArgumentParser(description="模型对比测试")
    
    # 模型路径
    parser.add_argument("--base-model", required=True, help="基础模型路径")
    parser.add_argument("--finetuned-model", required=True, help="微调模型路径")
    parser.add_argument("--device", default="auto", help="设备")
    
    # 运行模式
    parser.add_argument("--interactive", action="store_true", help="交互式对比模式")
    parser.add_argument("--questions-file", help="测试问题文件")
    parser.add_argument("--output", help="输出文件路径")
    
    # 生成参数
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大新token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p")
    parser.add_argument("--top-k", type=int, default=50, help="top-k")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚")
    
    args = parser.parse_args()
    
    # 检查模型路径
    # 基础模型可以是本地路径或Hugging Face模型标识符
    if "/" in args.base_model and not Path(args.base_model).exists():
        # 可能是Hugging Face模型标识符，尝试在线加载
        print(f"将从Hugging Face加载基础模型: {args.base_model}")
    elif not Path(args.base_model).exists():
        print(f"基础模型路径不存在: {args.base_model}")
        sys.exit(1)
    
    if not Path(args.finetuned_model).exists():
        print(f"微调模型路径不存在: {args.finetuned_model}")
        sys.exit(1)
    
    # 初始化对比器
    comparator = ModelComparison(
        base_model_path=args.base_model,
        finetuned_model_path=args.finetuned_model,
        device=args.device
    )
    
    # 更新生成配置
    comparator.generation_config.max_new_tokens = args.max_new_tokens
    comparator.generation_config.temperature = args.temperature
    comparator.generation_config.top_p = args.top_p
    comparator.generation_config.top_k = args.top_k
    comparator.generation_config.repetition_penalty = args.repetition_penalty
    
    # 根据模式运行
    if args.interactive:
        comparator.interactive_comparison()
    elif args.questions_file:
        if not Path(args.questions_file).exists():
            print(f"问题文件不存在: {args.questions_file}")
            sys.exit(1)
        
        questions = load_test_questions(args.questions_file)
        print(f"加载了 {len(questions)} 个测试问题")
        
        comparator.batch_comparison(questions, args.output)
    else:
        print("请指定运行模式: --interactive 或 --questions-file")
        parser.print_help()


if __name__ == "__main__":
    main()