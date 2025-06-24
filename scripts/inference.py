#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型推理脚本
与微调后的Qwen模型进行交互式对话
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer
)
from peft import PeftModel


class QwenInference:
    """Qwen模型推理器"""
    
    def __init__(self, model_path: str, base_model_path: Optional[str] = None, device: str = "auto"):
        self.model_path = model_path
        self.base_model_path = base_model_path
        self.device = device
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        
        # 对话历史
        self.conversation_history = []
        
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """加载模型和分词器"""
        print("正在加载模型和分词器...")
        
        # 加载分词器
        tokenizer_path = self.base_model_path if self.base_model_path else self.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left"  # 推理时使用左填充
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        if self.base_model_path:
            print(f"加载基础模型: {self.base_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device if self.device != "auto" else "auto"
            )
            
            # 加载LoRA权重
            print(f"加载LoRA权重: {self.model_path}")
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            self.model = self.model.merge_and_unload()  # 合并权重
        else:
            # 直接加载微调后的完整模型
            print(f"加载完整模型: {self.model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map=self.device if self.device != "auto" else "auto"
            )
        
        self.model.eval()
        
        # 设置默认生成配置
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        print("模型和分词器加载完成！")
        
        # 显示模型信息
        if torch.cuda.is_available():
            print(f"使用设备: {next(self.model.parameters()).device}")
            print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B")
    
    def update_generation_config(self, **kwargs):
        """更新生成配置"""
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
                print(f"已更新 {key} = {value}")
            else:
                print(f"警告: 未知参数 {key}")
    
    def format_prompt(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        """格式化输入提示"""
        if system_prompt:
            prompt = f"System: {system_prompt}\n\nUser: {user_input}\n\nAssistant:"
        else:
            prompt = f"User: {user_input}\n\nAssistant:"
        
        return prompt
    
    def generate_response(self, prompt: str, stream: bool = False) -> str:
        """生成回复"""
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        
        # 移动到正确的设备
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 设置流式输出
        streamer = None
        if stream:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # 生成回复
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                streamer=streamer
            )
        
        # 解码输出
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self, user_input: str, system_prompt: Optional[str] = None, stream: bool = False) -> str:
        """单轮对话"""
        prompt = self.format_prompt(user_input, system_prompt)
        response = self.generate_response(prompt, stream=stream)
        
        # 保存对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        return response
    
    def multi_turn_chat(self, user_input: str, max_history: int = 5) -> str:
        """多轮对话"""
        # 构建包含历史的提示
        conversation = []
        
        # 添加历史对话（限制数量）
        recent_history = self.conversation_history[-max_history:] if max_history > 0 else []
        for turn in recent_history:
            conversation.append(f"User: {turn['user']}")
            conversation.append(f"Assistant: {turn['assistant']}")
        
        # 添加当前用户输入
        conversation.append(f"User: {user_input}")
        conversation.append("Assistant:")
        
        prompt = "\n\n".join(conversation)
        response = self.generate_response(prompt)
        
        # 保存对话历史
        self.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        return response
    
    def batch_inference(self, inputs: List[str], batch_size: int = 8) -> List[str]:
        """批量推理"""
        responses = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_prompts = [self.format_prompt(inp) for inp in batch_inputs]
            
            # 批量编码
            encoded = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    generation_config=self.generation_config
                )
            
            # 解码批量输出
            for j, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[encoded['input_ids'][j].shape[0]:],
                    skip_special_tokens=True
                )
                responses.append(response.strip())
        
        return responses
    
    def save_conversation(self, filepath: str):
        """保存对话历史"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        print(f"对话历史已保存到: {filepath}")
    
    def load_conversation(self, filepath: str):
        """加载对话历史"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"对话历史已从 {filepath} 加载")
        else:
            print(f"文件不存在: {filepath}")
    
    def clear_conversation(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空")
    
    def interactive_chat(self):
        """交互式对话模式"""
        print("\n=== Qwen交互式对话 ===")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空对话历史")
        print("输入 'save <filename>' 保存对话")
        print("输入 'load <filename>' 加载对话")
        print("输入 'config <param>=<value>' 修改生成参数")
        print("输入 'stream on/off' 开启/关闭流式输出")
        print("-" * 50)
        
        stream_mode = False
        
        while True:
            try:
                user_input = input("\n用户: ").strip()
                
                if not user_input:
                    continue
                
                # 处理特殊命令
                if user_input.lower() in ['quit', 'exit']:
                    print("再见！")
                    break
                
                elif user_input.lower() == 'clear':
                    self.clear_conversation()
                    continue
                
                elif user_input.startswith('save '):
                    filename = user_input[5:].strip()
                    if not filename:
                        filename = f"conversation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                    self.save_conversation(filename)
                    continue
                
                elif user_input.startswith('load '):
                    filename = user_input[5:].strip()
                    self.load_conversation(filename)
                    continue
                
                elif user_input.startswith('config '):
                    config_str = user_input[7:].strip()
                    try:
                        key, value = config_str.split('=')
                        key = key.strip()
                        value = value.strip()
                        
                        # 尝试转换数值
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                        
                        self.update_generation_config(**{key: value})
                    except ValueError:
                        print("格式错误，请使用: config <参数>=<值>")
                    continue
                
                elif user_input.startswith('stream '):
                    mode = user_input[7:].strip().lower()
                    if mode == 'on':
                        stream_mode = True
                        print("流式输出已开启")
                    elif mode == 'off':
                        stream_mode = False
                        print("流式输出已关闭")
                    else:
                        print("请使用: stream on 或 stream off")
                    continue
                
                # 生成回复
                print("\n助手: ", end="" if stream_mode else "")
                response = self.multi_turn_chat(user_input)
                
                if not stream_mode:
                    print(response)
                
            except KeyboardInterrupt:
                print("\n\n对话被中断")
                break
            except Exception as e:
                print(f"\n错误: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Qwen模型推理")
    parser.add_argument("--model-path", required=True, help="模型路径")
    parser.add_argument("--base-model", help="基础模型路径（用于LoRA）")
    parser.add_argument("--device", default="auto", help="设备")
    parser.add_argument("--interactive", action="store_true", help="交互式模式")
    parser.add_argument("--input", help="单次输入")
    parser.add_argument("--batch-file", help="批量推理文件")
    parser.add_argument("--output", help="输出文件")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    
    # 生成参数
    parser.add_argument("--max-new-tokens", type=int, default=512, help="最大新token数")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p")
    parser.add_argument("--top-k", type=int, default=50, help="top-k")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚")
    
    args = parser.parse_args()
    
    # 检查模型路径
    if not Path(args.model_path).exists():
        print(f"模型路径不存在: {args.model_path}")
        sys.exit(1)
    
    # 初始化推理器
    inference = QwenInference(
        model_path=args.model_path,
        base_model_path=args.base_model,
        device=args.device
    )
    
    # 更新生成配置
    inference.update_generation_config(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty
    )
    
    # 根据模式运行
    if args.interactive:
        inference.interactive_chat()
    
    elif args.input:
        print(f"用户: {args.input}")
        print("助手: ", end="" if args.stream else "")
        response = inference.chat(args.input, stream=args.stream)
        if not args.stream:
            print(response)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(response)
    
    elif args.batch_file:
        if not Path(args.batch_file).exists():
            print(f"批量文件不存在: {args.batch_file}")
            sys.exit(1)
        
        # 读取批量输入
        with open(args.batch_file, 'r', encoding='utf-8') as f:
            if args.batch_file.endswith('.json'):
                inputs = json.load(f)
            else:
                inputs = [line.strip() for line in f if line.strip()]
        
        print(f"开始批量推理，共 {len(inputs)} 个输入...")
        responses = inference.batch_inference(inputs)
        
        # 保存结果
        output_file = args.output or "batch_results.json"
        results = []
        for inp, resp in zip(inputs, responses):
            results.append({"input": inp, "output": resp})
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"批量推理完成，结果已保存到: {output_file}")
    
    else:
        print("请指定运行模式: --interactive, --input, 或 --batch-file")
        parser.print_help()


if __name__ == "__main__":
    import pandas as pd  # 添加缺失的导入
    main()