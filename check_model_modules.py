#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Qwen模型的模块结构，找到正确的LoRA目标模块
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def check_qwen_modules():
    print("正在加载Qwen模型...")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-7B-Chat",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 使用CPU避免显存问题
        attn_implementation="eager"
    )
    
    print("\n=== Qwen模型结构分析 ===")
    print(f"模型类型: {type(model).__name__}")
    
    # 打印所有模块名称
    print("\n=== 所有模块名称 ===")
    for name, module in model.named_modules():
        if len(name.split('.')) <= 3:  # 只显示前3层
            print(f"{name}: {type(module).__name__}")
    
    # 查找包含attention和MLP的模块
    print("\n=== 可能的LoRA目标模块 ===")
    attention_modules = []
    mlp_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'Linear' in module_type:
            if any(keyword in name.lower() for keyword in ['attn', 'attention', 'q_', 'k_', 'v_', 'o_']):
                attention_modules.append(name)
            elif any(keyword in name.lower() for keyword in ['mlp', 'feed', 'gate', 'up', 'down']):
                mlp_modules.append(name)
    
    print("\n注意力相关模块:")
    for module in attention_modules[:10]:  # 只显示前10个
        print(f"  {module}")
    
    print("\nMLP相关模块:")
    for module in mlp_modules[:10]:  # 只显示前10个
        print(f"  {module}")
    
    # 分析第一个transformer层的结构
    print("\n=== 第一个Transformer层结构 ===")
    for name, module in model.named_modules():
        if 'transformer.h.0' in name and len(name.split('.')) == 4:
            print(f"{name}: {type(module).__name__}")
    
    # 推荐的LoRA目标模块
    print("\n=== 推荐的LoRA目标模块 ===")
    recommended_modules = []
    
    # 查找实际的模块名称模式
    for name, module in model.named_modules():
        if 'transformer.h.0' in name and type(module).__name__ == 'Linear':
            module_name = name.split('.')[-1]
            if module_name not in recommended_modules:
                recommended_modules.append(module_name)
    
    print("建议的target_modules配置:")
    print(recommended_modules)
    
    return recommended_modules

if __name__ == "__main__":
    try:
        modules = check_qwen_modules()
        print(f"\n检查完成！建议使用的模块: {modules}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()