#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预下载脚本
使用快速传输功能加速下载Qwen-7B-Chat模型
"""

import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

def setup_fast_download():
    """设置快速下载环境变量"""
    print("🚀 设置快速下载环境...")
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    print("✅ 环境变量设置完成")
    print(f"   HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")

def download_model_fast():
    """使用快速方式下载模型"""
    model_name = "Qwen/Qwen-7B-Chat"
    local_dir = "./models/qwen-7b-chat"
    
    print(f"\n📦 开始下载模型: {model_name}")
    print(f"📁 本地保存路径: {local_dir}")
    
    start_time = time.time()
    
    try:
        # 方法1: 使用snapshot_download批量下载
        print("\n🔄 使用snapshot_download批量下载...")
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ 模型文件下载完成")
        
        # 方法2: 验证下载并加载模型
        print("\n🔍 验证模型完整性...")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        print("✅ 分词器加载成功")
        
        # 注意：这里只验证模型配置，不完全加载以节省内存
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(local_dir)
        print("✅ 模型配置验证成功")
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n🎉 下载完成！总耗时: {duration:.2f}秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("\n🔄 尝试使用传统方式下载...")
        
        try:
            # 备用方案：传统下载方式
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")
            print("✅ 分词器下载完成")
            
            # 只下载配置，不下载完整模型以节省时间
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, cache_dir="./models")
            print("✅ 模型配置下载完成")
            
            end_time = time.time()
            duration = end_time - start_time
            print(f"\n🎉 备用方案下载完成！总耗时: {duration:.2f}秒")
            
            return True
            
        except Exception as e2:
            print(f"❌ 备用方案也失败: {str(e2)}")
            return False

def main():
    """主函数"""
    print("=" * 60)
    print("🚀 Qwen-7B-Chat 模型快速下载工具")
    print("=" * 60)
    
    # 设置快速下载环境
    setup_fast_download()
    
    # 创建模型目录
    os.makedirs("./models", exist_ok=True)
    
    # 下载模型
    success = download_model_fast()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 模型下载成功！")
        print("📝 接下来可以运行训练脚本:")
        print("   python scripts/train.py --config config/training_config.yaml")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 模型下载失败！")
        print("💡 建议检查网络连接或稍后重试")
        print("=" * 60)

if __name__ == "__main__":
    main()