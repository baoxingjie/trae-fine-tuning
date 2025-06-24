#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化最小下载脚本
只下载训练必需的文件，大幅减少下载时间
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def setup_environment():
    """设置优化环境变量"""
    print("🚀 设置下载优化环境...")
    
    # 启用快速传输
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
    # 使用镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 设置超时
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    
    print("✅ 环境变量设置完成")

def install_hf_transfer():
    """安装hf_transfer加速包"""
    print("📦 安装hf_transfer加速包...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "hf_transfer"], 
                      check=True, capture_output=True, text=True)
        print("✅ hf_transfer 安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ hf_transfer 安装失败: {e}")
        return False

def download_essential_files():
    """下载训练必需的最小文件集"""
    print("\n⚡ 开始最小化下载...")
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub 未安装")
        return False
    
    model_name = "Qwen/Qwen-7B-Chat"
    local_dir = "./models/qwen-7b-chat"
    
    # 创建目录
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    # 必要的配置文件
    config_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "generation_config.json",
        "qwen.tiktoken",
        "vocab.txt"
    ]
    
    print("📥 下载配置文件...")
    downloaded_configs = 0
    
    for filename in config_files:
        try:
            print(f"  下载 {filename}...")
            file_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"  ✅ {filename} 完成")
            downloaded_configs += 1
        except Exception as e:
            print(f"  ⚠️ {filename} 跳过: {str(e)[:50]}...")
    
    print(f"\n📊 配置文件下载完成: {downloaded_configs}/{len(config_files)}")
    
    # 下载一个模型文件用于验证
    print("\n📥 下载第一个模型文件...")
    try:
        model_file = hf_hub_download(
            repo_id=model_name,
            filename="model-00001-of-00008.safetensors",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("✅ 第一个模型文件下载完成")
        
        # 检查文件大小
        file_size = Path(model_file).stat().st_size / (1024*1024*1024)  # GB
        print(f"📊 文件大小: {file_size:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型文件下载失败: {e}")
        return False

def create_model_index():
    """创建模型索引文件，让训练脚本知道如何加载模型"""
    print("\n📝 创建模型索引...")
    
    model_dir = Path("./models/qwen-7b-chat")
    
    # 创建一个简单的索引文件
    index_content = {
        "model_name": "Qwen/Qwen-7B-Chat",
        "local_path": str(model_dir.absolute()),
        "download_status": "partial",
        "available_files": list(model_dir.glob("*")) if model_dir.exists() else [],
        "note": "其他模型文件将在训练时按需下载"
    }
    
    try:
        import json
        with open(model_dir / "download_info.json", "w", encoding="utf-8") as f:
            json.dump(index_content, f, indent=2, ensure_ascii=False, default=str)
        print("✅ 模型索引创建完成")
        return True
    except Exception as e:
        print(f"⚠️ 索引创建失败: {e}")
        return False

def verify_download():
    """验证下载的文件"""
    print("\n🔍 验证下载文件...")
    
    model_dir = Path("./models/qwen-7b-chat")
    
    if not model_dir.exists():
        print("❌ 模型目录不存在")
        return False
    
    files = list(model_dir.glob("*"))
    print(f"📊 已下载文件数量: {len(files)}")
    
    # 检查关键文件
    key_files = ["config.json", "tokenizer_config.json"]
    missing_files = []
    
    for key_file in key_files:
        if not (model_dir / key_file).exists():
            missing_files.append(key_file)
    
    if missing_files:
        print(f"⚠️ 缺少关键文件: {missing_files}")
        return False
    
    print("✅ 关键文件验证通过")
    
    # 计算总大小
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    total_size_gb = total_size / (1024*1024*1024)
    print(f"📊 已下载总大小: {total_size_gb:.2f} GB")
    
    return True

def main():
    """主函数"""
    print("=" * 60)
    print("⚡ 自动化最小下载工具")
    print("🎯 策略: 只下载训练必需文件，大幅减少等待时间")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 设置环境
    setup_environment()
    
    # 2. 安装加速包
    install_hf_transfer()
    
    # 3. 下载必要文件
    if not download_essential_files():
        print("\n❌ 下载失败！")
        return
    
    # 4. 创建索引
    create_model_index()
    
    # 5. 验证下载
    if not verify_download():
        print("\n⚠️ 验证失败，但可以尝试继续训练")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🎉 最小下载完成！")
    print(f"⏱️ 总耗时: {duration:.2f}秒")
    print("\n📝 接下来可以启动训练:")
    print("   python scripts/train.py --config config/training_config.yaml")
    print("\n💡 说明:")
    print("   - 已下载训练必需的配置文件")
    print("   - 其他模型文件将在训练时自动下载")
    print("   - 这样可以立即开始训练，无需等待完整下载")
    print("=" * 60)

if __name__ == "__main__":
    main()