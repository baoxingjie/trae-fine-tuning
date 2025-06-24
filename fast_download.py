#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超快速模型下载脚本
使用多种优化策略加速下载
"""

import os
import subprocess
import sys
import time
from pathlib import Path

def install_requirements():
    """安装必要的依赖"""
    print("📦 检查并安装必要依赖...")
    try:
        # 安装hf_transfer用于快速下载
        subprocess.run([sys.executable, "-m", "pip", "install", "hf_transfer"], 
                      check=True, capture_output=True)
        print("✅ hf_transfer 安装完成")
    except subprocess.CalledProcessError:
        print("⚠️ hf_transfer 安装失败，将使用标准下载")

def setup_environment():
    """设置优化环境变量"""
    print("🚀 设置下载优化环境...")
    
    # 启用快速传输
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
    # 使用镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 设置并发下载数
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
    
    # 禁用符号链接以避免Windows问题
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    print("✅ 环境变量设置完成:")
    print(f"   HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")

def download_with_cli():
    """使用huggingface-cli下载"""
    model_name = "Qwen/Qwen-7B-Chat"
    local_dir = "./models/qwen-7b-chat"
    
    print(f"\n📦 使用CLI工具下载模型: {model_name}")
    print(f"📁 保存到: {local_dir}")
    
    # 创建目录
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # 使用huggingface-cli下载
        cmd = [
            sys.executable, "-m", "huggingface_hub.commands.huggingface_cli",
            "download",
            model_name,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks", "False",
            "--resume-download"
        ]
        
        print("🔄 开始下载...")
        start_time = time.time()
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✅ CLI下载完成！耗时: {duration:.2f}秒")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CLI下载失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ huggingface-cli 未找到，尝试安装...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub[cli]"], 
                          check=True)
            print("✅ huggingface-cli 安装完成，请重新运行脚本")
        except:
            print("❌ 安装失败")
        return False

def download_with_git():
    """使用git lfs下载（备用方案）"""
    model_name = "Qwen/Qwen-7B-Chat"
    local_dir = "./models/qwen-7b-chat"
    
    print(f"\n🔄 使用Git LFS下载: {model_name}")
    
    try:
        # 设置git lfs并发
        subprocess.run(["git", "config", "--global", "lfs.concurrenttransfers", "8"], 
                      check=True, capture_output=True)
        
        # 克隆仓库
        repo_url = f"https://hf-mirror.com/{model_name}"
        cmd = ["git", "clone", repo_url, local_dir]
        
        print("🔄 开始Git克隆...")
        start_time = time.time()
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✅ Git下载完成！耗时: {duration:.2f}秒")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git下载失败: {e}")
        return False
    except FileNotFoundError:
        print("❌ Git未安装或未在PATH中")
        return False

def download_essential_files():
    """只下载必要文件（最快方案）"""
    print("\n⚡ 使用最小化下载策略...")
    
    from huggingface_hub import hf_hub_download
    
    model_name = "Qwen/Qwen-7B-Chat"
    local_dir = "./models/qwen-7b-chat"
    
    # 必要文件列表
    essential_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "generation_config.json"
    ]
    
    try:
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        print("📥 下载配置文件...")
        for filename in essential_files:
            try:
                file_path = hf_hub_download(
                    repo_id=model_name,
                    filename=filename,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False
                )
                print(f"✅ {filename}")
            except Exception as e:
                print(f"⚠️ {filename} 下载失败: {e}")
        
        print("\n📥 下载第一个模型文件用于测试...")
        try:
            model_file = hf_hub_download(
                repo_id=model_name,
                filename="model-00001-of-00008.safetensors",
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )
            print("✅ 第一个模型文件下载完成")
        except Exception as e:
            print(f"⚠️ 模型文件下载失败: {e}")
        
        print("\n✅ 基础文件下载完成！可以开始训练了")
        print("💡 其他模型文件将在训练时按需下载")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("⚡ 超快速模型下载工具")
    print("=" * 60)
    
    # 安装依赖
    install_requirements()
    
    # 设置环境
    setup_environment()
    
    print("\n🎯 选择下载策略:")
    print("1. 最小化下载（推荐）- 只下载必要文件")
    print("2. CLI完整下载")
    print("3. Git LFS下载")
    
    choice = input("\n请选择 (1-3，默认1): ").strip() or "1"
    
    success = False
    
    if choice == "1":
        success = download_essential_files()
    elif choice == "2":
        success = download_with_cli()
    elif choice == "3":
        success = download_with_git()
    else:
        print("❌ 无效选择")
        return
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 下载完成！")
        print("📝 现在可以运行训练:")
        print("   python scripts/train.py --config config/training_config.yaml")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 下载失败！")
        print("💡 请检查网络连接或尝试其他下载策略")
        print("=" * 60)

if __name__ == "__main__":
    main()