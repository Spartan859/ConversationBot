"""
下载 faster-whisper 模型
使用 HF_MIRROR 加速下载
"""

import os
from pathlib import Path


def download_faster_whisper_models():
    """下载 faster-whisper 的 large 系列模型"""
    
    # 设置 HF 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("正在安装 huggingface_hub...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    # faster-whisper 模型在 HF 上的仓库 ID
    models = {
        "tiny": "Systran/faster-whisper-tiny",
        "base": "Systran/faster-whisper-base",
        "small": "Systran/faster-whisper-small",
        "medium": "Systran/faster-whisper-medium",
        "large-v1": "Systran/faster-whisper-large-v1",
        "large-v2": "Systran/faster-whisper-large-v2",
        "large-v3": "Systran/faster-whisper-large-v3",
    }
    
    print("=" * 80)
    print("📥 下载 faster-whisper 所有模型")
    print("=" * 80)
    print(f"镜像地址: {os.environ['HF_ENDPOINT']}")
    print(f"缓存目录: ~/.cache/huggingface/hub/")
    print("=" * 80)
    print()
    
    for model_name, repo_id in models.items():
        print(f"\n{'='*80}")
        print(f"📦 下载模型: {model_name}")
        print(f"   仓库: {repo_id}")
        print(f"{'='*80}")
        
        try:
            # 下载模型到缓存目录
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=None,  # 使用默认缓存目录
                resume_download=True,  # 支持断点续传
                local_files_only=False
            )
            
            print(f"✅ {model_name} 下载完成！")
            print(f"   本地路径: {local_path}")
            
        except Exception as e:
            print(f"❌ {model_name} 下载失败: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("✅ 所有模型下载完成！")
    print("=" * 80)
    print("\n使用方法:")
    print("  from faster_whisper import WhisperModel")
    print("  model = WhisperModel('large-v3', device='cuda')")
    print()


if __name__ == "__main__":
    download_faster_whisper_models()
