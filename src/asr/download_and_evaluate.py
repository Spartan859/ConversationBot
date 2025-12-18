"""
下载 AISHELL-1 测试数据集
使用 Hugging Face 镜像加速下载
"""

import os
import sys
from pathlib import Path

def download_aishell1_test():
    """下载 AISHELL-1 测试集"""
    
    # 设置镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("正在安装 datasets 库...")
        os.system("pip install datasets soundfile librosa")
        from datasets import load_dataset
    
    print("=" * 60)
    print("📥 下载 AISHELL-1 数据集")
    print("=" * 60)
    
    # 下载数据集（AISHELL-1 HF版本只有 train split）
    print("正在从 Hugging Face 下载数据集...")
    dataset = load_dataset(
        "AISHELL/AISHELL-1",
        split="train"
    )
    
    print(f"✅ 下载完成！样本数: {len(dataset)}")
    print(f"   数据字段: {dataset.features}")
    
    # 查看样例
    print("\n📝 样例数据:")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"   [{i}] ID: {sample.get('id', 'N/A')}")
        print(f"       文本: {sample.get('text', sample.get('sentence', 'N/A'))}")
        print()
    
    return dataset


def evaluate_with_hf_dataset(
    models: list = None,
    num_samples: int = 100,
    device: str = None
):
    """
    使用 Hugging Face 数据集进行评估
    """
    if models is None:
        models = ['tiny', 'base', 'small']  # 默认只测小模型
    
    # 设置镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import numpy as np
    import time
    
    # 添加项目路径
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    # 使用 faster-whisper 作为评估后端
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("正在安装 faster-whisper ...")
        os.system("pip install faster-whisper")
        from faster_whisper import WhisperModel
    from src.asr.evaluate_asr import calculate_cer, EvaluationResult
    from src.asr.number_converter import NumberConverter
    
    # 简体中文引导提示（要求保持中文数字格式）
    SIMPLIFIED_CHINESE_PROMPT = "以下是普通话的句子。请使用简体中文输出，保持中文数字格式，不要转换为阿拉伯数字。"
    
    print("=" * 60)
    print("🚀 faster-whisper 基准测试 (AISHELL-1)")
    print("=" * 60)
    
    # 下载 transcript 文件
    print("正在下载 transcript 文件...")
    transcript_path = hf_hub_download(
        repo_id="AISHELL/AISHELL-1",
        filename="data_aishell/transcript/aishell_transcript_v0.8.txt",
        repo_type="dataset"
    )
    
    # 解析 transcript
    print("正在解析 transcript...")
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                # 移除文本中的空格（中文文本）
                text = text.replace(' ', '')
                transcripts[utt_id] = text
    
    print(f"   已加载 {len(transcripts)} 条转录文本")
    
    # 加载数据集（AISHELL-1 只有 train split，从中抽样测试）
    print("正在加载音频数据集...")
    dataset = load_dataset(
        "AISHELL/AISHELL-1",
        split="train"
    )
    
    print(f"   已加载 {len(dataset)} 条音频样本")
    
    # 检查缓存文件
    cache_file = Path(__file__).parent / "aishell1_valid_indices_cache.json"
    
    if cache_file.exists():
        print(f"✅ 发现缓存文件，直接加载索引...")
        import json
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
            valid_indices = cache_data['indices']
            id_map = cache_data['id_map']  # {index: utt_id}
        print(f"   从缓存加载了 {len(valid_indices)} 条有效样本索引")
    else:
        # 过滤出有 transcript 的样本
        print("正在匹配音频和文本（首次运行，构建索引缓存）...")
        valid_indices = []
        id_map = {}
        for idx, item in enumerate(dataset):
            if (idx + 1) % 1000 == 0:
                print(f"   进度: {idx+1}/{len(dataset)}, 已匹配: {len(valid_indices)} 条")
            
            key = item['__key__']
            # 从路径中提取 ID（如 train/S0002/BAC009S0002W0122 -> BAC009S0002W0122）
            utt_id = key.split('/')[-1]
            if utt_id in transcripts:
                valid_indices.append(idx)
                id_map[str(idx)] = utt_id  # JSON key 必须是字符串
        
        print(f"✅ 匹配完成！匹配到 {len(valid_indices)} 条有效样本")
        
        # 保存索引到缓存
        print(f"💾 保存索引到缓存文件: {cache_file}")
        import json
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'indices': valid_indices,
                'id_map': id_map
            }, f, ensure_ascii=False, indent=2)
        print(f"   缓存已保存，下次运行将直接使用缓存（秒级加载）")
    
    print(f"📊 最终有效样本数: {len(valid_indices)}")
    
    # 随机抽样索引
    np.random.seed(42)
    sampled_indices = np.random.choice(
        valid_indices, 
        min(num_samples, len(valid_indices)), 
        replace=False
    )
    
    print(f"测试样本数: {len(sampled_indices)}")
    print(f"测试模型: {', '.join(models)}")
    print("=" * 60)
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"📊 评估模型: {model_name}")
        print(f"{'='*60}")
        
        # 初始化 faster-whisper 模型
        try:
            # faster-whisper 支持的模型名: tiny, base, small, medium, large-v1, large-v2, large-v3
            # 注意: "large" 需要显式指定版本，这里不做映射，由用户明确指定
            use_device = device if device else ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu")
            compute_type = "float16" if (use_device == "cuda") else "int8"
            model = WhisperModel(model_name, device=use_device, compute_type=compute_type)
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            continue
        
        result = EvaluationResult(model_name=model_name)
        total_audio_duration = 0.0
        
        # 保存每个样本的详细结果
        sample_results = []
        
        for i, idx in enumerate(sampled_indices):
            if (i + 1) % 20 == 0:
                print(f"   进度: {i+1}/{len(sampled_indices)}, 当前 CER: {result.cer:.2%}" if result.total_chars_ref > 0 else f"   进度: {i+1}/{len(sampled_indices)}")
            
            try:
                # 直接从 dataset 索引加载样本（按需加载，不复制）
                sample = dataset[int(idx)]
                key = sample['__key__']
                utt_id = key.split('/')[-1]
                reference = transcripts[utt_id]
                # 获取音频数据 - Hugging Face datasets 的 Audio 对象
                wav_data = sample['wav']
                
                # 检查是否是字典格式（datasets Audio feature 自动解码后的格式）
                if isinstance(wav_data, dict) and 'array' in wav_data:
                    audio_array = wav_data['array'].astype(np.float32)
                    sample_rate = wav_data.get('sampling_rate', 16000)
                else:
                    # AudioDecoder 对象 - 使用 torchcodec API
                    try:
                        # 调用 get_all_samples() 获取 AudioSamples 对象
                        audio_samples = wav_data.get_all_samples()
                        # AudioSamples.data 是 torch.Tensor，shape: (channels, num_samples)
                        audio_tensor = audio_samples.data
                        sample_rate = int(audio_samples.sample_rate)
                        
                        # 转换为 numpy array
                        audio_array = audio_tensor.cpu().numpy().astype(np.float32)
                        
                        # 如果是多声道 (channels, samples)，转为 (samples, channels) 并取第一声道
                        if audio_array.ndim == 2:
                            audio_array = audio_array[0]  # 取第一个声道
                        
                        if i == 0:
                            print(f"   ✅ AudioDecoder: sample_rate={sample_rate}, shape={audio_array.shape}")
                            
                    except Exception as e:
                        print(f"   ⚠️ AudioDecoder 解码失败: {e}")
                        continue
                
                # 重采样到 16kHz（如果需要）
                if sample_rate != 16000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
                    sample_rate = 16000
                
                # 计算音频时长
                audio_duration = len(audio_array) / sample_rate
                total_audio_duration += audio_duration
                
                # 语音识别（faster-whisper）
                start_time = time.time()
                segments, info = model.transcribe(
                    audio_array,
                    language="zh",
                    beam_size=5,
                    vad_filter=False,
                    without_timestamps=True,
                    initial_prompt=SIMPLIFIED_CHINESE_PROMPT
                )
                hypothesis = ''.join(segment.text for segment in segments).strip()
                
                # 数字后处理：将阿拉伯数字转换为中文数字
                hypothesis = NumberConverter.convert_text(hypothesis)
                
                elapsed = time.time() - start_time
                # reference 已经在循环开始时从 transcripts 获取
                
                # 计算 CER
                cer, sub, dele, ins, ref_len = calculate_cer(reference, hypothesis)
                
                # 保存样本结果
                sample_results.append({
                    'utt_id': utt_id,
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'cer': cer,
                    'substitutions': sub,
                    'deletions': dele,
                    'insertions': ins
                })
                
                # 累加统计
                result.total_samples += 1
                result.total_chars_ref += ref_len
                result.total_substitutions += sub
                result.total_deletions += dele
                result.total_insertions += ins
                result.total_time += elapsed
                result.calculate_cer()
                
            except Exception as e:
                print(f"   ⚠️ 处理样本失败: {e}")
                continue
        
        # 计算 RTF
        if total_audio_duration > 0:
            result.rtf = result.total_time / total_audio_duration
        
        results[model_name] = result
        
        # 保存详细结果到文件（eval 子文件夹）
        eval_dir = Path(__file__).parent / "eval"
        eval_dir.mkdir(exist_ok=True)
        output_file = eval_dir / f"aishell1_results_{model_name}.txt"
        avg_time = result.total_time / result.total_samples if result.total_samples > 0 else 0
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"AISHELL-1 评估结果 - 模型: {model_name} (faster-whisper)\n")
            f.write(f"样本数: {len(sample_results)}, CER: {result.cer:.2%}, RTF: {result.rtf:.3f}, 平均耗时: {avg_time:.3f}秒\n")
            f.write(f"后处理: 阿拉伯数字转中文数字\n")
            f.write("=" * 100 + "\n\n")
            
            for idx, sample in enumerate(sample_results, 1):
                f.write(f"[{idx}] {sample['utt_id']}\n")
                f.write(f"GT:  {sample['reference']}\n")
                f.write(f"识别: {sample['hypothesis']}\n")
                f.write(f"CER: {sample['cer']:.2%} (替换:{sample['substitutions']}, 删除:{sample['deletions']}, 插入:{sample['insertions']})\n")
                f.write("-" * 100 + "\n")
        
        print(f"   💾 详细结果已保存到: {output_file}")
        
        # 计算平均耗时
        avg_time_per_sample = result.total_time / result.total_samples if result.total_samples > 0 else 0
        
        print(f"\n📈 {model_name} 评估结果:")
        print(f"   CER: {result.cer:.2%}")
        print(f"   RTF: {result.rtf:.3f}")
        print(f"   平均耗时: {avg_time_per_sample:.3f}秒/样本")
    
    # 打印汇总
    print("\n" + "=" * 100)
    print("📊 评估结果汇总 (AISHELL-1 测试集)")
    print("=" * 100)
    print(f"{'模型':<12} {'CER':>10} {'替换':>8} {'删除':>8} {'插入':>8} {'RTF':>8} {'平均耗时(秒)':>14}")
    print("-" * 100)
    
    for model_name, result in results.items():
        avg_time = result.total_time / result.total_samples if result.total_samples > 0 else 0
        print(f"{model_name:<12} {result.cer:>9.2%} {result.total_substitutions:>8} "
              f"{result.total_deletions:>8} {result.total_insertions:>8} "
              f"{result.rtf:>8.3f} {avg_time:>14.3f}")
    
    print("=" * 100)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载并测试 AISHELL-1')
    parser.add_argument('--download_only', action='store_true',
                        help='仅下载数据集')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['tiny', 'base', 'small'],
                        help='要测试的模型')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='测试样本数')
    parser.add_argument('--device', type=str, default='cuda',
                        help='运行设备')
    
    args = parser.parse_args()
    
    if args.download_only:
        download_aishell1_test()
    else:
        evaluate_with_hf_dataset(
            models=args.models,
            num_samples=args.num_samples,
            device=args.device
        )
