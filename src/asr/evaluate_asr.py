"""
ASR 模型评估脚本
在 AISHELL-1 数据集上评估 Whisper 各版本模型的准确率

评估指标：
- CER (Character Error Rate): 字符错误率，中文ASR的主要评估指标
- WER (Word Error Rate): 词错误率（参考）

数据集：AISHELL-1
- 开源中文语音数据集
- 178小时高质量录音
- 400+说话人
- 下载地址: https://www.openslr.org/33/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import re

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.asr.whisper_asr import WhisperASR


@dataclass
class EvaluationResult:
    """评估结果"""
    model_name: str
    total_samples: int = 0
    total_chars_ref: int = 0
    total_substitutions: int = 0
    total_deletions: int = 0
    total_insertions: int = 0
    total_time: float = 0.0
    cer: float = 0.0
    rtf: float = 0.0  # Real-Time Factor
    errors: List[dict] = field(default_factory=list)
    
    def calculate_cer(self):
        """计算 CER"""
        if self.total_chars_ref > 0:
            self.cer = (self.total_substitutions + self.total_deletions + self.total_insertions) / self.total_chars_ref
        return self.cer


def normalize_text(text: str) -> str:
    """
    文本规范化处理
    
    处理步骤：
    1. 转为小写（英文）
    2. 移除标点符号
    3. 移除多余空格
    4. 统一全角/半角字符
    """
    if not text:
        return ""
    
    # 全角转半角
    text = text.translate(str.maketrans(
        '０１２３４５６７８９ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ',
        '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    ))
    
    # 转小写
    text = text.lower()
    
    # 移除标点符号（保留中文字符、英文字母、数字）
    text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbfa-z0-9]', '', text)
    
    return text


def levenshtein_distance(ref: str, hyp: str) -> Tuple[int, int, int, int]:
    """
    计算 Levenshtein 编辑距离
    
    使用动态规划计算将 ref 转换为 hyp 所需的最小编辑操作数
    
    Args:
        ref: 参考文本（标准答案）
        hyp: 假设文本（模型输出）
        
    Returns:
        (编辑距离, 替换数, 删除数, 插入数)
    """
    m, n = len(ref), len(hyp)
    
    # dp[i][j] = 将 ref[:i] 转换为 hyp[:j] 的最小编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i  # 删除 i 个字符
    for j in range(n + 1):
        dp[0][j] = j  # 插入 j 个字符
    
    # 动态规划填表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j-1] + 1,  # 替换
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1     # 插入
                )
    
    # 回溯计算各类错误数量
    substitutions = deletions = insertions = 0
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    
    return dp[m][n], substitutions, deletions, insertions


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, int, int, int, int]:
    """
    计算字符错误率 (CER)
    
    CER = (S + D + I) / N
    其中：
    - S: 替换错误数
    - D: 删除错误数
    - I: 插入错误数
    - N: 参考文本字符数
    
    Args:
        reference: 参考文本
        hypothesis: 识别结果
        
    Returns:
        (CER, 替换数, 删除数, 插入数, 参考字符数)
    """
    # 文本规范化
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0, 0, 0, len(hyp), 0
    
    distance, sub, dele, ins = levenshtein_distance(ref, hyp)
    cer = distance / len(ref)
    
    return cer, sub, dele, ins, len(ref)


class AISHELL1Dataset:
    """
    AISHELL-1 数据集加载器
    
    数据集结构：
    data_aishell/
    ├── wav/
    │   ├── train/
    │   ├── dev/
    │   └── test/
    └── transcript/
        └── aishell_transcript_v0.8.txt
    """
    
    def __init__(self, data_dir: str, split: str = "test"):
        """
        初始化数据集
        
        Args:
            data_dir: 数据集根目录
            split: 数据集划分 (train/dev/test)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = []
        
        self._load_transcripts()
    
    def _load_transcripts(self):
        """加载转录文本"""
        # 转录文件路径
        transcript_file = self.data_dir / "transcript" / "aishell_transcript_v0.8.txt"
        
        if not transcript_file.exists():
            # 尝试其他可能的路径
            alt_paths = [
                self.data_dir / "aishell_transcript_v0.8.txt",
                self.data_dir / "transcript.txt",
            ]
            for alt in alt_paths:
                if alt.exists():
                    transcript_file = alt
                    break
        
        if not transcript_file.exists():
            raise FileNotFoundError(f"找不到转录文件: {transcript_file}")
        
        # 读取转录文本
        transcripts = {}
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    # 移除文本中的空格（中文文本）
                    text = text.replace(' ', '')
                    transcripts[utt_id] = text
        
        # 查找音频文件
        wav_dir = self.data_dir / "wav" / self.split
        if not wav_dir.exists():
            wav_dir = self.data_dir / self.split
        
        if not wav_dir.exists():
            raise FileNotFoundError(f"找不到音频目录: {wav_dir}")
        
        # 遍历音频文件
        for wav_file in wav_dir.rglob("*.wav"):
            utt_id = wav_file.stem
            if utt_id in transcripts:
                self.samples.append({
                    'id': utt_id,
                    'audio_path': str(wav_file),
                    'text': transcripts[utt_id]
                })
        
        print(f"📁 加载 AISHELL-1 {self.split} 集: {len(self.samples)} 条样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_subset(self, n: int, seed: int = 42) -> List[dict]:
        """
        获取数据子集（用于快速测试）
        
        Args:
            n: 样本数量
            seed: 随机种子
            
        Returns:
            样本列表
        """
        np.random.seed(seed)
        indices = np.random.choice(len(self.samples), min(n, len(self.samples)), replace=False)
        return [self.samples[i] for i in indices]


def evaluate_model(
    model_name: str,
    samples: List[dict],
    device: str = None,
    verbose: bool = True
) -> EvaluationResult:
    """
    评估单个模型
    
    Args:
        model_name: 模型名称
        samples: 测试样本列表
        device: 运行设备
        verbose: 是否打印详细信息
        
    Returns:
        评估结果
    """
    print(f"\n{'='*60}")
    print(f"📊 评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 初始化模型
    try:
        asr = WhisperASR(model_name=model_name, device=device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    result = EvaluationResult(model_name=model_name)
    total_audio_duration = 0.0
    
    for i, sample in enumerate(samples):
        if verbose and (i + 1) % 10 == 0:
            print(f"   进度: {i+1}/{len(samples)}")
        
        try:
            # 计算音频时长
            import soundfile as sf
            audio_data, sr = sf.read(sample['audio_path'])
            audio_duration = len(audio_data) / sr
            total_audio_duration += audio_duration
            
            # 语音识别
            start_time = time.time()
            asr_result = asr.transcribe(
                sample['audio_path'],
                language="zh",
                verbose=False
            )
            elapsed = time.time() - start_time
            
            # 获取识别结果
            hypothesis = asr_result.get('text', '')
            reference = sample['text']
            
            # 计算 CER
            cer, sub, dele, ins, ref_len = calculate_cer(reference, hypothesis)
            
            # 累加统计
            result.total_samples += 1
            result.total_chars_ref += ref_len
            result.total_substitutions += sub
            result.total_deletions += dele
            result.total_insertions += ins
            result.total_time += elapsed
            
            # 记录错误样本（CER > 20%）
            if cer > 0.2:
                result.errors.append({
                    'id': sample['id'],
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'cer': cer
                })
                
        except Exception as e:
            print(f"   ⚠️ 处理样本 {sample['id']} 失败: {e}")
            continue
    
    # 计算最终指标
    result.calculate_cer()
    
    # 计算实时因子 (RTF)
    if total_audio_duration > 0:
        result.rtf = result.total_time / total_audio_duration
    
    # 打印结果
    print(f"\n📈 评估结果 - {model_name}")
    print(f"   样本数: {result.total_samples}")
    print(f"   总字符数: {result.total_chars_ref}")
    print(f"   替换错误: {result.total_substitutions}")
    print(f"   删除错误: {result.total_deletions}")
    print(f"   插入错误: {result.total_insertions}")
    print(f"   CER: {result.cer:.2%}")
    print(f"   总耗时: {result.total_time:.2f}s")
    print(f"   RTF: {result.rtf:.3f} (< 1.0 表示快于实时)")
    
    return result


def run_benchmark(
    data_dir: str,
    models: List[str] = None,
    num_samples: int = 100,
    device: str = None,
    output_file: str = None
) -> Dict[str, EvaluationResult]:
    """
    运行完整基准测试
    
    Args:
        data_dir: AISHELL-1 数据集目录
        models: 要测试的模型列表
        num_samples: 测试样本数量
        device: 运行设备
        output_file: 结果输出文件
        
    Returns:
        各模型评估结果
    """
    if models is None:
        models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    
    print("=" * 60)
    print("🚀 Whisper ASR 基准测试")
    print("=" * 60)
    print(f"数据集: AISHELL-1")
    print(f"测试模型: {', '.join(models)}")
    print(f"样本数量: {num_samples}")
    print(f"运行设备: {device or '自动选择'}")
    print("=" * 60)
    
    # 加载数据集
    try:
        dataset = AISHELL1Dataset(data_dir, split="test")
        samples = dataset.get_subset(num_samples)
    except FileNotFoundError as e:
        print(f"❌ 数据集加载失败: {e}")
        print("\n请确保 AISHELL-1 数据集已正确下载并解压")
        print("下载地址: https://www.openslr.org/33/")
        return {}
    
    # 评估各模型
    results = {}
    for model_name in models:
        result = evaluate_model(model_name, samples, device=device)
        if result:
            results[model_name] = result
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print("📊 评估结果汇总")
    print("=" * 80)
    print(f"{'模型':<12} {'CER':>10} {'替换':>8} {'删除':>8} {'插入':>8} {'RTF':>8} {'耗时':>10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        print(f"{model_name:<12} {result.cer:>9.2%} {result.total_substitutions:>8} "
              f"{result.total_deletions:>8} {result.total_insertions:>8} "
              f"{result.rtf:>8.3f} {result.total_time:>9.1f}s")
    
    print("=" * 80)
    
    # 保存结果
    if output_file:
        output_data = {
            'dataset': 'AISHELL-1',
            'num_samples': num_samples,
            'results': {
                name: {
                    'cer': res.cer,
                    'substitutions': res.total_substitutions,
                    'deletions': res.total_deletions,
                    'insertions': res.total_insertions,
                    'rtf': res.rtf,
                    'total_time': res.total_time
                }
                for name, res in results.items()
            }
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存至: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Whisper ASR 评估脚本')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='AISHELL-1 数据集目录')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                        help='要测试的模型列表')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='测试样本数量 (默认: 100)')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='结果输出文件')
    
    args = parser.parse_args()
    
    run_benchmark(
        data_dir=args.data_dir,
        models=args.models,
        num_samples=args.num_samples,
        device=args.device,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
