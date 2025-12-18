"""
Whisper 语音识别模块
基于 OpenAI Whisper 实现语音转文字
"""

import whisper
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Dict, Any
import time


class WhisperASR:
    """Whisper 语音识别类"""
    
    # 可用模型及其特点
    MODELS = {
        'tiny':     {'params': '39M',  'vram': '~1GB',  'speed': '~32x'},
        'base':     {'params': '74M',  'vram': '~1GB',  'speed': '~16x'},
        'small':    {'params': '244M', 'vram': '~2GB',  'speed': '~6x'},
        'medium':   {'params': '769M', 'vram': '~5GB',  'speed': '~2x'},
        'large':    {'params': '1550M','vram': '~10GB', 'speed': '~1x'},
        'large-v2': {'params': '1550M','vram': '~10GB', 'speed': '~1x'},
        'large-v3': {'params': '1550M','vram': '~10GB', 'speed': '~1x'},
    }
    
    def __init__(
        self,
        model_name: str = "base",
        device: Optional[str] = None,
        download_root: Optional[str] = None
    ):
        """
        初始化 Whisper ASR
        
        Args:
            model_name: 模型名称 (tiny/base/small/medium/large/large-v2/large-v3)
            device: 运行设备 (cuda/cpu)，None则自动选择
            download_root: 模型下载目录
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🔧 初始化 Whisper ASR...")
        print(f"   模型: {model_name}")
        print(f"   设备: {self.device}")
        
        if model_name in self.MODELS:
            info = self.MODELS[model_name]
            print(f"   参数量: {info['params']}, 显存需求: {info['vram']}, 相对速度: {info['speed']}")
        
        # 加载模型
        start_time = time.time()
        self.model = whisper.load_model(
            model_name,
            device=self.device,
            download_root=download_root
        )
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成！耗时: {load_time:.2f}秒")
        
    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """列出所有可用模型"""
        return cls.MODELS
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> Dict[str, Any]:
        """
        语音转文字
        
        Args:
            audio: 音频文件路径或numpy数组
            language: 语言代码 (zh/en/ja等)，None则自动检测
            task: 任务类型 (transcribe=转录 / translate=翻译成英文)
            **kwargs: 其他Whisper参数
            
        Returns:
            识别结果字典，包含 text, segments, language 等
        """
        print(f"🎯 开始语音识别...")
        start_time = time.time()
        
        # 如果是numpy数组，确保是float32类型
        if isinstance(audio, np.ndarray):
            audio = audio.astype(np.float32)
        
        # 执行识别
        result = self.model.transcribe(
            audio,
            language=language,
            task=task,
            **kwargs
        )
        
        elapsed_time = time.time() - start_time
        
        print(f"✅ 识别完成！")
        print(f"   检测语言: {result['language']}")
        print(f"   耗时: {elapsed_time:.2f}秒")
        print(f"   识别结果: {result['text']}")
        
        # 添加额外信息
        result['elapsed_time'] = elapsed_time
        result['model'] = self.model_name
        result['device'] = self.device
        
        return result
    
    def transcribe_with_timestamps(
        self,
        audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        word_timestamps: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带时间戳的语音识别
        
        Args:
            audio: 音频文件路径或numpy数组
            language: 语言代码
            word_timestamps: 是否返回词级时间戳
            **kwargs: 其他参数
            
        Returns:
            识别结果，包含详细时间戳
        """
        return self.transcribe(
            audio,
            language=language,
            word_timestamps=word_timestamps,
            **kwargs
        )
    
    def detect_language(self, audio: Union[str, np.ndarray]) -> tuple:
        """
        检测音频语言
        
        Args:
            audio: 音频文件路径或numpy数组
            
        Returns:
            (语言代码, 概率)
        """
        # 加载音频
        if isinstance(audio, str):
            audio_array = whisper.load_audio(audio)
        else:
            audio_array = audio.astype(np.float32)
        
        # 只取前30秒用于语言检测
        audio_array = whisper.pad_or_trim(audio_array)
        
        # 计算mel频谱
        mel = whisper.log_mel_spectrogram(audio_array).to(self.device)
        
        # 检测语言
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        
        print(f"🌐 检测到语言: {detected_lang} (置信度: {probs[detected_lang]:.2%})")
        
        return detected_lang, probs[detected_lang]
    
    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'device': self.device,
            'model': self.model_name,
        }
        
        if self.device == 'cuda':
            info['cuda_available'] = torch.cuda.is_available()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_total'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        
        return info


# 测试代码
if __name__ == "__main__":
    # 初始化ASR
    asr = WhisperASR(model_name="base")
    
    # 打印设备信息
    print("\n设备信息:")
    for k, v in asr.get_device_info().items():
        print(f"  {k}: {v}")
    
    # 如果有测试音频文件，进行识别测试
    test_file = "test_recording.wav"
    if Path(test_file).exists():
        result = asr.transcribe(test_file, language="zh")
        print(f"\n识别结果: {result['text']}")
