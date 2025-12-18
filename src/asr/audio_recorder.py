"""
音频录制模块
使用 sounddevice 实现麦克风音频录制
"""

import numpy as np
import sounddevice as sd
import soundfile as sf
from pathlib import Path
from typing import Optional
import tempfile
import time


class AudioRecorder:
    """音频录制器类"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = 'float32'
    ):
        """
        初始化音频录制器
        
        Args:
            sample_rate: 采样率，Whisper推荐16kHz
            channels: 声道数，单声道=1
            dtype: 数据类型
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.recording = False
        self.audio_data = []
        
    def list_devices(self) -> list:
        """列出所有可用的音频设备"""
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        return input_devices
    
    def record_fixed_duration(
        self,
        duration: float,
        device: Optional[int] = None
    ) -> np.ndarray:
        """
        录制固定时长的音频
        
        Args:
            duration: 录制时长（秒）
            device: 音频设备索引，None则使用默认设备
            
        Returns:
            音频数据 numpy数组
        """
        print(f"🎤 开始录音，时长 {duration} 秒...")
        
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=device
        )
        sd.wait()  # 等待录制完成
        
        print("✅ 录音完成！")
        return audio.flatten()
    
    def start_recording(self, device: Optional[int] = None):
        """
        开始录音（非阻塞）
        
        Args:
            device: 音频设备索引
        """
        self.audio_data = []
        self.recording = True
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"录音状态: {status}")
            if self.recording:
                self.audio_data.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=device,
            callback=callback
        )
        self.stream.start()
        print("🎤 录音已开始，调用 stop_recording() 停止...")
        
    def stop_recording(self) -> np.ndarray:
        """
        停止录音并返回音频数据
        
        Returns:
            音频数据 numpy数组
        """
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0)
            print(f"✅ 录音停止，时长: {len(audio) / self.sample_rate:.2f} 秒")
            return audio.flatten()
        else:
            print("⚠️ 未录制到音频数据")
            return np.array([])
    
    def save_audio(
        self,
        audio: np.ndarray,
        filepath: str,
        sample_rate: Optional[int] = None
    ):
        """
        保存音频到文件
        
        Args:
            audio: 音频数据
            filepath: 保存路径
            sample_rate: 采样率
        """
        sr = sample_rate or self.sample_rate
        sf.write(filepath, audio, sr)
        print(f"💾 音频已保存到: {filepath}")
        
    def load_audio(self, filepath: str) -> tuple:
        """
        从文件加载音频
        
        Args:
            filepath: 音频文件路径
            
        Returns:
            (音频数据, 采样率)
        """
        audio, sr = sf.read(filepath)
        print(f"📂 已加载音频: {filepath}, 采样率: {sr}, 时长: {len(audio)/sr:.2f}秒")
        return audio, sr


# 测试代码
if __name__ == "__main__":
    recorder = AudioRecorder()
    
    # 列出设备
    print("可用音频设备:")
    for device in recorder.list_devices():
        print(f"  [{device['index']}] {device['name']}")
    
    # 录制5秒音频
    audio = recorder.record_fixed_duration(5)
    
    # 保存音频
    recorder.save_audio(audio, "test_recording.wav")
