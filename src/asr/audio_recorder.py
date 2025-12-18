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
        dtype: str = 'float32',
        device: Optional[int] = None
    ):
        """
        初始化音频录制器
        
        Args:
            sample_rate: 采样率，Whisper推荐16kHz
            channels: 声道数，单声道=1
            dtype: 数据类型
            device: 默认录音设备索引，None则使用系统默认
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device = device
        self.recording = False
        self.audio_data = []
        
    def list_devices(self, show: bool = True) -> list:
        """
        列出所有可用的音频输入设备
        
        Args:
            show: 是否打印设备列表
            
        Returns:
            输入设备列表
        """
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
        
        if show:
            print("\n🎙️ 可用录音设备:")
            print("-" * 60)
            for dev in input_devices:
                marker = " *" if dev['index'] == self.device else ""
                print(f"  [{dev['index']:2d}] {dev['name'][:45]:<45}{marker}")
            print("-" * 60)
            if self.device is not None:
                print(f"  (* 当前选择的设备)")
            else:
                print(f"  (未指定设备，将使用系统默认)")
            print()
        
        return input_devices
    
    def select_device(self, device_index: Optional[int] = None) -> int:
        """
        选择录音设备
        
        Args:
            device_index: 设备索引，None则交互式选择
            
        Returns:
            选择的设备索引
        """
        if device_index is not None:
            # 验证设备索引是否有效
            input_devices = self.list_devices(show=False)
            valid_indices = [d['index'] for d in input_devices]
            if device_index not in valid_indices:
                raise ValueError(f"无效的设备索引: {device_index}，可用索引: {valid_indices}")
            self.device = device_index
            device_name = next(d['name'] for d in input_devices if d['index'] == device_index)
            print(f"✅ 已选择录音设备: [{device_index}] {device_name}")
            return device_index
        
        # 交互式选择
        input_devices = self.list_devices(show=True)
        
        if not input_devices:
            raise RuntimeError("未找到任何录音设备！")
        
        while True:
            try:
                choice = input("请输入设备编号 (直接回车使用默认设备): ").strip()
                if choice == "":
                    self.device = None
                    print("✅ 将使用系统默认录音设备")
                    return sd.default.device[0]  # 返回默认输入设备
                
                device_index = int(choice)
                valid_indices = [d['index'] for d in input_devices]
                if device_index not in valid_indices:
                    print(f"❌ 无效的设备编号，请选择: {valid_indices}")
                    continue
                    
                self.device = device_index
                device_name = next(d['name'] for d in input_devices if d['index'] == device_index)
                print(f"✅ 已选择录音设备: [{device_index}] {device_name}")
                return device_index
                
            except ValueError:
                print("❌ 请输入有效的数字")
    
    def get_current_device(self) -> dict:
        """
        获取当前选择的设备信息
        
        Returns:
            设备信息字典
        """
        if self.device is None:
            default_input = sd.default.device[0]
            if default_input is not None:
                device_info = sd.query_devices(default_input)
                return {
                    'index': default_input,
                    'name': device_info['name'],
                    'channels': device_info['max_input_channels'],
                    'sample_rate': device_info['default_samplerate'],
                    'is_default': True
                }
            return None
        else:
            device_info = sd.query_devices(self.device)
            return {
                'index': self.device,
                'name': device_info['name'],
                'channels': device_info['max_input_channels'],
                'sample_rate': device_info['default_samplerate'],
                'is_default': False
            }
    
    def record_fixed_duration(
        self,
        duration: float,
        device: Optional[int] = None
    ) -> np.ndarray:
        """
        录制固定时长的音频
        
        Args:
            duration: 录制时长（秒）
            device: 音频设备索引，None则使用实例默认设备
            
        Returns:
            音频数据 numpy数组
        """
        use_device = device if device is not None else self.device
        device_info = self.get_current_device() if use_device is None else sd.query_devices(use_device)
        device_name = device_info['name'] if isinstance(device_info, dict) else device_info.get('name', '未知')
        
        print(f"🎤 开始录音，时长 {duration} 秒...")
        print(f"   设备: {device_name}")
        
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            device=use_device
        )
        sd.wait()  # 等待录制完成
        
        print("✅ 录音完成！")
        return audio.flatten()
    
    def start_recording(self, device: Optional[int] = None):
        """
        开始录音（非阻塞）
        
        Args:
            device: 音频设备索引，None则使用实例默认设备
        """
        use_device = device if device is not None else self.device
        
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
            device=use_device,
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
    
    # 列出并选择设备
    print("=" * 60)
    print("        音频录制测试")
    print("=" * 60)
    
    # 交互式选择录音设备
    recorder.select_device()
    
    # 显示当前设备信息
    current = recorder.get_current_device()
    if current:
        print(f"\n当前设备信息:")
        print(f"  名称: {current['name']}")
        print(f"  通道数: {current['channels']}")
        print(f"  默认采样率: {current['sample_rate']}")
    
    # 录制5秒音频
    print("\n准备录制 5 秒音频...")
    input("按回车键开始录音...")
    
    audio = recorder.record_fixed_duration(5)
    
    # 保存音频
    recorder.save_audio(audio, "test_recording.wav")
    print("\n测试完成！")
