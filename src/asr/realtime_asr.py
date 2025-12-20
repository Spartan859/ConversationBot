"""
实时语音识别模块
基于 Whisper 实现准实时的语音识别
"""

import numpy as np
import tempfile
import shutil
import subprocess
import os
import sounddevice as sd
import threading
import queue
import time
import signal
import sys
from typing import Optional, Callable, Generator
from collections import deque

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("warning: faster-whisper 未安装，将使用标准 whisper（速度较慢）")
    print("   安装命令: pip install faster-whisper")

import whisper
import torch


class RealtimeASR:
    """
    实时语音识别类
    
    实现方式：
    1. 分段录音 - 将音频分成小段进行识别
    2. VAD 检测 - 检测语音活动，静音时触发识别
    3. 滑动窗口 - 保持上下文连续性
    """
    
    # 简体中文引导提示
    SIMPLIFIED_CHINESE_PROMPT = "以下是普通话的句子，使用简体中文输出。"
    
    def __init__(
        self,
        model_name: str = "base",
        language: str = "zh",
        sample_rate: int = 16000,
        use_faster_whisper: bool = True,
        device: Optional[str] = None,
        audio_device: Optional[int] = None,
        initial_prompt: Optional[str] = None,
        use_simplified_chinese: bool = True
    ):
        """
        初始化实时ASR
        
        Args:
            model_name: 模型名称 (tiny/base/small)，实时场景建议用小模型
            language: 识别语言
            sample_rate: 采样率
            use_faster_whisper: 是否使用 faster-whisper 加速
            device: 运行设备 (cuda/cpu)
            audio_device: 录音设备索引
            initial_prompt: 初始提示文本，引导模型输出风格
            use_simplified_chinese: 是否使用简体中文提示（language="zh"时生效）
        """
        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.audio_device = audio_device
        
        # 设置 initial_prompt
        if initial_prompt is not None:
            self.initial_prompt = initial_prompt
        elif language == "zh" and use_simplified_chinese:
            self.initial_prompt = self.SIMPLIFIED_CHINESE_PROMPT
        else:
            self.initial_prompt = None
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=int(sample_rate * 30))  # 最多30秒
        self.audio_queue = queue.Queue()
        
        # 状态控制
        self.is_running = False
        self.is_speaking = False
        self._stream = None
        
        # VAD 参数
        self.vad_threshold = 0.01  # 语音活动阈值
        self.silence_duration = 0.8  # 静音持续时间(秒)触发识别
        self.min_speech_duration = 0.5  # 最小语音时长(秒)
        
        # 加载模型
        self._load_model(use_faster_whisper)
    
    def list_audio_devices(self, show: bool = True) -> list:
        """
        列出所有可用的录音设备
        
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
            print("\n可用录音设备:")
            print("-" * 60)
            for dev in input_devices:
                marker = " *" if dev['index'] == self.audio_device else ""
                print(f"  [{dev['index']:2d}] {dev['name'][:45]:<45}{marker}")
            print("-" * 60)
            if self.audio_device is not None:
                print(f"  (* 当前选择的设备)")
            else:
                print(f"  (未指定设备，将使用系统默认)")
            print()
        
        return input_devices
    
    def select_audio_device(self, device_index: Optional[int] = None) -> int:
        """
        选择录音设备
        
        Args:
            device_index: 设备索引，None则交互式选择
            
        Returns:
            选择的设备索引
        """
        if device_index is not None:
            input_devices = self.list_audio_devices(show=False)
            valid_indices = [d['index'] for d in input_devices]
            if device_index not in valid_indices:
                raise ValueError(f"无效的设备索引: {device_index}，可用索引: {valid_indices}")
            self.audio_device = device_index
            device_name = next(d['name'] for d in input_devices if d['index'] == device_index)
            print(f"[OK] 已选择录音设备: [{device_index}] {device_name}")
            return device_index
        
        # 交互式选择
        input_devices = self.list_audio_devices(show=True)
        
        if not input_devices:
            raise RuntimeError("未找到任何录音设备！")
        
        while True:
            try:
                choice = input("请输入设备编号 (直接回车使用默认设备): ").strip()
                if choice == "":
                    self.audio_device = None
                    print("[OK] 将使用系统默认录音设备")
                    return sd.default.device[0]
                
                device_index = int(choice)
                valid_indices = [d['index'] for d in input_devices]
                if device_index not in valid_indices:
                    print(f"[Error] 无效的设备编号，请选择: {valid_indices}")
                    continue
                    
                self.audio_device = device_index
                device_name = next(d['name'] for d in input_devices if d['index'] == device_index)
                print(f"[OK] 已选择录音设备: [{device_index}] {device_name}")
                return device_index
                
            except ValueError:
                print("[Error] 请输入有效的数字")
        
    def _load_model(self, use_faster_whisper: bool):
        """加载语音识别模型"""
        print(f"[Init] 初始化实时 ASR...")
        print(f"   模型: {self.model_name}")
        print(f"   设备: {self.device}")
        
        start_time = time.time()
        
        if use_faster_whisper and FASTER_WHISPER_AVAILABLE:
            # 使用 faster-whisper（推荐，速度快4倍）
            # 注意：CPU 模式下 int8 量化首次加载很慢，large-v3 可能需要数分钟
            # GPU 模式使用 float16，CPU 模式使用 float32（避免量化耗时）
            if self.device == "cuda":
                compute_type = "float16"
            else:
                # CPU 模式：int8 量化加载很慢但推理快，float32 加载快但推理慢
                # 对于 large 模型在 CPU 上建议用 float32 避免长时间等待
                if self.model_name in ["large", "large-v2", "large-v3", "medium"]:
                    compute_type = "float32"
                    print(f"   [提示] CPU 模式下大模型使用 float32 以加快加载速度")
                else:
                    compute_type = "int8"
            
            print(f"   计算类型: {compute_type}")
            print(f"   正在加载模型，请稍候...")
            
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type
            )
            self.use_faster = True
            print(f"   引擎: faster-whisper (加速版)")
        else:
            # 使用标准 whisper
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.use_faster = False
            print(f"   引擎: openai-whisper (标准版)")
        
        load_time = time.time() - start_time
        print(f"[OK] 模型加载完成！耗时: {load_time:.2f}秒")
    
    def _transcribe(self, audio: np.ndarray) -> str:
        print("进入_trasncribe")
        """执行识别"""
        if len(audio) < self.sample_rate * 0.3:  # 少于0.3秒不识别
            return ""
        
        audio = audio.astype(np.float32)
        
        if self.use_faster:
            # faster-whisper API
            segments, _ = self.model.transcribe(
                audio,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=self.initial_prompt
            )
            text = " ".join([seg.text for seg in segments])
        else:
            # 标准 whisper API
            result = self.model.transcribe(
                audio,
                language=self.language,
                fp16=(self.device == "cuda"),
                initial_prompt=self.initial_prompt
            )
            text = result['text']
        
        return text.strip()

    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样音频：

        优先使用 scipy.signal.resample_poly（效率高、抗伪影好），若不可用则回退到 librosa.resample。
        返回值为 float32 numpy 数组，长度大约为 target_sr / orig_sr * len(audio)。
        """
        # 优先使用 scipy 的 resample_poly（避免 STFT 的相位问题）
        try:
            import scipy.signal as sps
            from math import gcd
            g = gcd(orig_sr, target_sr)
            up = target_sr // g
            down = orig_sr // g
            audio = audio.astype(np.float32)
            res = sps.resample_poly(audio, up, down)
            return res
        except Exception:
            try:
                import librosa
                return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            except Exception:
                raise RuntimeError("需要 scipy 或 librosa 进行重采样，请安装 scipy 或 librosa")

    def transcribe(self, source) -> str:
        """兼容接口：接受文件路径（str）、(sample_rate, np.array) 或 numpy 数组并返回识别文本

        - 如果是文件路径，会使用 soundfile 读取并在需要时用 librosa 重采样
        - 如果是 (sr, array) 元组，会根据实例的 sample_rate 进行重采样（如需）
        - 如果是 numpy 数组，则视为已经是目标采样率的数据
        """
        try:
            # 文件路径
            if isinstance(source, str):
                import soundfile as sf
                orig_path = source
                audio, sr = sf.read(orig_path, dtype='float32')
                print(f"读取音频采样率: {sr}, 目标采样率: {self.sample_rate}")
                if audio.ndim > 1:
                    print(f"输入音频有 {audio.shape[1]} 个通道，已转换为单声道")
                    audio = np.mean(audio, axis=1)
                
                if sr != self.sample_rate:
                    # 优先使用系统 ffmpeg 进行快速重采样
                    ffmpeg_exe = shutil.which("ffmpeg")
                    if ffmpeg_exe:
                        print("使用 ffmpeg 进行快速重采样...")
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmpf:
                            tmp_out = tmpf.name
                        try:
                            # -ar 采样率, -ac 声道, -sample_fmt f32 保持 float32
                            subprocess.run(
                                [ffmpeg_exe, '-y', '-i', orig_path, '-ar', str(self.sample_rate), '-ac', '1', tmp_out],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            audio, sr = sf.read(tmp_out, dtype='float32')
                        finally:
                            try:
                                os.remove(tmp_out)
                            except Exception:
                                pass
                    else:
                        print("正在重采样音频...")
                        audio = self._resample_audio(audio, sr, self.sample_rate)
            # Gradio 返回的 (sr, array)
            elif isinstance(source, tuple) or isinstance(source, list):
                sr, audio = source
                audio = np.asarray(audio, dtype=np.float32)
                if sr != self.sample_rate:
                    # 如果系统有 ffmpeg，先写临时 wav 再用 ffmpeg 转换
                    ffmpeg_exe = shutil.which("ffmpeg")
                    if ffmpeg_exe:
                        print("使用 ffmpeg 进行快速重采样 (临时文件)...")
                        import soundfile as sf
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_in:
                            in_path = tmp_in.name
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                            out_path = tmp_out.name
                        try:
                            sf.write(in_path, audio, sr, format='WAV')
                            subprocess.run(
                                [ffmpeg_exe, '-y', '-i', in_path, '-ar', str(self.sample_rate), '-ac', '1', '-acodec', 'pcm_s16le', out_path],
                                check=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL
                            )
                            audio, _ = sf.read(out_path, dtype='float32')
                        finally:
                            for p in (in_path, out_path):
                                try:
                                    os.remove(p)
                                except Exception:
                                    pass
                    else:
                        print("正在重采样音频...")
                        audio = self._resample_audio(audio, sr, self.sample_rate)
            # numpy 数组
            elif isinstance(source, np.ndarray):
                audio = source.astype(np.float32)
            else:
                raise ValueError("Unsupported audio source type for transcribe()")

            return self._transcribe(audio)
        except Exception as e:
            print(f"[Error] transcribe 失败: {e}")
            return ""
    
    def _calculate_energy(self, audio: np.ndarray) -> float:
        """计算音频能量（用于VAD）"""
        return np.sqrt(np.mean(audio ** 2))
    
    def _audio_callback(self, indata, frames, time_info, status):
        """音频输入回调"""
        if status:
            print(f"[Warning] 音频状态: {status}")
        if self.is_running:
            self.audio_queue.put(indata.copy())
    
    def _setup_signal_handler(self):
        """设置信号处理器（只在主线程注册）"""
        # signal.signal 只能在主线程注册，否则会抛出 ValueError
        import threading
        if threading.current_thread() is not threading.main_thread():
            print("[Warning] 信号处理器只能在主线程注册，跳过信号注册（在子线程中运行）")
            return

        def signal_handler(signum, frame):
            print("\n\n[Stop] 收到停止信号，正在退出...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def stop(self):
        """停止识别"""
        self.is_running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass
            self._stream = None
        print("[Stop] 识别已停止")
    
    def start_realtime(
        self,
        callback: Callable[[str], None],
        device: Optional[int] = None,
        chunk_duration: float = 0.5
    ):
        """
        启动实时识别（基于VAD）
        
        Args:
            callback: 识别结果回调函数 callback(text)
            device: 录音设备索引，None则使用实例默认设备
            chunk_duration: 音频块时长(秒)
        """
        self._setup_signal_handler()
        self.is_running = True
        
        use_device = device if device is not None else self.audio_device
        chunk_size = int(self.sample_rate * chunk_duration)
        
        print("\n" + "=" * 50)
        print("[Start] 实时语音识别已启动")
        print("   说话后会自动识别")
        print("   按 Ctrl+C 或输入 q 停止")
        print("=" * 50 + "\n")
        
        # 启动音频输入流
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            device=use_device,
            blocksize=chunk_size,
            callback=self._audio_callback
        )
        
        # 状态变量
        speech_buffer = []
        silence_frames = 0
        silence_threshold = int(self.silence_duration / chunk_duration)
        min_speech_frames = int(self.min_speech_duration / chunk_duration)
        
        try:
            self._stream.start()
            
            while self.is_running:
                try:
                    # 获取音频块，设置较短超时以便响应停止信号
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                    audio_chunk = audio_chunk.flatten()
                    
                    # 计算能量
                    energy = self._calculate_energy(audio_chunk)
                    
                    if energy > self.vad_threshold:
                        # 检测到语音
                        if not self.is_speaking:
                            self.is_speaking = True
                            print("[Voice] 检测到语音...", end="", flush=True)
                        
                        speech_buffer.append(audio_chunk)
                        silence_frames = 0
                    else:
                        # 静音
                        if self.is_speaking:
                            silence_frames += 1
                            speech_buffer.append(audio_chunk)  # 保留一些静音
                            
                            if silence_frames >= silence_threshold:
                                # 静音时间足够，触发识别
                                if len(speech_buffer) >= min_speech_frames:
                                    audio = np.concatenate(speech_buffer)
                                    print(f" 识别中...")
                                    
                                    text = self._transcribe(audio)
                                    if text:
                                        callback(text)
                                else:
                                    print(" (太短，已忽略)")
                                
                                # 重置状态
                                speech_buffer = []
                                self.is_speaking = False
                                silence_frames = 0
                
                except queue.Empty:
                    continue
                        
        except Exception as e:
            print(f"\n[Error] 发生错误: {e}")
        finally:
            self.stop()
    
    def start_continuous(
        self,
        callback: Callable[[str], None],
        device: Optional[int] = None,
        segment_duration: float = 3.0,
        overlap: float = 0.5
    ):
        """
        启动连续识别（固定时间分段）
        
        Args:
            callback: 识别结果回调函数
            device: 录音设备索引，None则使用实例默认设备
            segment_duration: 每段时长(秒)
            overlap: 重叠时长(秒)，用于保持上下文
        """
        self._setup_signal_handler()
        self.is_running = True
        
        use_device = device if device is not None else self.audio_device
        
        print("\n" + "=" * 50)
        print("[Start] 连续语音识别已启动")
        print(f"   每 {segment_duration} 秒识别一次")
        print("   按 Ctrl+C 停止")
        print("=" * 50 + "\n")
        
        segment_samples = int(self.sample_rate * segment_duration)
        
        try:
            while self.is_running:
                # 录制一段音频
                print("[Rec] 录音中...", end="", flush=True)
                audio = sd.rec(
                    segment_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    device=use_device
                )
                sd.wait()
                
                if not self.is_running:
                    break
                    
                audio = audio.flatten()
                
                # 识别
                print(" 识别中...", end="", flush=True)
                text = self._transcribe(audio)
                
                if text:
                    print(f" [OK]")
                    callback(text)
                else:
                    print(f" (无语音)")
                    
        except Exception as e:
            print(f"\n[Error] 发生错误: {e}")
        finally:
            self.is_running = False
            print("\n[Stop] 连续识别已停止")
    
    def transcribe_stream(
        self,
        device: Optional[int] = None,
        segment_duration: float = 3.0
    ) -> Generator[str, None, None]:
        """
        流式识别生成器
        
        Args:
            device: 录音设备索引
            segment_duration: 每段时长
            
        Yields:
            识别的文本
        """
        self.is_running = True
        use_device = device if device is not None else self.audio_device
        segment_samples = int(self.sample_rate * segment_duration)
        
        try:
            while self.is_running:
                audio = sd.rec(
                    segment_samples,
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    device=use_device
                )
                sd.wait()
                
                if not self.is_running:
                    break
                    
                audio = audio.flatten()
                
                text = self._transcribe(audio)
                if text:
                    yield text
                    
        except GeneratorExit:
            pass
        finally:
            self.is_running = False


# 测试代码
if __name__ == "__main__":
    print("=" * 50)
    print("    实时语音识别测试")
    print("=" * 50)
    
    # 初始化实时ASR
    asr = RealtimeASR(
        model_name="large-v3",  # 实时场景用小模型
        language="zh",
        use_faster_whisper=True  # 推荐开启加速
    )
    
    # 选择录音设备
    asr.select_audio_device()
    
    # 定义回调函数
    def on_text(text: str):
        print(f"[Result] 识别结果: {text}")
    
    print("\n选择识别模式:")
    print("  1. VAD 实时识别（说完自动识别）")
    print("  2. 连续分段识别（每3秒识别一次）")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        asr.start_realtime(callback=on_text)
    else:
        asr.start_continuous(callback=on_text, segment_duration=3.0)
