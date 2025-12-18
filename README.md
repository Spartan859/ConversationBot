# 语音对话系统 - 大作业报告

> **课程**：人工智能导论  
> **作业**：搭建语音对话系统  
> **日期**：2025年12月

---

## 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [模块一：语音识别 (ASR)](#模块一语音识别-asr)
4. [模块二：对话模型](#模块二对话模型)
5. [模块三：语音合成 (TTS)](#模块三语音合成-tts)
6. [系统集成与演示](#系统集成与演示)
7. [实验结果与分析](#实验结果与分析)
8. [总结与感悟](#总结与感悟)

---

## 项目概述

本项目旨在搭建一个完整的语音对话系统，实现**语音输入 → 智能对话 → 语音输出**的端到端交互体验。

### 实现要求

- ✅ 基于开源框架和模型搭建
- ✅ 输入为语音，输出为语音
- ✅ 支持多轮对话
- ✅ 支持声音克隆（加分项）

### 所用工具与资源

| 类别 | 工具/模型 | 版本 |
|------|-----------|------|
| 语音识别 | OpenAI Whisper | base/small |
| 对话模型 | OpenAI GPT / Ollama | - |
| 语音合成 | Edge-TTS | 6.1.9+ |
| 知识库 | KdConv (清华) | - |
| Web框架 | Gradio | 4.0+ |
| 编程语言 | Python | 3.10+ |

### 计算资源

| 资源 | 配置 |
|------|------|
| CPU | Intel/AMD 多核处理器 |
| GPU | NVIDIA GPU (可选，加速推理) |
| 内存 | 8GB+ |
| 显存 | 2GB+ (使用small模型) |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     语音对话系统架构                          │
└─────────────────────────────────────────────────────────────┘

     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │   ASR    │      │ Dialogue │      │   TTS    │
     │ (Whisper)│ ──▶  │  (LLM)   │ ──▶  │(Edge-TTS)│
     └──────────┘      └──────────┘      └──────────┘
          ▲                 │                 │
          │                 ▼                 ▼
     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │ 麦克风   │      │ KdConv   │      │  扬声器  │
     │ 输入     │      │ 知识库   │      │  输出    │
     └──────────┘      └──────────┘      └──────────┘
```

---

## 模块一：语音识别 (ASR)

### 1.1 技术选型

经过调研对比，选择 **OpenAI Whisper** 作为语音识别引擎：

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **Whisper** | 多语言、高准确率、开源 | 需要GPU加速 | ✅ |
| FunASR | 中文优化、实时性好 | 文档较少 | |
| Google API | 简单易用 | 需联网、有限制 | |

**选择理由**：
1. Whisper 支持 99 种语言，中英文效果都很好
2. 完全开源免费，可本地部署
3. 提供多种模型尺寸，可根据硬件选择

### 1.2 Whisper 模型对比

| 模型 | 参数量 | 显存需求 | 相对速度 | 准确率 |
|------|--------|----------|----------|--------|
| tiny | 39M | ~1GB | ~32x | ⭐⭐ |
| base | 74M | ~1GB | ~16x | ⭐⭐⭐ |
| small | 244M | ~2GB | ~6x | ⭐⭐⭐⭐ |
| medium | 769M | ~5GB | ~2x | ⭐⭐⭐⭐⭐ |
| large-v3 | 1550M | ~10GB | ~1x | ⭐⭐⭐⭐⭐ |

**本项目选择**：`base` 或 `small` 模型，在速度和准确率之间取得平衡。

### 1.3 搭建过程

#### 步骤1：安装依赖

```bash
# 创建 conda 虚拟环境
conda create -n conversation-bot python=3.10 -y
conda activate conversation-bot

# 安装 PyTorch (根据CUDA版本选择)
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# 或 CPU 版本: conda install pytorch torchaudio cpuonly -c pytorch

# 安装其他依赖
pip install openai-whisper
pip install sounddevice soundfile
```

#### 步骤2：实现音频录制模块

创建 `src/asr/audio_recorder.py`：

```python
import sounddevice as sd
import numpy as np

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
    
    def record_fixed_duration(self, duration):
        """录制固定时长音频"""
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()
```

**技巧**：
- 采样率设置为 16kHz，这是 Whisper 的推荐值
- 使用单声道录制，减少数据量
- `sd.wait()` 确保录制完成

#### 步骤3：实现 Whisper 语音识别

创建 `src/asr/whisper_asr.py`：

```python
import whisper
import torch

class WhisperASR:
    def __init__(self, model_name="base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)
    
    def transcribe(self, audio, language=None):
        """语音转文字"""
        result = self.model.transcribe(
            audio,
            language=language,
            task="transcribe"
        )
        return result
```

### 1.4 遇到的问题与解决

#### 问题1：模型下载缓慢

**现象**：首次加载模型时下载速度很慢

**解决方案**：
```python
# 方法1：设置代理
import os
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

# 方法2：手动下载模型到本地
whisper.load_model("base", download_root="./models")
```

#### 问题2：音频格式不兼容

**现象**：传入音频数据报错 `expected float32`

**解决方案**：
```python
# 确保音频数据类型正确
audio = audio.astype(np.float32)
```

#### 问题3：GPU显存不足

**现象**：使用 large 模型时 CUDA OOM

**解决方案**：
```python
# 使用较小模型
model = whisper.load_model("small")  # 而非 "large"

# 或强制使用 CPU
model = whisper.load_model("base", device="cpu")
```

### 1.5 测试与效果

#### 测试代码

```python
from src.asr import AudioRecorder, WhisperASR

# 录制音频
recorder = AudioRecorder()
audio = recorder.record_fixed_duration(5)  # 录5秒

# 语音识别
asr = WhisperASR(model_name="base")
result = asr.transcribe(audio, language="zh")
print(f"识别结果: {result['text']}")
```

#### 识别效果对比

| 测试内容 | tiny | base | small |
|----------|------|------|-------|
| "你好，世界" | ✅ 你好世界 | ✅ 你好，世界 | ✅ 你好，世界 |
| "今天天气怎么样" | ✅ 今天天气怎么样 | ✅ 今天天气怎么样 | ✅ 今天天气怎么样 |
| 英文混合 | ⚠️ 部分错误 | ✅ 正确 | ✅ 正确 |
| 识别速度 | ~0.3s | ~0.8s | ~2s |

### 1.6 模块总结

**ASR模块实现效果**：
- ✅ 成功集成 Whisper 语音识别
- ✅ 支持实时录音和文件识别
- ✅ 支持中英文及多语言
- ✅ 支持自动语言检测

**代码结构**：
```
src/asr/
├── __init__.py
├── audio_recorder.py   # 音频录制
└── whisper_asr.py      # Whisper 语音识别
```

---

## 模块二：对话模型

> 🚧 待实现...

---

## 模块三：语音合成 (TTS)

> 🚧 待实现...

---

## 系统集成与演示

> 🚧 待实现...

---

## 实验结果与分析

> 🚧 待完善...

---

## 总结与感悟

### 搭建感悟

1. **模块化设计的重要性**：将系统拆分为 ASR、Dialogue、TTS 三个模块，便于独立开发和测试

2. **开源生态的便利**：Whisper、Edge-TTS 等开源工具大大降低了开发门槛

3. **硬件资源的权衡**：需要根据实际硬件选择合适的模型尺寸

### 收获

- 掌握了语音识别系统的搭建流程
- 了解了 Whisper 模型的原理和使用
- 学会了音频信号处理的基本方法

### 待改进

- [ ] 优化识别延迟
- [ ] 添加噪声抑制
- [ ] 支持流式识别

---

## 参考资料

1. [OpenAI Whisper](https://github.com/openai/whisper)
2. [Whisper 论文](https://arxiv.org/abs/2212.04356)
3. [KdConv 数据集](https://github.com/thu-coai/KdConv)
4. [Edge-TTS](https://github.com/rany2/edge-tts)
5. [Gradio 文档](https://gradio.app/docs)
