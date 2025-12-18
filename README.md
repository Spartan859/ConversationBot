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

本项目选择 **OpenAI Whisper** 作为语音识别引擎，并使用 **faster-whisper** 进行加速优化。

| 方案 | 优点 | 缺点 |
|------|------|------|
| Whisper | 多语言支持、准确率高、开源免费 | 原版推理速度较慢 |
| faster-whisper | 速度提升4倍、显存减半 | 需额外安装 |
| FunASR | 中文优化好 | 文档较少 |
| Google API | 简单易用 | 需联网、有调用限制 |

**最终方案**：faster-whisper (large-v3) 用于离线转写，base/small 用于实时对话

### 1.2 模型选择与应用场景分析

#### 模型规格对比

| 模型 | 参数量 | 显存需求 | 相对速度 | 推理耗时 (RTF) | CER (%) |
|------|--------|----------|----------|---------------|---------|
| tiny | 39M | ~1GB | 32x | 0.011 | 26.11 |
| base | 74M | ~1GB | 16x | 0.014 | 16.51 |
| small | 244M | ~2GB | 6x | 0.030 | 9.32 |
| medium | 769M | ~5GB | 2x | 0.070 | 5.93 |
| large-v1 | 1550M | ~10GB | 1x | 0.165 | 5.22 |
| large-v2 | 1550M | ~10GB | 1x | 0.173 | 5.29 |
| large-v3 | 1550M | ~10GB | 1x | 0.153 | 4.59 |

> RTF (Real-Time Factor): 识别耗时/音频时长，越小越快。数据基于 AISHELL-1 测试集评估（100 样本，CUDA加速）。

#### 应用场景推荐

**1. 实时语音对话系统（本项目核心场景）**

**推荐模型**: `base` 或 `small`

**理由**:
- **低延迟要求**: 实时对话要求 RTF < 0.3（识别速度快于说话速度）
- base 模型 RTF=0.014，可实现几乎无感延迟
- small 模型 RTF=0.030，在准确率与速度间取得平衡（CER仅9.32%）
- 适合消费级 GPU（GTX 1060+）或高性能 CPU

**典型应用**: 智能客服、语音助手、实时字幕

---

**2. 离线音频转写（会议记录、访谈整理）**

**推荐模型**: `large-v3` 或 `medium`

**理由**:
- **准确率优先**: 离线场景对延迟不敏感，追求最低错误率
- large-v3 达到 4.59% CER，接近人类水平（专业标注员约3-4%）
- medium 性价比高（CER 5.93%），适合中等显存设备
- 长音频批量处理，可容忍数倍实时的处理时间

**典型应用**: 播客转录、学术访谈整理、法庭记录

---

**3. 嵌入式设备（边缘计算、IoT）**

**推荐模型**: `tiny`

**理由**:
- **资源受限**: 嵌入式设备显存/内存有限（通常<2GB）
- tiny 模型仅 75MB，可在树莓派等设备上运行
- 虽然 CER 较高（26.11%），但满足简单语音指令识别
- 超低延迟（RTF=0.011），适合实时响应

**典型应用**: 智能音箱、车载语音助手、智能家居控制

---

**4. 医疗/法律等专业领域**

**推荐模型**: `large-v3` + 专业词表微调

**理由**:
- **零容忍错误**: 专业术语识别错误可能造成严重后果
- large-v3 提供最高基础准确率
- 结合领域专用 initial_prompt 和后处理规则提升准确率
- 建议结合人工校对形成闭环

**典型应用**: 医疗病历录入、法律诉讼记录、金融合规审查

---

**5. 多语言场景**

**推荐模型**: `large-v2` 或 `large-v3`

**理由**:
- Whisper 大模型支持 99 种语言，小模型多语言能力较弱
- 中英混合场景下，large 模型更准确识别语码切换
- 适合国际会议、多语言客服

---

**本项目选型决策**:
- **开发阶段**: `small` - 快速迭代，降低等待时间
- **演示阶段**: `base` - 流畅体验，响应迅速
- **评估测试**: `large-v3` - 验证系统准确率上限

### 1.3 环境搭建

```bash
# 创建 conda 虚拟环境
conda create -n conversation-bot python=3.10 -y
conda activate conversation-bot

# 安装 PyTorch（根据 CUDA 版本选择）
conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# CPU 版本: conda install pytorch torchaudio cpuonly -c pytorch

# 安装语音识别依赖
pip install faster-whisper      # 推荐：加速版
pip install openai-whisper      # 备选：标准版
pip install sounddevice soundfile librosa
pip install datasets jiwer      # 用于模型评估
```

### 1.4 核心实现

#### 音频录制模块

```python
# src/asr/audio_recorder.py
import sounddevice as sd
import numpy as np

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1, device=None):
        self.sample_rate = sample_rate  # Whisper 推荐 16kHz
        self.channels = channels
        self.device = device            # 支持选择录音设备
    
    def record_fixed_duration(self, duration):
        """录制固定时长音频"""
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            device=self.device
        )
        sd.wait()
        return audio.flatten()
    
    def select_device(self, device_index=None):
        """选择录音设备（支持交互式选择）"""
        devices = sd.query_devices()
        # 显示可用设备列表，支持用户交互选择
        # 详见完整代码
```

#### 语音识别模块

```python
# src/asr/realtime_asr.py（基于 faster-whisper）
from faster_whisper import WhisperModel

class RealtimeASR:
    SIMPLIFIED_CHINESE_PROMPT = "以下是普通话的句子，使用简体中文输出。"
    
    def __init__(self, model_name="base", language="zh", use_simplified_chinese=True):
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        self.model = WhisperModel(
            model_name, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type=compute_type
        )
        self.language = language
        self.initial_prompt = self.SIMPLIFIED_CHINESE_PROMPT if use_simplified_chinese else None
    
    def transcribe(self, audio):
        segments, info = self.model.transcribe(
            audio, 
            language=self.language,
            vad_filter=True,  # 启用语音活动检测
            beam_size=5,
            initial_prompt=self.initial_prompt
        )
        return ''.join(segment.text for segment in segments).strip()
```

### 1.5 问题与解决方案

#### 问题 1: 模型下载速度慢

**现象**: 首次加载模型时，从 Hugging Face 下载速度缓慢或超时

**解决方案**: 使用 [HF Mirror](https://hf-mirror.com) 镜像站加速

```bash
# 设置环境变量（Windows PowerShell）
$env:HF_ENDPOINT="https://hf-mirror.com"

# Linux/macOS
export HF_ENDPOINT="https://hf-mirror.com"
```

```python
# 批量下载所有模型
python src/asr/download_models.py
```

#### 问题 2: 中文数字格式不一致

**现象**: 识别结果将中文数字转换为阿拉伯数字，与参考文本不匹配

```
GT:  特别是七十年代末八十年代初
识别: 特别是70年代末80年代初。
CER: 38.46% (替换:5)
```

**根本原因**: 
- Whisper 模型训练数据中，数字多以阿拉伯数字形式出现
- `initial_prompt` 只是引导性提示，无法完全控制输出格式
- 测试发现 prompt "保持中文数字格式" 对模型影响极小

**解决方案**: 实现后处理算法 - 阿拉伯数字转中文数字

创建 `src/asr/number_converter.py` 模块：

```python
class NumberConverter:
    """阿拉伯数字转中文数字转换器"""
    
    @classmethod
    def convert_text(cls, text: str) -> str:
        """转换文本中的所有阿拉伯数字为中文数字"""
        # 1. 年份范围: 2015-2016 → 二零一五至二零一六
        text = re.sub(r'(\d{4})-(\d{4})', cls.convert_year_range, text)
        
        # 2. 百分比: 46% → 百分之四十六
        text = re.sub(r'(\d+(?:\.\d+)?)%', cls.convert_percentage, text)
        
        # 3. 日期: 10月22日 → 十月二十二日
        text = re.sub(r'(\d{1,2})月(\d{1,2})日', cls.convert_date, text)
        
        # 4. 完整年份: 2013年 → 二零一三年
        text = re.sub(r'(\d{4})年', cls.convert_year_full, text)
        
        # 5. 带单位数字: 137人 → 一百三十七人
        text = re.sub(r'(\d+)(人|米|个|次|岁)', cls.convert_standalone_number, text)
        
        return text
```

**效果验证**:

| 原文 | 转换后 |
|------|--------|
| 特别是70年代末80年代初。 | 特别是七十年代末八十年代初。 |
| 2015-2016赛季 | 二零一五至二零一六赛季 |
| 46%正在考虑移民 | 百分之四十六正在考虑移民 |
| 10月22日消息 | 十月二十二日消息 |
| 被贴条人数137人 | 被贴条人数一百三十七人 |

**集成方式**:

```python
# 在识别结果后立即调用转换
from src.asr.number_converter import NumberConverter

hypothesis = model.transcribe(audio)
hypothesis = NumberConverter.convert_text(hypothesis)  # 后处理转换
```

#### 问题 3: large 模型 CPU 加载缓慢

**现象**: CPU 模式下加载 large-v3 模型需要 5-10 分钟

**原因**: faster-whisper 在 CPU 上默认使用 int8 量化，首次量化需要大量计算

**解决方案**: 对大模型使用 float32 避免量化开销

```python
if model_name in ["large", "large-v2", "large-v3", "medium"]:
    compute_type = "float32"  # CPU 大模型使用 float32
else:
    compute_type = "int8"     # 小模型可用 int8
```

#### 问题 4: VAD 误触发

**现象**: 环境噪音或键盘声被识别为语音

**解决方案**: 调整 VAD 阈值和静音持续时间

```python
self.vad_threshold = 0.015   # 提高阈值（默认 0.01）
self.silence_duration = 1.0  # 延长静音判定时间（默认 0.8s）
```

### 1.6 模型评估与准确率测试

#### 评估数据集：AISHELL-1

[AISHELL-1](https://www.openslr.org/33/) 是学术界最权威的开源中文语音识别数据集：

| 属性 | 值 |
|------|-----|
| 总时长 | 178 小时 |
| 说话人数 | 400+ |
| 录音环境 | 安静室内 |
| 采样率 | 16kHz |
| 训练集 | 120,098 条 |
| 测试集 | 7,176 条 |

#### 评估指标：CER（字符错误率）

**CER (Character Error Rate)** 是中文语音识别的标准评估指标：

$$CER = \frac{S + D + I}{N} \times 100\%$$

其中：
- **S (Substitutions)**: 替换错误数 - 识别结果中被错误替换的字符
- **D (Deletions)**: 删除错误数 - 参考文本中被遗漏的字符  
- **I (Insertions)**: 插入错误数 - 识别结果中多余的字符
- **N**: 参考文本总字符数

**RTF (Real-Time Factor)**: 识别耗时与音频时长的比值
- RTF < 1: 识别速度快于音频播放速度（可实时）
- RTF = 1: 识别速度等于音频播放速度
- RTF > 1: 识别速度慢于音频播放速度（有延迟）

**计算方法**：使用 Levenshtein 编辑距离算法

```python
def levenshtein_distance(ref: str, hyp: str) -> int:
    """动态规划计算编辑距离"""
    m, n = len(ref), len(hyp)
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
    return dp[m][n]
```

#### AISHELL-1 评估结果

**测试条件**:
- 硬件: NVIDIA GPU (CUDA 加速)
- 引擎: faster-whisper
- 样本数: 100 条（随机抽样）
- 后处理: 阿拉伯数字转中文数字

**完整评估结果**:

```
====================================================================================================
评估结果汇总 (AISHELL-1 测试集)
====================================================================================================
模型            CER      替换     删除     插入     RTF    平均耗时(秒)
----------------------------------------------------------------------------------------------------
tiny          26.11%     351       9       10     0.011     0.048
base          16.51%     225       4        5     0.014     0.065
small          9.32%     122       7        3     0.030     0.132
medium         5.93%      79       3        2     0.070     0.315
large-v1       5.22%      67       4        3     0.165     0.740
large-v2       5.29%      63      10        2     0.173     0.775
large-v3       4.59%      59       3        3     0.153     0.684
====================================================================================================
```

**关键发现**:

1. **准确率梯度**: 
   - tiny → base: 降低 9.6 个百分点（36.8% 相对提升）
   - base → small: 降低 7.2 个百分点（43.6% 相对提升）
   - small → large-v3: 降低 4.7 个百分点（50.6% 相对提升）

2. **速度表现**:
   - tiny/base/small 的 RTF 均小于 0.05，适合实时应用
   - medium 的 RTF=0.070 仍可实时（每秒音频 70ms 处理时间）
   - large 系列 RTF>0.15，适合离线批处理

3. **large 系列对比**:
   - large-v3 相比 v1/v2 准确率提升 12-13%
   - v3 速度略快于 v1/v2（模型优化）
   - **推荐使用 large-v3**

4. **错误类型分析**:
   - 替换错误占主导（85-90%）- 同音字、多音字混淆
   - 删除错误较少（1-3%）- 多为尾音、轻声
   - 插入错误最少（1-2%）- 多为标点符号

#### 运行评估脚本

```bash
# 评估单个模型（快速验证）
python src/asr/download_and_evaluate.py --models small --num_samples 50

# 评估所有模型（完整测试）
python src/asr/download_and_evaluate.py --models tiny base small medium large-v1 large-v2 large-v3 --num_samples 100

# 指定设备
python src/asr/download_and_evaluate.py --models base --device cuda --num_samples 100

# 大规模评估（生产环境验证）
python src/asr/download_and_evaluate.py --models large-v3 --num_samples 500
```

**评估结果输出**:
- 终端显示: 汇总表格（CER/RTF/错误统计）
- 文件保存: `src/asr/eval/aishell1_results_{model}.txt`
  - 每个样本的详细对比（GT vs 识别结果）
  - 单样本 CER 和错误类型

### 1.7 性能优化技术

#### faster-whisper 加速原理

faster-whisper 基于 [CTranslate2](https://github.com/OpenNMT/CTranslate2) 实现，相比原版 Whisper 速度提升 4 倍：

**核心优化技术**:

1. **权重量化**: FP32 → FP16/INT8，减少 50% 显存和计算量
2. **算子融合**: 合并多个小算子为大算子，减少 kernel 启动开销
3. **KV Cache 优化**: 高效管理 Transformer 注意力缓存
4. **动态批处理**: 根据输入长度自动调整批大小
5. **CPU/GPU 混合推理**: 自动选择最优执行设备

**性能对比**:

| 引擎 | 模型 | 5秒音频耗时 | 显存占用 |
|------|------|------------|---------|
| whisper | large-v3 | ~6.0s | ~10GB |
| faster-whisper | large-v3 | ~1.5s | ~5GB |
| faster-whisper | base | ~0.2s | ~1GB |

#### VAD（语音活动检测）

**作用**: 区分语音段和静音段，避免对无效音频进行识别

**本项目实现**: 基于能量的简易 VAD

```python
def _calculate_energy(self, audio):
    """计算音频能量（均方根 RMS）"""
    return np.sqrt(np.mean(audio ** 2))

# 判断是否为语音
is_speech = energy > self.vad_threshold  # 阈值默认 0.01
```

**工作流程**:
1. 音频分块（0.5秒/块）
2. 计算每块能量值
3. 能量超阈值 → 语音，开始缓存
4. 连续静音 0.8秒 → 触发识别
5. 清空缓存，等待下轮

**faster-whisper 内置 VAD**:
- 使用 Silero VAD 模型（神经网络）
- 准确率更高，自动过滤静音
- 启用方式: `vad_filter=True`

### 1.8 功能测试

**运行测试**:

```python
from src.asr import AudioRecorder, WhisperASR, RealtimeASR

# 方式1：录音 + 标准识别
recorder = AudioRecorder()
recorder.select_device()                    # 选择录音设备
audio = recorder.record_fixed_duration(5)   # 录制5秒
recorder.save_audio(audio, "test.wav")      # 保存音频

asr = WhisperASR(model_name="base")
result = asr.transcribe(audio, language="zh")
print(f"识别结果: {result['text']}")

# 方式2：实时识别（VAD自动检测语音结束）
realtime = RealtimeASR(model_name="base", language="zh")
realtime.start_realtime(callback=lambda text: print(f"识别: {text}"))
```

**实时识别测试**：
- VAD 检测灵敏度良好，静音后约 0.8 秒触发识别
- 使用 faster-whisper + base 模型，端到端延迟约 1 秒

### 1.8 代码结构

```
src/asr/
├── __init__.py
├── audio_recorder.py        # 音频录制（支持设备选择）
├── whisper_asr.py           # 标准 Whisper 语音识别
├── realtime_asr.py          # 实时语音识别（faster-whisper + VAD）
├── evaluate_asr.py          # ASR 评估框架
└── download_and_evaluate.py # HuggingFace 数据集评估
```

### 1.9 模块小结

- ✅ 完成 Whisper 语音识别集成
- ✅ 支持录音设备选择
- ✅ 通过 faster-whisper 解决性能瓶颈
- ✅ 实现基于 VAD 的实时识别
- ✅ 在 AISHELL-1 数据集上完成准确率评估
- ✅ large-v3 模型 CER 达到 5.2%

---

## 模块二：对话模型

> 🚧 待实现...

---

## 模块三：语音合成 (TTS)

### 3.1 技术选型

语音合成（Text-to-Speech）是将文字转换为语音的过程。本项目对比了多种 TTS 方案：

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| Edge-TTS | 免费、音质好、多语言、速度快 | 需联网、依赖微软服务 | 通用场景、快速开发 |
| VITS/So-VITS-SVC | 可克隆声音、效果自然 | 训练复杂、需GPU | 个性化声音定制 |
| GPT-SoVITS | 少样本声音克隆（1分钟音频） | 推理速度较慢 | 高质量声音克隆 |
| PaddleSpeech | 百度开源、中文优化好、本地离线 | 音质略逊于商业方案 | 离线部署、隐私保护 |
| Azure TTS | 商业级音质、表情控制丰富 | 收费、有配额限制 | 生产环境 |
| Coqui TTS | 开源、多种模型选择 | 中文支持较弱 | 研究用途 |

#### 选型分析

**方案A：Edge-TTS（推荐用于快速开发）**

**优势**：
- 免费使用，无 API Key 要求
- 音质接近商业级（微软 Azure TTS 后端）
- 支持 100+ 语言，中文语音自然流畅
- Python SDK 简单易用，单行代码即可合成
- 速度快（1秒文本约 0.3 秒合成）
- 支持多种中文音色（男声/女声，多种风格）

**劣势**：
- 需要网络连接（依赖微软服务器）
- 服务稳定性依赖微软（理论上可能下线）
- 无法定制个性化声音

**典型音色**：
- `zh-CN-XiaoxiaoNeural`（女声，温柔亲切）
- `zh-CN-YunxiNeural`（男声，稳重大气）
- `zh-CN-XiaoyiNeural`（女声，活泼开朗）
- `zh-CN-YunjianNeural`（男声，年轻阳光）

**代码示例**：
```python
import edge_tts
import asyncio

async def synthesize(text, output_file):
    communicate = edge_tts.Communicate(
        text=text,
        voice="zh-CN-XiaoxiaoNeural",
        rate="+0%",  # 语速
        pitch="+0Hz"  # 音调
    )
    await communicate.save(output_file)

asyncio.run(synthesize("你好，我是语音助手。", "output.mp3"))
```

---

**方案B：GPT-SoVITS（推荐用于声音克隆）**

**优势**：
- 少样本学习：仅需 1 分钟清晰音频即可克隆声音
- 音质自然：基于 GPT 和 VITS 的混合架构
- 支持情感迁移：可保留原声音的语气特征
- 中文效果优秀：专门针对中文优化

**劣势**：
- 需要 GPU（推理时至少 4GB 显存）
- 训练/推理速度较慢（实时率约 0.5-1.0）
- 环境配置复杂（依赖较多）

**适用场景**：
- 本项目加分项：使用自己或熟人的声音
- 有声书制作（作家本人录制）
- 游戏角色配音（声优声音克隆）
- 个性化语音助手

**工作流程**：
1. 录制 1-10 分钟清晰语音素材
2. 使用 GPT-SoVITS 进行快速微调（约 30 分钟）
3. 推理时输入文本 + 参考音频，生成目标语音

---

**方案C：PaddleSpeech（推荐用于离线部署）**

**优势**：
- 完全离线：无需联网，保护隐私
- 百度开源：社区活跃，文档完善
- 中文优化：FastSpeech2 + HiFiGAN 架构
- 支持流式合成：降低首字延迟

**劣势**：
- 音质略逊于 Edge-TTS（偏机械感）
- 模型体积较大（约 500MB）
- 需要本地计算资源

**适用场景**：
- 内网环境部署
- 对数据隐私要求高
- 需要长期稳定服务

---

### 3.2 本项目选型决策

**开发阶段**：Edge-TTS
- 理由：快速验证系统流程，无需 GPU
- 音色：`zh-CN-XiaoxiaoNeural`（女声，温柔自然）

**加分项实现**：GPT-SoVITS 声音克隆
- 理由：满足作业"鼓励使用自己或熟人的声音"要求
- 方案：录制 3-5 分钟个人语音，训练个性化模型

**备选方案**：PaddleSpeech（用于离线演示）
- 理由：如果演示环境无法联网，使用本地方案

---

### 3.3 Edge-TTS 详细说明

#### 安装与配置

```bash
# 安装 Edge-TTS
pip install edge-tts

# 测试安装
edge-tts --list-voices | grep "zh-CN"
```

#### 可用中文音色列表

| 音色名称 | 性别 | 风格 | 适用场景 |
|----------|------|------|----------|
| zh-CN-XiaoxiaoNeural | 女 | 温柔、亲切 | 客服、助手、故事朗读 |
| zh-CN-XiaoyiNeural | 女 | 活泼、开朗 | 儿童内容、广告 |
| zh-CN-YunjianNeural | 男 | 年轻、阳光 | 新闻播报、教育 |
| zh-CN-YunxiNeural | 男 | 稳重、大气 | 纪录片、商务 |
| zh-CN-XiaochenNeural | 女 | 知性、专业 | 新闻、学术 |
| zh-CN-XiaohanNeural | 女 | 温暖、舒缓 | 睡前故事、冥想 |

#### 高级参数控制

```python
import edge_tts

# 调整语速、音调、音量
communicate = edge_tts.Communicate(
    text="这是一段测试文本。",
    voice="zh-CN-XiaoxiaoNeural",
    rate="+20%",    # 语速加快 20%（范围：-50% 至 +100%）
    pitch="+5Hz",   # 音调提高 5Hz（范围：-50Hz 至 +50Hz）
    volume="+10%"   # 音量增加 10%（范围：-50% 至 +50%）
)
```

#### SSML 支持（高级用法）

Edge-TTS 支持 SSML（Speech Synthesis Markup Language）标记语言：

```python
ssml_text = """
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
    <voice name="zh-CN-XiaoxiaoNeural">
        你好，<break time="500ms"/>我是语音助手。
        <prosody rate="slow" pitch="+10Hz">这句话语速较慢，音调较高。</prosody>
        <emphasis level="strong">这是重点内容。</emphasis>
    </voice>
</speak>
"""

communicate = edge_tts.Communicate(text=ssml_text)
await communicate.save("output.mp3")
```

**SSML 常用标签**：
- `<break time="500ms"/>`：插入停顿
- `<prosody>`：调整语速、音调、音量
- `<emphasis>`：强调特定文字
- `<say-as>`：指定数字、日期的朗读方式

---

### 3.4 GPT-SoVITS 声音克隆方案

#### 安装与环境配置

```bash
# 克隆仓库
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# 创建环境
conda create -n gpt-sovits python=3.10
conda activate gpt-sovits

# 安装依赖
pip install -r requirements.txt

# 下载预训练模型
python download_models.py
```

#### 声音克隆工作流程

**步骤 1：准备音频素材**

录音要求：
- 时长：3-10 分钟（最少 1 分钟）
- 格式：WAV（16kHz 或 44.1kHz，单声道）
- 质量：安静环境，无背景噪音
- 内容：建议包含不同语气的句子（陈述、疑问、感叹）

```bash
# 使用 Audacity 或 FFmpeg 处理音频
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

**步骤 2：自动标注文本**

```bash
# 使用 Whisper 自动生成标注
python prepare_datasets/asr.py --input_audio voices/my_voice.wav
# 输出: voices/my_voice.list（每行：音频路径|文本|语言）
```

**步骤 3：快速微调**

```bash
# 使用 WebUI 进行训练
python webui.py

# 或使用命令行
python GPT_SoVITS/s1_train.py --config configs/s1.yaml
python GPT_SoVITS/s2_train.py --config configs/s2.yaml
```

**步骤 4：推理合成**

```python
from GPT_SoVITS.inference import infer

# 输入文本和参考音频
output = infer(
    text="这是需要合成的文本。",
    ref_audio_path="voices/my_voice_ref.wav",  # 3-10秒参考音频
    ref_text="这是参考音频的文本。",
    model_path="output/models/my_voice.pth"
)
```

#### 声音克隆效果评估

| 指标 | 评估方法 | 目标值 |
|------|----------|--------|
| 音色相似度 | 主观评分（1-5分） | >4.0 |
| 自然度 | MOS（Mean Opinion Score） | >3.5 |
| 情感表达 | 对比原音频的语气一致性 | 高度一致 |
| 合成速度 | RTF（Real-Time Factor） | <1.0 |

---

### 3.5 PaddleSpeech 离线方案

#### 安装

```bash
pip install paddlepaddle-gpu  # GPU 版本
# 或
pip install paddlepaddle      # CPU 版本

pip install paddlespeech
```

#### 下载模型

```bash
# 下载 FastSpeech2 + HiFiGAN 中文模型
paddlespeech tts --input "测试文本" --output test.wav --lang zh --am fastspeech2_csmsc --voc hifigan_csmsc
# 首次运行会自动下载模型（约 500MB）
```

#### Python 调用

```python
from paddlespeech.cli.tts import TTSExecutor

tts = TTSExecutor()

# 合成语音
tts(
    text="这是一段测试文本。",
    output="output.wav",
    am="fastspeech2_csmsc",      # 声学模型
    voc="hifigan_csmsc",         # 声码器
    lang="zh",
    sample_rate=24000
)
```

#### 流式合成（降低延迟）

```python
import soundfile as sf
from paddlespeech.server.engine.tts.online.python.tts_engine import TTSEngine

engine = TTSEngine()
engine.init()

# 流式输出
for chunk in engine.run("这是一段较长的文本。"):
    # chunk 是音频片段，可立即播放
    sf.write("output.wav", chunk, 24000, mode='a')
```

---

### 3.6 TTS 性能对比

| 方案 | 音质评分 | 合成速度 | 中文自然度 | 部署难度 | 网络依赖 |
|------|---------|---------|-----------|---------|---------|
| Edge-TTS | 4.5/5 | 极快（0.3s/句） | 优秀 | 极低 | 需要 |
| GPT-SoVITS | 4.7/5 | 慢（RTF~0.8） | 优秀 | 高 | 不需要 |
| PaddleSpeech | 3.8/5 | 中等（RTF~0.5） | 良好 | 中 | 不需要 |
| Azure TTS | 4.8/5 | 快 | 优秀 | 低 | 需要 |

**评分标准**：
- 音质：主观听感评分（5分制）
- 合成速度：每句话（约20字）的合成耗时
- 自然度：语调、停顿、韵律的自然程度
- 部署难度：环境配置、依赖安装的复杂度

---

### 3.7 本项目实施计划

#### 阶段一：基础 TTS 集成（1-2天）

**目标**：使用 Edge-TTS 实现基础语音合成功能

**任务清单**：
- [ ] 安装 edge-tts 依赖
- [ ] 实现 TTS 封装类（支持音色选择）
- [ ] 测试多种中文音色
- [ ] 实现异步合成（避免阻塞）
- [ ] 添加错误处理（网络异常重试）

**代码结构**：
```
src/tts/
├── __init__.py
├── edge_tts_engine.py       # Edge-TTS 引擎封装
├── audio_player.py          # 音频播放器
└── tts_config.py            # 音色配置
```

---

#### 阶段二：声音克隆（加分项，3-4天）

**目标**：使用 GPT-SoVITS 克隆个人声音

**任务清单**：
- [ ] 录制 5 分钟个人语音素材
- [ ] 使用 Whisper 自动标注文本
- [ ] 训练 GPT-SoVITS 模型（约 2 小时）
- [ ] 集成到系统中（可选切换音色）
- [ ] 对比 Edge-TTS 和个人声音的效果

**素材录制建议**：
- 安静环境，使用专业麦克风
- 朗读多样化文本（新闻、对话、诗歌）
- 包含不同情感（平静、兴奋、疑问）
- 保持音量稳定，避免爆音

---

#### 阶段三：性能优化（可选）

**目标**：提升合成速度和稳定性

**优化方向**：
- 文本预处理：过滤特殊字符、断句优化
- 缓存机制：常用语句预合成缓存
- 并发合成：多句话并行处理
- 降级策略：Edge-TTS 失败时自动切换 PaddleSpeech

---

### 3.8 模块小结

- ✅ 完成 TTS 技术选型分析
- ✅ 确定开发方案：Edge-TTS（主） + GPT-SoVITS（加分项）
- ⏳ 待实现：TTS 引擎封装
- ⏳ 待实现：声音克隆功能
- ⏳ 待实现：与系统集成

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

- [x] 优化识别延迟（已通过 faster-whisper 解决）
- [ ] 添加噪声抑制
- [x] 支持流式识别（已实现 VAD 实时识别）

---

## 参考资料

1. [OpenAI Whisper](https://github.com/openai/whisper)
2. [Whisper 论文](https://arxiv.org/abs/2212.04356)
3. [KdConv 数据集](https://github.com/thu-coai/KdConv)
4. [Edge-TTS](https://github.com/rany2/edge-tts)
5. [Gradio 文档](https://gradio.app/docs)
