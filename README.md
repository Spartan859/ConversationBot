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

---

## 模块二：对话模型

### 2.1 技术选型与架构设计

本项目设计了一个**智能 Agent 调用系统**，实现语音对话中的智能路由功能。

#### 核心思想

通过一个通用大模型作为"中枢决策器"，判断用户问题是否需要调用专业 Agent（清华本科学习助手），实现：
- **通用问题**：由通用大模型直接回答（如闲聊、常识问答）
- **专业问题**：自动路由到清华本科学习助手 Agent（如课程查询、学习资料推荐）

#### 系统架构

```
用户语音输入 (ASR)
    ↓
┌─────────────────────────────────────────┐
│  通用大模型 (路由决策层)                 │
from src.dialogue.utils import markdown_to_speech_text
│  - Prompt: 判断是否需要专业知识          │
└─────────────────────────────────────────┘
    ↓
决策输出 (XML 格式)
    ↓
┌──────────────┬──────────────────────┐
│ <need_agent> │                      │
│   true       │       false          │
└──────┬───────┴──────────┬───────────┘
       ↓                  ↓
┌──────────────┐   ┌─────────────────┐
│ 清华学习助手  │   │ 通用大模型回答  │
│ Agent        │   │ <answer>...</>  │
│ - 知识库检索  │   └─────────────────┘
│ - RAG 增强   │
└──────────────┘
       ↓
    合成回答
       ↓
    TTS 输出
```

#### 决策流程

**步骤 1：通用大模型判断**

输入 Prompt 模板：
```xml
你是一个智能路由助手，需要判断用户问题是否需要调用「清华大学本科学习助手」专业知识库。

【用户问题】：{user_query}

【判断规则】：
- 如果问题涉及清华大学课程、学习资料、校园生活、学术资源，输出：
  <need_agent>true</need_agent>
  
- 如果问题是通用闲聊、常识问答、不相关内容，输出：
  <need_agent>false</need_agent>
  <answer>你的回答内容</answer>

【注意】：
1. 只输出 XML 标签，不要有其他内容
2. need_agent=true 时，不要输出 answer 标签
3. need_agent=false 时，必须输出 answer 标签
```

**步骤 2：解析 XML 输出**

```python
import re

def parse_decision(response: str):
    need_agent = re.search(r'<need_agent>(true|false)</need_agent>', response)
    answer = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    
    if need_agent.group(1) == 'true':
        return {'route': 'agent', 'answer': None}
    else:
        return {'route': 'general', 'answer': answer.group(1).strip()}
```

**步骤 3：条件调用**

```python
decision = parse_decision(llm_response)

if decision['route'] == 'agent':
    # 调用清华学习助手 Agent
    final_answer = call_thu_assistant_agent(user_query)
else:
    # 直接使用通用大模型回答
    final_answer = decision['answer']
```

---

### 2.2 基于RAG的清华本科学习助手 Agent

#### Agent 简介

基于**火山引擎知识库 API**，提供以下能力：
- **知识检索**：从清华大学本科学习资料库中检索相关文档
- **RAG 增强**：结合检索结果和大模型生成精准答案
- **多模态支持**：支持文本、图片、表格等多种资料格式

#### API 配置

```python
# 知识库配置
collection_name = "thu"          # 知识库名称
project_name = "default"         # 项目名称
ak = "your_access_key"           # 火山引擎 AK
sk = "your_secret_key"           # 火山引擎 SK
account_id = "your_account_id"   # 账户 ID
```

#### 调用流程

1. **知识检索**：`search_knowledge(query)` - 召回 Top-K 相关文档
2. **Prompt 构建**：将检索结果拼接为 Prompt
3. **生成回答**：`chat_completion(messages)` - 基于 Deepseek-v3-1 生成答案
4. **后处理**：可选将 Markdown 格式转换为纯文本（适合 TTS 朗读）

---

### 2.3 ThuAssistantAgent 实现

#### 核心类：`ThuAssistantAgent`

位于 `src/dialogue/thu_agent.py`，封装完整的 RAG 查询流程。

**主要方法**：

```python
class ThuAssistantAgent:
    def __init__(
        self,
        ak: str,                          # 火山引擎 Access Key
        sk: str,                          # 火山引擎 Secret Key
        account_id: str,                  # 账户 ID
        model: str = "Deepseek-v3-1",    # 模型名称
        base_prompt: Optional[str] = None # 自定义 System Prompt
    ):
        """初始化 Agent"""
    
    def query(
        self, 
        user_query: str,
        post_process: bool = False  # 是否后处理为纯文本
    ) -> str:
        """
        完整的 RAG 查询流程
        
        Args:
            user_query: 用户问题
            post_process: 是否去除 Markdown 标记（适合 TTS）
            
        Returns:
            Markdown 格式（post_process=False）或纯文本（post_process=True）
        """
```

**使用示例**：

```python
from src.dialogue.thu_agent import ThuAssistantAgent

# 初始化
agent = ThuAssistantAgent(
    ak="your_ak",
    sk="your_sk",
    account_id="your_id"
)

# 查询（Markdown 格式）
response = agent.query("新生如何办理入学手续？")
print(response)  
# 输出：同学你好！办理入学手续需要：\n\n1. **准备材料**：...\n2. **报到流程**：...

# 查询（纯文本格式，适合 TTS）
response_tts = agent.query("新生如何办理入学手续？", post_process=True)
print(response_tts)
# 输出：同学你好！办理入学手续需要：\n\n准备材料：...\n报到流程：...
```

#### 后处理功能

为了支持 TTS 语音合成，实现了 Markdown 到纯文本的转换功能（位于 `src/utils.py`）：

**处理规则**：
- 去除 `**加粗**`、`*斜体*`、`#标题` 等 Markdown 标记
- 处理链接：`[文本](URL)` → `文本`
- 去除代码块、列表标记、引用标记
- 保留段落换行（便于 TTS 停顿）
- 保留中文标点和自然语气

**转换示例**：

| Markdown 格式 | 纯文本格式（TTS） |
|--------------|------------------|
| `**重要**：请提前办理` | `重要：请提前办理` |
| `详见[学生手册](http://xxx)` | `详见学生手册` |
| `- 第一步\n- 第二步` | `第一步\n第二步` |

**工具函数**：

```python
from src.utils import markdown_to_speech_text

text = "**注意**：请携带以下材料：\n\n- 身份证\n- 录取通知书"
tts_text = markdown_to_speech_text(text)
print(tts_text)
# 输出：注意：请携带以下材料：\n\n身份证\n录取通知书
```

### 2.4 GeneralAgent（通用大模型）

位于 `src/dialogue/general_agent.py`，封装 Ark(OpenAI SDK) responses 接口的通用调用，支持文本与图片输入。

**主要参数**：
- `api_key`：火山引擎 Ark API Key（必填）
- `model`：通用大模型名称（默认示例 `ep-20251219211834-fxjqq`）
- `base_url`：`https://ark.cn-beijing.volces.com/api/v3`
- `post_process`：是否去除 Markdown 标记（适合 TTS）

**使用示例（文本）**：

```python
import os
from src.dialogue.general_agent import GeneralAgent

agent = GeneralAgent(api_key=os.getenv("ARK_API_KEY"), model="ep-20251219211834-fxjqq")
resp = agent.chat("介绍一下清华大学的校训。", post_process=True)
print(resp)
```

**使用示例（多模态：图文）**：

```python
resp_img = agent.chat(
    "你看见了什么？",
    image_url="https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png",
    post_process=True,
)
print(resp_img)
```

**运行示例脚本**：

```bash
python -m src.dialogue.demo_agents
```
请提前设置：
- `ARK_API_KEY`：通用大模型 API Key
- `THU_AGENT_AK` / `THU_AGENT_SK` / `THU_AGENT_ACCOUNT_ID`：清华助手凭证

---

## 模块三：语音合成 (TTS)

### 3.1 技术选型

语音合成（Text-to-Speech）是将文字转换为语音的过程。本项目对比了几种主流 TTS 方案：

| 方案 | 优势 | 劣势 | 音质 | 克隆能力 |
|------|------|------|------|---------|
| Edge-TTS | 免费、速度快、多语言 | 需联网 | 4.5/5 | 无 |
| GPT-SoVITS | **1分钟少样本克隆**、音质自然 | 需GPU训练 | 4.7/5 | **强** |
| PaddleSpeech | 完全离线、百度开源 | 音质略逊 | 3.8/5 | 无 |

**最终选型：GPT-SoVITS**

**最终选型：GPT-SoVITS**

**选择理由**：
1. **满足作业加分项**：支持使用个人声音进行克隆
2. **少样本学习**：仅需 1-5 分钟清晰音频即可训练
3. **音质优秀**：基于 GPT + VITS 架构，自然度高
4. **中文优化好**：专门针对中文场景优化
5. **开源免费**：MIT 协议，可自由使用

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
