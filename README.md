# 语音对话系统 - 大作业报告
---
2023011002 自35 李翔宇

## 项目概述

本项目旨在搭建一个完整的语音对话系统，实现**语音输入 → 智能对话 → 语音输出**的端到端交互体验。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     语音对话系统架构                          │
└─────────────────────────────────────────────────────────────┘

     ┌──────────┐      ┌──────────┐      ┌──────────┐
     │   ASR    │      │ Dialogue │      │   TTS    │
     │ (Whisper)│ ──▶ │  (LLM)   │ ──▶│(GPT-SoVITS)│
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

#### 音频录制模块设计思路

**核心类：AudioRecorder**

**初始化参数**：
- 采样率：16000 Hz（Whisper 推荐标准）
- 声道数：单声道（节省资源）
- 录音设备：支持手动指定或交互式选择

**主要功能**：

1. **固定时长录音**
   - 输入：录制时长（秒）
   - 处理流程：
     * 计算所需采样点数 = 时长 × 采样率
     * 调用音频库开始录制
     * 阻塞等待录制完成
     * 将多维数组展平为一维
   - 输出：float32 格式音频数组

2. **设备选择功能**
   - 查询系统可用音频设备
   - 列出设备编号、名称、通道数
   - 支持用户交互式选择或指定设备索引
   - 验证设备有效性

#### 语音识别模块设计思路

**核心类：RealtimeASR**

**预设常量**：
- 简体中文提示词："以下是普通话的句子，使用简体中文输出。"

**初始化流程**：

1. **自动检测计算资源**
   - 检查 CUDA GPU 是否可用
   - GPU 可用 → 使用 float16 精度（速度快）
   - 仅 CPU → 使用 int8 量化（节省内存）

2. **加载 Whisper 模型**
   - 指定模型大小（tiny/base/small/medium/large）
   - 设置运行设备（cuda 或 cpu）
   - 应用对应的计算精度类型

3. **配置识别参数**
   - 设置目标语言（默认中文）
   - 启用简体中文提示词（引导输出格式）

**转录处理流程**：

1. 接收音频数据（numpy 数组）
2. 调用模型转录接口：
   - 指定语言代码
   - 启用 VAD 过滤（自动去除静音段）
   - 使用 beam search（beam_size=5，提升准确率）
   - 传入初始提示词
3. 模型返回分段结果和元信息
4. 拼接所有段落文本
5. 去除首尾空白字符
6. 返回完整识别文本

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

**算法设计思路**（`src/asr/number_converter.py`）：

**核心类：NumberConverter**

**主方法：convert_text** - 按优先级依次处理文本中的数字

**转换规则（按执行顺序）**：

1. **年份范围转换**
   - 匹配模式：`2015-2016` 形式
   - 转换策略：每个年份逐位转换
   - 输出示例：`二零一五至二零一六`

2. **百分比转换**
   - 匹配模式：`46%` 或 `46.5%`
   - 转换策略：数字转中文 + "百分之"前缀
   - 输出示例：`百分之四十六`

3. **日期转换**
   - 匹配模式：`10月22日`
   - 转换策略：月份和日期分别转换
   - 输出示例：`十月二十二日`

4. **完整年份转换**
   - 匹配模式：`2013年`
   - 转换策略：年份逐位转换
   - 输出示例：`二零一三年`

5. **带单位数字转换**
   - 匹配模式：`137人`、`50米` 等
   - 转换策略：数字转中文（支持十、百、千位）
   - 输出示例：`一百三十七人`

**实现要点**：
- 使用正则表达式匹配不同模式
- 按从复杂到简单的顺序处理（避免误匹配）
- 每种模式对应独立的转换函数
- 支持整数和小数处理

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

**计算方法**：Levenshtein 编辑距离算法

**算法原理**：动态规划（Dynamic Programming）

**核心思路**：

1. **初始化 DP 表**
   - 创建二维数组 dp[m+1][n+1]
   - m = 参考文本长度，n = 识别结果长度

2. **边界条件初始化**
   - dp[i][0] = i：参考文本前 i 个字符都需要删除
   - dp[0][j] = j：识别结果前 j 个字符都是插入

3. **动态规划递推**
   对于每个位置 (i, j)：
   
   - **如果字符相同**：ref[i-1] == hyp[j-1]
     * 无需编辑操作
     * dp[i][j] = dp[i-1][j-1]
   
   - **如果字符不同**：
     * 替换：dp[i-1][j-1] + 1
     * 删除：dp[i-1][j] + 1
     * 插入：dp[i][j-1] + 1
     * 取三者最小值

4. **返回结果**
   - dp[m][n] 即为最小编辑距离
   - 将编辑距离除以参考文本长度得到 CER

**复杂度分析**：
- 时间复杂度：O(m × n)
- 空间复杂度：O(m × n)

**示例演示**：

参考："清华大学" (m=4)  
识别："清芯大学" (n=4)  
编辑距离：1（将"华"替换为"芯"）  
CER = 1/4 = 25%

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

**本项目实现思路**: 基于能量的简易 VAD

**能量计算方法**：
- 算法：均方根（RMS）
- 步骤：
  1. 对音频信号的每个采样点求平方
  2. 计算所有平方值的平均值
  3. 对平均值开平方根
- 物理意义：反映音频信号的平均功率

**语音判定逻辑**：
- 计算当前音频块的能量值
- 与预设阈值比较（默认 0.01）
- 能量 > 阈值 → 判定为语音
- 能量 ≤ 阈值 → 判定为静音

**完整工作流程**：

1. **音频分块处理**
   - 将连续音频流切分为小块（每块 0.5 秒）
   - 逐块计算能量值

2. **语音检测循环**
   - 当检测到语音（能量超阈值）→ 开始缓存音频
   - 持续缓存所有语音块

3. **静音检测**
   - 监测连续静音时长
   - 连续静音达到 0.8 秒 → 判定说话结束

4. **触发识别**
   - 将缓存的所有音频块合并
   - 送入 Whisper 模型识别

5. **重置状态**
   - 清空音频缓存
   - 等待下一轮语音输入

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

**解析逻辑**：

1. **提取 need_agent 标签**
   - 使用正则表达式匹配：`<need_agent>true/false</need_agent>`
   - 提取标签内的布尔值（true 或 false）

2. **提取 answer 标签**（可选）
   - 使用正则表达式匹配：`<answer>...</answer>`
   - 提取标签内的完整文本内容（支持多行）

3. **构建决策结果**
   - 如果 need_agent = true：
     * 返回路由类型为 'agent'
     * answer 字段为空
   - 如果 need_agent = false：
     * 返回路由类型为 'general'
     * 提取并去除首尾空白后返回 answer 内容

**步骤 3：条件调用**

**路由执行逻辑**：

1. **调用解析函数**
   - 将 LLM 返回的文本传入解析器
   - 获得结构化的决策结果

2. **判断路由类型**
   - 检查 decision['route'] 的值

3. **分支执行**：
   - **路由 = 'agent'**：
     * 将用户原始问题传递给清华学习助手
     * 触发知识库检索 + RAG 流程
     * 返回专业答案
   
   - **路由 = 'general'**：
     * 直接使用 decision['answer'] 中的内容
     * 无需额外 API 调用
     * 返回通用回答

#### DialogueRouter 设计思路

完整的路由器实现位于 `src/dialogue/router.py`。

**核心类：DialogueRouter**

**类常量：决策提示词模板**

定义路由决策的 Prompt 模板，包含：
- 角色定位：智能路由助手
- 输入占位符：{user_query}
- 判断规则：
  * 清华相关（课程/资料/校园/学术）→ `<need_agent>true</need_agent>`
  * 通用问题（闲聊/常识/技术/其他）→ `<need_agent>false</need_agent>` + `<answer>...</answer>`
- 输出约束：纯 XML 标签，口语化风格

**初始化方法**

**输入参数**：
- general_agent：通用大模型 Agent 实例
- thu_agent：清华助手 Agent 实例
- verbose：是否输出详细日志（默认 False）

**存储内容**：
- 保存两个 Agent 的引用
- 保存日志开关状态

**核心方法：route**

**输入参数**：
- user_query：用户问题（字符串）
- post_process：是否将 Markdown 转为纯文本（布尔值）

**执行流程**：

**第一步：构建决策 Prompt**
- 将用户问题插入模板的 {user_query} 占位符
- 生成完整的决策提示词

**第二步：调用通用大模型判断**
- 如果启用 verbose → 打印"正在判断..."
- 将决策 Prompt 发送给 GeneralAgent
- 获取 XML 格式的决策响应
- 保持原始格式（post_process=False）

**第三步：解析决策结果**
- 使用工具函数检查是否存在 `<need_agent>true</need_agent>`
- 提取布尔判断结果

**第四步：条件路由执行**

**分支 A：need_agent = true**
- 如果启用 verbose → 打印"路由到 ThuAssistantAgent"
- 调用 thu_agent.query(user_query)
- 传递 post_process 参数
- 返回专业知识库的答案

**分支 B：need_agent = false**
- 如果启用 verbose → 打印"使用 GeneralAgent 回答"
- 提取 XML 中的 `<answer>` 标签内容
- 如果 post_process=True：
  * 将 Markdown 格式转为纯文本
  * 去除加粗、斜体、列表标记等
- 返回通用回答

**异常处理**：
- 如果无法提取 answer 标签
- 返回完整的原始响应文本

**调用示例流程**：

**场景 1：初始化系统组件**

1. 从环境变量获取认证信息：
   - ARK_API_KEY：通用大模型的 API Key
   - 清华助手三件套：AK, SK, Account ID

2. 创建两个 Agent 实例：
   - GeneralAgent：使用 API Key 和模型名初始化
   - ThuAssistantAgent：使用三个认证参数初始化

3. 构建智能路由器：
   - 将两个 Agent 注入路由器
   - 开启 verbose 模式便于调试

**场景 2：处理不同类型的问题**

- **通用问题**（例："什么是人工智能？"）
  * 路由器判断为非清华相关
  * 直接使用 GeneralAgent 回答
  * 无需访问知识库

- **清华相关问题**（例："新生入学需要准备哪些材料？"）
  * 路由器判断为清华相关
  * 调用 ThuAssistantAgent
  * 触发 RAG 流程（知识检索 + 生成）

**场景 3：启用 TTS 后处理**

- 在 route 方法中设置 post_process=True
- 系统自动将 Markdown 格式转为纯文本
- 去除特殊标记（加粗、斠序列表等）
- 返回适合 TTS 语音合成的文本

**运行测试**：

```bash
# 设置环境变量
$env:ARK_API_KEY="your-key"
$env:THU_AGENT_AK="your-ak"
$env:THU_AGENT_SK="your-sk"
$env:THU_AGENT_ACCOUNT_ID="your-id"

# 运行完整测试（6 个测试用例）
python src/dialogue/demo_router.py
```

**测试输出示例**：

```
======================================================================
  初始化 Agent 和 Router
======================================================================

1. 初始化 GeneralAgent...
   GeneralAgent 初始化完成
2. 初始化 ThuAssistantAgent...
   ThuAssistantAgent 初始化完成
3. 初始化 DialogueRouter...
   DialogueRouter 初始化完成

======================================================================
  测试 1：通用闲聊问题 (预期路由: GeneralAgent)
======================================================================

问题: 你好，今天天气怎么样？

[Router] 用户问题: 你好，今天天气怎么样？
[Router] 正在判断是否需要专业 Agent...
[Router] 使用 GeneralAgent 回答 (通用知识)

回答:
我是 Claude AI 助手。虽然我无法告诉你实时天气信息，但我可以建议你...

======================================================================
  测试 3：清华课程问题 (预期路由: ThuAssistantAgent)
======================================================================

问题: 清华大学有哪些人工智能相关的课程？

[Router] 用户问题: 清华大学有哪些人工智能相关的课程？
[Router] 正在判断是否需要专业 Agent...
[Router] 路由到 ThuAssistantAgent (专业知识库)

回答:
清华大学在人工智能领域提供以下主要课程...
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

**使用流程**：

**步骤 1：创建 Agent 实例**
- 从环境变量或配置文件读取认证信息
- 初始化 ThuAssistantAgent，传入 AK/SK/Account ID

**步骤 2：Markdown 格式查询**
- 调用 agent.query(问题)
- 默认返回 Markdown 格式文本
- 适合网页展示、终端输出
- 示例输出：
  ```
  同学你好！办理入学手续需要：
  
  1. **准备材料**：...
  2. **报到流程**：...
  ```

**步骤 3：纯文本格式查询（适合 TTS）**
- 调用 agent.query(问题, post_process=True)
- 系统自动去除 Markdown 标记
- 返回适合语音合成的纯文本
- 示例输出：
  ```
  同学你好！办理入学手续需要：
  
  准备材料：...
  报到流程：...
  ```

**后处理效果**：
- 去除 `**加粗**`、`*斜体*`
- 处理 `[文本](URL)` → `文本`
- 去除无序列表符号 `-`、`•`
- 保留段落换行（TTS 停顿）

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

**接口调用说明**：

**基本文本查询流程**：

1. **初始化 Agent**
   - 从环境变量获取 ARK_API_KEY
   - 指定模型 endpoint（如 "ep-20251219211834-fxjqq"）

2. **发起对话**
   - 调用 agent.chat(提问)
   - 设置 post_process=True 去除 Markdown 格式
   - 获取纯文本回答

3. **结果处理**
   - 打印或存储回答内容
   - 可直接用于 TTS 合成

**多模态（图文）查询流程**：

1. **准备输入**
   - 文本提问：如 "你看见了什么？"
   - 图片 URL：公网可访问的图片地址

2. **调用接口**
   - agent.chat(文本, image_url=图片URL)
   - 启用 post_process 去除格式

3. **获取结果**
   - 模型返回基于图像的文本描述
   - 适合多模态问答场景

**注意事项**：
- 必须提前设置 ARK_API_KEY 环境变量
- 图片 URL 需可公网访问
- post_process 适合 TTS，不适合网页展示

**运行示例脚本**：

```bash
python -m src.dialogue.demo_agents
```
请提前设置：
- `ARK_API_KEY`：通用大模型 API Key
- `THU_AGENT_AK` / `THU_AGENT_SK` / `THU_AGENT_ACCOUNT_ID`：清华助手凭证

### 2.5 模型评估系统

为了系统性地评估不同模型在清华本科学习助手场景下的性能，我们实现了完整的模型评估系统。

#### 评估模型列表

| 模型名称 | 模型版本 | Thinking 模式 |
|---------|---------|--------------|
| kimi-k2-thinking-251104 | 251104 | 支持 |
| doubao-seed-1-6-flash-250828 | 250828 | 支持 |
| doubao-seed-1-6-thinking-250715 | 250715 | 支持 |
| deepseek-v3-2-251201 | 251201 | 支持 |

每个模型将分别在**启用 thinking** 和**关闭 thinking** 两种模式下进行评估，共 **8 个配置组合**。

#### 评估指标

**1. 性能指标**
- **响应时间**：从发送请求到收到完整响应的耗时（秒）
- **输出长度**：生成文本的字符数（适度长度更好）
- **成功率**：成功返回有效回答的比例

**2. 质量指标**
- **长度合理性** (0-1)：根据问题难度评估输出长度是否适当
- **关键词覆盖率** (0-1)：参考答案中的关键实体是否被提及
- **结构完整性** (0-1)：是否包含建议、步骤等结构化元素
- **综合质量分数** (0-1)：加权平均得分（覆盖率 50% + 长度 30% + 结构 20%）

**3. 难度分析**
- 基础题 (basic)：简单事实性问题
- 中级题 (intermediate)：规则应用与简单建议
- 高级题 (advanced)：场景化解决方案
- 综合题 (comprehensive)：多因素决策与长期规划

#### 评估题库

评估使用 40 个精心设计的问题（`thu_agent_evaluation_questions.json`），覆盖：
- 10 个基础题：如"清华大学的校训是什么？"
- 10 个中级题：如"如何选择合适的选修课？"
- 10 个高级题：如"大一新生如何平衡学习和社团活动？"
- 10 个综合题：如"如何制定四年完整的学业规划？"

#### 快速开始

**1. 快速评估（每个配置测试 5 题）**

```bash
# 设置环境变量
$env:THU_AGENT_AK="your-ak"
$env:THU_AGENT_SK="your-sk"
$env:THU_AGENT_ACCOUNT_ID="your-id"

# 运行快速评估
python src/dialogue/quick_evaluate.py
```

**2. 完整评估（全部 40 题）**

```bash
# 运行完整评估
python src/dialogue/evaluate_models.py

# 自定义采样数量
python src/dialogue/evaluate_models.py --sample 20

# 指定输出文件名
python src/dialogue/evaluate_models.py --output my_results.json
```

**3. 结果可视化**

```bash
# 查看评估结果（终端输出）
python src/dialogue/visualize_results.py evaluation_results/evaluation_results_20251220_123456.json

# 同时生成 Markdown 报告
python src/dialogue/visualize_results.py evaluation_results/evaluation_results_20251220_123456.json --markdown
```

#### 评估结果示例

**对比表格**：

```
模型性能对比表
================================================================================
模型名称                          | 思考模式    | 成功率  | 平均响应时间 | 平均长度 | 平均质量
deepseek-v3-2-251201              | thinking_enabled  | 100.0% | 3.45s   | 425    | 0.782
kimi-k2-thinking-251104           | thinking_enabled  | 100.0% | 2.87s   | 398    | 0.765
doubao-seed-1-6-thinking-250715   | thinking_enabled  | 98.0%  | 4.12s   | 456    | 0.743
doubao-seed-1-6-flash-250828      | thinking_disabled | 100.0% | 1.23s   | 312    | 0.698
================================================================================
```

**最佳配置排行榜**：

```
最佳配置排行榜
================================================================================

【质量最优】前3名:
  1. deepseek-v3-2-251201 (thinking_enabled) - 质量分数: 0.782
  2. kimi-k2-thinking-251104 (thinking_enabled) - 质量分数: 0.765
  3. doubao-seed-1-6-thinking-250715 (thinking_enabled) - 质量分数: 0.743

【速度最快】前3名:
  1. doubao-seed-1-6-flash-250828 (thinking_disabled) - 响应时间: 1.23s
  2. kimi-k2-thinking-251104 (thinking_disabled) - 响应时间: 1.98s
  3. kimi-k2-thinking-251104 (thinking_enabled) - 响应时间: 2.87s

【综合最佳】前3名 (质量70% + 速度30%):
  1. kimi-k2-thinking-251104 (thinking_enabled) - 综合分数: 0.724
  2. deepseek-v3-2-251201 (thinking_enabled) - 综合分数: 0.689
  3. doubao-seed-1-6-flash-250828 (thinking_disabled) - 综合分数: 0.645
================================================================================
```

**Thinking 模式对比**：

```
Thinking 模式对比
================================================================================

模型: kimi-k2-thinking-251104
  质量分数: 0.723 → 0.765 (+0.042, +5.8%)
  响应时间: 1.98s → 2.87s (+0.89s, +44.9%)
  推荐: 开启 thinking 模式（质量提升明显，速度损失可接受）

模型: doubao-seed-1-6-thinking-250715
  质量分数: 0.698 → 0.743 (+0.045, +6.4%)
  响应时间: 2.34s → 4.12s (+1.78s, +76.1%)
  权衡: thinking 模式提升质量，但速度慢一倍以上

模型: deepseek-v3-2-251201
  质量分数: 0.756 → 0.782 (+0.026, +3.4%)
  响应时间: 2.89s → 3.45s (+0.56s, +19.4%)
  推荐: 开启 thinking 模式（质量提升明显，速度损失可接受）
================================================================================
```

#### 评估系统架构

**核心类设计**：`ModelEvaluator` (`src/dialogue/evaluate_models.py`)

**类初始化参数**：
- ak, sk, account_id：火山引擎认证信息
- questions_file：评估题库 JSON 文件路径
- output_dir：结果输出目录

**核心方法设计**：

1. **_load_questions**
   - 输入：JSON 文件路径
   - 处理：解析 JSON，提取 questions 数组
   - 输出：问题列表（包含id/category/difficulty/question/reference_answer）

2. **_evaluate_single_question**
   - 输入：Agent 实例、问题字典、thinking 模式
   - 流程：
     * 记录开始时间戳
     * 调用 agent.query(问题, enable_thinking=xxx)
     * 记录结束时间戳
     * 计算响应耗时
     * 解析响应 JSON，提取文本内容
     * 计算输出长度
     * 调用质量评分函数
   - 输出：单问题评估结果字典

3. **_calculate_text_quality_score**
   - 输入：生成文本、参考答案、问题难度
   - 评分维度：
     * 长度合理性：根据难度判断长度是否合适
     * 关键词覆盖率：提取参考答案关键词，计算出现比例
     * 结构完整性：检测"建议"、"步骤"等结构化元素
   - 输出：各维度分数 + 加权综合分

4. **_calculate_statistics**
   - 输入：所有问题的评估结果列表
   - 计算内容：
     * 成功率
     * 响应时间统计（平均/最小/最大/总和）
     * 输出长度统计
     * 质量分数统计
     * 按难度分层统计
   - 输出：统计数据字典

5. **evaluate_model_config**
   - 输入：模型配置、thinking 模式、采样数量
   - 流程：
     * 打印评估开始信息
     * 创建 ThuAssistantAgent 实例
     * 选择评估问题（全部或采样）
     * 循环评估每个问题
     * 打印实时进度和结果
     * 计算统计数据
   - 输出：单个配置的完整评估结果

6. **run_full_evaluation**
   - 输入：采样数量、输出文件名
   - 流程：
     * 打印评估概览（模型数/配置数/题库大小）
     * 双层循环：每个模型 × 每种 thinking 模式
     * 调用 evaluate_model_config 评估每个配置
     * 收集所有结果
     * 生成汇总报告
     * 保存 JSON 文件
     * 打印汇总信息
   - 输出：结果文件路径

**使用流程**：

1. 导入类并创建实例
2. 设置认证参数和配置
3. 调用 run_full_evaluation
4. 系统自动执行所有评估
5. 获取 JSON 结果文件路径

**输出结构**：

```json
{
  "metadata": {
    "evaluation_date": "2025-12-20T15:30:00",
    "total_models": 4,
    "total_configurations": 8,
    "sample_size": 40
  },
  "summary": {
    "configurations": [
      {
        "model_name": "kimi-k2-thinking-251104",
        "thinking_mode": "thinking_enabled",
        "success_rate": 1.0,
        "avg_response_time": 2.87,
        "avg_output_length": 398,
        "avg_quality_score": 0.765
      }
    ],
    "best_configuration": {...}
  },
  "detailed_results": [
    {
      "model_name": "...",
      "statistics": {...},
      "detailed_results": [
        {
          "question_id": 1,
          "question": "...",
          "response_time_seconds": 2.5,
          "output_length_chars": 320,
          "quality_scores": {
            "length_appropriateness": 0.9,
            "keyword_coverage": 0.8,
            "structure_completeness": 0.7,
            "overall_quality": 0.765
          },
          "output_text": "...",
          "reference_answer": "...",
          "success": true
        }
      ]
    }
  ]
}
```

#### 人工质量评判

由于自动评分系统的区分度不足（所有模型质量评分集中在 0.15-0.6 区间），我们进行了人工质量评判。从4个难度等级各抽取1题（Q3基础题、Q12中级题、Q21高级题、Q40综合题），对8个模型配置的回答进行定性分析。

**评估工具**：`compare_model_outputs.py` - 提取并对比不同模型对同一题目的输出

```bash
# 提取第3题所有模型回答
python src/dialogue/compare_model_outputs.py evaluation_results/results.json -q 3 -o eval_q3.txt

# 查看所有题目列表
python src/dialogue/compare_model_outputs.py evaluation_results/results.json --list
```

---

##### Q3（基础题）：缓考申请需提交哪些材料？

**参考答案**：需提交《课程缓考申请表》，因病缓考还需提供校医院或二甲以上医院诊断证明。

**最佳答案：Doubao-seed-1-6 (thinking enabled) - 评分 (5/5)**

> "申请缓考需要提交两样材料哦：一是填写好的《课程缓考申请表》，这个表可以在清华大学教学门户下载；如果是因为生病申请缓考的话，还得同时提交校医院或者二级甲等以上医院开的诊断证明。"

**评判理由**：
- 信息完整（两类材料全覆盖）
- 简洁口语化（881字，适中）
- 逻辑清晰（"一是...如果..."）
- 完美适配TTS语音输出

**最差答案：Doubao-seed-1-6-flash (thinking enabled) - 评分 (1/5)**

> 长达3315字，声称"资料中未直接提及缓考申请流程及材料"（完全错误！），使用大量"### 标题""**加粗**"等Markdown格式，完全不适合语音输出。

**其他模型表现**：
- **Deepseek-v3-1** (快速/思考)：信息准确，但使用列表格式略显僵硬
- **Doubao-seed-1-6** (快速)：与思考模式表现相当，语气稍调整
- **Doubao-seed-1-6-thinking**：简洁准确，口语化自然

---

##### Q12（中级题）：哪些课程不能申请免修？

**参考答案**：思想政治理论课、体育课、实验课、实践类课程、军事训练、课程设计及综合论文训练不可免修。

**最佳答案：Doubao系列 (thinking enabled) - 评分 (5/5)**

> "思想政治理论课、体育课、实验课、实践类课程、军事理论与技能训练、课程设计和综合论文训练这些课程不能申请免修。"

**评判理由**：
- 关键词100%覆盖（7类课程全部列举）
- 一句话说清（最简洁的正确答案）
- 自动评分最高（0.508，Deepseek为0.483）
- 完美口语化

**次优答案：Deepseek-v3-1 - 评分 (4/5)**

> 使用数字列表格式（1. 2. 3...），信息完整但略显僵硬，不太适合语音播报。

**关键发现**：
- Doubao模型在简洁性上优势明显
- 自动评分的"结构完整性"指标（Doubao 0.12 vs Deepseek 0.00）准确捕捉到这一差异
- 中级题对简洁性和完整性平衡要求高

---

##### Q21（高级题）：我觉得在清华学习很难受，一直很不开心，你能帮帮我吗？

**参考答案**：建议联系班主任/辅导员、心理中心、学习中心等。

**最佳答案：Deepseek-v3-1 (thinking enabled) - 评分 (5/5)**

> "同学你好，听到你在清华学习感到难受和不开心，我非常理解你的心情。清华的学习环境确实充满挑战，但请记住，你并不孤单，学校有很多资源可以帮助你。
> 
> 首先，我强烈建议你联系**清华大学学生心理发展指导中心**（电话：010-62782007，24小时心理热线：010-62785252）...
> 
> 别忘了你身边的支持网络——班主任和辅导员都很愿意帮助你...
> 
> 寻求帮助是强者的表现，希望你愿意迈出这一步。清华有很多人关心你的成长，包括我。需要我帮你具体介绍任何一项服务吗？"

**评判理由**：
- **情感共鸣强**（"我非常理解你的心情""你并不孤单"）
- **信息全面**（心理中心+学习中心+班主任+辅导员）
- **具体可操作**（提供电话号码、公众号名称、地址）
- **鼓励性语气**（"寻求帮助是强者的表现"）
- **开放式结尾**（"需要我帮你具体介绍..."）

**次优答案：Doubao-seed-1-6-flash (thinking enabled) - 评分 (4/5)**

> 信息完整，但情感连接较弱，缺少"我非常理解"这类共情表达。

**其他模型表现**：
- **Doubao-seed-1-6** (思考)：热线电话准确，但情感铺垫不足（1442字）
- **Doubao-seed-1-6** (快速)：语气活泼（"同学你好呀"），但显得不够严肃
- **Deepseek-v3-1** (快速)：简洁版，保留核心共情元素（656字）

**关键发现**：
- 高级题需要情感共鸣+实用建议双重能力
- Deepseek在情感理解上表现优异
- 自动评分低估了情感维度的重要性

---

##### Q40（综合题）：发现某门课程评分不公，如何理性申诉？

**参考答案**：下学期第一周提交《成绩复议申请表》，切勿直接骚扰教师。

**最佳答案：Doubao-seed-1-6 (快速模式) - 评分 (5/5)**

> "如果你发现某门课程评分不公，可以这样理性申诉：首先得在下学期开学第一周内申请复议...去清华大学教学门户下载《成绩复议申请表》提交给开课院系...记住哦，不能直接找任课老师要求改成绩，严重干扰的话学校会依规处理的。"

**评判理由**：
- **流程清晰**（时间→下载→提交→审核，逻辑线明确）
- **重点突出**（强调复议期限、禁止骚扰教师）
- **口语化自然**（"记住哦""会依规处理的"）
- **长度适中**（1318字，综合题可接受）
- **TTS友好**（无格式污染）

**最差答案：Doubao-seed-1-6-flash (thinking enabled) - 评分 (1/5)**

> 长达3233字，使用大量"### 一、二、三"章节格式，完全破坏语音连贯性。过度展开（包含"课程咨询委员会""学生申诉委员会"等原问题未涉及内容）。

**其他模型表现**：
- **Deepseek-v3-1** (思考)：1183字，4步流程清晰，但使用数字列表
- **Deepseek-v3-1** (快速)：610字，简洁高效，但缺少"禁止骚扰"警示
- **Doubao-seed-1-6-thinking** (思考)：1053字，流程完整，语气友好

**关键发现**：
- 综合题需要结构化思维+适度展开
- Doubao快速模式在复杂场景下表现稳定
- Flash版本过度展开倾向明显，需要prompt优化

---

##### 人工评判总结

| 难度等级 | 最佳模型 | 次优模型 | 关键能力要求 |
|---------|---------|---------|------------|
| **基础题**（Q3） | Doubao-seed-1-6（思考） | Deepseek-v3-1 | 简洁性、准确性 |
| **中级题**（Q12） | Doubao系列（思考） | Deepseek-v3-1 | 完整性、口语化 |
| **高级题**（Q21） | Deepseek-v3-1（思考） | Doubao-flash（思考） | 情感共鸣、实用性 |
| **综合题**（Q40） | Doubao-seed-1-6（快速） | Deepseek-v3-1（思考） | 结构化、重点突出 |

**核心发现**：

1. **自动评分的局限性**
   - 无法评估情感共鸣（Q21中Deepseek优势被忽视）
   - 未惩罚格式污染（Flash模型大量Markdown未扣分）
   - 长度惩罚不足（3000+字回答仍得0.5分）
   - 结构完整性指标有效（Q12准确区分）

2. **模型特性对比**

| 模型 | 优势场景 | 劣势场景 | 推荐用途 |
|------|---------|---------|---------|
| **Doubao-seed-1-6** | 基础题、中级题 | 高级题情感偏弱 | 日常事务查询 |
| **Deepseek-v3-1** | 高级题、复杂决策 | 格式偶尔僵硬 | 深度咨询场景 |
| **Doubao-flash** | 所有场景均过度展开 | 全面 | **不推荐用于TTS** |
| **Doubao-thinking** | 综合题、流程说明 | 速度较慢 | 需要深度思考的问题 |

3. **TTS适配性排名**
   - **第一名：Doubao-seed-1-6**（思考/快速均优秀）
   - **第二名：Deepseek-v3-1**（快速模式更好）
   - **第三名：Doubao-thinking**（可用但较慢）
   - **不推荐：Doubao-flash**（格式污染严重，不推荐）

4. **Thinking模式效果**
   - **Doubao系列**：思考模式提升质量，保持简洁
   - **Deepseek**：思考模式增强情感理解
   - **Flash版本**：思考模式反而导致过度展开

---

#### 应用建议（基于人工评判）

**1. 实时对话场景（优先速度+简洁）**
- **推荐**：`Doubao-seed-1-6`（快速模式）
- **特点**：响应快、简洁口语化、适合TTS
- **适用**：基础查询、快问快答

**2. 高质量咨询场景（优先情感共鸣）**
- **推荐**：`Deepseek-v3-1`（思考模式）
- **特点**：情感理解强、建议全面、鼓励性强
- **适用**：心理咨询、职业规划、重大决策

**3. 复杂流程说明（优先结构化）**
- **推荐**：`Doubao-seed-1-6`（快速或思考模式）
- **特点**：流程清晰、重点突出、TTS友好
- **适用**：办事指南、操作步骤、规则解读

**4. 平衡场景（质量+速度+简洁）**
- **推荐**：`Doubao-seed-1-6`（快速模式）或 `Deepseek-v3-1`（快速模式）
- **特点**：综合表现最优，响应时间可接受
- **适用**：通用语音助手

**不推荐使用**：
- `Doubao-seed-1-6-flash`（任何模式）：过度展开、格式污染、不适合TTS
- 任何模型的长篇回复（>1500字）：需要prompt优化控制长度

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

**选择理由**：
1. **满足作业加分项**：支持使用个人声音进行克隆
2. **少样本学习**：仅需 1-5 分钟清晰音频即可训练
3. **音质优秀**：基于 GPT + VITS 架构，自然度高
4. **中文优化好**：专门针对中文场景优化
5. **开源免费**：MIT 协议，可自由使用

---

### 3.2 GPT-SoVITS 环境部署

#### 3.2.1 创建 Conda 环境

```bash
# 创建 Python 3.10 环境
conda create -n GPTSoVits python=3.10 -y

# 激活环境
conda activate GPTSoVits
```

#### 3.2.2 安装依赖

```bash
# 使用 HF-Mirror 镜像源安装依赖（适用于 CUDA 12.6）
bash install.sh --device CU126 --source HF-Mirror
```

**注意事项**：
- `--device` 参数根据你的 CUDA 版本选择：`CU118`、`CU121`、`CU126` 等
- `--source HF-Mirror` 使用国内 Hugging Face 镜像，加速下载
- 如果是 CPU 部署，使用 `--device CPU`

#### 3.2.3 手动下载 Gradio 依赖（重要）

由于网络问题，需要手动下载 Gradio 的 frpc 组件：

```bash
# 下载地址：
# https://cdn-media.huggingface.co/frpc-gradio-0.2/frpc_linux_amd64

# 将下载的文件放置到 Conda 环境的指定路径：
# {$conda_env_path}/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2

# 添加执行权限
chmod +x {$conda_env_path}/lib/python3.10/site-packages/gradio/frpc_linux_amd64_v0.2
```

**获取 Conda 环境路径**：
```bash
conda env list | grep GPTSoVits
# 输出示例：GPTSoVits    /home/user/miniconda3/envs/GPTSoVits
```

#### 3.2.4 启动 WebUI

```bash
# 启动 GPT-SoVITS WebUI（开启公网分享）
is_share=True python webui.py zh_CN
```

**参数说明**：
- `is_share=True`：开启 Gradio 公网分享链接（可选）
- `zh_CN`：使用中文界面

启动成功后，终端会显示本地访问地址（如 `http://127.0.0.1:7860`）。

---

### 3.3 数据集准备与预处理

#### 3.3.1 数据集来源

本项目使用了以下两个数据集：

| 数据集 | 来源 | 时长 | 音频质量 | 说话人 |
|--------|------|------|---------|--------|
| 陶建华教授演讲 | [BV1Ed4y1r74g](https://www.bilibili.com/video/BV1Ed4y1r74g) | 约30分钟 | 高清 | 中年男性，专业学术发音 |
| MyGO高松灯干声素材 | [BV1EK411t7cc](https://www.bilibili.com/video/BV1EK411t7cc) | 约60分钟 | 纯人声 | 年轻女性，动画角色声线 |

**数据集选择标准**：
- 音频质量清晰，无明显噪音
- 发音标准，语速适中
- 情绪变化适度（不宜过于激动或平淡）
- 单说话人（避免多人对话）

#### 3.3.2 音频预处理（Adobe Audition）

使用 **Adobe Audition** 对原始音频进行标准化处理：

**步骤 1：音量归一化**
1. 导入音频文件到 Adobe Audition
2. 选择 `效果` → `振幅和压限` → `匹配响度`
3. 设置目标响度：**-9 dB 到 -6 dB**
4. 应用效果

**步骤 2：噪音处理**
1. 选择一段纯噪音区域（无人声）
2. `效果` → `降噪/恢复` → `降噪（处理）`
3. 捕捉噪音样本并应用（降噪量：50%-70%）

**步骤 3：删除异常片段**
- 删除过高音量片段（峰值 > -3 dB）
- 删除过长的静音片段（> 2 秒）
- 删除非目标说话人音频

**步骤 4：导出**
- 格式：**WAV (未压缩)**
- 采样率：**44.1 kHz 或 48 kHz**
- 位深度：**16-bit 或 24-bit**
- 声道：**单声道（Mono）**
- 保存到：`dataset/{speaker_name}/` 文件夹

**文件命名规范**：
```
dataset/
├── tao_jianhua/          # 说话人1
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── takamatsu_tomori/     # 说话人2
    ├── audio_001.wav
    └── ...
```

---

### 3.4 自动语音识别（ASR）打标

使用 **faster-whisper** 对音频进行自动转写（生成训练所需的文本标注）。

#### 3.4.1 在 GPT-SoVITS WebUI 中打标

1. 打开 WebUI，进入 `数据处理` 标签
2. 点击 `自动标注` 功能
3. 选择音频文件夹：`dataset/{speaker_name}/`
4. 选择 ASR 模型：`faster-whisper`（推荐使用 `medium` 或 `large` 模型）
5. 点击 `开始标注`

#### 3.4.2 打标结果检查

打标完成后，会在音频同级目录生成 `.list` 文件：

```
dataset/tao_jianhua/
├── audio_001.wav
├── audio_001.wav.list    # 对应的文本标注
├── audio_002.wav
└── audio_002.wav.list
```

**`.list` 文件格式**：
```
dataset/tao_jianhua/audio_001.wav|tao_jianhua|ZH|今天我们讨论的主题是基于因果推理和强化学习的目标行为预测技术。
```

**字段说明**：
- 字段1：音频文件路径
- 字段2：说话人ID
- 字段3：语言代码（`ZH` 中文，`EN` 英文）
- 字段4：转写文本

**手动校对**：
- 检查 10-20 条样本的转写准确性
- 修正明显错误（专有名词、标点符号）
- 删除质量差的音频及其标注

---

### 3.5 模型训练

#### 3.5.1 训练参数配置

在 GPT-SoVITS WebUI 的 `训练` 标签中进行配置：

| 参数 | 设置值 | 说明 |
|------|--------|------|
| 实验名称 | `{speaker_name}` | 如 `tao_jianhua` 或 `takamatsu_tomori` |
| 音频文件夹 | `dataset/{speaker_name}/` | 预处理后的音频路径 |
| 标注文件夹 | `dataset/{speaker_name}/` | `.list` 文件所在路径 |
| Batch Size | **6** | 单张 RTX 4090 (24GB) 推荐值 |
| 训练轮数 | **10** | 少样本场景下足够 |
| DPO 训练 | **开启** | Direct Preference Optimization，提升音质 |
| 学习率 | `2e-4`（默认） | 一般无需调整 |
| 保存频率 | 每 2 轮保存一次 | 用于选择最佳 checkpoint |

**硬件要求**：
- **GPU**：NVIDIA RTX 4090 (24GB) 或同等算力
- **显存**：至少 16GB（batch_size=6）
- **训练时长**：10 轮约 2-4 小时（取决于数据量）

#### 3.5.2 DPO 训练说明

**DPO（Direct Preference Optimization）**：
- 一种基于偏好学习的优化方法
- 通过对比"更好"和"较差"的合成样本来改进模型
- 显著提升音质自然度和情感表达

**开启条件**：
- 数据量 > 30 分钟
- GPU 显存 > 16GB

#### 3.5.3 开始训练

1. 确认所有参数配置正确
2. 点击 `开始训练` 按钮
3. 观察终端日志，确认训练正常进行

**训练日志示例**：
```
Epoch 1/10 | Step 100/500 | Loss: 1.234 | LR: 0.0002
Epoch 2/10 | Step 200/500 | Loss: 0.987 | LR: 0.0002
...
Epoch 10/10 | Step 500/500 | Loss: 0.345 | LR: 0.0002
Training completed! Best checkpoint saved.
```

#### 3.5.4 训练监控

- **Loss 下降趋势**：正常应持续下降，最终稳定在 0.3-0.5
- **过拟合检测**：如果 Loss 突然上升，可能需要减少轮数
- **音频抽检**：每 2 轮测试合成一段音频，听感评估

---

### 3.6 模型导出与推理

#### 3.6.1 导出模型

训练完成后，在 `模型管理` 标签中：

1. 选择最佳 checkpoint（通常是最后一轮）
2. 点击 `导出模型`
3. 导出路径：`GPT_SoVITS/pretrained_models/{speaker_name}/`

**导出文件结构**：
```
GPT_SoVITS/pretrained_models/tao_jianhua/
├── {speaker_name}_e10_s500.pth    # VITS 声学模型
└── {speaker_name}_e10_s500.ckpt   # GPT 文本模型
```

#### 3.6.2 推理配置

在 `推理` 标签中：

1. **加载模型**：
   - GPT 模型路径：`pretrained_models/{speaker_name}/*.ckpt`
   - SoVITS 模型路径：`pretrained_models/{speaker_name}/*.pth`

2. **参考音频设置**：
   - 上传 3-10 秒的参考音频（来自训练集）
   - 输入参考音频的文本内容

3. **合成参数**：
   - **Temperature**：`0.6`（控制随机性，0.3-1.0）
   - **Top P**：`0.9`（采样策略）
   - **Top K**：`20`（候选token数量）
   - **语速**：`1.0`（0.5-2.0可调）

#### 3.6.3 生成语音

1. 在文本框中输入要合成的文本
2. 点击 `生成音频`
3. 试听生成结果，调整参数直至满意

**最佳实践**：
- 参考音频选择情绪适中的片段
- 合成文本长度控制在 50-200 字
- 避免生成过长的连续音频（建议分段）

---

### 3.7 集成到对话系统

#### 3.7.1 启动本地推理服务

本项目使用 **GPT-SoVITS-V4-Inference** 推理特化整合包，通过 HTTP API 进行调用。

**步骤 1：启动推理服务**

```cmd
# 进入 GPT-SoVITS-V4-Inference 目录
cd GPT-SoVITS-V4-Inference

# 启动 API 服务（默认端口 8000）
gsvi.bat
```

服务启动后，会显示：
```
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**步骤 2：准备参考音频**

将训练集中的参考音频下载到本地的 `custom_refs/` 目录。

**步骤 3：确认模型可用**

**实现思路**：
1. 向 API 服务发送 GET 请求：`http://127.0.0.1:8000/classic_model_list/v4`
2. 解析返回的 JSON 数据，获取 `gpt` 和 `sovits` 字段
3. 输出可用的模型列表：
   - GPT 模型列表（如 `["【GSVI】jianhua_tao-e10"]`）
   - SoVITS 模型列表（如 `["【GSVI】jianhua_tao_e10_s540_l32"]`）

#### 3.7.2 Python API 调用

**实现思路**：使用 `src/tts/gpt_sovits_tts.py` 中的 `GPTSoVITSTTS` 类进行调用

**步骤 1：初始化 TTS 客户端**
- 导入 `GPTSoVITSTTS` 类
- 配置初始化参数：
  - API 服务地址：`http://127.0.0.1:8000`
  - GPT 模型名称：从模型列表中选择
  - SoVITS 模型名称：对应的声学模型
  - 参考音频路径：服务端相对路径（`./custom_refs/xxx.wav`）
  - 参考文本：音频对应的文字内容
  - 参考文本语言：`中文` 或 `英文`

**步骤 2：定义合成函数**
- 创建语音合成函数，接收两个参数：
  - `text`：待合成的文本内容
  - `output_path`：本地保存路径
- 调用 TTS 客户端的 `synthesize()` 方法：
  - 传入文本内容
  - 指定输出路径
  - 配置合成参数（温度、采样策略、语速等）
- 返回生成的音频文件路径

**步骤 3：执行合成**
- 准备待合成的文本（如"清华大学是一所综合性大学..."）
- 调用合成函数，生成音频文件
- 输出保存路径确认

**API 参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | str | 必填 | 要合成的文本，支持 `\n` 换行表示停顿 |
| `output_path` | str | None | 本地保存路径，若为 None 则返回服务器 URL |
| `temperature` | float | 1.0 | 温度参数，控制随机性（0.1-2.0） |
| `top_p` | float | 1.0 | Top-P 采样策略 |
| `top_k` | int | 10 | Top-K 采样策略 |
| `speed` | float | 1.0 | 语速倍率（0.5-2.0） |
| `text_split_method` | str | "按标点符号切" | 文本切分方式 |
| `media_type` | str | "wav" | 输出格式（wav/mp3） |

#### 3.7.3 流式合成（实时对话）

**实现思路**：边生成文本边合成语音，实现低延迟的实时交互

**核心流程**：
```
┌─────────────────────────────────────────────────────────┐
│  LLM 流式输出                                            │
│    "清"→"华"→"大"→"学"→...→"。"                      │
└───────────────────┬─────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────────┐
│  文本缓冲与句子检测                                      │
│  1. 累积字符到缓冲区                                     │
│  2. 检测句子结束符（。！？.!?）                         │
│  3. 触发时提取完整句子                                   │
└───────────────────┬───────────────────────────────────┘
                    ↓
┌───────────────────────────────────────────────────────┐
│  TTS 合成与播放                                          │
│  1. 调用 TTS API 合成当前句子                            │
│  2. 保存为音频文件（stream_audio_001.wav）              │
│  3. 使用音频播放库播放                                   │
│  4. 清空缓冲区，继续接收下一句                           │
└───────────────────────────────────────────────────────┘
```

**关键实现要点**：
1. **异步处理**：使用异步生成器接收 LLM 的流式输出
2. **句子切分**：定义句子结束符列表，实时检测边界
3. **并行处理**：TTS 合成与音频播放可并行，提升流畅度
4. **文件命名**：使用序号索引确保音频顺序（`stream_audio_001.wav`）
5. **缓冲管理**：合成后清空缓冲区，准备接收下一句

#### 3.7.4 对话系统完整集成

**系统架构设计**：

```
┌──────────────────────────────────────────────────────────┐
│                  VoiceDialogueSystem                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ ASR 模块     │  │ 对话路由     │  │ TTS 模块     │  │
│  │ RealtimeASR  │  │ DialogRouter │  │ GPTSoVITS    │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │           │
└─────────┼─────────────────┼─────────────────┼───────────┘
          ↓                 ↓                 ↓
     语音识别           智能对话           语音合成
```

**初始化流程**：
1. **创建 ASR 实例**
   - 模型大小：`base`（平衡速度与准确率）
   - 目标语言：中文

2. **创建对话路由实例**
   - 加载清华知识库（如已配置）
   - 初始化通用对话 Agent

3. **创建 TTS 客户端**
   - 连接到本地推理服务（端口 8000）
   - 配置 GPT/SoVITS 模型名称
   - 设置参考音频和文本

**对话处理流程**：

**输入**：用户语音文件路径（如 `user_question.wav`）

**步骤 1：语音识别**
- 调用 ASR 模块的 `transcribe()` 方法
- 输入：音频文件路径
- 输出：识别出的文本（如"清华的校训是什么？"）
- 日志：打印用户问题

**步骤 2：对话生成**
- 调用对话路由的 `route()` 方法
- 输入：用户问题文本
- 参数：`post_process=True`（移除 Markdown 格式）
- 处理：
  - 判断是否为清华相关问题
  - 路由到对应 Agent（ThuAssistant 或 General）
  - 生成回答文本
  - 去除 Markdown 标记（适合 TTS 朗读）
- 输出：纯文本回答
- 日志：打印系统回答

**步骤 3：语音合成**
- 调用 TTS 客户端的 `synthesize()` 方法
- 输入：回答文本
- 参数：
  - `output_path`：保存路径（如 `response.wav`）
  - `temperature`：1.0（自然度）
  - `speed`：1.0（正常语速）
- 输出：语音文件路径

**返回**：生成的语音文件路径

**完整数据流**：
```
用户语音文件 (user_question.wav)
    ↓ [ASR.transcribe()]
用户问题文本 ("清华的校训是什么？")
    ↓ [DialogueRouter.route(post_process=True)]
系统回答文本 ("清华大学的校训是...")
    ↓ [GPTSoVITSTTS.synthesize()]
系统语音文件 (response.wav)
```

**关键优化要点**：
- **格式处理**：开启 `post_process=True` 移除 Markdown 标记
- **参数调优**：`temperature=1.0` 保证自然度，`speed=1.0` 正常语速
- **服务复用**：使用本地推理服务，避免重复加载模型
- **错误处理**：各阶段添加异常捕获，确保系统稳定性

---

### 3.8 性能优化

#### 3.8.1 推理加速

**方案1：服务端 GPU 加速**

GPT-SoVITS-V4-Inference 推理服务默认使用 GPU，确保服务端正确配置：

```bash
# 检查 GPU 是否可用
nvidia-smi

# 启动服务时会自动使用 GPU
gsvi.bat
```

**方案2：调整批处理参数**

**实现思路**：在调用 `synthesize()` 时调整批处理参数提升速度

**优化参数**：
- `batch_size`：增加到 20（默认 10），充分利用 GPU 并行能力
- `parallel_infer`：设置为 True，开启并行推理
- `text_split_method`：设置为"按标点符号切"，自动切分长文本

**效果**：长文本合成速度提升 30-50%

**方案3：减少采样步数**

**实现思路**：适度降低采样步数可加速生成，音质略有下降

**优化参数**：
- `sample_steps`：从默认 16 降至 12-14
- 速度提升：约 20-30%
- 音质影响：轻微（普通用户难以察觉）

#### 3.8.2 音频缓存

**优化目标**：对于常见回复（如问候语），预先生成并缓存，避免重复合成

**实现思路**：

**步骤 1：设计缓存包装类**
- 类名：`CachedTTS`
- 成员变量：
  - `tts`：原始 TTS 客户端实例
  - `cache_dir`：缓存目录路径（默认 `audio_cache`）
- 初始化：创建缓存目录（如不存在）

**步骤 2：实现缓存合成方法**
- 方法签名：`synthesize(text: str) -> str`
- 输入：待合成的文本
- 返回：音频文件路径

**步骤 3：缓存键生成**
- 计算文本的 MD5 哈希值作为唯一标识
- 构造缓存文件路径：`cache_dir/{hash}.wav`

**步骤 4：缓存命中检查**
```
IF 缓存文件存在：
    输出日志："使用缓存音频"
    直接返回缓存文件路径
ELSE：
    调用原始 TTS 合成新音频
    保存到缓存路径
    返回文件路径
```

**使用流程**：
1. **创建缓存包装器**：传入原始 TTS 实例
2. **首次调用**：文本"你好，有什么可以帮助你的吗？"
   - 缓存未命中 → 调用 TTS 合成 → 保存到缓存
   - 耗时：约 3-5 秒
3. **再次调用**：相同文本
   - 缓存命中 → 直接返回缓存文件
   - 耗时：< 0.01 秒

**性能提升**：
- 加速比：300-500 倍
- 适用场景：固定问候语、常见回复、系统提示音

#### 3.8.3 批量推理

**实现思路**：处理多段文本时，使用批量合成方法减少网络开销

**批量合成流程**：

**输入准备**：
- 文本列表：`["文本1", "文本2", "文本3", ...]`
- 输出目录：`batch_outputs`
- 文件前缀：`audio`
- 合成参数：温度、语速等

**执行步骤**：
```
对于每个文本（索引 i）：
  1. 构造输出路径：{output_dir}/{prefix}_{i:03d}.wav
     示例：batch_outputs/audio_001.wav
  
  2. 调用 TTS 合成：
     - 输入：texts[i]
     - 输出：audio_files[i]
     - 参数：temperature, speed 等
  
  3. 错误处理：
     IF 合成失败：
       记录警告日志
       audio_files[i] = None
     ELSE：
       audio_files[i] = 文件路径
```

**输出统计**：
- 统计成功生成的文件数量（过滤掉 None 值）
- 输出日志："成功生成 X/Y 个音频文件"

**优势**：
- 统一参数配置，减少重复代码
- 自动文件命名（序号递增）
- 批量错误处理，不会因单个失败而中断

#### 3.8.4 异步请求

**实现思路**：对于非阻塞场景，使用异步请求提升并发性能

**异步合成架构**：

```
┌──────────────────────────────────────────────────────┐
│  主协程：batch_synthesize_async()                     │
│                                                      │
│  文本列表 → 创建多个异步任务                          │
└───────────────────┬──────────────────────────────────┘
                    ↓
        ┌───────────┴───────────┐
        ↓           ↓           ↓
   ┌────────┐  ┌────────┐  ┌────────┐
   │ 任务1  │  │ 任务2  │  │ 任务3  │
   │ 文本1  │  │ 文本2  │  │ 文本3  │
   └───┬────┘  └───┬────┘  └───┬────┘
       ↓           ↓           ↓
   ┌─────────────────────────────────┐
   │  并发执行（asyncio.gather）      │
   └─────────────────────────────────┘
       ↓           ↓           ↓
   audio_1.wav audio_2.wav audio_3.wav
```

**单个异步任务流程**：

**函数**：`async_synthesize(api_url, payload, output_path)`

**步骤 1：创建异步 HTTP 会话**
- 使用 `aiohttp.ClientSession` 管理连接池

**步骤 2：发送合成请求**
- 目标 URL：`{api_url}/infer_classic`
- 方法：POST
- 数据：JSON payload（包含文本、模型参数等）
- 超时：300 秒
- 异步等待响应，解析 JSON 获取 `audio_url`

**步骤 3：下载音频文件**
- 向 `audio_url` 发送 GET 请求
- 异步读取音频二进制数据
- 写入本地文件（`output_path`）

**步骤 4：返回文件路径**

**批量异步处理流程**：

**函数**：`batch_synthesize_async(texts)`

1. **创建任务列表**：
   - 遍历文本列表，为每个文本创建异步任务
   - 构造对应的 payload（包含文本、模型配置）
   - 指定输出路径（`output_0.wav`, `output_1.wav`, ...）

2. **并发执行**：
   - 使用 `asyncio.gather(*tasks)` 并发执行所有任务
   - 等待所有任务完成

3. **返回结果列表**：
   - 包含所有生成的音频文件路径

**性能优势**：
- **并发度**：3 个文本并行处理，总耗时 ≈ 单个耗时
- **资源利用**：网络 I/O 等待时 CPU 可处理其他任务
- **适用场景**：批量生成、实时响应、高并发服务

---

### 3.9 使用示例

完整的使用示例代码位于 [examples/tts_usage_example.py](examples/tts_usage_example.py)，包含以下场景：

| 示例 | 说明 | 关键技术 |
|------|------|----------|
| 示例1 | 基础语音合成 | `synthesize()` 基本用法 |
| 示例2 | 多行文本合成 | 使用 `\n` 创建停顿 |
| 示例3 | 参数调优 | 调整语速、温度等参数 |
| 示例4 | 批量合成 | `batch_synthesize()` 方法 |
| 示例5 | 获取可用模型 | `get_model_list()` API |
| 示例6 | 缓存加速 | 使用哈希缓存重复文本 |

**快速运行**：

```bash
# 1. 启动 GPT-SoVITS-V4-Inference 服务
cd GPT-SoVITS-V4-Inference
gsvi.bat

# 2. 运行示例（在新终端）
python examples/tts_usage_example.py
```

**预期输出**：

```
============================== GPT-SoVITS TTS 使用示例 ==============================

=== 示例5：获取可用模型 ===
✓ 连接成功
  可用的 GPT 模型 (1 个):
    - 【GSVI】jianhua_tao-e10
  可用的 SoVITS 模型 (1 个):
    - 【GSVI】jianhua_tao_e10_s540_l32

=== 示例1：基础语音合成 ===
正在合成语音，文本长度: 26 字符...
合成完成，耗时: 3.45 秒
音频已保存到: outputs/example_1.wav
✓ 音频已保存: outputs/example_1.wav

=== 示例2：多行文本合成 ===
...

✓ 所有示例运行完成！
  生成的音频文件位于 outputs/ 目录
================================================================================
```

---


## 系统集成与演示

### 完整工作流

整个语音对话系统的工作流程：

```
用户语音输入（麦克风）
    ↓
┌─────────────────────────────────────┐
│ 模块一：语音识别 (ASR)               │
│ 使用 faster-whisper (base/small)     │
│ 输出：用户问题文本                   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 模块二：智能路由 (DialogueRouter)    │
│ 步骤 1：通用大模型判断               │
│ 步骤 2：解析 XML 决策标签             │
│ 步骤 3：路由到合适的 Agent           │
│ 输出：对话回答 (Markdown)            │
└─────────────────────────────────────┘
    ↓
    两条分支：
    ├─ 清华相关问题：ThuAssistantAgent  
    │  （知识库检索 + RAG + Deepseek）   
    │                                   
    └─ 通用问题：GeneralAgent           
       （直接调用通用大模型）           
    ↓
┌─────────────────────────────────────┐
│ 模块三：语音合成 (TTS)               │
│ - 文本后处理（Markdown → 纯文本）    │
│ - 文本转语音 (Edge-TTS 或 GPT-SoVITS)│
│ 输出：语音信号→扬声器                │
└─────────────────────────────────────┘
```

### 完整集成思路

**第一步：初始化 ASR 模块**
- 创建 RealtimeASR 实例
- 选择 base 模型（平衡速度与准确率）
- 设置目标语言为中文

**第二步：初始化对话 Agent**

1. **创建 GeneralAgent**
   - 从环境变量读取 ARK_API_KEY
   - 指定通用大模型 endpoint

2. **创建 ThuAssistantAgent**
   - 从环境变量读取 AK/SK/Account ID
   - 连接清华知识库 API

**第三步：初始化智能路由器**
- 将两个 Agent 注入 DialogueRouter
- 启用 verbose 模式显示决策过程

**第四步：端到端对话流程**

**定义对话循环函数**：

1. **语音输入 + ASR**
   - 打印提示信息
   - 调用 ASR 的 transcribe_from_microphone
   - 设置超时 5 秒（自动 VAD 检测）
   - 打印识别结果

2. **智能路由处理**
   - 打印处理状态
   - 调用 router.route(识别文本)
   - 启用 post_process=True 转为 TTS 格式
   - 路由器自动决策并调用合适的 Agent
   - 打印最终回答

3. **TTS 语音合成**
   - 创建 TTS 引擎实例
   - 调用 speak 方法
   - 播放语音回答

4. **循环控制**
   - 持续运行直到用户中断（Ctrl+C）
   - 异常处理：捕获中断信号，优雅退出

**主程序入口**：
- 检查 __name__ == "__main__"
- 调用 conversation_loop 启动系统

**系统运行流程**：
```
用户说话 → VAD检测静音 → Whisper识别 → Router决策 → Agent处理 → TTS合成 → 扬声器播放
↑                                                                      ↓
└───────────────────────────── 循环继续 ───────────────────────────────┘
```

---

### Web 应用部署

为了提供更友好的交互体验，本项目提供了基于 **Gradio** 的 Web 界面，支持：
- 🎤 音频输入设备选择（麦克风录音 / 文件上传）
- 🤖 GPT 和 SoVITS 模型选择
- 💬 实时对话文本显示
- 🔊 自动播放合成语音
- 📝 对话历史记录

#### Web 应用架构

```
┌──────────────────────────────────────────────────────┐
│                  Gradio Web 界面                      │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ 系统配置   │  │ 语音对话   │  │ 使用说明   │    │
│  │            │  │            │  │            │    │
│  │ - API配置  │  │ - 音频输入 │  │ - 操作步骤 │    │
│  │ - 模型选择 │  │ - 文本显示 │  │ - 故障排除 │    │
│  │ - 参考音频 │  │ - 音频播放 │  │            │    │
│  │ - 初始化   │  │ - 对话历史 │  │            │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└──────────────────┬───────────────────────────────────┘
                   ↓
    ┌──────────────────────────────────────┐
    │     VoiceDialogueWebApp              │
    │  ┌────────┐ ┌────────┐ ┌────────┐   │
    │  │  ASR   │ │ Router │ │  TTS   │   │
    │  └────────┘ └────────┘ └────────┘   │
    └──────────────────────────────────────┘
```

#### 部署步骤

**步骤 1：安装依赖**

确保已安装 Gradio：

```bash
pip install gradio soundfile
```

**步骤 2：启动 TTS 服务**

在新终端中启动 GPT-SoVITS-V4-Inference 服务：

```bash
cd GPT-SoVITS-V4-Inference
python api.py
```

等待服务启动，确认看到：
```
INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**步骤 3：启动 Web 应用**

在项目根目录运行：

```bash
python app_web.py
```

启动成功后，终端会显示：

```
======================================================================
语音对话系统 Web 应用
======================================================================

📋 启动前检查清单：
  ✓ 确保已启动 GPT-SoVITS-V4-Inference 服务
  ✓ 确保已安装所有依赖包（pip install -r requirements.txt）
  ✓ 确保已配置好 API 密钥（火山引擎、Deepseek 等）

🚀 正在启动 Web 应用...

Running on local URL:  http://127.0.0.1:7860
```

**步骤 4：访问 Web 界面**

在浏览器中打开：`http://127.0.0.1:7860`

#### 使用流程

**1. 系统配置（首次使用必需）**

进入 **"系统配置"** 标签页：

```
┌─────────────────────────────────────────────┐
│ 1. TTS 服务配置                              │
│    - API 地址: http://127.0.0.1:8000        │
│    - 点击 "刷新模型列表"                     │
│                                             │
│ 2. 模型选择                                 │
│    - GPT 模型: 【GSVI】jianhua_tao-e10      │
│    - SoVITS 模型: ...e10_s540_l32           │
│                                             │
│ 3. 参考音频配置                             │
│    - 路径: ./custom_refs/ref.wav            │
│    - 文本: 参考音频的对应文本               │
│                                             │
│ 4. 点击 "初始化系统" 按钮                   │
└─────────────────────────────────────────────┘
```

**实现思路**：
- 前端提供表单输入 TTS API 地址、模型名称、参考音频配置
- 点击"刷新模型列表"时，发送 GET 请求到 `/classic_model_list/v4`
- 解析返回的模型列表，填充到下拉框
- 点击"初始化系统"时，实例化 ASR、DialogueRouter、GPTSoVITSTTS
- 显示初始化状态（成功/失败）

**2. 语音对话**

进入 **"语音对话"** 标签页：

**方式 A：麦克风录音**
```
1. 点击 🎤 麦克风图标
2. 开始说话（系统自动录音）
3. 再次点击图标停止录音
4. 点击 "▶️ 处理音频" 按钮
```

**方式 B：上传音频文件**
```
1. 点击 "上传" 按钮
2. 选择本地音频文件（WAV/MP3）
3. 点击 "▶️ 处理音频" 按钮
```

**处理流程**：
```
音频输入
    ↓
┌──────────────────────────────┐
│ 👤 用户说                     │
│  显示识别的文本               │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│ 🤖 系统回复                   │
│  显示生成的回答文本           │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│ 🔊 合成语音                   │
│  自动播放音频（autoplay）     │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│ 📝 对话历史                   │
│  === 对话 1 (14:30:25) ===   │
│  👤 用户: 清华的校训是什么？  │
│  🤖 系统: 清华大学的校训是... │
└──────────────────────────────┘
```

**3. 对话历史管理**

- **查看历史**：右侧面板实时显示所有对话记录
- **清空历史**：点击 "🗑️ 清空历史" 按钮

#### 功能特性

| 功能 | 说明 | 实现方式 |
|------|------|----------|
| **设备选择** | 支持麦克风录音和文件上传 | Gradio Audio 组件的 `sources` 参数 |
| **模型选择** | 动态获取并选择 GPT/SoVITS 模型 | 下拉框组件 + API 调用 |
| **文本显示** | 实时显示识别和回复文本 | Textbox 组件（非交互模式） |
| **自动播放** | 语音合成完成后自动播放 | Audio 组件的 `autoplay=True` |
| **历史记录** | 保存所有对话内容和时间戳 | 内存列表 + 格式化输出 |
| **错误处理** | 友好的错误提示 | Try-catch + 状态文本框 |

#### 代码结构

```
app_web.py                        # Web 应用主文件
├── VoiceDialogueWebApp          # 后端处理类
│   ├── __init__()               # 初始化
│   ├── initialize_system()       # 初始化 ASR/Router/TTS
│   ├── get_available_models()    # 获取模型列表
│   ├── process_audio()           # 处理音频输入
│   └── _format_history()         # 格式化对话历史
│
└── create_interface()            # 创建 Gradio 界面
    ├── Tab 1: 系统配置
    │   ├── TTS API 配置
    │   ├── 模型选择下拉框
    │   ├── 参考音频设置
    │   └── 初始化按钮
    │
    ├── Tab 2: 语音对话
    │   ├── 音频输入（麦克风/上传）
    │   ├── 处理按钮
    │   ├── 文本显示区域
    │   ├── 音频播放器
    │   └── 对话历史面板
    │
    └── Tab 3: 使用说明
```

#### 高级配置

**1. 修改服务端口**

编辑 `app_web.py` 的 `demo.launch()` 部分：

```python
demo.launch(
    server_port=7860,  # 改为其他端口（如 8080）
    share=False        # 改为 True 开启公网分享
)
```

**2. 开启公网访问**

设置 `share=True` 后，Gradio 会生成一个公网 URL：

```
Running on public URL: https://xxxxx.gradio.live
```

分享此链接即可让他人访问（链接 72 小时有效）。

**3. 自定义主题**

修改 `gr.Blocks()` 的 `theme` 参数：

```python
gr.Blocks(theme=gr.themes.Glass())  # 可选: Soft, Glass, Monochrome
```

#### 故障排除

**问题 1：无法连接到 TTS 服务**

**症状**：点击"刷新模型列表"显示连接失败

**解决方案**：
1. 确认 TTS 服务已启动（检查终端输出）
2. 访问 `http://127.0.0.1:8000/classic_model_list/v4` 验证
3. 检查防火墙设置
4. 尝试修改 API 地址为 `http://localhost:8000`

**问题 2：麦克风无法录音**

**症状**：点击麦克风图标无反应

**解决方案**：
1. 检查浏览器麦克风权限（地址栏左侧图标）
2. 使用 HTTPS 或 localhost（部分浏览器限制）
3. 尝试上传音频文件代替录音

**问题 3：音频播放失败**

**症状**：语音合成成功但无声音

**解决方案**：
1. 检查浏览器音量设置
2. 确认 `outputs/` 目录有生成的 WAV 文件
3. 手动下载音频文件测试播放
4. 检查浏览器控制台错误日志

**问题 4：系统初始化失败**

**症状**：点击"初始化系统"显示错误

**解决方案**：
1. 检查是否已配置环境变量（ARK_API_KEY 等）
2. 确认模型名称拼写正确
3. 验证参考音频路径存在（相对 TTS 服务端）
4. 查看终端日志获取详细错误

**问题 5：对话响应慢**

**优化建议**：
1. 使用 `faster-whisper` 的 `small` 模型（已默认使用 `base`）
2. 调整 TTS 参数：`sample_steps=12`（默认 16）
3. 启用音频缓存（常见问候语）
4. 使用 GPU 加速（确保 CUDA 可用）

#### 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **ASR 延迟** | 1-3 秒 | 使用 faster-whisper base 模型 |
| **对话生成** | 2-5 秒 | 取决于问题复杂度和网络延迟 |
| **TTS 合成** | 3-8 秒 | 取决于文本长度和模型参数 |
| **端到端** | 6-16 秒 | 从用户说话到听到回复 |
| **内存占用** | 2-4 GB | ASR + LLM + TTS 服务 |

---

## 总结与感悟
本项目搭建并验证了一个端到端的语音对话原型，覆盖从语音识别（ASR）、对话生成（Dialogue）到语音合成（TTS）的完整链路。实现上遵循模块化设计，使各模块可以独立替换与调试：

- 核心成果：完成 ASR（基于 faster-whisper）、对话路由、以及与 GPT-SoVITS TTS 服务的对接；实现模型动态选择、音频上传/录音输入、结果展示与对话历史保存。
- 关键问题与解决思路：通过使用 ffmpeg/soundfile 加速音频预处理、引入数字格式后处理（阿拉伯数字 ↔ 中文数字），以及在模型选择上权衡实时性与准确性，显著提升了系统的健壮性与可用性。
- 工程教训：实时流式交互虽然提升体验，但会带来并发与状态同步复杂性。优先保证按需处理流程的稳定性更利于迭代与调试；良好的日志、状态反馈与边界条件处理能极大降低运维成本。

### 评估测试与对比分析

在开发过程中，我们开展了一系列评估测试与对比实验，以量化各模块的性能并指导工程取舍：

- ASR 基准测试：在 AISHELL-1 上测量了各型号的 CER 与 RTF（见上文表格），结果显示 large-v3 在准确率上优于小模型（CER 最低），但在实时场景中 `base` / `small` 在 RTF 与可用性上更优，适合在线对话部署。

- faster-whisper vs openai-whisper：faster-whisper 在推理速度上有显著优势（加速数倍），因此作为实时识别主选；openai-whisper 可作为精度参考但延迟较高。

- TTS 延迟对比：通过多段文本测试，TTS 合成时间随文本长度近似线性增长（本项目测得范围约 3–8 秒），短句合成延迟可接受，中长文本建议分段或异步合成以改善交互体验。

- 端到端延迟：在默认配置下（base/small ASR + 网络化 LLM + GPT-SoVITS TTS），端到端响应时间通常在 6–16 秒区间，影响因素主要为模型选择、网络延迟与 TTS 文本长度。

- VAD 与噪声鲁棒性测试：通过对比不同 VAD 阈值和静默判定时长，选择更高的阈值和略长的静默时长可以显著降低误触发率，兼顾识别召回。

- 文本后处理效果验证：引入阿拉伯数字 ↔ 中文数字转换后对若干样本进行对比，示例见上文，后处理在特定任务（如年代、数量）上能显著降低表面错误，提升可读性与下游评估指标。

- 音频预处理与兼容性：使用 `ffmpeg` / `soundfile` 做重采样与格式转换，提高了对多种输入文件的兼容性与处理速度；在多设备（Windows、主流浏览器）上验证了录音/播放链路的稳定性。

- 稳定性与失败模式分析：进行了长时间压力测试与异常注入（TTS 服务断连、模型加载失败、设备变更），并基于结果改进了错误提示、重试策略与资源释放逻辑。

以上评估与对比为工程决策提供了数据支撑：在保证体验的前提下优先选择延迟更低的组件（如 faster-whisper、较小的 ASR 模型），并通过后处理与参数调优弥补精度损失。

未来方向展望：

1. 在保证稳定性的前提下，逐步引入流式/并发能力，并用明确的状态机或队列机制管理并发请求。
2. 将知识库（如 KdConv）以向量检索方式接入，采用 RAG 策略提升领域问答质量。
3. 优化延迟（模型量化、推理加速、音频缓存）并扩展 TTS 的多说话人/少样本克隆能力。

总体而言，本项目达成了工程可运行且易于扩展的原型目标，为后续深入优化与功能扩展打下了良好基础。

---

## 参考资料

- OpenAI Whisper (原始论文与代码)：https://github.com/openai/whisper
- faster-whisper（Whisper 的推理加速实现）：https://github.com/guillaumekln/faster-whisper
- GPT-SoVITS-V4-Inference（TTS 服务实现，项目中使用的 TTS 接口）：https://github.com/RVC-Boss/GPT-SoVITS-V4-Inference
- SoVITS-SVC / VITS（语音合成与声码器相关项目）：https://github.com/jaywalnut310/vits
- KdConv（清华 CoAI 的知识驱动对话数据集）：https://github.com/thu-coai/KdConv
- AISHELL-1（中文语音识别评估数据集）：https://www.openslr.org/33/
- FunASR（阿里开源的语音识别工具链）：https://github.com/alibaba-damo-academy/FunASR
- Gradio（快速构建 Web UI）：https://gradio.app
- ffmpeg（音频处理与转码工具）：https://ffmpeg.org
- soundfile（Python 音频读写库）：https://pysoundfile.readthedocs.io
- librosa（音频特征与处理库）：https://librosa.org
- Hugging Face（模型与数据集托管）：https://huggingface.co
- Edge-TTS（微软 Edge 的 TTS 客户端实现）：https://github.com/rany2/edge-tts


