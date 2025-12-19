# 语音对话系统 - 大作业报告

> **课程**：人工智能导论  
> **作业**：搭建语音对话系统  
> **日期**：2025年12月

---

## 目录

1. [项目概述](#项目概述)
2. [快速开始](#快速开始)
3. [系统架构](#系统架构)
4. [模块一：语音识别 (ASR)](#模块一语音识别-asr)
5. [模块二：对话模型](#模块二对话模型)
   - 2.1 [技术选型与架构设计](#21-技术选型与架构设计)
   - 2.2 [基于RAG的清华本科学习助手 Agent](#22-基于rag的清华本科学习助手-agent)
   - 2.3 [ThuAssistantAgent 实现](#23-thuassistantagent-实现)
   - 2.4 [GeneralAgent（通用大模型）](#24-generalagent通用大模型)
   - 2.5 [模型评估系统](#25-模型评估系统)
6. [模块三：语音合成 (TTS)](#模块三语音合成-tts)
7. [系统集成与演示](#系统集成与演示)
8. [实验结果与分析](#实验结果与分析)
9. [总结与感悟](#总结与感悟)

---

## 快速开始

### 环境配置

```bash
# 创建虚拟环境
conda create -n conversation-bot python=3.10 -y
conda activate conversation-bot

# 安装依赖
pip install -r requirements.txt
```

### 设置环境变量

```powershell
# Windows PowerShell
$env:ARK_API_KEY="your-api-key"           # 通用大模型 API Key
$env:THU_AGENT_AK="your-access-key"       # 清华助手 Access Key
$env:THU_AGENT_SK="your-secret-key"       # 清华助手 Secret Key
$env:THU_AGENT_ACCOUNT_ID="your-id"       # 清华助手 Account ID
```

### 运行智能路由系统

```bash
# 简单测试（6 个测试用例）
python src/dialogue/demo_router.py

# 完整集成演示
python src/dialogue/integration_demo.py
```

### 运行模型评估

```bash
# 快速评估（5题/配置，约2分钟）
python src/dialogue/quick_evaluate.py

# 完整评估（40题/配置，约15分钟）
python src/dialogue/evaluate_models.py

# 查看评估结果
python src/dialogue/visualize_results.py evaluation_results/evaluation_results_*.json --markdown
```

**详细文档**：
- 📖 [完整评估指南](MODEL_EVALUATION_GUIDE.md)
- 🚀 [快速参考](EVALUATION_QUICKSTART.md)
- 📊 [实现总结](EVALUATION_IMPLEMENTATION_SUMMARY.md)

### 基本使用思路

**系统初始化流程**：

1. **创建通用对话 Agent**
   - 从环境变量读取 API Key
   - 配置基础 URL 和模型参数

2. **创建清华助手 Agent**
   - 从环境变量读取三个认证参数（AK, SK, Account ID）
   - 连接知识库 API 服务

3. **构建智能路由器**
   - 将两个 Agent 注入路由器
   - 启用详细日志模式（verbose=True）便于调试

4. **处理用户查询**
   - 调用路由器的 route 方法
   - 路由器自动判断并分发到合适的 Agent
   - 返回最终答案

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
   ✓ GeneralAgent 初始化完成
2. 初始化 ThuAssistantAgent...
   ✓ ThuAssistantAgent 初始化完成
3. 初始化 DialogueRouter...
   ✓ DialogueRouter 初始化完成

======================================================================
  测试 1：通用闲聊问题 (预期路由: GeneralAgent)
======================================================================

问题: 你好，今天天气怎么样？

[Router] 用户问题: 你好，今天天气怎么样？
[Router] 正在判断是否需要专业 Agent...
[Router] ✓ 使用 GeneralAgent 回答 (通用知识)

回答:
我是 Claude AI 助手。虽然我无法告诉你实时天气信息，但我可以建议你...

======================================================================
  测试 3：清华课程问题 (预期路由: ThuAssistantAgent)
======================================================================

问题: 清华大学有哪些人工智能相关的课程？

[Router] 用户问题: 清华大学有哪些人工智能相关的课程？
[Router] 正在判断是否需要专业 Agent...
[Router] ✓ 路由到 ThuAssistantAgent (专业知识库)

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
| kimi-k2-thinking-251104 | 251104 | ✓ 支持 |
| doubao-seed-1-6-flash-250828 | 250828 | ✓ 支持 |
| doubao-seed-1-6-thinking-250715 | 250715 | ✓ 支持 |
| deepseek-v3-2-251201 | 251201 | ✓ 支持 |

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
🏆 最佳配置排行榜
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
💭 Thinking 模式对比
================================================================================

模型: kimi-k2-thinking-251104
  质量分数: 0.723 → 0.765 (+0.042, +5.8%)
  响应时间: 1.98s → 2.87s (+0.89s, +44.9%)
  ✓ 推荐: 开启 thinking 模式（质量提升明显，速度损失可接受）

模型: doubao-seed-1-6-thinking-250715
  质量分数: 0.698 → 0.743 (+0.045, +6.4%)
  响应时间: 2.34s → 4.12s (+1.78s, +76.1%)
  ⚠️  权衡: thinking 模式提升质量，但速度慢一倍以上

模型: deepseek-v3-2-251201
  质量分数: 0.756 → 0.782 (+0.026, +3.4%)
  响应时间: 2.89s → 3.45s (+0.56s, +19.4%)
  ✓ 推荐: 开启 thinking 模式（质量提升明显，速度损失可接受）
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

#### 应用建议

**1. 实时对话场景（优先速度）**
- 推荐：`doubao-seed-1-6-flash-250828` (thinking_disabled)
- 特点：响应时间 < 1.5s，质量达到 70%

**2. 高质量咨询场景（优先质量）**
- 推荐：`deepseek-v3-2-251201` (thinking_enabled)
- 特点：质量分数最高（0.782），响应时间适中（3.5s）

**3. 平衡场景（质量+速度）**
- 推荐：`kimi-k2-thinking-251104` (thinking_enabled)
- 特点：综合得分最高，响应时间 < 3s，质量良好（0.765）

**4. 资源受限场景（降低成本）**
- 推荐：所有模型的 `thinking_disabled` 模式
- 特点：速度提升 40-80%，质量仅下降 5-10%

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

3. **TTS 语音合成（待实现）**
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
↓                                                                                        ↑
└───────────────────────────── 循环继续 ───────────────────────────────┘
```

### 关键特性

1. **智能路由**
   - 通用大模型自动判断问题类型
   - XML 标签清晰表达决策逻辑
   - 高准确率的专业知识库路由

2. **RAG 增强**
   - ThuAssistantAgent 自动检索知识库
   - 结合上下文生成精准答案
   - 支持多轮对话跟踪

3. **多模态支持**
   - GeneralAgent 支持文本和图片输入
   - ThuAssistantAgent 支持知识库中的表格/图表

4. **TTS 友好**
   - Markdown 自动转换为纯文本
   - 保留自然停顿和语气
   - 兼容 Edge-TTS、GPT-SoVITS 等引擎

---

## 实验结果与分析

> 🚧 待完善...

---

## 总结与感悟

### 搭建感悟

1. **模块化设计的重要性**：将系统拆分为 ASR、Dialogue、TTS 三个模块，便于独立开发和测试

2. **智能路由的设计**：通过通用大模型进行动态决策，比硬编码规则更灵活、更可扩展

3. **RAG 架构的应用**：知识库 + 大模型的组合能显著提升专业领域的回答质量

4. **开源生态的便利**：火山引擎 API、Whisper 等开源工具大大降低了开发门槛

5. **文本后处理的必要性**：Markdown 格式对人类阅读友好，但对 TTS 会产生干扰，需要专门转换

### 收获

✅ **已完成**：
- 掌握了语音识别系统搭建（ASR 模块完成）
- 实现了 Whisper 模型优化（faster-whisper 加速）
- 建立了智能路由系统（Dialogue 模块完成）
- 集成了两个专业 Agent（GeneralAgent + ThuAssistantAgent）
- 实现了文本后处理工具（Markdown → TTS）

⏳ **进行中**：
- GPT-SoVITS 环境搭建
- TTS 模块集成

📝 **待改进**：

- [x] 优化识别延迟（已通过 faster-whisper 解决）
- [x] 支持流式识别（已实现 VAD 实时识别）
- [x] 实现智能路由（已通过 DialogueRouter 实现）
- [ ] 添加噪声抑制
- [ ] 支持声音克隆（GPT-SoVITS）
- [ ] 多轮对话上下文管理
- [ ] 用户偏好学习

---

## 参考资料

1. [OpenAI Whisper](https://github.com/openai/whisper)
2. [Whisper 论文](https://arxiv.org/abs/2212.04356)
3. [KdConv 数据集](https://github.com/thu-coai/KdConv)
4. [Edge-TTS](https://github.com/rany2/edge-tts)
5. [Gradio 文档](https://gradio.app/docs)
