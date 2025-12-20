"""
语音对话系统 Web 应用
使用 Gradio 构建交互界面，支持音频输入设备选择、模型选择、实时对话和音频播放
"""

from typing import Optional
import gradio as gr
import os
import time
import requests
import threading
import queue
import numpy as np
from pathlib import Path
from src.asr.realtime_asr import RealtimeASR
from src.dialogue.router import DialogueRouter
from src.dialogue.general_agent import GeneralAgent
from src.dialogue.thu_agent import ThuAssistantAgent
from src.tts import GPTSoVITSTTS


class VoiceDialogueWebApp:
    """语音对话系统 Web 应用类"""
    def __init__(self):
        self.asr = None
        self.router = None
        self.tts = None
        self.tts_api_url = "http://127.0.0.1:8000"
        self.conversation_history = []
        
        # 实时模式相关变量
        self.realtime_mode = False
        self.realtime_thread = None
        self.audio_queue = queue.Queue()
        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speaking = False
        self.chunk_duration = 0.5  # 音频块时长(秒)
        self.silence_threshold = 6  # 静音帧数阈值
        self.min_speech_frames = 4  # 最小语音帧数
        
    def initialize_system(self, tts_api_url, gpt_model, sovits_model, ref_audio_path, ref_text):
        """
        初始化语音对话系统
        
        Args:
            tts_api_url: TTS API 服务地址
            gpt_model: GPT 模型名称
            sovits_model: SoVITS 模型名称
            ref_audio_path: 参考音频路径
            ref_text: 参考文本
        """
        try:
            # 初始化 ASR（语音识别）
            if self.asr is None:
                self.asr = RealtimeASR(model_name="large-v3", language="zh")
                print("✓ ASR 模块初始化成功")
            
            # 初始化对话路由
            if self.router is None:
                # 从环境变量读取 API 密钥
                ark_api_key = os.getenv("ARK_API_KEY")
                if not ark_api_key:
                    raise ValueError("请设置环境变量 ARK_API_KEY")
                
                volc_ak = os.getenv("THU_AGENT_AK")
                volc_sk = os.getenv("THU_AGENT_SK")
                volc_account_id = os.getenv("THU_AGENT_ACCOUNT_ID")
                if not all([volc_ak, volc_sk, volc_account_id]):
                    raise ValueError("请设置环境变量 THU_AGENT_AK, THU_AGENT_SK, THU_AGENT_ACCOUNT_ID")
                
                # 创建 GeneralAgent
                general_agent = GeneralAgent(api_key=ark_api_key)
                
                # 创建 ThuAssistantAgent
                thu_agent = ThuAssistantAgent(
                    ak=volc_ak,
                    sk=volc_sk,
                    account_id=volc_account_id
                )
                
                # 创建 DialogueRouter
                self.router = DialogueRouter(
                    general_agent=general_agent,
                    thu_agent=thu_agent,
                    verbose=True
                )
                print("✓ 对话路由初始化成功")
            
            # 初始化 TTS（语音合成）
            self.tts_api_url = tts_api_url
            self.tts = GPTSoVITSTTS(
                api_url=tts_api_url,
                gpt_model_name=gpt_model,
                sovits_model_name=sovits_model,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                ref_text_lang="中文"
            )
            print("✓ TTS 模块初始化成功")
            
            # 初始化成功后获取设备状态信息（设备下拉由刷新按钮填充）
            _, device_status_msg = self.get_audio_devices()
            return "✓ 系统初始化成功！可以开始对话了。", device_status_msg
        
        except Exception as e:
            # 初始化失败时也返回失败信息
            return f"✗ 初始化失败: {str(e)}", str(e)
    
    def get_available_models(self, tts_api_url):
        """
        获取可用的 TTS 模型列表
        
        Args:
            tts_api_url: TTS API 服务地址
        
        Returns:
            (gpt_models, sovits_models, status_message)
        """
        try:
            response = requests.get(f"{tts_api_url}/classic_model_list/v4", timeout=10)
            response.raise_for_status()
            models = response.json()
            
            gpt_models = models.get('gpt', [])
            sovits_models = models.get('sovits', [])
            
            if not gpt_models or not sovits_models:
                return [], [], "⚠ 未找到可用模型，请检查 TTS 服务是否正常运行"
            
            status = f"✓ 找到 {len(gpt_models)} 个 GPT 模型和 {len(sovits_models)} 个 SoVITS 模型"
            return gpt_models, sovits_models, status
        
        except requests.RequestException as e:
            return [], [], f"✗ 无法连接到 TTS 服务 ({tts_api_url}): {str(e)}"

    def get_audio_devices(self):
        """获取本地录音设备列表，返回 (choices, status_message)"""
        try:
            if self.asr is None:
                return [], "✗ 请先初始化 ASR 模块 (点击初始化系统)"
            devices = self.asr.list_audio_devices(show=False)
            choices = [f"{d['index']} - {d['name']}" for d in devices]
            status = f"✓ 找到 {len(choices)} 个录音输入设备"
            return choices, status
        except Exception as e:
            return [], f"✗ 获取设备失败: {e}"
    
    def process_audio(self, audio_input):
        """
        处理音频输入，返回对话文本和合成的语音
        
        Args:
            audio_input: 音频输入（文件路径或 tuple）
        
        Returns:
            (user_text, bot_text, output_audio_path, conversation_log)
        """
        if not self.asr or not self.router or not self.tts:
            return "", "请先初始化系统！", None, self._format_history()
        
        try:
            # 处理不同的音频输入格式
            if isinstance(audio_input, tuple):
                # Gradio Audio 组件返回 (sample_rate, audio_array)
                audio_path = "temp_input.wav"
                import soundfile as sf
                sf.write(audio_path, audio_input[1], audio_input[0])
            else:
                # 直接是文件路径
                audio_path = audio_input
            
            # 步骤1：语音识别
            print(f"[ASR] 正在识别音频: {audio_path}")
            user_text = self.asr.transcribe(audio_path)
            print(f"[ASR] 识别结果: {user_text}")
            
            if not user_text or user_text.strip() == "":
                return "", "未识别到有效语音，请重试。", None, self._format_history()
            
            # 步骤2：对话生成
            print(f"[Dialogue] 正在生成回复...")
            bot_text = self.router.route(
                user_query=user_text,
                post_process=True  # 移除 Markdown 格式
            )
            print(f"[Dialogue] 回复: {bot_text}")
            
            # 步骤3：语音合成
            print(f"[TTS] 正在合成语音...")
            timestamp = int(time.time())
            output_audio_path = f"outputs/response_{timestamp}.wav"
            os.makedirs("outputs", exist_ok=True)
            
            self.tts.synthesize(
                text=bot_text,
                output_path=output_audio_path,
                temperature=1.0,
                speed=1.0
            )
            print(f"[TTS] 音频已保存: {output_audio_path}")
            
            # 记录对话历史
            self.conversation_history.append({
                "user": user_text,
                "bot": bot_text,
                "timestamp": time.strftime("%H:%M:%S")
            })
            
            return user_text, bot_text, output_audio_path, self._format_history()
        
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return "", error_msg, None, self._format_history()
    
    def _format_history(self):
        """格式化对话历史为显示文本"""
        if not self.conversation_history:
            return "暂无对话记录"
        
        history_text = ""
        for i, item in enumerate(self.conversation_history, 1):
            history_text += f"=== 对话 {i} ({item['timestamp']}) ===\n"
            history_text += f"👤 用户: {item['user']}\n"
            history_text += f"🤖 系统: {item['bot']}\n\n"
        
        return history_text
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        return "对话历史已清空", self._format_history()
    
    def start_realtime_mode(self):
        """启动实时语音对话模式"""
        if not self.asr or not self.router or not self.tts:
            return "❌ 请先初始化系统！", self._format_history(), None
        
        if self.realtime_mode:
            return "⚠️ 实时模式已经在运行中", self._format_history(), None
        
        self.realtime_mode = True
        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speaking = False
        
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        return "✅ 实时模式已启动，开始说话吧！", self._format_history(), None
    
    def stop_realtime_mode(self):
        """停止实时语音对话模式"""
        self.realtime_mode = False
        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speaking = False
        
        # 清空音频队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        return "⏹️ 实时模式已停止", self._format_history(), None
    
    def process_realtime_audio(self, audio_chunk):
        """
        处理实时音频流
        
        Args:
            audio_chunk: 音频数据 (sample_rate, audio_array)
        
        Yields:
            (status_text, conversation_log, audio_output)
        """
        if not self.realtime_mode:
            yield "⏹️ 实时模式未启动", self._format_history(), None
            return
        
        if audio_chunk is None:
            yield "⚠️ 未收到音频数据", self._format_history(), None
            return
        
        try:
            # 解析音频数据
            sample_rate, audio_data = audio_chunk
            
            # 转换为单声道 float32
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)
            
            # 重采样到 16kHz（如果需要）
            if sample_rate != 16000:
                import scipy.signal
                audio_data = scipy.signal.resample(
                    audio_data,
                    int(len(audio_data) * 16000 / sample_rate)
                )
                sample_rate = 16000
            
            # 计算音频能量
            energy = np.sqrt(np.mean(audio_data ** 2))
            vad_threshold = self.asr.vad_threshold if hasattr(self.asr, 'vad_threshold') else 0.01
            
            if energy > vad_threshold:
                # 检测到语音
                if not self.is_speaking:
                    self.is_speaking = True
                    yield "🎤 检测到语音...", self._format_history(), None
                
                self.speech_buffer.append(audio_data)
                self.silence_frames = 0
            else:
                # 静音
                if self.is_speaking:
                    self.silence_frames += 1
                    self.speech_buffer.append(audio_data)  # 保留一些静音
                    
                    if self.silence_frames >= self.silence_threshold:
                        # 静音时间足够，触发识别和对话
                        if len(self.speech_buffer) >= self.min_speech_frames:
                            yield "🔄 处理中...", self._format_history(), None
                            
                            # 合并音频数据
                            audio = np.concatenate(self.speech_buffer)
                            
                            # 保存临时音频文件
                            import soundfile as sf
                            timestamp = int(time.time())
                            temp_audio_path = f"temp_realtime_{timestamp}.wav"
                            sf.write(temp_audio_path, audio, sample_rate)
                            
                            # 处理音频（识别 + 对话 + 合成）
                            user_text, bot_text, output_audio_path, conv_log = self.process_audio(temp_audio_path)
                            
                            # 清理临时文件
                            try:
                                os.remove(temp_audio_path)
                            except:
                                pass
                            
                            # 返回结果
                            if output_audio_path:
                                yield f"✅ 完成\n👤: {user_text}\n🤖: {bot_text}", conv_log, output_audio_path
                            else:
                                yield f"⚠️ {bot_text}", conv_log, None
                        else:
                            yield "⚠️ 语音太短，已忽略", self._format_history(), None
                        
                        # 重置状态
                        self.speech_buffer = []
                        self.is_speaking = False
                        self.silence_frames = 0
            
        except Exception as e:
            yield f"❌ 处理错误: {str(e)}", self._format_history(), None
            # 重置状态
            self.speech_buffer = []
            self.is_speaking = False
            self.silence_frames = 0




def create_interface():
    """创建 Gradio 界面"""
    
    app = VoiceDialogueWebApp()
    
    with gr.Blocks(title="语音对话系统") as demo:
        gr.Markdown("""
        # 🎙️ 语音对话系统
        
        一个集成了语音识别、智能对话和语音合成的端到端系统
        """)
        
        with gr.Tab("⚙️ 系统配置"):
            gr.Markdown("### 1. TTS 服务配置")
            
            with gr.Row():
                tts_api_url_input = gr.Textbox(
                    label="TTS API 地址",
                    value="http://127.0.0.1:8000",
                    placeholder="http://127.0.0.1:8000"
                )
                refresh_models_btn = gr.Button("🔄 刷新模型列表", size="sm")
            
            model_status = gr.Textbox(label="状态", interactive=False)
            
            gr.Markdown("### 2. 模型选择")
            
            with gr.Row():
                gpt_model_dropdown = gr.Dropdown(
                    label="GPT 模型",
                    choices=[],
                    interactive=True
                )
                sovits_model_dropdown = gr.Dropdown(
                    label="SoVITS 模型",
                    choices=[],
                    interactive=True
                )
            
            gr.Markdown("### 3. 参考音频配置")
            
            ref_audio_path_input = gr.Textbox(
                label="参考音频路径（相对服务端）",
                value="./custom_refs/jianhua_tao.wav",
                placeholder="./custom_refs/your_ref_audio.wav"
            )
            
            ref_text_input = gr.Textbox(
                label="参考音频文本",
                value="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享",
                placeholder="参考音频对应的文本内容",
                lines=2
            )
            
            init_btn = gr.Button("🚀 初始化系统", variant="primary", size="lg")
            init_status = gr.Textbox(label="初始化状态", interactive=False)
            device_status_sys = gr.Textbox(label="设备状态（系统）", interactive=False)
            
            # 刷新模型列表
            def update_models(api_url):
                gpt_models, sovits_models, status = app.get_available_models(api_url)
                return (
                    gr.Dropdown(choices=gpt_models, value=gpt_models[0] if gpt_models else None),
                    gr.Dropdown(choices=sovits_models, value=sovits_models[0] if sovits_models else None),
                    status
                )
            
            refresh_models_btn.click(
                fn=update_models,
                inputs=[tts_api_url_input],
                outputs=[gpt_model_dropdown, sovits_model_dropdown, model_status]
            )
            
            # 初始化系统
            init_btn.click(
                fn=app.initialize_system,
                inputs=[
                    tts_api_url_input,
                    gpt_model_dropdown,
                    sovits_model_dropdown,
                    ref_audio_path_input,
                    ref_text_input
                ],
                outputs=[init_status, device_status_sys]
            )
        
        with gr.Tab("💬 语音对话"):
            gr.Markdown("### 开始对话")
            gr.Markdown("点击麦克风图标录音，或上传音频文件")
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        label="🎤 音频输入",
                        sources=["microphone", "upload"],
                        type="filepath"
                    )
                    process_btn = gr.Button("▶️ 处理音频", variant="primary", size="lg")
                    gr.Markdown("### 对话内容")
                    user_text_output = gr.Textbox(
                        label="👤 用户说",
                        interactive=False,
                        lines=2
                    )
                    bot_text_output = gr.Textbox(
                        label="🤖 系统回复",
                        interactive=False,
                        lines=4
                    )
                    audio_output = gr.Audio(
                        label="🔊 合成语音",
                        autoplay=True,
                        type="filepath"
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 对话历史")
                    conversation_log = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=20,
                        max_lines=30
                    )
                    clear_btn = gr.Button("🗑️ 清空历史", size="sm")
            # 处理音频
            process_btn.click(
                fn=app.process_audio,
                inputs=[audio_input],
                outputs=[user_text_output, bot_text_output, audio_output, conversation_log]
            )
            # 清空历史
            clear_btn.click(
                fn=app.clear_history,
                inputs=[],
                outputs=[bot_text_output, conversation_log]
            )
        
        with gr.Tab("🎙️ 实时对话"):
            gr.Markdown("### 实时语音对话")
            gr.Markdown("启动后持续录音，检测到语音并识别到静音后自动处理并回复")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 实时音频流")
                    realtime_audio_input = gr.Audio(
                        label="",
                        sources=["microphone"],
                        streaming=True,
                        type="numpy"
                    )
                    
                    with gr.Row():
                        start_realtime_btn = gr.Button("🟢 启动实时模式", variant="primary", size="lg")
                        stop_realtime_btn = gr.Button("🔴 停止实时模式", variant="stop", size="lg")
                    
                    realtime_status = gr.Textbox(
                        label="📊 状态",
                        interactive=False,
                        lines=5
                    )
                    
                    realtime_audio_output = gr.Audio(
                        label="🔊 合成语音",
                        autoplay=True,
                        type="filepath"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 对话历史")
                    realtime_conversation_log = gr.Textbox(
                        label="",
                        interactive=False,
                        lines=25,
                        max_lines=30
                    )
                    realtime_clear_btn = gr.Button("🗑️ 清空历史", size="sm")
            
            # 启动实时模式
            start_realtime_btn.click(
                fn=app.start_realtime_mode,
                inputs=[],
                outputs=[realtime_status, realtime_conversation_log, realtime_audio_output]
            )
            
            # 停止实时模式
            stop_realtime_btn.click(
                fn=app.stop_realtime_mode,
                inputs=[],
                outputs=[realtime_status, realtime_conversation_log, realtime_audio_output]
            )
            
            # 处理实时音频流
            realtime_audio_input.stream(
                fn=app.process_realtime_audio,
                inputs=[realtime_audio_input],
                outputs=[realtime_status, realtime_conversation_log, realtime_audio_output]
            )
            
            # 清空历史
            realtime_clear_btn.click(
                fn=app.clear_history,
                inputs=[],
                outputs=[realtime_status, realtime_conversation_log]
            )
        
        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 📖 使用步骤
            
            ### 1️⃣ 启动 TTS 服务
            
            在使用本系统前，需要先启动 GPT-SoVITS-V4-Inference 服务：
            
            ```bash
            cd GPT-SoVITS-V4-Inference
            python api.py
            ```
            
            ### 2️⃣ 配置系统
            
            前往 **"系统配置"** 标签页：
            
            1. 点击 **"刷新模型列表"** 获取可用模型
            2. 选择 **GPT 模型** 和 **SoVITS 模型**
            3. 配置 **参考音频路径** 和 **参考文本**
            4. 点击 **"初始化系统"** 完成配置
            
            ### 3️⃣ 开始对话
            
            前往 **"语音对话"** 标签页：
            
            1. **录音方式**：点击麦克风图标开始录音，再次点击停止
            2. **上传方式**：点击上传按钮，选择本地音频文件
            3. 点击 **"处理音频"** 按钮
            4. 系统将自动：
               - 识别语音内容
               - 生成智能回复
               - 合成语音并播放
            5. 查看右侧 **对话历史** 了解完整对话记录
            
            ### 4️⃣ 功能说明
            
            - **自动播放**：合成的语音会自动播放
            - **对话历史**：所有对话记录会保存在右侧面板
            - **清空历史**：点击 "清空历史" 按钮清除所有记录
            - **模型切换**：可随时在配置页面切换模型并重新初始化
            
            ### ⚠️ 注意事项
            
            1. **首次使用**：请确保已按照 README.md 完成环境部署
            2. **TTS 服务**：必须先启动 TTS 服务，否则无法合成语音
            3. **音频格式**：支持 WAV、MP3 等常见格式
            4. **网络环境**：TTS 服务需要能够访问本地 API（默认 http://127.0.0.1:8000）
            
            ### 🛠️ 故障排除
            
            **问题：无法获取模型列表**
            - 检查 TTS 服务是否启动
            - 确认 API 地址正确
            - 尝试访问 http://127.0.0.1:8000/classic_model_list/v4
            
            **问题：语音识别失败**
            - 确保音频清晰，无明显噪音
            - 录音时长建议 2-10 秒
            - 检查麦克风权限
            
            **问题：语音合成失败**
            - 检查模型是否正确加载
            - 确认参考音频路径存在
            - 查看终端日志获取详细错误信息
            
            ## 🎙️ 实时对话模式使用说明
            
            ### 什么是实时对话模式？
            
            实时对话模式允许你像真人对话一样，无需手动点击按钮，系统会自动检测你的语音、识别内容、生成回复并播放。
            
            ### 使用方法
            
            1. 前往 **"实时对话"** 标签页
            2. 点击 **"启动实时模式"** 按钮
            3. 允许浏览器访问麦克风权限
            4. 开始正常说话，系统会自动：
               - 检测到你的语音开始
               - 等待你说话结束（静音约 3 秒）
               - 自动识别、生成回复、播放语音
               - 继续监听下一轮对话
            5. 点击 **"停止实时模式"** 结束对话
            
            ### 工作原理
            
            - **VAD 检测**：使用语音活动检测（Voice Activity Detection）判断是否在说话
            - **自动分段**：静音超过阈值后自动触发处理
            - **连续对话**：处理完成后自动继续监听，无需手动操作
            
            ### 注意事项
            
            - 说话时保持环境安静，避免背景噪音
            - 每句话之间留有明显停顿（约 3 秒）
            - 如果误触发，可以点击停止后重新启动
            - 实时模式下对话历史会自动更新
            """)
    
    return demo


def main():
    """主函数：启动 Web 应用"""
    print("=" * 60)
    print("语音对话系统 Web 应用")
    print("=" * 60)
    print("\n📋 启动前检查清单：")
    print("  ✓ 确保已启动 GPT-SoVITS-V4-Inference 服务")
    print("  ✓ 确保已安装所有依赖包（pip install -r requirements.txt）")
    print("  ✓ 确保已配置好 API 密钥（火山引擎、Deepseek 等）")
    print("\n🚀 正在启动 Web 应用...\n")
    
    demo = create_interface()
    
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 不创建公网链接（可改为 True）
        show_error=True,        # 显示详细错误信息
        quiet=False,            # 显示启动日志
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()
