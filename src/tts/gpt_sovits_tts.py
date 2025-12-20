"""
GPT-SoVITS 语音合成模块
使用 GPT-SoVITS-V4-Inference 推理特化整合包的 API 接口
"""

import requests
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path


class GPTSoVITSTTS:
    """GPT-SoVITS TTS 客户端，通过 HTTP API 调用本地推理服务"""
    
    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        gpt_model_name: Optional[str] = None,
        sovits_model_name: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_text_lang: str = "中文",
        timeout: int = 300
    ):
        """
        初始化 GPT-SoVITS TTS 客户端
        
        Args:
            api_url: API 服务地址
            gpt_model_name: GPT 模型名称，如 "【GSVI】jianhua_tao-e10"
            sovits_model_name: SoVITS 模型名称，如 "【GSVI】jianhua_tao_e10_s540_l32"
            ref_audio_path: 参考音频路径（相对于服务端）
            ref_text: 参考音频对应的文本
            ref_text_lang: 参考文本语言，默认 "中文"
            timeout: 请求超时时间（秒）
        """
        self.api_url = api_url.rstrip('/')
        self.gpt_model_name = gpt_model_name
        self.sovits_model_name = sovits_model_name
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.ref_text_lang = ref_text_lang
        self.timeout = timeout
        
        # 如果未指定模型，尝试自动获取第一个可用模型
        if not gpt_model_name or not sovits_model_name:
            self._auto_select_models()
    
    def _auto_select_models(self):
        """自动选择第一个可用的模型"""
        try:
            models = self.get_model_list()
            if models and models.get('gpt') and models.get('sovits'):
                if not self.gpt_model_name:
                    self.gpt_model_name = models['gpt'][0]
                    print(f"自动选择 GPT 模型: {self.gpt_model_name}")
                if not self.sovits_model_name:
                    self.sovits_model_name = models['sovits'][0]
                    print(f"自动选择 SoVITS 模型: {self.sovits_model_name}")
        except Exception as e:
            print(f"自动选择模型失败: {e}")
    
    def get_model_list(self) -> Dict[str, Any]:
        """
        获取可用的模型列表
        
        Returns:
            包含 gpt 和 sovits 模型列表的字典
            
        Example:
            >>> tts = GPTSoVITSTTS()
            >>> models = tts.get_model_list()
            >>> print(models)
            {
                "msg": "获取模型列表成功",
                "gpt": ["【GSVI】jianhua_tao-e10"],
                "sovits": ["【GSVI】jianhua_tao_e10_s540_l32"]
            }
        """
        url = f"{self.api_url}/classic_model_list/v4"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"获取模型列表失败: {e}")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        text_lang: str = "中文",
        top_k: int = 10,
        top_p: float = 1.0,
        temperature: float = 1.0,
        speed: float = 1.0,
        text_split_method: str = "按中文句号。切",
        batch_size: int = 10,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        fragment_interval: float = 0.3,
        media_type: str = "wav",
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        seed: int = -1,
        sample_steps: int = 16,
        if_sr: bool = False,
        gpt_model_name: Optional[str] = None,
        sovits_model_name: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_text_lang: Optional[str] = None
    ) -> str:
        """
        将文本合成为语音
        
        Args:
            text: 要合成的文本（支持多行，\\n 会被识别为停顿）
            output_path: 输出音频文件路径（本地保存路径），如果为 None 则返回服务器 URL
            text_lang: 文本语言，"中文" 或 "英文"
            top_k: 采样策略 Top-K
            top_p: 采样策略 Top-P（nucleus sampling）
            temperature: 温度参数，控制随机性（0.1-2.0）
            speed: 语速倍率（0.5-2.0）
            text_split_method: 文本切分方法，"按标点符号切" 或 "不切"
            batch_size: 批处理大小
            batch_threshold: 批处理阈值
            split_bucket: 是否按桶切分
            fragment_interval: 片段间隔时间（秒）
            media_type: 输出格式，"wav" 或 "mp3"
            parallel_infer: 是否并行推理
            repetition_penalty: 重复惩罚系数
            seed: 随机种子，-1 表示随机
            sample_steps: 采样步数
            if_sr: 是否启用超分辨率
            gpt_model_name: GPT 模型名称（覆盖初始化参数）
            sovits_model_name: SoVITS 模型名称（覆盖初始化参数）
            ref_audio_path: 参考音频路径（覆盖初始化参数）
            ref_text: 参考文本（覆盖初始化参数）
            ref_text_lang: 参考文本语言（覆盖初始化参数）
        
        Returns:
            如果指定 output_path，返回本地文件路径；否则返回服务器音频 URL
            
        Example:
            >>> tts = GPTSoVITSTTS(
            ...     gpt_model_name="【GSVI】jianhua_tao-e10",
            ...     sovits_model_name="【GSVI】jianhua_tao_e10_s540_l32",
            ...     ref_audio_path="./custom_refs/jianhua_tao_1.wav",
            ...     ref_text="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享"
            ... )
            >>> audio_path = tts.synthesize(
            ...     text="清华大学的校训是自强不息，厚德载物。",
            ...     output_path="output.wav"
            ... )
            >>> print(audio_path)
        """
        # 使用传入参数或初始化参数
        gpt_model = gpt_model_name or self.gpt_model_name
        sovits_model = sovits_model_name or self.sovits_model_name
        ref_audio = ref_audio_path or self.ref_audio_path
        ref_prompt = ref_text or self.ref_text
        ref_lang = ref_text_lang or self.ref_text_lang
        
        # 验证必需参数
        if not gpt_model or not sovits_model:
            raise ValueError("必须指定 GPT 和 SoVITS 模型名称")
        if not ref_audio or not ref_prompt:
            raise ValueError("必须指定参考音频路径和参考文本")
        
        # 构建请求数据
        payload = {
            "dl_url": self.api_url,
            "version": "v4",
            "gpt_model_name": gpt_model,
            "sovits_model_name": sovits_model,
            "ref_audio_path": ref_audio,
            "prompt_text": ref_prompt,
            "prompt_text_lang": ref_lang,
            "text": text,
            "text_lang": text_lang,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "split_bucket": split_bucket,
            "speed_facter": speed,  # 注意：API 使用 "speed_facter" 而非 "speed_factor"
            "fragment_interval": fragment_interval,
            "media_type": media_type,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
            "sample_steps": sample_steps,
            "if_sr": if_sr
        }
        
        # 发送合成请求
        url = f"{self.api_url}/infer_classic"
        try:
            print(f"正在合成语音，文本长度: {len(text)} 字符...")
            start_time = time.time()
            
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            elapsed_time = time.time() - start_time
            print(f"合成完成，耗时: {elapsed_time:.2f} 秒")
            
            if result.get('msg') != '合成成功':
                raise RuntimeError(f"合成失败: {result.get('msg', '未知错误')}")
            
            audio_url = result.get('audio_url')
            if not audio_url:
                raise RuntimeError("服务器未返回音频 URL")
            
            # 如果指定了输出路径，下载音频文件
            if output_path:
                return self._download_audio(audio_url, output_path)
            else:
                return audio_url
                
        except requests.Timeout:
            raise RuntimeError(f"请求超时（{self.timeout} 秒），请尝试增加 timeout 参数或减少文本长度")
        except requests.RequestException as e:
            raise RuntimeError(f"合成请求失败: {e}")
    
    def _download_audio(self, audio_url: str, output_path: str) -> str:
        """
        从服务器下载音频文件
        
        Args:
            audio_url: 音频 URL
            output_path: 本地保存路径
        
        Returns:
            本地文件路径
        """
        try:
            # 创建输出目录
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 下载音频
            response = requests.get(audio_url, timeout=30)
            response.raise_for_status()
            
            # 保存到本地
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"音频已保存到: {output_path}")
            return output_path
            
        except requests.RequestException as e:
            raise RuntimeError(f"下载音频失败: {e}")
    
    def batch_synthesize(
        self,
        texts: list[str],
        output_dir: str = "outputs",
        prefix: str = "audio",
        **kwargs
    ) -> list[str]:
        """
        批量合成多段文本
        
        Args:
            texts: 文本列表
            output_dir: 输出目录
            prefix: 文件名前缀
            **kwargs: 传递给 synthesize 的其他参数
        
        Returns:
            生成的音频文件路径列表
            
        Example:
            >>> tts = GPTSoVITSTTS(...)
            >>> texts = [
            ...     "清华大学位于北京市海淀区。",
            ...     "清华有多个学院和研究所。"
            ... ]
            >>> audio_files = tts.batch_synthesize(texts, output_dir="outputs")
        """
        os.makedirs(output_dir, exist_ok=True)
        audio_files = []
        
        for i, text in enumerate(texts, 1):
            output_path = os.path.join(output_dir, f"{prefix}_{i:03d}.wav")
            try:
                audio_path = self.synthesize(text, output_path=output_path, **kwargs)
                audio_files.append(audio_path)
            except Exception as e:
                print(f"警告：第 {i} 段文本合成失败: {e}")
                audio_files.append(None)
        
        return audio_files


def main():
    """测试示例"""
    # 初始化 TTS 客户端
    tts = GPTSoVITSTTS(
        api_url="http://127.0.0.1:8000",
        gpt_model_name="【GSVI】jianhua_tao-e10",
        sovits_model_name="【GSVI】jianhua_tao_e10_s540_l32",
        ref_audio_path="./custom_refs/jianhua_tao.wav",
        ref_text="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享",
        ref_text_lang="中文"
    )
    
    # 获取模型列表
    print("=== 获取模型列表 ===")
    models = tts.get_model_list()
    print(f"可用的 GPT 模型: {models['gpt']}")
    print(f"可用的 SoVITS 模型: {models['sovits']}")
    
    # 合成单段文本
    print("\n=== 合成单段文本 ===")
    text = """清华大学的校训是"自强不息，厚德载物"，这句话源自《周易》里的两句："天行健，君子以自强不息"和"地势坤，君子以厚德载物"。

简单说，"自强不息"就是要像天体运行不停歇那样，永远努力向上、不服输，不断提升自己；"厚德载物"则是像大地能承载万物一样，要有宽厚的品德，能包容、担当，对人对事都有胸怀。

这个校训一直影响着清华的学生，鼓励大家既要拼搏进取，又要修德立身～"""
    
    audio_path = tts.synthesize(
        text=text,
        output_path="test_output.wav",
        temperature=1.0,
        top_p=1.0,
        top_k=10,
        speed=1.0
    )
    print(f"生成的音频: {audio_path}")
    
    # 批量合成
    print("\n=== 批量合成 ===")
    texts = [
        "清华大学位于北京市海淀区。",
        "清华有多个学院和研究所。",
        "清华培养了大批优秀人才。"
    ]
    audio_files = tts.batch_synthesize(texts, output_dir="batch_outputs")
    print(f"生成了 {len([f for f in audio_files if f])} 个音频文件")


if __name__ == "__main__":
    main()
