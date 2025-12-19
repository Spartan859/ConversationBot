"""
通用对话 Agent
基于火山引擎 Ark(OpenAI SDK) 的 responses 接口封装，支持文本 / 图片输入
"""

from typing import List, Dict, Optional, Any
from openai import OpenAI

from .utils import markdown_to_speech_text


class GeneralAgent:
    """通用大模型 Agent（不依赖知识库）"""

    DEFAULT_SYSTEM_PROMPT = (
        "你是一个友善的中文助手，请用简洁、口语化的方式回答用户问题。"
    )

    def __init__(
        self,
        api_key: str,
        model: str = "ep-20251219211834-fxjqq",
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        enable_thinking: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_thinking = enable_thinking

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _build_messages(
        self,
        user_query: str,
        image_url: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        构建 messages 列表，兼容文本 / 图片输入
        history: 形如 [ {"role": "assistant", "content": [{"type": "output_text", "text": "..."}]}, ... ]
        """
        messages: List[Dict[str, Any]] = []

        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": sys_prompt}
                    ],
                }
            )

        if history:
            messages.extend(history)

        content: List[Dict[str, Any]] = []
        if image_url:
            content.append({"type": "input_image", "image_url": image_url})
        content.append({"type": "input_text", "text": user_query})

        messages.append({"role": "user", "content": content})
        return messages

    def _extract_text(self, response: Any) -> str:
        """尽可能从 responses 对象中提取文本"""
        if hasattr(response, "output_text"):
            return response.output_text
        try:
            data = response.model_dump()
            if "output_text" in data:
                return data["output_text"]
            # Ark responses 结构下：output -> choices -> message -> content
            output = data.get("output") or {}
            choices = output.get("choices") or []
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content") or []
                if content and isinstance(content, list):
                    # 取第一个 text 片段
                    for part in content:
                        if part.get("type") in ("output_text", "text"):
                            return part.get("text", "")
            return str(response)
        except Exception:
            return str(response)

    def chat(
        self,
        user_query: str,
        image_url: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        post_process: bool = False,
    ) -> str:
        """
        调用通用大模型

        Args:
            user_query: 用户问题（文本）
            image_url: 可选图片 URL（多模态）
            history: 可选对话历史（responses 兼容格式）
            system_prompt: 可覆盖默认的 system prompt
            post_process: 是否去除 Markdown 标记，返回纯文本（适合 TTS）
        """
        messages = self._build_messages(
            user_query=user_query,
            image_url=image_url,
            history=history,
            system_prompt=system_prompt,
        )

        params = {
            "model": self.model,
            "input": messages,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.enable_thinking:
            params["extra_body"] = {"thinking": {"type": "enabled"}}

        response = self.client.responses.create(**params)
        text = self._extract_text(response)

        if post_process:
            return markdown_to_speech_text(text)
        return text

    def set_model(self, model: str):
        self.model = model

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt


    # 无 __main__，请使用 demo_agents.py 运行示例
