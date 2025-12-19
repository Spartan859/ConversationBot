"""
演示脚本：通用大模型 & 清华本科学习助手
运行方式：python -m src.dialogue.demo_agents
"""

import os
from pprint import pprint

from .general_agent import GeneralAgent
from .thu_agent import ThuAssistantAgent


def run_general_agent():
    api_key = os.getenv("ARK_API_KEY", "")
    agent = GeneralAgent(api_key=api_key,enable_thinking=True)

    print("=" * 80)
    print("GeneralAgent 文本示例")
    print("=" * 80)
    resp = agent.chat("介绍一下清华大学的校训。", post_process=True)
    print(resp)

    print("\n" + "=" * 80)
    print("GeneralAgent 多模态示例（如需图片）")
    print("=" * 80)
    resp_img = agent.chat(
        "你看见了什么？",
        image_url="https://ark-project.tos-cn-beijing.volces.com/doc_image/ark_demo_img_1.png",
        post_process=True,
    )
    print(resp_img)


def run_thu_agent():
    # 请在环境变量中配置 AK/SK/ACCOUNT_ID
    ak = os.getenv("THU_AGENT_AK", "")
    sk = os.getenv("THU_AGENT_SK", "")
    account_id = os.getenv("THU_AGENT_ACCOUNT_ID", "")

    agent = ThuAssistantAgent(
        ak=ak,
        sk=sk,
        account_id=account_id,
        model="Deepseek-v3-1",
        model_version="250821",
    )

    print("\n" + "=" * 80)
    print("ThuAssistantAgent 测试后处理功能（去除 Markdown 标记，适合 TTS）")
    print("=" * 80)
    response_tts = agent.query("新生因故不能按期入学应如何处理？", post_process=True)
    print(response_tts)

    print("\n" + "=" * 80)
    print("ThuAssistantAgent 测试自定义 Prompt")
    print("=" * 80)
    custom_prompt = """你是清华大学的智能助手，请用简洁明了的语言回答学生问题。
参考资料：
<context>
  {}
</context>"""
    agent.set_base_prompt(custom_prompt)
    response = agent.query("陶建华是谁？", post_process=True)
    print(response)


def main():
    # run_general_agent()
    run_thu_agent()


if __name__ == "__main__":
    main()
