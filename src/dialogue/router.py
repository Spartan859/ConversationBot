"""
智能路由器：根据用户问题自动选择合适的 Agent

工作流程：
1. 通用大模型判断是否需要专业知识
2. 解析 XML 标签决策结果
3. 如果 need_agent=true，调用 ThuAssistantAgent
4. 如果 need_agent=false，返回通用大模型的回答
"""

from typing import Optional

try:
    from .general_agent import GeneralAgent
    from .thu_agent import ThuAssistantAgent
    from .utils import extract_xml_tag, is_xml_tag_present, markdown_to_speech_text
except ImportError:
    from general_agent import GeneralAgent
    from thu_agent import ThuAssistantAgent
    from utils import extract_xml_tag, is_xml_tag_present, markdown_to_speech_text


class DialogueRouter:
    """智能路由器：根据用户问题自动选择合适的 Agent"""
    
    DECISION_PROMPT_TEMPLATE = """你是一个智能路由助手，需要判断用户问题是否需要调用「清华大学本科学习助手」专业知识库。

【用户问题】：{user_query}

【判断规则】：
- 如果问题涉及清华大学课程、学习资料、校园生活、学术资源、本科教育等内容，输出：
  <need_agent>true</need_agent>
  
- 如果问题是通用闲聊、常识问答、技术问题、非清华相关内容，输出：
  <need_agent>false</need_agent>
  <answer>你的回答内容（你需要保证输出的回复为口语，纯文本，需要适合用TTS模型转换为语音）</answer>

【注意】：
1. 只输出 XML 标签，不要有其他内容
2. need_agent=true 时，不要输出 answer 标签
3. need_agent=false 时，必须输出 answer 标签，且内容要口语化
4. 判断要准确，清华相关问题一定要路由到专业助手"""
    
    def __init__(
        self,
        general_agent: GeneralAgent,
        thu_agent: ThuAssistantAgent,
        verbose: bool = False
    ):
        """
        初始化路由器
        
        Args:
            general_agent: 通用大模型 Agent
            thu_agent: 清华本科学习助手 Agent
            verbose: 是否打印路由决策信息
        """
        self.general_agent = general_agent
        self.thu_agent = thu_agent
        self.verbose = verbose
    
    def route(
        self, 
        user_query: str,
        post_process: bool = False
    ) -> str:
        """
        智能路由：根据用户问题选择合适的 Agent
        
        工作流程：
        1. 通用大模型判断是否需要专业知识
        2. 解析 XML 标签决策结果
        3. 如果 need_agent=true，调用 ThuAssistantAgent
        4. 如果 need_agent=false，返回通用大模型的回答
        
        Args:
            user_query: 用户问题
            post_process: 是否后处理为纯文本（适合 TTS）
        
        Returns:
            最终回答（Markdown 或纯文本）
        """
        # 步骤 1：构建决策 Prompt
        decision_prompt = self._build_decision_prompt(user_query)
        
        # 步骤 2：通用大模型判断
        if self.verbose:
            print(f"[Router] 用户问题: {user_query}")
            print(f"[Router] 正在判断是否需要专业 Agent...")
        
        decision_response = self.general_agent.chat(
            decision_prompt,
            post_process=False  # 决策阶段不需要后处理
        )
        
        if self.verbose:
            print(f"[Router] 决策响应: {decision_response[:200]}...")
        
        # 步骤 3：解析决策结果
        need_agent = self._check_need_agent(decision_response)
        
        # 步骤 4：根据决策路由
        if need_agent:
            if self.verbose:
                print("[Router] ✓ 路由到 ThuAssistantAgent (专业知识库)")
            return self.thu_agent.query(user_query, post_process=post_process)
        else:
            if self.verbose:
                print("[Router] ✓ 使用 GeneralAgent 回答 (通用知识)")
            answer = extract_xml_tag(decision_response, 'answer')
            if answer:
                # 如果需要后处理，对提取的答案进行处理
                if post_process:
                    return markdown_to_speech_text(answer)
                return answer
            else:
                # 容错：如果没有 answer 标签，返回原始响应
                if self.verbose:
                    print("[Router] ⚠ 未找到 answer 标签，返回原始响应")
                return decision_response
    
    def _build_decision_prompt(self, user_query: str) -> str:
        """构建决策 Prompt"""
        return self.DECISION_PROMPT_TEMPLATE.format(user_query=user_query)
    
    def _check_need_agent(self, decision_response: str) -> bool:
        """
        检查是否需要调用 ThuAssistantAgent
        
        Args:
            decision_response: 决策响应（包含 XML 标签）
        
        Returns:
            True: 需要调用专业 Agent
            False: 使用通用大模型回答
        """
        # 使用 utils 中的 is_xml_tag_present 检查
        return is_xml_tag_present(decision_response, 'need_agent', 'true')


if __name__ == "__main__":
    # 简单测试
    import os
    
    print("正在初始化 Agent...")
    
    # 初始化 GeneralAgent
    general = GeneralAgent(
        api_key=os.getenv("ARK_API_KEY"),
        model="ep-20251219211834-fxjqq"
    )
    
    # 初始化 ThuAssistantAgent
    thu = ThuAssistantAgent(
        ak=os.getenv("THU_AGENT_AK"),
        sk=os.getenv("THU_AGENT_SK"),
        account_id=os.getenv("THU_AGENT_ACCOUNT_ID")
    )
    
    # 初始化 Router
    router = DialogueRouter(general, thu, verbose=True)
    
    # 测试通用问题
    print("\n" + "="*60)
    print("测试 1：通用问题（应路由到 GeneralAgent）")
    print("="*60)
    response1 = router.route("什么是人工智能？")
    print(f"\n回答: {response1}\n")
    
    # 测试清华相关问题
    print("\n" + "="*60)
    print("测试 2：清华相关问题（应路由到 ThuAssistantAgent）")
    print("="*60)
    response2 = router.route("清华大学有哪些特色课程？")
    print(f"\n回答: {response2}\n")
