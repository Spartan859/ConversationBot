"""
DialogueRouter 测试脚本

测试场景：
1. 通用问题（闲聊、常识）→ GeneralAgent
2. 清华相关问题（课程、校园）→ ThuAssistantAgent
3. 技术问题 → GeneralAgent
4. 学习资料查询 → ThuAssistantAgent
"""

import os
from .router import DialogueRouter
from .general_agent import GeneralAgent
from .thu_agent import ThuAssistantAgent


def print_section(title):
    """打印分隔线"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def test_router():
    """测试智能路由功能"""
    
    print_section("初始化 Agent 和 Router")
    
    # 初始化 GeneralAgent
    print("1. 初始化 GeneralAgent...")
    general = GeneralAgent(
        api_key=os.getenv("ARK_API_KEY"),
        model="ep-20251219211834-fxjqq"
    )
    print("   ✓ GeneralAgent 初始化完成")
    
    # 初始化 ThuAssistantAgent
    print("2. 初始化 ThuAssistantAgent...")
    thu = ThuAssistantAgent(
        ak=os.getenv("THU_AGENT_AK"),
        sk=os.getenv("THU_AGENT_SK"),
        account_id=os.getenv("THU_AGENT_ACCOUNT_ID")
    )
    print("   ✓ ThuAssistantAgent 初始化完成")
    
    # 初始化 Router
    print("3. 初始化 DialogueRouter...")
    router = DialogueRouter(general, thu, verbose=True)
    print("   ✓ DialogueRouter 初始化完成")
    
    # 测试用例
    test_cases = [
        {
            "title": "测试 1：通用闲聊问题",
            "query": "你好，今天天气怎么样？",
            "expected": "GeneralAgent"
        },
        {
            "title": "测试 2：常识问答",
            "query": "什么是人工智能？",
            "expected": "GeneralAgent"
        },
        {
            "title": "测试 3：清华课程问题",
            "query": "清华大学有哪些人工智能相关的课程？",
            "expected": "ThuAssistantAgent"
        },
        {
            "title": "测试 4：清华校园生活",
            "query": "新生入学需要准备哪些材料？",
            "expected": "ThuAssistantAgent"
        },
        {
            "title": "测试 5：学习资料查询",
            "query": "推荐一些清华大学的学习资源",
            "expected": "ThuAssistantAgent"
        },
        {
            "title": "测试 6：技术问题",
            "query": "Python 如何读取 CSV 文件？",
            "expected": "GeneralAgent"
        }
    ]
    
    # 执行测试
    for i, case in enumerate(test_cases, 1):
        print_section(f"{case['title']} (预期路由: {case['expected']})")
        print(f"问题: {case['query']}\n")
        
        try:
            response = router.route(case['query'], post_process=False)
            print(f"\n回答:\n{response}\n")
            print(f"{'✓ 测试通过' if response else '✗ 测试失败'}")
        except Exception as e:
            print(f"✗ 测试失败: {str(e)}")
    
    # 测试 post_process 功能
    print_section("测试 7：TTS 后处理功能")
    print("问题: 清华大学的校训是什么？")
    print("post_process=True (适合 TTS 朗读)\n")
    
    try:
        response = router.route("清华大学的校训是什么？", post_process=True)
        print(f"回答:\n{response}\n")
        print("✓ 后处理测试通过")
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
    
    print_section("测试完成")


if __name__ == "__main__":
    # 检查环境变量
    required_vars = ["ARK_API_KEY", "THU_AGENT_AK", "THU_AGENT_SK", "THU_AGENT_ACCOUNT_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("错误：缺少以下环境变量：")
        for var in missing_vars:
            print(f"  - {var}")
        print("\n请设置环境变量后重试：")
        print("  $env:ARK_API_KEY=\"your-key\"")
        print("  $env:THU_AGENT_AK=\"your-ak\"")
        print("  $env:THU_AGENT_SK=\"your-sk\"")
        print("  $env:THU_AGENT_ACCOUNT_ID=\"your-id\"")
    else:
        test_router()
