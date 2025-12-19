"""
智能路由系统集成示例

本脚本演示了完整的智能路由系统工作流：
1. 使用 DialogueRouter 自动判断问题类型
2. 根据决策路由到 GeneralAgent 或 ThuAssistantAgent
3. 展示不同问题场景的路由决策过程
"""

import os
from src.dialogue import GeneralAgent, ThuAssistantAgent, DialogueRouter


def check_env_vars():
    """检查必需的环境变量"""
    required = {
        "ARK_API_KEY": "通用大模型 API Key",
        "THU_AGENT_AK": "清华助手 Access Key",
        "THU_AGENT_SK": "清华助手 Secret Key",
        "THU_AGENT_ACCOUNT_ID": "清华助手 Account ID"
    }
    
    missing = {k: v for k, v in required.items() if not os.getenv(k)}
    
    if missing:
        print("❌ 缺少环境变量：")
        for key, desc in missing.items():
            print(f"   {key}: {desc}")
        print("\n请设置环境变量后重试：")
        print("   $env:ARK_API_KEY=\"your-key\"")
        print("   $env:THU_AGENT_AK=\"your-ak\"")
        print("   $env:THU_AGENT_SK=\"your-sk\"")
        print("   $env:THU_AGENT_ACCOUNT_ID=\"your-id\"")
        return False
    return True


def main():
    """主函数：演示路由系统"""
    
    if not check_env_vars():
        return
    
    print("="*70)
    print(" 智能路由系统集成演示")
    print("="*70)
    print()
    
    # 初始化 Agent
    print("📦 初始化 Agent...")
    print()
    
    general = GeneralAgent(
        api_key=os.getenv("ARK_API_KEY"),
        model="ep-20251219211834-fxjqq"
    )
    print("   ✓ GeneralAgent 初始化完成")
    
    thu = ThuAssistantAgent(
        ak=os.getenv("THU_AGENT_AK"),
        sk=os.getenv("THU_AGENT_SK"),
        account_id=os.getenv("THU_AGENT_ACCOUNT_ID")
    )
    print("   ✓ ThuAssistantAgent 初始化完成")
    
    router = DialogueRouter(general, thu, verbose=True)
    print("   ✓ DialogueRouter 初始化完成")
    print()
    
    # 测试场景
    test_scenarios = [
        {
            "category": "🗣️ 通用闲聊",
            "query": "你好，今天天气怎么样？",
            "expected": "GeneralAgent"
        },
        {
            "category": "🧠 常识问答",
            "query": "什么是人工智能？",
            "expected": "GeneralAgent"
        },
        {
            "category": "💻 技术问题",
            "query": "Python 如何读取 CSV 文件？",
            "expected": "GeneralAgent"
        },
        {
            "category": "🏫 清华课程查询",
            "query": "清华大学有哪些人工智能相关的课程？",
            "expected": "ThuAssistantAgent"
        },
        {
            "category": "📚 学习资料推荐",
            "query": "推荐一些清华大学的学习资源",
            "expected": "ThuAssistantAgent"
        },
        {
            "category": "🎓 校园生活",
            "query": "新生入学需要准备哪些材料？",
            "expected": "ThuAssistantAgent"
        }
    ]
    
    print("="*70)
    print(" 测试场景")
    print("="*70)
    print()
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"[{i}] {scenario['category']}")
        print(f"    问题: {scenario['query']}")
        print(f"    预期路由: {scenario['expected']}")
        print()
        
        try:
            response = router.route(scenario['query'], post_process=False)
            print(f"    ✓ 回答: {response[:100]}...")
        except Exception as e:
            print(f"    ✗ 错误: {str(e)}")
        
        print()
    
    print("="*70)
    print(" 高级功能演示：TTS 后处理")
    print("="*70)
    print()
    
    query = "清华大学的校训是什么？"
    print(f"问题: {query}")
    print(f"启用 TTS 后处理（post_process=True）")
    print()
    
    try:
        response = router.route(query, post_process=True)
        print(f"回答:\n{response}")
    except Exception as e:
        print(f"错误: {str(e)}")
    
    print()
    print("="*70)
    print(" 完成！")
    print("="*70)


if __name__ == "__main__":
    main()
