"""
快速评估脚本 - 仅测试每个配置的前5个问题
用于快速验证评估系统是否正常工作
"""

import os
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from .evaluate_models import ModelEvaluator


def main():
    """快速评估 - 每个配置只测试5个问题"""
    
    # 获取环境变量
    ak = os.getenv("THU_AGENT_AK")
    sk = os.getenv("THU_AGENT_SK")
    account_id = os.getenv("THU_AGENT_ACCOUNT_ID")
    
    if not all([ak, sk, account_id]):
        print("错误：缺少环境变量")
        print("请设置: THU_AGENT_AK, THU_AGENT_SK, THU_AGENT_ACCOUNT_ID")
        return
    
    print("\n" + "="*80)
    print("快速评估模式 - 每个配置测试 5 个问题")
    print("="*80 + "\n")
    
    # 创建评估器
    evaluator = ModelEvaluator(
        ak=ak,
        sk=sk,
        account_id=account_id,
        questions_file="thu_agent_evaluation_questions.json"
    )
    
    # 运行评估（只测试5个问题）
    output_path = evaluator.run_full_evaluation(
        sample_size=5,
        output_filename="quick_evaluation_results.json"
    )
    
    print(f"\n✓ 快速评估完成！")
    print(f"✓ 完整评估请运行: python src/dialogue/evaluate_models.py")


if __name__ == "__main__":
    main()
