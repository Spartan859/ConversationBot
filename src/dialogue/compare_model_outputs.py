#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型输出对比工具
用于提取和对比评估结果中不同模型对同一题目的回答
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def load_evaluation_results(file_path: str) -> Dict:
    """加载评估结果JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_question_outputs(results: Dict, question_id: int) -> List[Dict]:
    """
    提取指定题目的所有模型输出
    
    Args:
        results: 评估结果数据
        question_id: 题目ID
        
    Returns:
        包含所有模型输出的列表
    """
    outputs = []
    
    # 遍历所有配置的详细结果
    for config in results.get('detailed_results', []):
        model_name = config.get('model_name', 'Unknown')
        thinking_mode = config.get('thinking_mode', 'Unknown')
        thinking_enabled = config.get('thinking_enabled', False)
        
        # 注意：这里有嵌套的detailed_results
        questions = config.get('detailed_results', [])
        
        # 查找指定题目
        for question in questions:
            if question.get('question_id') == question_id:
                # 提取generated_answer
                output_text = question.get('output_text', '')
                generated_answer = ''
                reasoning_content = ''
                
                try:
                    # 尝试解析output_text中的JSON
                    if output_text.startswith('{'):
                        output_json = json.loads(output_text)
                        data = output_json.get('data', {})
                        generated_answer = data.get('generated_answer', '')
                        reasoning_content = data.get('reasoning_content', '')
                except json.JSONDecodeError:
                    generated_answer = output_text
                
                outputs.append({
                    'model_name': model_name,
                    'thinking_mode': thinking_mode,
                    'thinking_enabled': thinking_enabled,
                    'question': question.get('question', ''),
                    'category': question.get('category', ''),
                    'difficulty': question.get('difficulty', ''),
                    'generated_answer': generated_answer,
                    'reasoning_content': reasoning_content,
                    'response_time': question.get('response_time_seconds', 0),
                    'output_length': question.get('output_length_chars', 0),
                    'quality_scores': question.get('quality_scores', {}),
                    'reference_answer': question.get('reference_answer', '')
                })
                break
    
    return outputs


def format_output_comparison(outputs: List[Dict], show_reasoning: bool = False) -> str:
    """
    格式化输出对比结果
    
    Args:
        outputs: 模型输出列表
        show_reasoning: 是否显示推理过程
        
    Returns:
        格式化的对比文本
    """
    if not outputs:
        return "未找到指定题目的输出"
    
    # 题目信息（所有模型相同）
    first_output = outputs[0]
    result = []
    result.append("=" * 100)
    result.append(f"📝 题目 ID: {first_output.get('question', 'Unknown')}")
    result.append(f"📚 类别: {first_output.get('category', '')} ({first_output.get('difficulty', '')})")
    result.append(f"✅ 参考答案: {first_output.get('reference_answer', '')}")
    result.append("=" * 100)
    result.append("")
    
    # 按模型排序
    outputs_sorted = sorted(outputs, key=lambda x: (x['model_name'], not x['thinking_enabled']))
    
    # 输出每个模型的回答
    for idx, output in enumerate(outputs_sorted, 1):
        thinking_icon = "🧠" if output['thinking_enabled'] else "⚡"
        result.append(f"\n{'─' * 100}")
        result.append(f"{thinking_icon} 模型 {idx}: {output['model_name']} ({'思考模式' if output['thinking_enabled'] else '快速模式'})")
        result.append(f"{'─' * 100}")
        result.append(f"⏱️  响应时间: {output['response_time']:.2f}秒")
        result.append(f"📏 输出长度: {output['output_length']}字")
        
        # 质量评分
        quality = output['quality_scores']
        result.append(f"⭐ 质量评分: {quality.get('overall_quality', 0):.3f}")
        result.append(f"   ├─ 长度适当性: {quality.get('length_appropriateness', 0):.2f}")
        result.append(f"   ├─ 关键词覆盖: {quality.get('keyword_coverage', 0):.2f}")
        result.append(f"   └─ 结构完整性: {quality.get('structure_completeness', 0):.2f}")
        result.append("")
        result.append("💬 生成回答:")
        result.append("─" * 100)
        result.append(output['generated_answer'])
        result.append("")
        
        # 如果需要显示推理过程
        if show_reasoning and output['reasoning_content']:
            result.append("🤔 推理过程:")
            result.append("─" * 100)
            result.append(output['reasoning_content'][:500] + "..." if len(output['reasoning_content']) > 500 else output['reasoning_content'])
            result.append("")
    
    result.append("\n" + "=" * 100)
    return "\n".join(result)


def save_comparison_to_file(comparison_text: str, output_file: str):
    """保存对比结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(comparison_text)
    print(f"\n✅ 对比结果已保存到: {output_file}")


def list_all_questions(results: Dict):
    """列出所有题目"""
    if not results.get('detailed_results'):
        print("未找到评估结果")
        return
    
    # 从第一个配置中提取所有题目（注意嵌套结构）
    first_config = results['detailed_results'][0]
    questions = first_config.get('detailed_results', [])
    
    print("\n" + "=" * 100)
    print("📋 所有题目列表")
    print("=" * 100)
    
    current_category = None
    for q in questions:
        category = q.get('category', '')
        if category != current_category:
            current_category = category
            print(f"\n【{category}】")
        
        print(f"  ID {q['question_id']:2d}: {q['question']}")
        ref_answer = q['reference_answer'][:50] if q.get('reference_answer') else ''
        print(f"        难度: {q['difficulty']}, 参考答案: {ref_answer}...")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='模型输出对比工具 - 提取和对比评估结果中不同模型的回答',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 查看所有题目
  python compare_model_outputs.py evaluation_results.json --list
  
  # 对比第1题的所有模型输出
  python compare_model_outputs.py evaluation_results.json -q 1
  
  # 对比第21题并显示推理过程
  python compare_model_outputs.py evaluation_results.json -q 21 --reasoning
  
  # 对比第31题并保存到文件
  python compare_model_outputs.py evaluation_results.json -q 31 -o comparison_q31.txt
        """
    )
    
    parser.add_argument('results_file', 
                        help='评估结果JSON文件路径')
    parser.add_argument('-q', '--question-id', 
                        type=int,
                        help='要对比的题目ID')
    parser.add_argument('-o', '--output', 
                        help='保存对比结果的输出文件路径')
    parser.add_argument('-r', '--reasoning', 
                        action='store_true',
                        help='显示模型的推理过程')
    parser.add_argument('--list', 
                        action='store_true',
                        help='列出所有题目')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.results_file).exists():
        print(f"❌ 错误: 文件不存在 - {args.results_file}")
        return
    
    # 加载评估结果
    print(f"📂 正在加载评估结果: {args.results_file}")
    results = load_evaluation_results(args.results_file)
    print(f"✅ 已加载 {len(results.get('detailed_results', []))} 个模型配置的结果")
    
    # 如果只是列出题目
    if args.list:
        list_all_questions(results)
        return
    
    # 检查是否指定了题目ID
    if args.question_id is None:
        print("\n❌ 错误: 请使用 -q 参数指定题目ID，或使用 --list 查看所有题目")
        print("示例: python compare_model_outputs.py evaluation_results.json -q 1")
        return
    
    # 提取指定题目的输出
    print(f"\n🔍 正在提取题目 {args.question_id} 的所有模型输出...")
    outputs = extract_question_outputs(results, args.question_id)
    
    if not outputs:
        print(f"❌ 未找到题目ID {args.question_id} 的输出")
        return
    
    print(f"✅ 找到 {len(outputs)} 个模型配置的输出")
    
    # 格式化对比结果
    comparison = format_output_comparison(outputs, show_reasoning=args.reasoning)
    
    # 输出到控制台
    print("\n" + comparison)
    
    # 如果指定了输出文件，保存结果
    if args.output:
        save_comparison_to_file(comparison, args.output)


if __name__ == '__main__':
    main()
