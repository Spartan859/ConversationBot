"""
评估结果可视化脚本
读取评估结果 JSON，生成可视化报告
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def load_evaluation_results(json_file: str) -> Dict[str, Any]:
    """加载评估结果"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_comparison_table(results: Dict[str, Any]):
    """打印对比表格"""
    print("\n" + "="*120)
    print("模型性能对比表")
    print("="*120)
    
    # 表头
    header = f"{'模型名称':<35} | {'思考模式':<18} | {'成功率':<8} | {'平均响应时间':<12} | {'平均长度':<10} | {'平均质量':<10}"
    print(header)
    print("-"*120)
    
    # 数据行
    for config in results['summary']['configurations']:
        model = config['model_name']
        thinking = config['thinking_mode']
        success = f"{config['success_rate']:.1%}"
        time_s = f"{config['avg_response_time']:.2f}s"
        length = f"{config['avg_output_length']:.0f}"
        quality = f"{config['avg_quality_score']:.3f}"
        
        row = f"{model:<35} | {thinking:<18} | {success:<8} | {time_s:<12} | {length:<10} | {quality:<10}"
        print(row)
    
    print("="*120)


def print_best_configurations(results: Dict[str, Any]):
    """打印最佳配置"""
    configs = results['summary']['configurations']
    
    print("\n" + "="*80)
    print("🏆 最佳配置排行榜")
    print("="*80)
    
    # 按质量分数排序
    by_quality = sorted(configs, key=lambda x: x['avg_quality_score'], reverse=True)
    print("\n【质量最优】前3名:")
    for i, config in enumerate(by_quality[:3], 1):
        print(f"  {i}. {config['model_name']} ({config['thinking_mode']}) - 质量分数: {config['avg_quality_score']:.3f}")
    
    # 按响应时间排序
    by_speed = sorted(configs, key=lambda x: x['avg_response_time'])
    print("\n【速度最快】前3名:")
    for i, config in enumerate(by_speed[:3], 1):
        print(f"  {i}. {config['model_name']} ({config['thinking_mode']}) - 响应时间: {config['avg_response_time']:.2f}s")
    
    # 综合评分（质量*0.7 + 速度*0.3）
    for config in configs:
        # 归一化速度分数（越快越好）
        max_time = max(c['avg_response_time'] for c in configs)
        speed_score = 1 - (config['avg_response_time'] / max_time)
        config['综合分数'] = config['avg_quality_score'] * 0.7 + speed_score * 0.3
    
    by_overall = sorted(configs, key=lambda x: x['综合分数'], reverse=True)
    print("\n【综合最佳】前3名 (质量70% + 速度30%):")
    for i, config in enumerate(by_overall[:3], 1):
        print(f"  {i}. {config['model_name']} ({config['thinking_mode']}) - 综合分数: {config['综合分数']:.3f}")
    
    print("="*80)


def print_thinking_mode_comparison(results: Dict[str, Any]):
    """对比同一模型的不同 thinking 模式"""
    configs = results['summary']['configurations']
    
    print("\n" + "="*80)
    print("💭 Thinking 模式对比")
    print("="*80)
    
    # 按模型分组
    models = {}
    for config in configs:
        model_name = config['model_name']
        if model_name not in models:
            models[model_name] = {}
        models[model_name][config['thinking_mode']] = config
    
    for model_name, modes in models.items():
        print(f"\n模型: {model_name}")
        
        if 'thinking_enabled' in modes and 'thinking_disabled' in modes:
            enabled = modes['thinking_enabled']
            disabled = modes['thinking_disabled']
            
            # 质量差异
            quality_diff = enabled['avg_quality_score'] - disabled['avg_quality_score']
            quality_pct = (quality_diff / disabled['avg_quality_score']) * 100 if disabled['avg_quality_score'] > 0 else 0
            
            # 速度差异
            time_diff = enabled['avg_response_time'] - disabled['avg_response_time']
            time_pct = (time_diff / disabled['avg_response_time']) * 100 if disabled['avg_response_time'] > 0 else 0
            
            print(f"  质量分数: {disabled['avg_quality_score']:.3f} → {enabled['avg_quality_score']:.3f} "
                  f"({'+'if quality_diff >= 0 else ''}{quality_diff:.3f}, {quality_pct:+.1f}%)")
            print(f"  响应时间: {disabled['avg_response_time']:.2f}s → {enabled['avg_response_time']:.2f}s "
                  f"({'+'if time_diff >= 0 else ''}{time_diff:.2f}s, {time_pct:+.1f}%)")
            
            # 判断是否值得开启
            if quality_diff > 0.05 and time_pct < 50:
                print(f"  ✓ 推荐: 开启 thinking 模式（质量提升明显，速度损失可接受）")
            elif quality_diff < -0.02:
                print(f"  ✗ 不推荐: thinking 模式反而降低质量")
            elif time_pct > 100:
                print(f"  ⚠️  权衡: thinking 模式提升质量，但速度慢一倍以上")
            else:
                print(f"  ℹ️  中性: thinking 模式影响不明显")
    
    print("="*80)


def print_difficulty_analysis(results: Dict[str, Any]):
    """按难度分析性能"""
    print("\n" + "="*80)
    print("📊 不同难度问题的性能分析")
    print("="*80)
    
    for result in results['detailed_results']:
        model_name = result['model_name']
        thinking_mode = result['thinking_mode']
        
        print(f"\n{model_name} ({thinking_mode})")
        
        if 'by_difficulty' in result['statistics']:
            by_diff = result['statistics']['by_difficulty']
            
            for difficulty in ['basic', 'intermediate', 'advanced', 'comprehensive']:
                if difficulty in by_diff:
                    data = by_diff[difficulty]
                    print(f"  {difficulty:>15}: 平均时间 {data['avg_response_time']:.2f}s | "
                          f"平均质量 {data['avg_quality_score']:.3f}")
    
    print("="*80)


def generate_markdown_report(results: Dict[str, Any], output_file: str = "evaluation_report.md"):
    """生成 Markdown 格式的报告"""
    md_lines = []
    
    md_lines.append("# 清华助手 Agent 模型评估报告\n")
    md_lines.append(f"**评估时间**: {results['metadata']['evaluation_date']}\n")
    md_lines.append(f"**评估模型数**: {results['metadata']['total_models']}\n")
    md_lines.append(f"**评估配置数**: {results['metadata']['total_configurations']}\n")
    md_lines.append(f"**问题数量**: {results['metadata']['sample_size']}\n")
    
    # 对比表格
    md_lines.append("\n## 📊 模型性能对比\n")
    md_lines.append("| 模型名称 | 思考模式 | 成功率 | 平均响应时间 | 平均长度 | 平均质量 |\n")
    md_lines.append("|---------|---------|--------|------------|---------|--------|\n")
    
    for config in results['summary']['configurations']:
        md_lines.append(
            f"| {config['model_name']} | {config['thinking_mode']} | "
            f"{config['success_rate']:.1%} | {config['avg_response_time']:.2f}s | "
            f"{config['avg_output_length']:.0f} | {config['avg_quality_score']:.3f} |\n"
        )
    
    # 最佳配置
    md_lines.append("\n## 🏆 最佳配置\n")
    
    configs = results['summary']['configurations']
    by_quality = sorted(configs, key=lambda x: x['avg_quality_score'], reverse=True)
    
    md_lines.append("\n### 质量最优\n")
    for i, config in enumerate(by_quality[:3], 1):
        md_lines.append(f"{i}. **{config['model_name']}** ({config['thinking_mode']}) - 质量分数: {config['avg_quality_score']:.3f}\n")
    
    # 保存文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(md_lines)
    
    print(f"\n✓ Markdown 报告已生成: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化评估结果')
    parser.add_argument('json_file', type=str, 
                       help='评估结果 JSON 文件路径')
    parser.add_argument('--markdown', action='store_true',
                       help='生成 Markdown 报告')
    
    args = parser.parse_args()
    
    # 加载结果
    results = load_evaluation_results(args.json_file)
    
    # 打印各种分析
    print_comparison_table(results)
    print_best_configurations(results)
    print_thinking_mode_comparison(results)
    print_difficulty_analysis(results)
    
    # 生成 Markdown 报告
    if args.markdown:
        generate_markdown_report(results)


if __name__ == "__main__":
    main()
