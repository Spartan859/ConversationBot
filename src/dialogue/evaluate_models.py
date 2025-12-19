"""
清华助手 Agent 模型评估脚本

评估不同模型在清华本科学习助手场景下的性能表现
评估指标：响应时间、输出长度、文本质量
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from .thu_agent import ThuAssistantAgent


class ModelEvaluator:
    """模型评估器"""
    
    # 待评估的模型配置（基于 request_param_examples.txt）
    MODELS_TO_EVALUATE = [
        {
            "name": "Doubao-seed-1-6",
            "model": "Doubao-seed-1-6",
            "model_version": "251015"
        },
        {
            "name": "Doubao-seed-1-6-flash",
            "model": "Doubao-seed-1-6-flash",
            "model_version": "250828"
        },
        {
            "name": "Doubao-seed-1-6-thinking",
            "model": "Doubao-seed-1-6-thinking",
            "model_version": "250715"
        },
        {
            "name": "Deepseek-v3-1",
            "model": "Deepseek-v3-1",
            "model_version": "250821"
        }
    ]
    
    # 思考模式配置
    THINKING_MODES = [
        {"enabled": True, "label": "thinking_enabled"},
        {"enabled": False, "label": "thinking_disabled"}
    ]
    
    def __init__(
        self,
        ak: str,
        sk: str,
        account_id: str,
        questions_file: str = "thu_agent_evaluation_questions.json",
        output_dir: str = "evaluation_results"
    ):
        """
        初始化评估器
        
        Args:
            ak: 火山引擎 Access Key
            sk: 火山引擎 Secret Key
            account_id: 账户 ID
            questions_file: 评估题库文件路径
            output_dir: 结果输出目录
        """
        self.ak = ak
        self.sk = sk
        self.account_id = account_id
        self.questions_file = questions_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载评估题库
        self.questions = self._load_questions()
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """加载评估题库"""
        with open(self.questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['questions']
    
    def _calculate_text_quality_score(
        self, 
        output: str, 
        reference: str,
        question_difficulty: str
    ) -> Dict[str, Any]:
        """
        计算文本质量分数
        
        指标：
        1. 长度合理性：根据难度评估长度是否合理
        2. 关键词覆盖：参考答案中的关键实体是否出现
        3. 结构完整性：是否有清晰的建议步骤
        """
        scores = {}
        
        # 1. 长度合理性评分
        output_len = len(output)
        expected_lengths = {
            "basic": (50, 200),
            "intermediate": (100, 300),
            "advanced": (200, 500),
            "comprehensive": (300, 800)
        }
        min_len, max_len = expected_lengths.get(question_difficulty, (100, 400))
        
        if output_len < min_len:
            length_score = output_len / min_len
        elif output_len > max_len:
            length_score = max(0.5, 1 - (output_len - max_len) / max_len)
        else:
            length_score = 1.0
        
        scores['length_appropriateness'] = round(length_score, 3)
        
        # 2. 关键词覆盖率
        # 提取参考答案中的关键词（去除标点、数字）
        import re
        ref_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', reference))
        out_words = set(re.findall(r'[\u4e00-\u9fa5]{2,}', output))
        
        if ref_words:
            keyword_coverage = len(ref_words & out_words) / len(ref_words)
        else:
            keyword_coverage = 0.0
        
        scores['keyword_coverage'] = round(keyword_coverage, 3)
        
        # 3. 结构完整性（检查是否有建议、步骤等）
        structure_indicators = ['建议', '第一', '第二', '①', '②', '步骤', '首先', '其次']
        structure_score = sum(1 for ind in structure_indicators if ind in output) / len(structure_indicators)
        scores['structure_completeness'] = round(structure_score, 3)
        
        # 综合得分（加权平均）
        overall_score = (
            length_score * 0.3 +
            keyword_coverage * 0.5 +
            structure_score * 0.2
        )
        scores['overall_quality'] = round(overall_score, 3)
        
        return scores
    
    def _evaluate_single_question(
        self,
        model_config: Dict[str, str],
        question: Dict[str, Any],
        thinking_enabled: bool
    ) -> Dict[str, Any]:
        """
        评估单个问题
        
        Returns:
            包含评估指标的字典
        """
        question_text = question['question']
        reference_answer = question['reference_answer']
        difficulty = question['difficulty']
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 为每个线程创建独立的 Agent 实例（避免并发竞态）
            agent = ThuAssistantAgent(
                ak=self.ak,
                sk=self.sk,
                account_id=self.account_id,
                model=model_config['model'],
                model_version=model_config['model_version']
            )
            
            # 调用 Agent，传递 thinking 参数
            response = agent.query(
                user_query=question_text,
                post_process=False,  # 评估时保留原始输出
                max_tokens=32768,
                temperature=1.0,
                enable_thinking=thinking_enabled
            )
            
            # 记录结束时间
            end_time = time.time()
            response_time = end_time - start_time
            
            # 提取实际文本内容（处理 JSON 响应）
            try:
                response_data = json.loads(response)
                if "choices" in response_data and len(response_data["choices"]) > 0:
                    output_text = response_data["choices"][0].get("message", {}).get("content", "")
                else:
                    output_text = response
            except json.JSONDecodeError:
                output_text = response
            
            # 计算指标
            output_length = len(output_text)
            quality_scores = self._calculate_text_quality_score(
                output_text, 
                reference_answer,
                difficulty
            )
            
            result = {
                "question_id": question['id'],
                "question": question_text,
                "category": question['category'],
                "difficulty": difficulty,
                "response_time_seconds": round(response_time, 3),
                "output_length_chars": output_length,
                "quality_scores": quality_scores,
                "output_text": output_text[:500] + "..." if len(output_text) > 500 else output_text,  # 截断过长输出
                "reference_answer": reference_answer,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            result = {
                "question_id": question['id'],
                "question": question_text,
                "category": question['category'],
                "difficulty": difficulty,
                "response_time_seconds": 0,
                "output_length_chars": 0,
                "quality_scores": {
                    "length_appropriateness": 0,
                    "keyword_coverage": 0,
                    "structure_completeness": 0,
                    "overall_quality": 0
                },
                "output_text": "",
                "reference_answer": reference_answer,
                "success": False,
                "error": str(e)
            }
        
        return result
    
    def _calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计指标"""
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {
                "total_questions": len(results),
                "successful_questions": 0,
                "failed_questions": len(results),
                "success_rate": 0.0
            }
        
        response_times = [r['response_time_seconds'] for r in successful_results]
        output_lengths = [r['output_length_chars'] for r in successful_results]
        quality_scores = [r['quality_scores']['overall_quality'] for r in successful_results]
        
        stats = {
            "total_questions": len(results),
            "successful_questions": len(successful_results),
            "failed_questions": len(results) - len(successful_results),
            "success_rate": round(len(successful_results) / len(results), 3),
            
            "response_time": {
                "mean": round(sum(response_times) / len(response_times), 3),
                "min": round(min(response_times), 3),
                "max": round(max(response_times), 3),
                "total": round(sum(response_times), 3)
            },
            
            "output_length": {
                "mean": round(sum(output_lengths) / len(output_lengths), 1),
                "min": min(output_lengths),
                "max": max(output_lengths),
                "total": sum(output_lengths)
            },
            
            "quality_score": {
                "mean": round(sum(quality_scores) / len(quality_scores), 3),
                "min": round(min(quality_scores), 3),
                "max": round(max(quality_scores), 3)
            }
        }
        
        # 按难度统计
        by_difficulty = {}
        for difficulty in ['basic', 'intermediate', 'advanced', 'comprehensive']:
            difficulty_results = [r for r in successful_results if r['difficulty'] == difficulty]
            if difficulty_results:
                by_difficulty[difficulty] = {
                    "count": len(difficulty_results),
                    "avg_response_time": round(sum(r['response_time_seconds'] for r in difficulty_results) / len(difficulty_results), 3),
                    "avg_quality_score": round(sum(r['quality_scores']['overall_quality'] for r in difficulty_results) / len(difficulty_results), 3)
                }
        
        stats['by_difficulty'] = by_difficulty
        
        return stats
    
    def evaluate_model_config(
        self,
        model_config: Dict[str, str],
        thinking_mode: Dict[str, Any],
        sample_size: Optional[int] = None,
        max_workers: int = 8
    ) -> Dict[str, Any]:
        """
        评估单个模型配置（并行执行）
        
        Args:
            model_config: 模型配置
            thinking_mode: 思考模式配置
            sample_size: 采样大小（None 表示全部评估）
            max_workers: 最大并发线程数（默认 8）
        
        Returns:
            评估结果
        """
        model_name = model_config['name']
        thinking_label = thinking_mode['label']
        thinking_enabled = thinking_mode['enabled']
        
        print(f"\n{'='*80}")
        print(f"评估配置: {model_name} - {thinking_label}")
        print(f"{'='*80}")
        
        # 选择评估问题
        questions_to_eval = self.questions[:sample_size] if sample_size else self.questions
        
        print(f"  使用 {max_workers} 个线程并行评估 {len(questions_to_eval)} 个问题...")
        
        # 并行评估每个问题
        results = []
        completed_count = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_question = {
                executor.submit(
                    self._evaluate_single_question, 
                    model_config,
                    question, 
                    thinking_enabled
                ): (i, question) 
                for i, question in enumerate(questions_to_eval, 1)
            }
            
            # 按完成顺序处理结果
            for future in as_completed(future_to_question):
                i, question = future_to_question[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    results.append((i, result))  # 保存索引和结果
                    
                    # 显示进度和结果
                    print(f"  [{completed_count}/{len(questions_to_eval)}] 问题 {question['id']}: {question['question'][:30]}...")
                    if result['success']:
                        print(f"      ✓ 耗时: {result['response_time_seconds']:.2f}s | "
                              f"长度: {result['output_length_chars']} | "
                              f"质量: {result['quality_scores']['overall_quality']:.2f}")
                    else:
                        print(f"      ✗ 失败: {result['error']}")
                        
                except Exception as e:
                    print(f"  [{completed_count}/{len(questions_to_eval)}] 问题 {question['id']}: 执行异常 - {str(e)}")
                    # 创建失败结果
                    results.append((i, {
                        "question_id": question['id'],
                        "question": question['question'],
                        "category": question['category'],
                        "difficulty": question['difficulty'],
                        "response_time_seconds": 0,
                        "output_length_chars": 0,
                        "quality_scores": {
                            "length_appropriateness": 0,
                            "keyword_coverage": 0,
                            "structure_completeness": 0,
                            "overall_quality": 0
                        },
                        "output_text": "",
                        "reference_answer": question['reference_answer'],
                        "success": False,
                        "error": f"执行异常: {str(e)}"
                    }))
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        results = [r[1] for r in results]  # 只保留结果，去掉索引
        
        # 计算统计数据
        statistics = self._calculate_statistics(results)
        
        return {
            "model_name": model_name,
            "model": model_config['model'],
            "model_version": model_config['model_version'],
            "thinking_mode": thinking_label,
            "thinking_enabled": thinking_enabled,
            "evaluation_timestamp": datetime.now().isoformat(),
            "statistics": statistics,
            "detailed_results": results
        }
    
    def run_full_evaluation(
        self,
        sample_size: Optional[int] = None,
        output_filename: Optional[str] = None,
        max_workers: int = 8
    ):
        """
        运行完整评估（并行执行）
        
        Args:
            sample_size: 每个配置的采样大小（None 表示全部）
            output_filename: 输出文件名（None 则自动生成）
            max_workers: 最大并发线程数（默认 8）
        """
        print(f"\n{'#'*80}")
        print(f"# 清华助手 Agent 模型评估")
        print(f"# 评估模型数: {len(self.MODELS_TO_EVALUATE)}")
        print(f"# 思考模式数: {len(self.THINKING_MODES)}")
        print(f"# 总配置数: {len(self.MODELS_TO_EVALUATE) * len(self.THINKING_MODES)}")
        print(f"# 题库大小: {len(self.questions)}")
        print(f"# 采样大小: {sample_size if sample_size else '全部'}")
        print(f"{'#'*80}\n")
        
        all_results = []
        
        # 评估每个模型配置组合
        for model_config in self.MODELS_TO_EVALUATE:
            for thinking_mode in self.THINKING_MODES:
                result = self.evaluate_model_config(
                    model_config,
                    thinking_mode,
                    sample_size,
                    max_workers
                )
                all_results.append(result)
        
        # 生成汇总报告
        summary = self._generate_summary(all_results)
        
        # 保存结果
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"evaluation_results_{timestamp}.json"
        
        output_path = self.output_dir / output_filename
        
        final_output = {
            "metadata": {
                "evaluation_date": datetime.now().isoformat(),
                "total_models": len(self.MODELS_TO_EVALUATE),
                "total_configurations": len(all_results),
                "questions_file": self.questions_file,
                "sample_size": sample_size if sample_size else len(self.questions)
            },
            "summary": summary,
            "detailed_results": all_results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ 评估完成！")
        print(f"✓ 结果已保存至: {output_path}")
        print(f"{'='*80}\n")
        
        # 显示汇总
        self._print_summary(summary)
        
        return output_path
    
    def _generate_summary(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成汇总报告"""
        summary = {
            "configurations": []
        }
        
        for result in all_results:
            config_summary = {
                "model_name": result['model_name'],
                "thinking_mode": result['thinking_mode'],
                "success_rate": result['statistics']['success_rate'],
                "avg_response_time": result['statistics']['response_time']['mean'],
                "avg_output_length": result['statistics']['output_length']['mean'],
                "avg_quality_score": result['statistics']['quality_score']['mean']
            }
            summary['configurations'].append(config_summary)
        
        # 排序：按质量分数降序
        summary['configurations'].sort(key=lambda x: x['avg_quality_score'], reverse=True)
        
        # 最佳配置
        if summary['configurations']:
            summary['best_configuration'] = summary['configurations'][0]
        
        return summary
    
    def _print_summary(self, summary: Dict[str, Any]):
        """打印汇总信息"""
        print("\n" + "="*80)
        print("评估结果汇总")
        print("="*80)
        
        for config in summary['configurations']:
            print(f"\n模型: {config['model_name']} ({config['thinking_mode']})")
            print(f"  成功率: {config['success_rate']:.1%}")
            print(f"  平均响应时间: {config['avg_response_time']:.2f}s")
            print(f"  平均输出长度: {config['avg_output_length']:.0f} 字符")
            print(f"  平均质量分数: {config['avg_quality_score']:.3f}")
        
        if 'best_configuration' in summary:
            best = summary['best_configuration']
            print(f"\n{'='*80}")
            print(f"🏆 最佳配置: {best['model_name']} ({best['thinking_mode']})")
            print(f"   质量分数: {best['avg_quality_score']:.3f}")
            print(f"{'='*80}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='清华助手 Agent 模型评估')
    parser.add_argument('--sample', type=int, default=None, 
                       help='每个配置的采样数量（默认全部）')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件名（默认自动生成）')
    parser.add_argument('--workers', type=int, default=8,
                       help='并发线程数（默认 8）')
    
    args = parser.parse_args()
    
    # 获取环境变量
    ak = os.getenv("THU_AGENT_AK")
    sk = os.getenv("THU_AGENT_SK")
    account_id = os.getenv("THU_AGENT_ACCOUNT_ID")
    
    if not all([ak, sk, account_id]):
        print("错误：缺少环境变量")
        print("请设置: THU_AGENT_AK, THU_AGENT_SK, THU_AGENT_ACCOUNT_ID")
        return
    
    # 创建评估器
    evaluator = ModelEvaluator(
        ak=ak,
        sk=sk,
        account_id=account_id
    )
    
    # 运行评估
    evaluator.run_full_evaluation(
        sample_size=args.sample,
        output_filename=args.output,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
