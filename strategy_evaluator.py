import pandas as pd
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from rag_pipeline import AdvancedRAGPipeline
from prompt_templates import PromptTemplateManager

class StrategyComparator:
    """
    检索策略对比评估器
    
    功能: 对比不同检索策略的效果
    - simple: 简单向量检索
    - graph: 图增强检索
    - hybrid: 混合检索
    - hyde: HyDE检索
    - multi_query: 多查询融合
    - full: 完整流水线
    """
    
    def __init__(
        self,
        pipeline: AdvancedRAGPipeline,
        llm,
        tokenizer,
        judge_llm=None,
        embedding_model=None
    ):
        """
        初始化策略对比器
        
        Args:
            pipeline: AdvancedRAGPipeline实例
            llm: 本地LLM (ChatGLM3)
            tokenizer: 对应的tokenizer
            judge_llm: 评估用的LLM (可选，如DeepSeek)
            embedding_model: Embedding模型 (用于Ragas评估)
        """
        self.pipeline = pipeline
        self.llm = llm
        self.tokenizer = tokenizer
        self.judge_llm = judge_llm
        self.embedding_model = embedding_model
    
    def compare_strategies(
        self,
        test_questions: List[Dict[str, str]],
        strategies: List[str] = None,
        save_path: str = "strategy_comparison.csv"
    ) -> pd.DataFrame:
        """
        对比不同检索策略
        
        Args:
            test_questions: 测试问题列表，每个元素包含 {'question': ..., 'ground_truth': ...}
            strategies: 要对比的策略列表
            save_path: 结果保存路径
            
        Returns:
            对比结果DataFrame
        """
        if strategies is None:
            strategies = ["simple", "graph", "hybrid", "hyde", "multi_query", "full"]
        
        print(f"\n{'='*70}")
        print(f"开始策略对比评估")
        print(f"测试问题数: {len(test_questions)}")
        print(f"对比策略: {strategies}")
        print(f"{'='*70}\n")
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*60}")
            print(f"评估策略: {strategy}")
            print(f"{'='*60}")
            
            # 运行评估
            strategy_result = self._evaluate_strategy(
                test_questions,
                strategy
            )
            results[strategy] = strategy_result
        
        # 生成对比报告
        comparison_data = []
        for strategy in strategies:
            if strategy in results:
                comparison_data.append({
                    'Strategy': strategy,
                    'Avg_Response_Length': results[strategy].get('avg_length', 0),
                    'Avg_Retrieval_Docs': results[strategy].get('avg_docs', 0),
                    'Success_Rate': results[strategy].get('success_rate', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # 保存结果
        comparison_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"\n{'='*70}")
        print("策略对比报告")
        print(f"{'='*70}")
        print(comparison_df.to_string(index=False))
        print(f"\n结果已保存至: {save_path}")
        
        return comparison_df
    
    def _evaluate_strategy(
        self,
        test_questions: List[Dict[str, str]],
        strategy: str
    ) -> Dict:
        """评估单个策略"""
        total_length = 0
        total_docs = 0
        success_count = 0
        
        for i, item in enumerate(test_questions):
            question = item['question']
            print(f"\n[{i+1}/{len(test_questions)}] 问题: {question[:50]}...")
            
            try:
                # 检索
                docs = self.pipeline.retrieve(question, strategy=strategy, top_k=5)
                total_docs += len(docs)
                
                # 生成答案
                context_str = "\n\n".join([doc.page_content for doc in docs])
                prompt = PromptTemplateManager.build_rag_prompt(
                    question,
                    docs,
                    prompt_type="default"
                )
                
                response, _ = self.llm.chat(
                    self.tokenizer,
                    prompt,
                    history=[],
                    do_sample=False,
                    repetition_penalty=1.2
                )
                
                total_length += len(response)
                success_count += 1
                
                print(f"  ✓ 检索到 {len(docs)} 个文档，生成答案长度: {len(response)}")
                
            except Exception as e:
                print(f"  ✗ 评估失败: {e}")
        
        return {
            'avg_length': total_length / len(test_questions) if test_questions else 0,
            'avg_docs': total_docs / len(test_questions) if test_questions else 0,
            'success_rate': success_count / len(test_questions) if test_questions else 0
        }
    
    def evaluate_with_ragas(
        self,
        test_data: List[Dict[str, str]],
        strategy: str = "full"
    ):
        """
        使用Ragas框架评估
        
        需要judge_llm和embedding_model
        """
        if self.judge_llm is None or self.embedding_model is None:
            print("[StrategyComparator] Ragas评估需要judge_llm和embedding_model")
            return None
        
        print(f"\n开始Ragas评估 (策略: {strategy})...")
        
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for item in test_data:
            q = item["question"]
            gt = item["ground_truth"]
            
            # 检索
            docs = self.pipeline.retrieve(q, strategy=strategy, top_k=3)
            retrieved_text = [d.page_content for d in docs]
            
            # 生成答案
            context_str = "\n".join(retrieved_text)
            prompt = f"基于已知信息：\n{context_str}\n\n问题：{q}"
            ans, _ = self.llm.chat(self.tokenizer, prompt, history=[], do_sample=False)
            
            questions.append(q)
            answers.append(ans)
            contexts.append(retrieved_text)
            ground_truths.append(gt)
        
        # 构建数据集
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data_dict)
        
        # 运行评估
        results = evaluate(
            dataset=dataset,
            metrics=[
                context_recall,
                faithfulness,
                answer_relevancy,
                context_precision
            ],
            llm=self.judge_llm,
            embeddings=self.embedding_model
        )
        
        return results
