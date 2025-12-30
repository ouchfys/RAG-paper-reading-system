from typing import List, Dict, Optional
from langchain_core.documents import Document

class PromptTemplateManager:
    """
    Prompt模板管理器
    
    提供多种Prompt策略:
    1. Default: 标准RAG prompt
    2. Chain-of-Thought (CoT): 引导逐步推理
    3. ReAct: Reasoning + Acting框架
    4. Few-shot: 通过示例引导输出
    """
    
    @staticmethod
    def build_rag_prompt(
        query: str,
        contexts: List[Document],
        prompt_type: str = "default",
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        构建RAG prompt
        
        Args:
            query: 用户查询
            contexts: 检索到的文档列表
            prompt_type: prompt类型 (default/cot/react/few_shot)
            examples: Few-shot示例 (仅prompt_type="few_shot"时需要)
            
        Returns:
            构建好的prompt字符串
        """
        if prompt_type == "default":
            return PromptTemplateManager._default_prompt(query, contexts)
        elif prompt_type == "cot":
            return PromptTemplateManager._chain_of_thought_prompt(query, contexts)
        elif prompt_type == "react":
            return PromptTemplateManager._react_prompt(query, contexts)
        elif prompt_type == "few_shot":
            return PromptTemplateManager._few_shot_prompt(query, contexts, examples)
        else:
            print(f"[PromptTemplate] 未知prompt类型 '{prompt_type}'，使用default")
            return PromptTemplateManager._default_prompt(query, contexts)
    
    @staticmethod
    def _default_prompt(query: str, contexts: List[Document]) -> str:
        """
        标准RAG prompt
        
        特点: 清晰的指令 + 结构化的上下文
        """
        context_str = "\n\n".join([
            f"【文档片段{i+1}】\n"
            f"来源: {doc.metadata.get('source', '未知')} - "
            f"{doc.metadata.get('chapter', '未知章节')} (第{doc.metadata.get('page', '?')}页)\n"
            f"内容: {doc.page_content}"
            for i, doc in enumerate(contexts)
        ])
        
        prompt = f"""你是一个专业的工业文档问答助手。请基于提供的文档片段回答用户问题。

# 回答要求
1. 必须基于文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说明"文档中未提及"
3. 引用时注明来源(文档片段编号)
4. 回答要专业、准确、简洁

# 已知文档
{context_str}

# 用户问题
{query}

# 你的回答
"""
        return prompt
    
    @staticmethod
    def _chain_of_thought_prompt(query: str, contexts: List[Document]) -> str:
        """
        Chain-of-Thought Prompt
        
        特点: 引导LLM逐步推理，提高复杂问题的回答质量
        """
        context_str = "\n\n".join([
            f"文档{i+1}: {doc.page_content}"
            for i, doc in enumerate(contexts)
        ])
        
        prompt = f"""你是一个专业的技术文档分析专家。请按照以下步骤回答问题:

# 文档内容
{context_str}

# 问题
{query}

# 请按以下步骤思考(Chain-of-Thought):

步骤1 - 理解问题: 
[分析问题的核心要点是什么]

步骤2 - 定位信息:
[在哪些文档片段中找到了相关信息?]

步骤3 - 提取关键点:
[从文档中提取的关键信息有哪些?]

步骤4 - 综合回答:
[基于以上分析，给出最终答案]

请开始你的分析:
"""
        return prompt
    
    @staticmethod
    def _react_prompt(query: str, contexts: List[Document]) -> str:
        """
        ReAct Prompt (Reasoning + Acting)
        
        特点: 适合需要多步推理的复杂查询
        """
        context_str = "\n\n".join([
            f"文档{i+1}: {doc.page_content}"
            for i, doc in enumerate(contexts)
        ])
        
        prompt = f"""你是一个具有推理能力的AI助手。使用ReAct框架回答问题。

# 可用文档
{context_str}

# 问题
{query}

# ReAct框架 (Thought -> Action -> Observation -> Answer)

Thought 1: [我需要理解问题的核心]
Action 1: [分析问题类型和所需信息]
Observation 1: [问题是关于...的，需要...信息]

Thought 2: [我需要在文档中查找相关信息]
Action 2: [检查文档1-{len(contexts)}]
Observation 2: [在文档X中找到了...信息]

Thought 3: [我需要综合信息得出答案]
Action 3: [整合文档中的信息]
Observation 3: [综合来看...]

Answer: [最终答案]

请开始:
"""
        return prompt
    
    @staticmethod
    def _few_shot_prompt(
        query: str,
        contexts: List[Document],
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Few-shot Prompting
        
        特点: 通过示例引导LLM输出格式和风格
        """
        if examples is None:
            # 默认示例
            examples = [
                {
                    "query": "什么是RAG?",
                    "context": "RAG(Retrieval-Augmented Generation)是一种结合检索和生成的技术...",
                    "answer": "RAG是检索增强生成技术，通过检索相关文档来辅助大语言模型生成更准确的答案。\n\n【来源】文档1"
                },
                {
                    "query": "如何优化检索性能?",
                    "context": "可以通过建立索引、使用缓存、优化查询语句等方式...",
                    "answer": "优化检索性能的主要方法包括:\n1. 建立合适的索引\n2. 使用缓存机制\n3. 优化查询语句\n\n【来源】文档2"
                }
            ]
        
        # 构建示例部分
        examples_str = "\n\n".join([
            f"示例{i+1}:\n问题: {ex['query']}\n文档: {ex['context']}\n回答: {ex['answer']}"
            for i, ex in enumerate(examples)
        ])
        
        # 构建上下文
        context_str = "\n\n".join([
            f"文档{i+1}: {doc.page_content}"
            for i, doc in enumerate(contexts)
        ])
        
        prompt = f"""你是一个专业的文档问答助手。请参考以下示例的回答风格和格式。

# 示例
{examples_str}

# 现在请回答新问题

文档:
{context_str}

问题: {query}

回答:
"""
        return prompt


class SelfConsistencyVerifier:
    """
    自洽性验证器
    
    核心思想: 生成多个答案，选择最一致的
    原理: 多数投票降低随机性，提高答案可靠性
    """
    
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
    
    def generate_with_consistency(
        self,
        prompt: str,
        num_samples: int = 3,
        temperature: float = 0.7
    ) -> str:
        """
        生成多个答案并选择最一致的
        
        Args:
            prompt: 输入prompt
            num_samples: 生成答案数量
            temperature: 采样温度
            
        Returns:
            最一致的答案
        """
        print(f"[SelfConsistency] 生成 {num_samples} 个答案进行一致性验证...")
        
        # Step 1: 生成多个答案
        answers = []
        for i in range(num_samples):
            print(f"[SelfConsistency] 生成答案 {i+1}/{num_samples}...")
            try:
                response, _ = self.llm.chat(
                    self.tokenizer,
                    prompt,
                    history=[],
                    do_sample=True,
                    temperature=temperature,
                    repetition_penalty=1.2
                )
                answers.append(response)
            except Exception as e:
                print(f"[SelfConsistency] 生成答案失败: {e}")
                continue
        
        if not answers:
            print("[SelfConsistency] 未能生成任何答案")
            return "抱歉，生成答案失败。"
        
        if len(answers) == 1:
            return answers[0]
        
        # Step 2: 选择最一致的答案
        print("[SelfConsistency] 选择最一致的答案...")
        best_answer = self._select_most_consistent(answers)
        
        return best_answer
    
    def _select_most_consistent(self, answers: List[str]) -> str:
        """
        选择最一致的答案
        
        策略: 计算每个答案与其他答案的相似度，选择平均相似度最高的
        """
        from difflib import SequenceMatcher
        
        scores = []
        for i, ans1 in enumerate(answers):
            similarity_sum = 0
            for j, ans2 in enumerate(answers):
                if i != j:
                    similarity = SequenceMatcher(None, ans1, ans2).ratio()
                    similarity_sum += similarity
            
            avg_similarity = similarity_sum / (len(answers) - 1) if len(answers) > 1 else 0
            scores.append((ans1, avg_similarity))
            print(f"[SelfConsistency] 答案{i+1}平均相似度: {avg_similarity:.3f}")
        
        # 返回平均相似度最高的
        best_answer = max(scores, key=lambda x: x[1])[0]
        print(f"[SelfConsistency] 选择了平均相似度最高的答案")
        
        return best_answer
