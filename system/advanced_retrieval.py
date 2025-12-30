from typing import List, Optional
from langchain_core.documents import Document
import re

class HyDERetriever:
    """
    HyDE (Hypothetical Document Embeddings) 检索器
    
    核心思想: 用LLM生成假想答案，用答案去检索而不是用问题
    原理: 答案和文档在embedding空间更接近，检索效果更好
    """
    
    def __init__(self, llm, tokenizer, vector_store):
        """
        初始化HyDE检索器
        
        Args:
            llm: ChatGLM3模型实例
            tokenizer: 对应的tokenizer
            vector_store: Neo4jVector实例
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.vector_store = vector_store
    
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        HyDE检索流程
        
        1. 用LLM生成假想答案
        2. 用假想答案的embedding去检索
        3. 返回真实文档
        """
        print(f"[HyDE] 开始HyDE检索: query='{query[:30]}...'")
        
        # Step 1: 生成假想答案
        hypothetical_answer = self._generate_hypothetical_answer(query)
        print(f"[HyDE] 生成假想答案: '{hypothetical_answer[:100]}...'")
        
        # Step 2: 用假想答案检索
        docs = self.vector_store.similarity_search(hypothetical_answer, k=k)
        print(f"[HyDE] 检索到 {len(docs)} 个文档")
        
        return docs
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """
        生成假想答案
        
        Prompt Engineering要点: 引导LLM生成包含专业术语的详细答案
        """
        prompt = f"""你是一个工业文档专家。请根据问题生成一个详细的、技术性的答案。

注意: 这个答案是用来检索相关文档的，所以要包含可能出现在文档中的专业术语和表述方式。

问题: {query}

请直接给出答案，不要说"我不知道"或"需要更多信息"。即使不确定，也请基于相关领域知识给出合理的推测:
"""
        
        try:
            # 调用LLM
            response, _ = self.llm.chat(
                self.tokenizer,
                prompt,
                history=[],
                do_sample=True,      # 开启采样，增加多样性
                temperature=0.7,     # 适中的温度
                max_length=512
            )
            return response
        except Exception as e:
            print(f"[HyDE] 生成假想答案失败: {e}")
            # 降级: 返回原问题
            return query


class QueryRewriter:
    """
    查询重写器
    
    功能:
    1. Multi-Query: 生成多个角度的查询
    2. Step-back: 生成更抽象的查询
    """
    
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
    
    def rewrite_multi_query(
        self, 
        query: str,
        num_queries: int = 3
    ) -> List[str]:
        """
        生成多个角度的查询
        
        使用Few-shot Prompting引导输出格式
        """
        print(f"[QueryRewriter] 开始多查询重写: query='{query[:30]}...'")
        
        prompt = f"""你是一个查询优化专家。给定一个用户问题，生成{num_queries}个不同角度的改写查询，用于检索相关文档。

# 示例 (Few-shot Learning)

原问题: 如何提高RAG系统的准确率?
改写1: RAG系统准确率优化方法
改写2: 检索增强生成的性能提升技术
改写3: 提高文档检索召回率的策略

原问题: Neo4j的索引类型有哪些?
改写1: Neo4j支持的索引种类
改写2: 图数据库索引机制
改写3: Neo4j性能优化之索引设计

# 你的任务

原问题: {query}
改写1:
改写2:
改写3:
"""
        
        try:
            response, _ = self.llm.chat(
                self.tokenizer,
                prompt,
                history=[],
                do_sample=False,  # 确保输出稳定
                temperature=0.3
            )
            
            # 解析输出
            queries = self._parse_queries(response, num_queries)
            print(f"[QueryRewriter] 生成了 {len(queries)} 个重写查询")
            return queries
        except Exception as e:
            print(f"[QueryRewriter] 多查询重写失败: {e}")
            return [query]  # 降级: 返回原查询
    
    def rewrite_step_back(self, query: str) -> str:
        """
        Step-back Prompting: 生成更抽象的查询
        
        例如: "ChatGLM3如何部署" -> "大语言模型部署方法"
        """
        print(f"[QueryRewriter] 开始Step-back重写: query='{query[:30]}...'")
        
        prompt = f"""给定一个具体的技术问题，生成一个更抽象、更通用的问题，用于检索背景知识。

# 示例

具体问题: ChatGLM3如何在AutoDL上部署?
抽象问题: 大语言模型的部署方法和环境配置

具体问题: Neo4j的NEXT_CHUNK关系如何建立?
抽象问题: 图数据库中关系的创建和管理

# 你的任务

具体问题: {query}
抽象问题:
"""
        
        try:
            response, _ = self.llm.chat(
                self.tokenizer,
                prompt,
                history=[],
                do_sample=False
            )
            
            # 提取抽象问题
            abstract_query = response.strip()
            # 如果包含"抽象问题:"，提取后面的内容
            if "抽象问题:" in abstract_query or "抽象问题：" in abstract_query:
                abstract_query = re.split(r'抽象问题[:：]\s*', abstract_query)[-1].strip()
            
            print(f"[QueryRewriter] Step-back查询: '{abstract_query[:50]}...'")
            return abstract_query
        except Exception as e:
            print(f"[QueryRewriter] Step-back重写失败: {e}")
            return query
    
    def _parse_queries(self, response: str, num: int) -> List[str]:
        """解析LLM输出的多个查询"""
        queries = []
        
        for i in range(1, num + 1):
            # 匹配 "改写1:" 或 "改写1："
            pattern = f"改写{i}[:：]\\s*(.+?)(?=改写{i+1}[:：]|$)"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                query_text = match.group(1).strip()
                # 只取第一行
                query_text = query_text.split('\n')[0].strip()
                if query_text:
                    queries.append(query_text)
        
        # 如果解析失败，返回原始响应的前几行
        if not queries:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            queries = lines[:num] if lines else [response]
        
        return queries


class LLMReranker:
    """
    LLM重排序器
    
    核心思想: 粗排(向量检索) + 精排(LLM打分)
    为什么需要: 向量相似度 ≠ 语义相关性
    """
    
    def __init__(self, llm, tokenizer):
        self.llm = llm
        self.tokenizer = tokenizer
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 3
    ) -> List[Document]:
        """
        用LLM对检索结果重新打分
        
        Args:
            query: 用户查询
            documents: 待排序的文档列表
            top_k: 返回top-k结果
            
        Returns:
            重排序后的文档列表
        """
        print(f"[Reranker] 开始重排序: {len(documents)} 个文档 -> top {top_k}")
        
        if not documents:
            return []
        
        scored_docs = []
        
        for i, doc in enumerate(documents):
            print(f"[Reranker] 评分文档 {i+1}/{len(documents)}...")
            score = self._score_relevance(query, doc.page_content)
            scored_docs.append((doc, score))
        
        # 按分数排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 打印排序结果
        print("[Reranker] 排序结果:")
        for i, (doc, score) in enumerate(scored_docs[:top_k]):
            print(f"  Rank {i+1}: score={score:.1f}, text='{doc.page_content[:50]}...'")
        
        return [doc for doc, score in scored_docs[:top_k]]
    
    def _score_relevance(self, query: str, document: str) -> float:
        """
        让LLM打分: 0-10
        
        Prompt Engineering重点: 清晰的评分标准
        """
        # 截断文档避免超长
        doc_snippet = document[:500] if len(document) > 500 else document
        
        prompt = f"""你是一个文档相关性评估专家。给定一个问题和一段文档，评估文档对回答问题的相关性。

评分标准:
- 10分: 文档直接回答了问题，包含所有必要信息
- 7-9分: 文档高度相关，包含部分答案
- 4-6分: 文档有一定相关性，但信息不完整
- 1-3分: 文档相关性较低
- 0分: 完全不相关

问题: {query}

文档:
{doc_snippet}

请只输出一个0-10的整数，不要有其他内容:
"""
        
        try:
            response, _ = self.llm.chat(
                self.tokenizer,
                prompt,
                history=[],
                do_sample=False,
                temperature=0
            )
            
            # 解析分数
            score_text = response.strip()
            # 提取数字
            numbers = re.findall(r'\d+', score_text)
            if numbers:
                score = float(numbers[0])
                return max(0, min(10, score))  # 限制在0-10
            else:
                return 5.0  # 默认中等分数
        except Exception as e:
            print(f"[Reranker] 评分失败: {e}")
            return 5.0
