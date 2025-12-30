from typing import List
from langchain_core.documents import Document
from graph_retrieval import GraphEnhancedRetriever
from advanced_retrieval import HyDERetriever, QueryRewriter, LLMReranker

class AdvancedRAGPipeline:
    """
    高级RAG检索流水线
    
    集成所有检索策略:
    - 图增强检索 (GraphEnhancedRetriever)
    - HyDE检索
    - 查询重写 (Multi-Query + Step-back)
    - LLM重排序
    
    支持多种检索模式，可根据场景选择
    """
    
    def __init__(
        self,
        llm,
        tokenizer,
        vector_store,
        graph
    ):
        """
        初始化高级RAG流水线
        
        Args:
            llm: ChatGLM3模型实例
            tokenizer: 对应的tokenizer
            vector_store: Neo4jVector实例
            graph: py2neo Graph实例
        """
        self.llm = llm
        self.tokenizer = tokenizer
        
        # 初始化各个检索器
        self.graph_retriever = GraphEnhancedRetriever(vector_store, graph)
        self.hyde_retriever = HyDERetriever(llm, tokenizer, vector_store)
        self.query_rewriter = QueryRewriter(llm, tokenizer)
        self.reranker = LLMReranker(llm, tokenizer)
        
        print("[AdvancedRAG] 高级RAG流水线初始化完成")
    
    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        top_k: int = 5
    ) -> List[Document]:
        """
        可配置的检索策略
        
        Args:
            query: 用户查询
            strategy: 检索策略
                - "simple": 简单向量检索 (基线)
                - "graph": 图增强检索 (上下文扩展)
                - "hybrid": 混合检索 (向量+关键词+图)
                - "hyde": HyDE检索
                - "multi_query": 多查询融合
                - "full": 完整流水线 (推荐)
            top_k: 返回文档数量
            
        Returns:
            检索到的文档列表
        """
        print(f"\n{'='*60}")
        print(f"[AdvancedRAG] 检索策略: {strategy}")
        print(f"[AdvancedRAG] 查询: {query}")
        print(f"{'='*60}\n")
        
        if strategy == "simple":
            return self._simple_retrieve(query, top_k)
        elif strategy == "graph":
            return self._graph_retrieve(query, top_k)
        elif strategy == "hybrid":
            return self._hybrid_retrieve(query, top_k)
        elif strategy == "hyde":
            return self._hyde_retrieve(query, top_k)
        elif strategy == "multi_query":
            return self._multi_query_retrieve(query, top_k)
        elif strategy == "full":
            return self._full_pipeline(query, top_k)
        else:
            print(f"[AdvancedRAG] 未知策略 '{strategy}'，使用默认hybrid策略")
            return self._hybrid_retrieve(query, top_k)
    
    def _simple_retrieve(self, query: str, top_k: int) -> List[Document]:
        """基线: 简单向量检索"""
        print("[AdvancedRAG] 使用简单向量检索")
        return self.graph_retriever.vector_store.similarity_search(query, k=top_k)
    
    def _graph_retrieve(self, query: str, top_k: int) -> List[Document]:
        """图增强检索 (带上下文扩展)"""
        print("[AdvancedRAG] 使用图增强检索")
        return self.graph_retriever.retrieve_with_context_expansion(
            query, k=top_k, expand_before=1, expand_after=2
        )
    
    def _hybrid_retrieve(self, query: str, top_k: int) -> List[Document]:
        """混合检索 (向量+关键词+图)"""
        print("[AdvancedRAG] 使用混合检索")
        return self.graph_retriever.hybrid_retrieve(
            query,
            vector_weight=0.5,
            keyword_weight=0.3,
            graph_weight=0.2,
            top_k=top_k
        )
    
    def _hyde_retrieve(self, query: str, top_k: int) -> List[Document]:
        """HyDE检索"""
        print("[AdvancedRAG] 使用HyDE检索")
        return self.hyde_retriever.retrieve(query, k=top_k)
    
    def _multi_query_retrieve(self, query: str, top_k: int) -> List[Document]:
        """多查询融合"""
        print("[AdvancedRAG] 使用多查询融合")
        
        # 生成多个查询
        queries = self.query_rewriter.rewrite_multi_query(query, num_queries=3)
        
        # 对每个查询进行检索
        all_docs = []
        for q in queries:
            docs = self.graph_retriever.vector_store.similarity_search(q, k=3)
            all_docs.extend(docs)
        
        # 去重
        unique_docs = self._deduplicate(all_docs)
        
        # 重排序
        if len(unique_docs) > top_k:
            return self.reranker.rerank(query, unique_docs, top_k=top_k)
        else:
            return unique_docs[:top_k]
    
    def _full_pipeline(self, query: str, top_k: int) -> List[Document]:
        """
        完整的高级检索流水线
        
        流程:
        1. Query Rewriting (Multi-Query + Step-back)
        2. 多策略检索 (向量 + HyDE + 关键词)
        3. 图扩展
        4. Re-ranking
        """
        print("[AdvancedRAG] 使用完整流水线")
        
        # Step 1: Query Rewriting
        print("\n[Pipeline Step 1] Query Rewriting...")
        rewritten_queries = self.query_rewriter.rewrite_multi_query(query, 3)
        step_back_query = self.query_rewriter.rewrite_step_back(query)
        all_queries = rewritten_queries + [step_back_query, query]
        print(f"[Pipeline] 生成了 {len(all_queries)} 个查询变体")
        
        # Step 2: 多策略检索
        print("\n[Pipeline Step 2] 多策略检索...")
        all_docs = []
        
        # 2.1 向量检索 (对所有查询)
        for q in all_queries:
            docs = self.graph_retriever.vector_store.similarity_search(q, k=2)
            all_docs.extend(docs)
        
        # 2.2 HyDE检索
        hyde_docs = self.hyde_retriever.retrieve(query, k=3)
        all_docs.extend(hyde_docs)
        
        # 2.3 关键词检索
        keyword_docs = self.graph_retriever.retrieve_by_keywords(query, top_k=3)
        all_docs.extend(keyword_docs)
        
        print(f"[Pipeline] 多策略检索共得到 {len(all_docs)} 个文档")
        
        # Step 3: 去重
        print("\n[Pipeline Step 3] 去重...")
        unique_docs = self._deduplicate(all_docs)
        print(f"[Pipeline] 去重后剩余 {len(unique_docs)} 个文档")
        
        # Step 4: 图扩展 (只对top5扩展，避免过多)
        print("\n[Pipeline Step 4] 图扩展...")
        expanded_docs = []
        for doc in unique_docs[:5]:
            context = self.graph_retriever._expand_context(
                doc.page_content,
                doc.metadata.get('source'),
                before=1,
                after=1
            )
            for ctx in context:
                expanded_docs.append(Document(
                    page_content=ctx['text'],
                    metadata={
                        'source': ctx['source'],
                        'page': ctx['page'],
                        'chapter': ctx['chapter']
                    }
                ))
        
        # 再次去重
        expanded_docs = self._deduplicate(expanded_docs)
        print(f"[Pipeline] 图扩展后共 {len(expanded_docs)} 个文档")
        
        # Step 5: Re-ranking
        print("\n[Pipeline Step 5] LLM重排序...")
        final_docs = self.reranker.rerank(query, expanded_docs, top_k=top_k)
        
        print(f"\n[Pipeline] 完整流水线完成，返回 {len(final_docs)} 个文档")
        return final_docs
    
    def _deduplicate(self, docs: List[Document]) -> List[Document]:
        """去重"""
        seen = set()
        unique = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique.append(doc)
        return unique
