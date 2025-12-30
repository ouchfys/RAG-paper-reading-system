from py2neo import Graph
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
import jieba.analyse

class GraphEnhancedRetriever:
    """
    图增强检索器 - 深度利用Neo4j图数据库
    
    核心功能:
    1. 上下文扩展: 通过NEXT_CHUNK关系自动扩展检索结果
    2. 关键词检索: 利用HAS_KEYWORD关系提升专有名词检索
    3. 混合检索: 融合向量、关键词、图遍历多种策略
    """
    
    def __init__(self, vector_store, graph: Graph):
        """
        初始化图增强检索器
        
        Args:
            vector_store: LangChain的Neo4jVector实例
            graph: py2neo的Graph实例
        """
        self.vector_store = vector_store
        self.graph = graph
    
    def retrieve_with_context_expansion(
        self, 
        query: str, 
        k: int = 3,
        expand_before: int = 1,
        expand_after: int = 2,
    ) -> List[Document]:
        """
        带上下文扩展的检索
        
        算法流程:
        1. 向量检索找到top-k相关chunks
        2. 对每个chunk通过NEXT_CHUNK关系扩展上下文
        3. 去重并按原文顺序排序
        
        Args:
            query: 用户查询
            k: 初始检索数量
            expand_before: 向前扩展chunk数量
            expand_after: 向后扩展chunk数量
            
        Returns:
            扩展后的Document列表
        """
        print(f"[GraphRetriever] 开始上下文扩展检索: query='{query[:30]}...', k={k}")
        
        # Step 1: 向量检索
        initial_docs = self.vector_store.similarity_search(query, k=k)
        print(f"[GraphRetriever] 向量检索得到 {len(initial_docs)} 个初始结果")
        
        # Step 2: 图扩展
        expanded_chunks = []
        for i, doc in enumerate(initial_docs):
            print(f"[GraphRetriever] 扩展第 {i+1} 个chunk的上下文...")
            context_chunks = self._expand_context(
                doc.page_content,
                doc.metadata.get('source'),
                expand_before,
                expand_after
            )
            expanded_chunks.extend(context_chunks)
        
        # Step 3: 去重 + 排序
        unique_chunks = self._deduplicate_and_sort(expanded_chunks)
        print(f"[GraphRetriever] 上下文扩展完成，最终得到 {len(unique_chunks)} 个unique chunks")
        
        return unique_chunks
    
    def _expand_context(
        self, 
        chunk_text: str, 
        source: str,
        before: int,
        after: int
    ) -> List[Dict]:
        """
        通过Cypher查询扩展上下文
        
        利用NEXT_CHUNK关系进行图遍历
        """
        # 注意: 由于data_import.py中使用text作为匹配条件，这里保持一致
        # 虽然这不是最优方案，但为了不修改现有代码，暂时使用
        
        query = """
        MATCH (current:Chunk {text: $text, source: $source})
        
        // 向前遍历 (找到指向current的chunks)
        OPTIONAL MATCH (before:Chunk)-[:NEXT_CHUNK*1..%d]->(current)
        WITH current, collect(DISTINCT before) as before_chunks
        
        // 向后遍历
        OPTIONAL MATCH (current)-[:NEXT_CHUNK*1..%d]->(after:Chunk)
        WITH current, before_chunks, collect(DISTINCT after) as after_chunks
        
        // 合并结果
        WITH before_chunks + [current] + after_chunks as all_chunks
        UNWIND all_chunks as chunk
        
        RETURN DISTINCT 
            chunk.text as text,
            chunk.page as page,
            chunk.chapter as chapter,
            chunk.source as source
        ORDER BY chunk.page
        """ % (before, after)
        
        try:
            result = self.graph.run(query, text=chunk_text, source=source)
            chunks = [dict(record) for record in result]
            return chunks if chunks else [{'text': chunk_text, 'page': 0, 'chapter': '未知', 'source': source}]
        except Exception as e:
            print(f"[GraphRetriever] 图扩展查询失败: {e}")
            # 降级: 返回原始chunk
            return [{'text': chunk_text, 'page': 0, 'chapter': '未知', 'source': source}]
    
    def retrieve_by_keywords(
        self, 
        query: str,
        top_k: int = 5
    ) -> List[Document]:
        """
        基于关键词的图检索
        
        算法:
        1. 提取query中的关键词
        2. 在图中找到包含这些关键词的chunks
        3. 按关键词匹配度排序
        
        适用场景: 查询包含专有名词时效果更好
        """
        print(f"[GraphRetriever] 开始关键词检索: query='{query[:30]}...'")
        
        # 提取关键词
        keywords = jieba.analyse.extract_tags(query, topK=5)
        print(f"[GraphRetriever] 提取到关键词: {keywords}")
        
        if not keywords:
            print("[GraphRetriever] 未提取到关键词，返回空列表")
            return []
        
        # Cypher查询: 找到包含关键词的chunks
        query_cypher = """
        MATCH (c:Chunk)-[:HAS_KEYWORD]->(k:Keyword)
        WHERE k.name IN $keywords
        
        // 计算每个chunk匹配了多少个关键词
        WITH c, count(DISTINCT k) as keyword_count
        ORDER BY keyword_count DESC
        LIMIT $top_k
        
        RETURN 
            c.text as text,
            c.page as page,
            c.chapter as chapter,
            c.source as source,
            keyword_count
        """
        
        try:
            result = self.graph.run(query_cypher, keywords=keywords, top_k=top_k)
            
            docs = []
            for record in result:
                docs.append(Document(
                    page_content=record['text'],
                    metadata={
                        'source': record['source'],
                        'page': record['page'],
                        'chapter': record['chapter'],
                        'keyword_match_count': record['keyword_count']
                    }
                ))
            
            print(f"[GraphRetriever] 关键词检索得到 {len(docs)} 个结果")
            return docs
        except Exception as e:
            print(f"[GraphRetriever] 关键词检索失败: {e}")
            return []
    
    def hybrid_retrieve(
        self,
        query: str,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.2,
        top_k: int = 5
    ) -> List[Document]:
        """
        混合检索策略: 向量 + 关键词 + 图扩展
        
        使用Reciprocal Rank Fusion (RRF)算法融合多个检索结果
        
        Args:
            query: 用户查询
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            graph_weight: 图扩展权重
            top_k: 返回top-k结果
            
        Returns:
            融合后的Document列表
        """
        print(f"[GraphRetriever] 开始混合检索: weights=({vector_weight}, {keyword_weight}, {graph_weight})")
        
        # 1. 向量检索
        vector_docs = self.vector_store.similarity_search(query, k=top_k)
        print(f"[GraphRetriever] 向量检索: {len(vector_docs)} 个结果")
        
        # 2. 关键词检索
        keyword_docs = self.retrieve_by_keywords(query, top_k=top_k)
        print(f"[GraphRetriever] 关键词检索: {len(keyword_docs)} 个结果")
        
        # 3. 图扩展(对向量检索结果)
        expanded_docs = self.retrieve_with_context_expansion(
            query, k=min(3, top_k), expand_before=1, expand_after=1
        )
        print(f"[GraphRetriever] 图扩展: {len(expanded_docs)} 个结果")
        
        # 4. Reciprocal Rank Fusion (RRF)
        scores = {}
        doc_map = {}  # text -> Document对象
        
        # 向量检索得分
        for rank, doc in enumerate(vector_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + vector_weight / (rank + 1)
            doc_map[key] = doc
        
        # 关键词检索得分
        for rank, doc in enumerate(keyword_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + keyword_weight / (rank + 1)
            doc_map[key] = doc
        
        # 图扩展得分
        for rank, doc in enumerate(expanded_docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + graph_weight / (rank + 1)
            doc_map[key] = doc
        
        # 5. 按得分排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 6. 转换回Document对象
        final_docs = [doc_map[text] for text, score in sorted_items]
        
        print(f"[GraphRetriever] 混合检索完成，返回 {len(final_docs)} 个结果")
        return final_docs
    
    def _deduplicate_and_sort(self, chunks: List[Dict]) -> List[Document]:
        """
        去重并转换为Document对象
        
        按页码排序，保持原文顺序
        """
        seen = set()
        unique = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if text and text not in seen:
                seen.add(text)
                unique.append(Document(
                    page_content=text,
                    metadata={
                        'source': chunk.get('source', '未知'),
                        'page': chunk.get('page', 0),
                        'chapter': chunk.get('chapter', '未知')
                    }
                ))
        
        # 按页码排序
        unique.sort(key=lambda x: x.metadata.get('page', 0))
        return unique
