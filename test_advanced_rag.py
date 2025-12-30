"""
高级RAG系统测试脚本

功能:
1. 测试各个检索策略
2. 对比不同策略的效果
3. 验证Prompt Engineering功能
"""

import sys
import os

# 添加system目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'system'))

from transformers import AutoModel, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from py2neo import Graph
import torch

from rag_pipeline import AdvancedRAGPipeline
from prompt_templates import PromptTemplateManager, SelfConsistencyVerifier
from strategy_evaluator import StrategyComparator

# ================= 配置 =================
MODEL_PATH = "/root/autodl-tmp/models/chatglm3-6b"
EMBEDDING_PATH = "/root/autodl-tmp/models/bge-large-zh-v1.5"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

def load_components():
    """加载所有组件"""
    print("⏳ 加载Embedding模型...")
    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_PATH,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    print("⏳ 连接Neo4j...")
    vector_store = Neo4jVector.from_existing_graph(
        embedding=embedding,
        url=NEO4J_URL,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="vector",
        node_label="Chunk",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
    
    graph = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    print("⏳ 加载ChatGLM3...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    ).eval()
    
    print("⏳ 初始化RAG流水线...")
    pipeline = AdvancedRAGPipeline(model, tokenizer, vector_store, graph)
    
    print("✅ 所有组件加载完成!\n")
    return model, tokenizer, pipeline, vector_store, graph


def test_retrieval_strategies(pipeline, query):
    """测试不同的检索策略"""
    print(f"\n{'='*70}")
    print(f"测试问题: {query}")
    print(f"{'='*70}\n")
    
    strategies = ["simple", "graph", "hybrid", "hyde", "multi_query", "full"]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"策略: {strategy}")
        print(f"{'='*60}")
        
        try:
            docs = pipeline.retrieve(query, strategy=strategy, top_k=3)
            print(f"\n检索到 {len(docs)} 个文档:")
            for i, doc in enumerate(docs):
                print(f"\n文档{i+1}:")
                print(f"  来源: {doc.metadata.get('source', '未知')}")
                print(f"  章节: {doc.metadata.get('chapter', '未知')}")
                print(f"  页码: {doc.metadata.get('page', '?')}")
                print(f"  内容: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"✗ 策略 {strategy} 失败: {e}")


def test_prompt_strategies(model, tokenizer, docs, query):
    """测试不同的Prompt策略"""
    print(f"\n{'='*70}")
    print(f"测试Prompt策略")
    print(f"问题: {query}")
    print(f"{'='*70}\n")
    
    prompt_types = ["default", "cot", "react"]
    
    for prompt_type in prompt_types:
        print(f"\n{'='*60}")
        print(f"Prompt类型: {prompt_type}")
        print(f"{'='*60}")
        
        try:
            prompt = PromptTemplateManager.build_rag_prompt(
                query,
                docs,
                prompt_type=prompt_type
            )
            
            print(f"\n生成的Prompt (前500字符):")
            print(prompt[:500] + "...\n")
            
            print("生成答案...")
            response, _ = model.chat(
                tokenizer,
                prompt,
                history=[],
                do_sample=False,
                repetition_penalty=1.2
            )
            
            print(f"\n答案:")
            print(response)
            
        except Exception as e:
            print(f"✗ Prompt类型 {prompt_type} 失败: {e}")


def test_self_consistency(model, tokenizer, docs, query):
    """测试自洽性验证"""
    print(f"\n{'='*70}")
    print(f"测试自洽性验证")
    print(f"{'='*70}\n")
    
    verifier = SelfConsistencyVerifier(model, tokenizer)
    
    prompt = PromptTemplateManager.build_rag_prompt(query, docs, "default")
    
    print("生成3个答案并选择最一致的...")
    best_answer = verifier.generate_with_consistency(
        prompt,
        num_samples=3,
        temperature=0.7
    )
    
    print(f"\n最一致的答案:")
    print(best_answer)


def run_strategy_comparison(pipeline, model, tokenizer):
    """运行策略对比评估"""
    print(f"\n{'='*70}")
    print(f"策略对比评估")
    print(f"{'='*70}\n")
    
    # 定义测试问题
    test_questions = [
        {
            "question": "什么是RAG系统?",
            "ground_truth": "RAG是检索增强生成系统"
        },
        {
            "question": "如何优化检索性能?",
            "ground_truth": "可以通过建立索引、使用缓存等方式优化"
        },
        {
            "question": "Neo4j有什么优势?",
            "ground_truth": "Neo4j是图数据库，擅长处理关系数据"
        }
    ]
    
    comparator = StrategyComparator(pipeline, model, tokenizer)
    
    # 运行对比
    results = comparator.compare_strategies(
        test_questions,
        strategies=["simple", "hybrid", "full"],
        save_path="strategy_comparison_test.csv"
    )
    
    print("\n对比完成!")
    return results


def main():
    """主函数"""
    print("\n" + "="*70)
    print("高级RAG系统测试")
    print("="*70 + "\n")
    
    # 加载组件
    model, tokenizer, pipeline, vector_store, graph = load_components()
    
    # 测试问题
    test_query = "什么是检索增强生成?"
    
    # 1. 测试检索策略
    print("\n" + "="*70)
    print("测试1: 检索策略对比")
    print("="*70)
    test_retrieval_strategies(pipeline, test_query)
    
    # 获取一些文档用于后续测试
    docs = pipeline.retrieve(test_query, strategy="hybrid", top_k=3)
    
    if docs:
        # 2. 测试Prompt策略
        print("\n" + "="*70)
        print("测试2: Prompt策略对比")
        print("="*70)
        test_prompt_strategies(model, tokenizer, docs, test_query)
        
        # 3. 测试自洽性验证
        print("\n" + "="*70)
        print("测试3: 自洽性验证")
        print("="*70)
        test_self_consistency(model, tokenizer, docs, test_query)
    
    # 4. 策略对比评估
    print("\n" + "="*70)
    print("测试4: 策略对比评估")
    print("="*70)
    run_strategy_comparison(pipeline, model, tokenizer)
    
    print("\n" + "="*70)
    print("所有测试完成!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
