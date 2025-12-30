import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.vectorstores import Neo4jVector
except ImportError:
    from langchain.vectorstores import Neo4jVector
from py2neo import Graph
import sys
import os

# 添加system目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'system'))

from rag_pipeline import AdvancedRAGPipeline
from prompt_templates import PromptTemplateManager, SelfConsistencyVerifier

# ================= 配置路径 =================
MODEL_PATH = "/root/autodl-tmp/models/chatglm3-6b"
EMBEDDING_PATH = "/root/autodl-tmp/models/bge-large-zh-v1.5"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 设置页面
st.set_page_config(page_title="高级RAG系统", page_icon="🚀", layout="wide")
st.title("🚀 高级工业文档知识问答系统 (Advanced RAG)")
st.markdown("### 🔥 图数据库深度利用 | HyDE | Query Rewriting | Re-ranking | Prompt Engineering")

# ================= 1. 加载模型 =================
@st.cache_resource
def load_models():
    print("⏳ [System] 正在加载 Embedding 模型...")
    try:
        embedding = HuggingFaceEmbeddings(
            model_name=EMBEDDING_PATH,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
    except Exception as e:
        st.error(f"Embedding 模型加载失败: {e}")
        return None, None, None, None, None
    
    print("⏳ [System] 正在连接 Neo4j...")
    try:
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
        
        # py2neo Graph连接
        graph = Graph(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    except Exception as e:
        st.error(f"Neo4j 连接失败: {e}")
        return None, None, None, None, None

    print("⏳ [System] 正在加载 ChatGLM3 模型...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            device_map="auto", 
            torch_dtype="auto"
        ).eval()
    except Exception as e:
        st.error(f"ChatGLM3 模型加载失败: {e}")
        return None, None, None, None, None
    
    print("⏳ [System] 初始化高级RAG流水线...")
    try:
        pipeline = AdvancedRAGPipeline(model, tokenizer, vector_store, graph)
        consistency_verifier = SelfConsistencyVerifier(model, tokenizer)
    except Exception as e:
        st.error(f"RAG流水线初始化失败: {e}")
        return None, None, None, None, None
    
    print("✅ [System] 所有组件加载完成!")
    return tokenizer, model, vector_store, pipeline, consistency_verifier

tokenizer, model, vector_store, pipeline, consistency_verifier = load_models()

# ================= 2. 状态管理 =================
if "history" not in st.session_state:
    st.session_state.history = []

# ================= 3. 侧边栏配置 =================
with st.sidebar:
    st.header("⚙️ 高级配置")
    
    # 检索策略选择
    st.subheader("🔍 检索策略")
    retrieval_strategy = st.selectbox(
        "选择检索策略",
        ["simple", "graph", "hybrid", "hyde", "multi_query", "full"],
        index=5,  # 默认选择full
        help="""
        - simple: 简单向量检索 (基线)
        - graph: 图增强检索 (上下文扩展)
        - hybrid: 混合检索 (向量+关键词+图)
        - hyde: HyDE检索 (假想答案)
        - multi_query: 多查询融合
        - full: 完整流水线 (推荐)
        """
    )
    
    # Prompt策略选择
    st.subheader("💬 Prompt策略")
    prompt_strategy = st.selectbox(
        "选择Prompt类型",
        ["default", "cot", "react", "few_shot"],
        index=0,
        help="""
        - default: 标准RAG prompt
        - cot: Chain-of-Thought (逐步推理)
        - react: ReAct框架 (推理+行动)
        - few_shot: Few-shot Learning (示例引导)
        """
    )
    
    # 自洽性验证
    use_consistency = st.checkbox(
        "启用自洽性验证",
        value=False,
        help="生成多个答案并选择最一致的 (耗时较长)"
    )
    
    if use_consistency:
        num_samples = st.slider("生成答案数量", 2, 5, 3)
    
    # 检索参数
    st.subheader("🎛️ 检索参数")
    top_k = st.slider("返回文档数量 (top_k)", 1, 10, 5)
    
    # 清空按钮
    st.divider()
    if st.button("🗑️ 清空对话历史"):
        st.session_state.history = []
        st.rerun()
    
    st.info("💡 使用高级检索策略和Prompt工程技术，提升问答质量")

# 显示历史消息
for query, response, metadata in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        st.markdown(response)
        if metadata:
            with st.expander("📊 检索详情"):
                st.write(f"**检索策略**: {metadata.get('strategy', 'N/A')}")
                st.write(f"**Prompt类型**: {metadata.get('prompt_type', 'N/A')}")
                st.write(f"**检索文档数**: {metadata.get('num_docs', 'N/A')}")

# ================= 4. 核心问答逻辑 =================
if prompt_text := st.chat_input("请输入您的问题..."):
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # 检索
    context_docs = []
    print(f"🔍 用户提问: {prompt_text}")
    
    try:
        with st.status("🔍 正在检索知识库...", expanded=True) as status:
            if pipeline:
                # 使用高级RAG流水线检索
                status.write(f"📌 使用检索策略: {retrieval_strategy}")
                context_docs = pipeline.retrieve(
                    prompt_text,
                    strategy=retrieval_strategy,
                    top_k=top_k
                )
                
                if context_docs:
                    status.write(f"✅ 检索到 {len(context_docs)} 个相关文档片段")
                    for i, doc in enumerate(context_docs[:3]):  # 只显示前3个
                        source = doc.metadata.get('source', '未知')
                        chapter = doc.metadata.get('chapter', '未知章节')
                        page = doc.metadata.get('page', '?')
                        status.markdown(f"**片段 {i+1}**: {source} - {chapter} (P{page})")
                        status.code(doc.page_content[:150] + "...", language="text")
                else:
                    status.write("⚠️ 未检索到相关内容")
                
                status.update(label="检索完成", state="complete", expanded=False)
            else:
                status.write("⚠️ RAG流水线未初始化")
                status.update(label="检索跳过", state="error", expanded=False)
                
    except Exception as e:
        st.error(f"检索出错: {e}")
        print(f"❌ 检索出错: {e}")

    # 构造 Prompt
    if context_docs:
        input_prompt = PromptTemplateManager.build_rag_prompt(
            prompt_text,
            context_docs,
            prompt_type=prompt_strategy
        )
    else:
        input_prompt = f"问题: {prompt_text}\n\n请基于你的知识回答:"

    # 生成答案
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        
        if model and tokenizer:
            try:
                if use_consistency:
                    # 使用自洽性验证
                    with st.spinner(f"🧠 正在生成 {num_samples} 个答案并验证一致性..."):
                        full_response = consistency_verifier.generate_with_consistency(
                            input_prompt,
                            num_samples=num_samples,
                            temperature=0.7
                        )
                        placeholder.markdown(full_response)
                else:
                    # 标准生成
                    for response, history, past_key_values in model.stream_chat(
                        tokenizer, 
                        input_prompt, 
                        history=[], 
                        do_sample=False,
                        repetition_penalty=1.2,
                        max_length=4096,
                        past_key_values=None,
                        return_past_key_values=True
                    ):
                        placeholder.markdown(response)
                        full_response = response
                
                # 保存到历史
                metadata = {
                    'strategy': retrieval_strategy,
                    'prompt_type': prompt_strategy,
                    'num_docs': len(context_docs)
                }
                st.session_state.history.append((prompt_text, full_response, metadata))
                
                # 显示元信息
                with st.expander("📊 本次检索详情"):
                    st.write(f"**检索策略**: {retrieval_strategy}")
                    st.write(f"**Prompt类型**: {prompt_strategy}")
                    st.write(f"**检索文档数**: {len(context_docs)}")
                    if use_consistency:
                        st.write(f"**自洽性验证**: 是 ({num_samples}个样本)")
                
            except Exception as e:
                st.error(f"生成出错: {e}")
                print(f"❌ 生成出错: {e}")
        else:
            st.error("模型未加载，无法生成回答。")
