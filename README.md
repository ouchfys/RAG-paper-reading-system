

# 高级工业文档智能问答系统 (Advanced RAG)

基于 **ChatGLM3-6B** 与 **Neo4j** 构建的垂直领域高级检索增强生成系统。相比基础版本，本项目引入了图谱深度利用、HyDE 检索、查询重写及自动化评估等机制，旨在解决长文档上下文丢失与复杂逻辑推理问题。

## 核心特性

### 1. 图数据库深度应用

* **上下文扩展**: 利用 Neo4j 的 `NEXT_CHUNK` 关系，自动检索并拼接相邻文本块，解决切片导致的语义断裂。
* **混合检索**: 结合向量相似度搜索与图谱关键词 (`HAS_KEYWORD`) 匹配，并使用 RRF 算法进行结果融合。

### 2. 高级检索策略

* **HyDE (Hypothetical Document Embeddings)**: 生成假想答案用于检索，提升对隐含语义的匹配能力。
* **Query Rewriting**: 包含多角度查询生成 (Multi-Query) 与回溯提示 (Step-back Prompting)。
* **Re-ranking**: 引入 LLM 对初步召回的文档进行二次精排，过滤低相关性内容。

### 3. Prompt Engineering

* **思维链 (CoT)**: 引导模型分步骤推理（理解-定位-提取-综合）。
* **ReAct 框架**: 结合推理与行动的逻辑处理复杂问题。
* **自洽性验证**: 通过生成多个回答并投票，提升输出结果的可靠性。

---

## 项目结构

```text
.
├── system/
│   ├── data_import.py          # 图谱构建：向量入库与关系建立
│   ├── file_extraction.py      # PDF 结构化解析
│   ├── graph_retrieval.py      # 图增强检索器实现
│   ├── advanced_retrieval.py   # HyDE、重写与重排序实现
│   ├── rag_pipeline.py         # 检索全流程集成
│   ├── prompt_templates.py     # Prompt 策略管理
│   └── strategy_evaluator.py   # 策略评估模块
├── web_demo_advanced.py        # 高级 Web 交互界面
├── test_advanced_rag.py        # 后端逻辑测试脚本
├── run_ingest.py               # 数据入库入口
└── requirements.txt            # 依赖清单

```

---

## 快速开始

### 1. 环境准备

确保已安装 Python 3.10+，并启动 Neo4j 数据库。

```bash
pip install transformers langchain langchain-community py2neo jieba ragas

```

### 2. 数据入库

解析 PDF 文档并构建图谱索引（向量+关系）。

```bash
python run_ingest.py

```

### 3. 启动服务

启动 Streamlit 前端界面。

```bash
streamlit run web_demo_advanced.py --server.port 6006

```

### 4. 运行测试

检查检索流水线各模块是否正常工作。

```bash
python test_advanced_rag.py

```

---

## 使用说明

Web 界面提供以下配置选项：

### 检索策略 (Retrieval Strategy)

* **simple**: 仅使用基础向量检索。
* **graph**: 启用图谱上下文扩展（推荐）。
* **hybrid**: 向量 + 关键词 + 图谱混合检索。
* **full**: 完整流水线（重写 + 混合检索 + 重排序）。

### 提示词策略 (Prompt Strategy)

* **default**: 标准问答格式。
* **cot**: 启用思维链，适合复杂推理问题。
* **react**: 启用推理+行动框架。
* **few_shot**: 加载预设示例引导回答风格。

### 高级选项

* **自洽性验证**: 开启后将生成 3 个独立回答并选取最优解，会增加响应时间。

---

## 系统逻辑简述

1. **数据层**: PDF 被解析为 Chunk 节点存入 Neo4j，通过 `NEXT_CHUNK` 链接上下文，通过 `HAS_KEYWORD` 链接实体节点。
2. **检索层**:
* 用户 Query 首先经过重写模块生成多个变体。
* 并行执行向量检索、关键词检索与 HyDE 检索。
* 结果经过 RRF 融合与去重。
* 通过图关系扩展上下文（向前/向后查找相邻 Chunk）。
* 最后通过 LLM Reranker 进行打分排序。


3. **生成层**: 依据选择的 Prompt 模板（如 CoT），将优化后的上下文输入 ChatGLM3 生成最终回答。
