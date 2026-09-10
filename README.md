# RAG 课程知识问答系统

> 基于 RAG 的《神经网络与深度学习》教材智能问答系统

## 📖 项目简介

用 RAG（检索增强生成）技术，把《神经网络与深度学习》（邱锡鹏）整本教材
做成一个可以提问的知识库问答系统。

**核心亮点**（随进度逐步实现）：
- RAG 全链路：文档切分 → Embedding → 向量检索 → 大模型生成
- 检索优化：hybrid search（BM25 + 向量）+ rerank，并有量化对比
- Agent / Function Call 能力
- FastAPI 后端 + 可交互界面

## 📁 目录结构

```
rag/
├── src/               # 源代码
│   └── week1_D3.py    # D3: PDF 文本提取
├── data/              # 数据（提取的文本等）
│   └── book.txt       # 教材全文（从 PDF 提取）
├── .gitignore
└── README.md
```

## 🔧 环境

- Python 3.10
- 核心依赖：`pypdf` `chromadb` `sentence-transformers` `langchain` `fastapi`

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install chromadb sentence-transformers langchain langchain-community fastapi uvicorn streamlit

# 2. 提取 PDF 文本（D3）
python src/week1_D3.py
```

## 📅 进度

| 步骤 | 内容 | 状态 |
|------|------|------|
| D1 | 环境确认 + 项目目录 | ✅ |
| D2 | 安装依赖库 | ✅ |
| D3 | PDF 文本提取（444页 → 48万字符）| ✅ |
| D4 | 文档切分 | 进行中 |
| D5 | d2l 学习 | 待开始 |
| D6 | Embedding + 向量库 | 待开始 |
| D7 | 最小检索 demo | 待开始 |
