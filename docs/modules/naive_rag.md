# 标准RAG模块 (naive_rag)

## 概述
标准RAG模块提供基于向量检索的检索增强生成能力，是RAG-Factory的基础实现。

## 核心功能
- 文本分块与向量化
- 向量相似度检索
- 基于检索结果的生成

## 快速使用
```python
from llama_index.core import VectorStoreIndex
from rag_factory.llms import OpenAICompatible

# 初始化LLM
llm = OpenAICompatible(
    api_base="http://your-llm-server/v1",
    model="your-model"
)

# 创建向量索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("你的问题")
```

## 配置参数
| 参数 | 类型 | 说明 |
|------|------|------|
| chunk_size | int | 文本分块大小 |
| chunk_overlap | int | 分块重叠大小 | 
| similarity_top_k | int | 检索结果数量 |

## 存储后端
支持多种向量数据库：
- Qdrant
- FAISS
- LanceDB

## 性能优化建议
- 调整chunk_size平衡检索精度和速度
- 使用更高效的embedding模型
- 增加相似度检索的top_k值提高召回率