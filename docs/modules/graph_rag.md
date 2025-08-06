# 图RAG模块 (graph_rag)

## 概述
图RAG模块基于知识图谱实现检索增强生成，能够捕获实体间的复杂关系。

## 核心功能
- 知识三元组抽取
- 图结构存储与检索
- 基于图路径的推理

## 快速使用
```python
from rag_factory.graph_constructor import GraphRAGConstructor
from llama_index.core import PropertyGraphIndex

# 初始化图构造器
kg_extractor = GraphRAGConstructor(
    llm=llm,
    max_paths_per_chunk=2
)

# 创建图索引
index = PropertyGraphIndex(
    nodes=nodes,
    kg_extractors=[kg_extractor],
    property_graph_store=graph_store
)

# 构建社区
index.property_graph_store.build_communities()

# 创建查询引擎
query_engine = GraphRAGQueryEngine(
    graph_store=index.property_graph_store,
    llm=llm,
    similarity_top_k=5
)
response = query_engine.query("实体间的关系是什么?")
```

## 配置参数
| 参数 | 类型 | 说明 |
|------|------|------|
| max_paths_per_chunk | int | 每个chunk抽取的最大三元组数 |
| max_cluster_size | int | 图聚类的最大社区大小 |
| similarity_top_k | int | 检索的路径数量 |

## 存储后端
- Neo4j
- NebulaGraph
- NetworkX (内存模式)

## 性能优化建议
- 调整max_paths_per_chunk平衡质量和性能
- 优化社区发现算法参数
- 使用图嵌入增强检索效果