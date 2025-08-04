# 多模态RAG模块 (mm_rag)

## 概述
多模态RAG模块支持同时处理文本和图像数据，实现跨模态的检索增强生成。

## 核心功能
- 多模态数据统一处理
- 跨模态检索
- 多模态内容生成

## 快速使用
```python
from llama_index.core import MultiModalVectorStoreIndex
from rag_factory.multi_modal_llms import OpenAICompatibleMultiModal

# 初始化多模态LLM
llm = OpenAICompatibleMultiModal(
    api_base="http://your-llm-server/v1",
    model="your-multimodal-model"
)

# 创建多模态索引
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    image_embed_model="clip:ViT-B/32"
)

# 创建查询引擎
query_engine = index.as_query_engine(
    text_qa_template=MULTIMODAL_QA_TMPL
)
response = query_engine.query("描述这张图片中的内容")
```

## 配置参数
| 参数 | 类型 | 说明 |
|------|------|------|
| image_embed_model | str | 图像嵌入模型 |
| text_embed_model | str | 文本嵌入模型 |
| similarity_top_k | int | 每种模态的检索结果数量 |

## 支持的数据类型
- 文本
- 图像
- 图文混合内容

## 性能优化建议
- 选择合适的图像嵌入模型
- 调整不同模态的检索权重
- 优化多模态提示模板