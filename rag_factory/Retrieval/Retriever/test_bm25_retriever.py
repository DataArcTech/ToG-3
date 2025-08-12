from Retriever_BM25 import BM25Retriever
from rag_factory.Retrieval.RetrieverBase import Document
from typing import List
import logging
logger = logging.getLogger(__name__)

def chinese_preprocessing_func(text: str) -> List[str]:
    """中文文本预处理函数
    
    Args:
        text: 输入的中文文本
        
    Returns:
        分词后的词语列表
    """
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        logger.warning("jieba 未安装，使用默认分词方法。请安装: pip install jieba")
        return text.split()


if __name__ == "__main__":
    # 构造测试数据
    texts = [
        "这是第一个测试文档",
        "这是第二个测试文档，内容稍有不同",
        "这是第三个文档，讨论不同的主题",
        "第四个文档包含更多详细信息",
        "最后一个文档作为总结"
    ]

    print(chinese_preprocessing_func("这是第一个测试文档"))
    metadatas = [
        {"source": "doc1", "type": "test"},
        {"source": "doc2", "type": "test"},
        {"source": "doc3", "type": "example"},
        {"source": "doc4", "type": "detailed"},
        {"source": "doc5", "type": "summary"},
    ]
    ids = [f"doc{i+1}" for i in range(len(texts))]

    # 创建BM25Retriever
    retriever = BM25Retriever.from_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        preprocess_func=chinese_preprocessing_func,
        k=3  # top3
    )

    # 查询
    query = "第二个测试文档内容"
    print(f"\n查询: {query}")
    
    # retriever.update_k(4)
    results = retriever.invoke(query, k=4)
    print("\n召回结果:")
    for i, (doc) in enumerate(results):
        # print(f"{i+1}. 分数: {score:.4f}")
        print(f"   ID: {doc.id}")
        print(f"   内容: {doc.content}")
        print(f"   元数据: {doc.metadata}")
