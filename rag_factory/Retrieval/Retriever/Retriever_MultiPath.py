from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from ..RetrieverBase import Document
from ..RetrieverBase import BaseRetriever
from ..utils.Fusion import FusionMethod, RRFusion, RetrievalResult


class MultiPathRetriever(BaseRetriever):
    """多路检索器"""
    
    def __init__(self, 
                 retrievers: List[BaseRetriever],
                 fusion_method: Optional[FusionMethod] = None,
                 top_k_per_retriever: int = 50):
        """
        Args:
            retrievers: 检索器列表，每个检索器需要实现retrieve方法
            fusion_method: 融合方法，默认为RRF
            top_k_per_retriever: 每个检索器返回的结果数量
        """
        self.retrievers = retrievers
        self.fusion_method = fusion_method or RRFusion()
        self.top_k_per_retriever = top_k_per_retriever
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        获取与查询相关的文档
        
        Args:
            query: 查询字符串
            **kwargs: 其他参数，包括top_k等
            
        Returns:
            相关文档列表
        """
        top_k = kwargs.get('top_k', 10)
        
        # 从每个检索器获取结果
        all_results = []
        for retriever in self.retrievers:
            try:
                # 使用BaseRetriever的invoke方法
                documents = retriever.invoke(query, **{**kwargs, 'k': self.top_k_per_retriever})
                
                # 转换为RetrievalResult格式
                formatted_results = []
                for i, doc in enumerate(documents):
                    if isinstance(doc, Document):
                        # 如果是Document对象
                        retrieval_result = RetrievalResult(
                            document=doc,
                            score=getattr(doc, 'score', 1.0),
                            rank=i + 1
                        )
                    elif isinstance(doc, dict):
                        # 如果返回的是字典格式，需要转换为Document对象
                        content = doc.get('content', '')
                        metadata = doc.get('metadata', {})
                        doc_id = doc.get('id')
                        
                        document = Document(
                            content=content,
                            metadata=metadata,
                            id=doc_id
                        )
                        
                        retrieval_result = RetrievalResult(
                            document=document,
                            score=doc.get('score', 1.0),
                            rank=i + 1
                        )
                    else:
                        # 如果是字符串或其他格式，转换为Document对象
                        document = Document(
                            content=str(doc),
                            metadata={},
                            id=None
                        )
                        
                        retrieval_result = RetrievalResult(
                            document=document,
                            score=1.0,
                            rank=i + 1
                        )
                    formatted_results.append(retrieval_result)
                
                all_results.append(formatted_results)
                
            except Exception as e:
                print(f"检索器 {type(retriever).__name__} 执行失败: {e}")
                all_results.append([])
        
        # 使用融合方法合并结果
        if not all_results or all(len(results) == 0 for results in all_results):
            return []
        
        fused_results = self.fusion_method.fuse(all_results, top_k)
        
        # 转换回Document格式
        documents = []
        for result in fused_results:
            doc = result.document
            # 将score和rank添加到metadata中以便保留
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata['score'] = result.score
            doc.metadata['rank'] = result.rank
            documents.append(doc)
        
        return documents

    
    def add_retriever(self, retriever: BaseRetriever):
        """添加新的检索器"""
        self.retrievers.append(retriever)
    
    def remove_retriever(self, name: str):
        """移除指定名称的检索器"""
        for i, retriever in enumerate(self.retrievers):
            if hasattr(retriever, '__class__') and retriever.__class__.__name__ == name:
                self.retrievers.pop(i)
                break
    
    def set_fusion_method(self, fusion_method: FusionMethod):
        """设置融合方法"""
        self.fusion_method = fusion_method


# 示例用法
if __name__ == "__main__":
    # 假设我们有两个检索器
    class MockBM25Retriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
            # 模拟BM25检索结果
            return [
                Document(content="BM25 result 1", metadata={"source": "bm25"}),
                Document(content="BM25 result 2", metadata={"source": "bm25"}),
                Document(content="Vector result 1", metadata={"source": "bm25"}),
            ]
    
    class MockVectorRetriever(BaseRetriever):
        def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
            # 模拟向量检索结果
            return [
                Document(content="Vector result 1", metadata={"source": "vector"}),
                Document(content="Vector result 2", metadata={"source": "vector"}),
            ]
    
    # 创建检索器实例
    bm25_retriever = MockBM25Retriever()
    vector_retriever = MockVectorRetriever()
    
    # 创建多路检索器
    multi_retriever = MultiPathRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        fusion_method=RRFusion(k=60.0),
        top_k_per_retriever=5
    )
    
    # 执行检索
    results = multi_retriever.invoke("test query", top_k=4)
    
    # 打印结果
    print(results)
    # for i, result in enumerate(results):
    #     score = result.metadata.get('score', 0.0)
    #     print(f"Rank {i+1}: {result.content} (Score: {score:.8f})")
