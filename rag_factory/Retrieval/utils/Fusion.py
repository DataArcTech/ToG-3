from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict

from ..RetrieverBase import Document


@dataclass
class RetrievalResult:
    """检索结果的数据类"""
    document: Document
    score: float
    rank: int = 0


class FusionMethod(ABC):
    """融合方法的抽象基类"""
    
    @abstractmethod
    def fuse(self, results: List[List[RetrievalResult]], top_k: int) -> List[RetrievalResult]:
        """
        融合多个检索器的结果
        
        Args:
            results: 每个检索器的结果列表
            top_k: 返回的最终结果数量
            
        Returns:
            融合后的结果列表
        """
        pass


class RRFusion(FusionMethod):
    """Reciprocal Rank Fusion (RRF) 方法"""
    
    def __init__(self, k: float = 60.0):
        """
        Args:
            k: RRF中的常数，默认为60.0
        """
        self.k = k
    
    def fuse(self, results: List[List[RetrievalResult]], top_k: int) -> List[RetrievalResult]:
        # 为每个结果分配rank
        for retriever_results in results:
            for i, result in enumerate(retriever_results):
                result.rank = i + 1
        
        # 计算RRF分数
        rrf_scores = defaultdict(float)
        document_map = {}
        
        for retriever_results in results:
            for result in retriever_results:
                rrf_score = 1.0 / (self.k + result.rank)
                # 使用文档内容作为key来去重
                content_key = result.document.content
                rrf_scores[content_key] += rrf_score
                document_map[content_key] = result.document
        
        # 按RRF分数排序
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 构建最终结果
        fused_results = []
        for i, (content, rrf_score) in enumerate(sorted_items[:top_k]):
            result = RetrievalResult(
                document=document_map[content],
                score=rrf_score,
                rank=i + 1
            )
            fused_results.append(result)
        
        return fused_results
