from abc import ABC, abstractmethod
from typing import List

from rag_factory.data_model.retrieval import RetrievalResult

class FusionBase(ABC):
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