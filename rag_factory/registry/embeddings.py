from typing import Dict, Type, Any, Optional, List
import logging
from rag_factory.embeddings.base import Embeddings
from rag_factory.embeddings.huggingface import HuggingFaceEmbeddings


logger = logging.getLogger(__name__)

class EmbeddingRegistry:
    """嵌入模型注册表，支持智能创建逻辑"""
    _embeddings: Dict[str, Type[Embeddings]] = {}

    @classmethod
    def register(cls, name: str, embedding_class: Type[Embeddings]):
        """注册嵌入模型类"""
        cls._embeddings[name] = embedding_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Embeddings:
        """创建嵌入模型实例"""
        if name not in cls._embeddings:
            available_embeddings = list(cls._embeddings.keys())
            raise ValueError(f"嵌入模型 '{name}' 未注册。可用的模型: {available_embeddings}")
        
        try:
            # 根据嵌入模型类型选择创建方式
            if name == "huggingface":
                return cls._create_huggingface(**kwargs)
            else:
                # 默认构造函数
                embedding_class = cls._embeddings[name]
                return embedding_class(**kwargs)
        except Exception as e:
            logger.error(f"创建嵌入模型 '{name}' 失败: {str(e)}")
            raise Exception(f"无法创建嵌入模型 '{name}': {str(e)}") from e
    
    @classmethod
    def _create_huggingface(cls, **kwargs) -> HuggingFaceEmbeddings:
        """创建HuggingFace嵌入模型"""
        return HuggingFaceEmbeddings(**kwargs)

    @classmethod
    def list_embeddings(cls) -> List[str]:
        """列出所有已注册的嵌入模型名称
        
        Returns:
            已注册的模型名称列表
        """
        return list(cls._embeddings.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查模型是否已注册
        
        Args:
            name: 模型名称
            
        Returns:
            如果已注册返回True，否则返回False
        """
        return name in cls._embeddings

    @classmethod
    def unregister(cls, name: str) -> bool:
        """取消注册模型
        
        Args:
            name: 模型名称
            
        Returns:
            成功取消注册返回True，模型不存在返回False
        """
        if name in cls._embeddings:
            del cls._embeddings[name]
            return True
        return False


# 注册默认的嵌入模型
EmbeddingRegistry.register("huggingface", HuggingFaceEmbeddings)