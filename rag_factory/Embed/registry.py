from typing import Dict, Type, Any, Optional, List
import logging
from .Embedding_Huggingface import HuggingFaceEmbeddings
from .Embedding_Base import Embeddings

class EmbeddingRegistry:
    """嵌入模型注册器，用于管理和创建不同类型的嵌入模型"""
    _embeddings: Dict[str, Type[Embeddings]] = {}

    @classmethod
    def register(cls, name: str, embedding_class: Type[Embeddings]):
        """注册嵌入模型类
        
        Args:
            name: 模型名称
            embedding_class: 嵌入模型类
        """
        cls._embeddings[name] = embedding_class

    @classmethod
    def create(cls, name: str, **kwargs) -> Embeddings:
        """获取嵌入模型实例
        
        Args:
            name: 模型名称
            **kwargs: 模型初始化参数
            
        Returns:
            嵌入模型实例
            
        Raises:
            ValueError: 当模型名称不存在时
        """
        if name not in cls._embeddings:
            available_embeddings = list(cls._embeddings.keys())
            raise ValueError(f"嵌入模型 '{name}' 未注册。可用的模型: {available_embeddings}")
        
        embedding_class = cls._embeddings[name]
        return embedding_class(**kwargs)

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