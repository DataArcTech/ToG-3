from typing import Dict, Type, Any, Optional, List
import logging
from .Retriever_BM25 import BM25Retriever
from .Retriever_MultiPath import MultiPathRetriever
from .Retriever_VectorStore import VectorStoreRetriever
from ..RetrieverBase import BaseRetriever

logger = logging.getLogger(__name__)

# TODO 统一所有检索器的调用方式

class RetrieverRegistry:
    """检索器注册表，用于管理和创建不同类型的检索器"""
    
    _retrievers: Dict[str, Type[BaseRetriever]] = {}

    @classmethod
    def register(cls, name: str, retriever_class: Type[BaseRetriever]):
        """
        注册检索器类
        
        Args:
            name: 检索器名称
            retriever_class: 检索器类
            
        Raises:
            ValueError: 当检索器类不是BaseRetriever的子类时
            TypeError: 当name不是字符串时
        """
        if not isinstance(name, str):
            raise TypeError("检索器名称必须是字符串类型")
        
        if not issubclass(retriever_class, BaseRetriever):
            raise ValueError(f"检索器类 {retriever_class} 必须继承自 BaseRetriever")
            
        if name in cls._retrievers:
            logger.warning(f"检索器 '{name}' 已存在，将被覆盖")
            
        cls._retrievers[name] = retriever_class
        logger.info(f"检索器 '{name}' 注册成功")

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseRetriever:
        """
        创建检索器实例
        
        Args:
            name: 检索器名称
            **kwargs: 传递给检索器构造函数的参数
            
        Returns:
            BaseRetriever: 检索器实例
            
        Raises:
            ValueError: 当检索器未注册时
            Exception: 当检索器创建失败时
        """
        if name not in cls._retrievers:
            available = ', '.join(cls.list_available())
            raise ValueError(f"检索器 '{name}' 未注册。可用的检索器: {available}")
            
        retriever_class = cls._retrievers[name]
        try:
            return retriever_class(**kwargs)
        except Exception as e:
            logger.error(f"创建检索器 '{name}' 失败: {str(e)}")
            raise Exception(f"无法创建检索器 '{name}': {str(e)}") from e
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        获取所有可用的检索器名称
        
        Returns:
            List[str]: 检索器名称列表
        """
        return list(cls._retrievers.keys())
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        取消注册检索器
        
        Args:
            name: 检索器名称
            
        Returns:
            bool: 是否成功取消注册
        """
        if name in cls._retrievers:
            del cls._retrievers[name]
            logger.info(f"检索器 '{name}' 取消注册成功")
            return True
        return False
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        检查检索器是否已注册
        
        Args:
            name: 检索器名称
            
        Returns:
            bool: 是否已注册
        """
        return name in cls._retrievers
    
    @classmethod
    def clear_all(cls):
        """清除所有已注册的检索器"""
        cls._retrievers.clear()
        logger.info("所有检索器注册已清除")

# 预注册内置检索器
RetrieverRegistry.register("bm25", BM25Retriever)
RetrieverRegistry.register("multipath", MultiPathRetriever) 
RetrieverRegistry.register("vectorstore", VectorStoreRetriever)
