from typing import Dict, Type, List, Callable
import logging

from rag_factory.retrieval.retriever.text.base import BaseRetriever
from rag_factory.retrieval.retriever.text.bm25 import BM25Retriever
from rag_factory.retrieval.retriever.text.vectorstore import VectorStoreRetriever
from rag_factory.retrieval.retriever.text.multipath import MultiPathRetriever

logger = logging.getLogger(__name__)


class RetrieverRegistry:
    """检索器注册表，用于管理和创建不同类型的检索器"""
    
    _retrievers: Dict[str, Type[BaseRetriever]] = {}

    @classmethod
    def register(cls, name: str, retriever_class: Type[BaseRetriever]):
        """注册检索器类"""
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
        """自动创建检索器实例"""
        if name not in cls._retrievers:
            available = ', '.join(cls.list_available())
            raise ValueError(f"检索器 '{name}' 未注册。可用的检索器: {available}")
            
        try:
            # 根据检索器类型选择创建方式
            if name == "bm25":
                return cls._create_bm25(**kwargs)
            elif name == "vectorstore":
                return cls._create_vectorstore(**kwargs)
            elif name == "multipath":
                return cls._create_multipath(**kwargs)
            else:
                # 默认构造函数
                retriever_class = cls._retrievers[name]
                return retriever_class(**kwargs)
                
        except Exception as e:
            logger.error(f"创建检索器 '{name}' 失败: {str(e)}")
            raise Exception(f"无法创建检索器 '{name}': {str(e)}") from e
    
    @classmethod
    def _create_bm25(cls, data_path: str = None, documents: List = None, 
                     preprocess_func: Callable = None, **kwargs) -> BM25Retriever:
        """智能创建BM25检索器"""
        if documents is not None:
            # 直接从文档创建
            return BM25Retriever.from_documents(
                documents=documents,
                preprocess_func=preprocess_func,
                **kwargs
            )
        else:
            # 创建空检索器（需要后续加载文档）
            return BM25Retriever(preprocess_func=preprocess_func, **kwargs)
    
    @classmethod
    def _create_vectorstore(cls, vectorstore, **kwargs) -> VectorStoreRetriever:
        """创建向量存储检索器"""
        return VectorStoreRetriever(vectorstore=vectorstore, **kwargs)
    
    @classmethod
    def _create_multipath(cls, retrievers: List[BaseRetriever], **kwargs) -> MultiPathRetriever:
        """创建多路检索器"""
        return MultiPathRetriever(retrievers=retrievers, **kwargs)
    
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
