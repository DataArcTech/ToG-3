# VectorStore/registry.py
from typing import Dict, Type, Any, Optional, List
from .VectorStoreBase import VectorStore
from ...Embed.Embedding_Base import Embeddings
from .VectorStore_Faiss import FaissVectorStore


class VectorStoreRegistry:
    """向量存储注册表"""
    
    _stores: Dict[str, Type[VectorStore]] = {}
    
    @classmethod
    def register(cls, name: str, store_class: Type[VectorStore]):
        """注册向量存储类"""
        cls._stores[name] = store_class
    
    @classmethod
    def create(cls, name: str, embedding: Embeddings, **kwargs) -> VectorStore:
        """创建向量存储实例"""
        if name not in cls._stores:
            raise ValueError(f"未注册的向量存储类型: {name}")
        
        return cls._stores[name](embedding=embedding, **kwargs)

    @classmethod
    def load(cls, name: str, folder_path: str, embedding: Embeddings, **kwargs) -> VectorStore:
        """加载已保存的向量存储实例"""
        if name not in cls._stores:
            raise ValueError(f"未注册的向量存储类型: {name}")
        store_class = cls._stores[name]
        if hasattr(store_class, "load_local"):
            return store_class.load_local(folder_path, embeddings=embedding, **kwargs)
        else:
            raise ValueError(f"{store_class.__name__} 不支持从本地加载")

    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出可用的向量存储类型"""
        return list(cls._stores.keys())


# 注册默认的向量存储
VectorStoreRegistry.register("faiss", FaissVectorStore)