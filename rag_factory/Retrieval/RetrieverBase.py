from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
from collections import Counter

from pydantic import ConfigDict


@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


class BaseRetriever(ABC):
    """检索器基类
    
    一个检索系统被定义为能够接受字符串查询并从某个源返回最"相关"文档的系统。
    
    使用方法:
    检索器遵循标准的可运行接口，应通过 `invoke`, `ainvoke` 等标准方法使用。
    
    实现:
    实现自定义检索器时，类应该实现 `_get_relevant_documents` 方法来定义检索文档的逻辑。
    可选地，可以通过重写 `_aget_relevant_documents` 方法提供异步原生实现。
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **kwargs):
        """初始化检索器
        
        Args:
            **kwargs: 其他参数，如 search_kwargs, tags, metadata 等
        """
        self.search_kwargs = kwargs.get("search_kwargs", {})
        self.tags = kwargs.get("tags")
        self.metadata = kwargs.get("metadata")
    
    def invoke(self, input: str, **kwargs: Any) -> List[Document]:
        """调用检索器获取相关文档
        
        同步检索器调用的主要入口点。
        
        Args:
            input: 查询字符串
            **kwargs: 传递给检索器的其他参数
            
        Returns:
            相关文档列表
            
        Examples:
            >>> retriever.invoke("query")
        """
        return self._get_relevant_documents(input, **kwargs)
    
    async def ainvoke(self, input: str, **kwargs: Any) -> List[Document]:
        """异步调用检索器获取相关文档
        
        异步检索器调用的主要入口点。
        
        Args:
            input: 查询字符串
            **kwargs: 传递给检索器的其他参数
            
        Returns:
            相关文档列表
            
        Examples:
            >>> await retriever.ainvoke("query")
        """
        return await self._aget_relevant_documents(input, **kwargs)
    
    @abstractmethod
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """获取与查询相关的文档
        
        Args:
            query: 用于查找相关文档的字符串
            **kwargs: 其他参数
            
        Returns:
            相关文档列表
        """
        pass
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """异步获取与查询相关的文档
        
        Args:
            query: 用于查找相关文档的字符串
            **kwargs: 其他参数
            
        Returns:
            相关文档列表
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, self._get_relevant_documents, query, **kwargs
            )
    
    def get_name(self) -> str:
        """获取检索器名称"""
        return self.__class__.__name__
