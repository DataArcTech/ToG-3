from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, ClassVar, Collection
from collections import Counter
import logging

from pydantic import ConfigDict, Field, model_validator
from Retrieval.RetrieverBase import BaseRetriever, Document
from Store.VectorStore.VectorStoreBase import VectorStore

logger = logging.getLogger(__name__)


class VectorStoreRetriever(BaseRetriever):
    """向量数据库检索器
    
    基于向量数据库的检索器实现，支持多种搜索类型：
    - similarity: 相似性搜索
    - similarity_score_threshold: 带分数阈值的相似性搜索
    - mmr: 最大边际相关性搜索
    """
    
    vectorstore: 'VectorStore'
    """用于检索的向量数据库实例"""
    
    search_type: str = "similarity"
    """执行的搜索类型，默认为 'similarity'"""
    
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """传递给搜索函数的关键字参数"""
    
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold", 
        "mmr",
    )
    """允许的搜索类型"""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, vectorstore: 'VectorStore', **kwargs):
        """初始化向量数据库检索器
        
        Args:
            vectorstore: 向量数据库实例
            search_type: 搜索类型，默认为 "similarity"
            search_kwargs: 搜索参数字典
            **kwargs: 其他参数
        """
        self.vectorstore = vectorstore
        self.search_type = kwargs.get("search_type", "similarity")
        self.search_kwargs = kwargs.get("search_kwargs", {})
        
        # 验证搜索类型
        self._validate_search_config()
        
        # 调用父类初始化
        super().__init__(**kwargs)
    
    def _validate_search_config(self) -> None:
        """验证搜索配置
        
        Raises:
            ValueError: 如果搜索类型不在允许的类型中
            ValueError: 如果使用 similarity_score_threshold 但未指定有效的 score_threshold
        """
        if self.search_type not in self.allowed_search_types:
            msg = (
                f"search_type '{self.search_type}' 不被允许。"
                f"有效值为: {self.allowed_search_types}"
            )
            raise ValueError(msg)
        
        if self.search_type == "similarity_score_threshold":
            score_threshold = self.search_kwargs.get("score_threshold")
            if (score_threshold is None or 
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "使用 'similarity_score_threshold' 搜索类型时，"
                    "必须在 search_kwargs 中指定有效的 score_threshold (0~1 之间的浮点数)"
                )
                raise ValueError(msg)
    
    @model_validator(mode="before")
    @classmethod
    def validate_search_type(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """验证搜索类型（Pydantic 验证器）
        
        Args:
            values: 要验证的值
            
        Returns:
            验证后的值
            
        Raises:
            ValueError: 如果搜索类型无效
        """
        search_type = values.get("search_type", "similarity")
        if search_type not in cls.allowed_search_types:
            msg = (
                f"search_type '{search_type}' 不被允许。"
                f"有效值为: {cls.allowed_search_types}"
            )
            raise ValueError(msg)
            
        if search_type == "similarity_score_threshold":
            search_kwargs = values.get("search_kwargs", {})
            score_threshold = search_kwargs.get("score_threshold")
            if (score_threshold is None or 
                not isinstance(score_threshold, (int, float)) or
                not (0 <= score_threshold <= 1)):
                msg = (
                    "使用 'similarity_score_threshold' 搜索类型时，"
                    "必须在 search_kwargs 中指定有效的 score_threshold (0~1 之间的数值)"
                )
                raise ValueError(msg)
        
        return values
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """获取与查询相关的文档
        
        Args:
            query: 查询字符串
            **kwargs: 额外的搜索参数
            
        Returns:
            相关文档列表
            
        Raises:
            ValueError: 如果搜索类型无效
        """
        # 合并搜索参数
        search_params = {**self.search_kwargs, **kwargs}
        
        try:
            if self.search_type == "similarity":
                docs = self.vectorstore.similarity_search(query, **search_params)
                
            elif self.search_type == "similarity_score_threshold":
                docs_and_similarities = (
                    self.vectorstore.similarity_search_with_relevance_scores(
                        query, **search_params
                    )
                )
                docs = [doc for doc, _ in docs_and_similarities]
                
            elif self.search_type == "mmr":
                docs = self.vectorstore.max_marginal_relevance_search(
                    query, **search_params
                )
                
            else:
                msg = f"不支持的搜索类型: {self.search_type}"
                raise ValueError(msg)
            
            logger.debug(f"检索到 {len(docs)} 个文档，搜索类型: {self.search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"检索文档时发生错误: {e}")
            raise
    
    async def _aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """异步获取与查询相关的文档
        
        Args:
            query: 查询字符串
            **kwargs: 额外的搜索参数
            
        Returns:
            相关文档列表
            
        Raises:
            ValueError: 如果搜索类型无效
        """
        # 合并搜索参数
        search_params = {**self.search_kwargs, **kwargs}
        
        try:
            if self.search_type == "similarity":
                docs = await self.vectorstore.asimilarity_search(query, **search_params)
                
            elif self.search_type == "similarity_score_threshold":
                docs_and_similarities = (
                    await self.vectorstore.asimilarity_search_with_relevance_scores(
                        query, **search_params
                    )
                )
                docs = [doc for doc, _ in docs_and_similarities]
                
            elif self.search_type == "mmr":
                docs = await self.vectorstore.amax_marginal_relevance_search(
                    query, **search_params
                )
                
            else:
                msg = f"不支持的搜索类型: {self.search_type}"
                raise ValueError(msg)
            
            logger.debug(f"异步检索到 {len(docs)} 个文档，搜索类型: {self.search_type}")
            return docs
            
        except Exception as e:
            logger.error(f"异步检索文档时发生错误: {e}")
            raise
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """向向量数据库添加文档
        
        Args:
            documents: 要添加的文档列表
            **kwargs: 其他关键字参数
            
        Returns:
            添加文档的ID列表
        """
        try:
            ids = self.vectorstore.add_documents(documents, **kwargs)
            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            return ids
        except Exception as e:
            logger.error(f"添加文档时发生错误: {e}")
            raise
    
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """异步向向量数据库添加文档
        
        Args:
            documents: 要添加的文档列表
            **kwargs: 其他关键字参数
            
        Returns:
            添加文档的ID列表
        """
        try:
            ids = await self.vectorstore.aadd_documents(documents, **kwargs)
            logger.info(f"成功异步添加 {len(documents)} 个文档到向量数据库")
            return ids
        except Exception as e:
            logger.error(f"异步添加文档时发生错误: {e}")
            raise
    
    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """从向量数据库删除文档
        
        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他关键字参数
            
        Returns:
            删除是否成功
        """
        try:
            result = self.vectorstore.delete(ids, **kwargs)
            if ids:
                logger.info(f"删除了 {len(ids)} 个文档")
            else:
                logger.info("删除了所有文档")
            return result
        except Exception as e:
            logger.error(f"删除文档时发生错误: {e}")
            raise
    
    async def adelete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """异步从向量数据库删除文档
        
        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他关键字参数
            
        Returns:
            删除是否成功
        """
        try:
            result = await self.vectorstore.adelete(ids, **kwargs)
            if ids:
                logger.info(f"异步删除了 {len(ids)} 个文档")
            else:
                logger.info("异步删除了所有文档")
            return result
        except Exception as e:
            logger.error(f"异步删除文档时发生错误: {e}")
            raise
    
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """根据ID获取文档
        
        Args:
            ids: 要获取的文档ID列表
            
        Returns:
            文档列表
        """
        try:
            docs = self.vectorstore.get_by_ids(ids)
            logger.debug(f"根据ID获取了 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.error(f"根据ID获取文档时发生错误: {e}")
            raise
    
    async def aget_by_ids(self, ids: List[str]) -> List[Document]:
        """异步根据ID获取文档
        
        Args:
            ids: 要获取的文档ID列表
            
        Returns:
            文档列表
        """
        try:
            docs = await self.vectorstore.aget_by_ids(ids)
            logger.debug(f"异步根据ID获取了 {len(docs)} 个文档")
            return docs
        except Exception as e:
            logger.error(f"异步根据ID获取文档时发生错误: {e}")
            raise
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """获取向量数据库信息
        
        Returns:
            包含向量数据库信息的字典
        """
        info = {
            "vectorstore_class": self.vectorstore.__class__.__name__,
            "search_type": self.search_type,
            "search_kwargs": self.search_kwargs,
            "allowed_search_types": list(self.allowed_search_types),
        }
        
        # 如果向量数据库有嵌入信息，添加到信息中
        if hasattr(self.vectorstore, 'embeddings') and self.vectorstore.embeddings:
            info["embedding_class"] = self.vectorstore.embeddings.__class__.__name__
        elif hasattr(self.vectorstore, 'embedding'):
            info["embedding_class"] = self.vectorstore.embedding.__class__.__name__
        
        return info
    
    def get_name(self) -> str:
        """获取检索器名称"""
        return f"{self.vectorstore.__class__.__name__}Retriever"
    
    def update_search_params(self, **kwargs: Any) -> None:
        """更新搜索参数
        
        Args:
            **kwargs: 要更新的搜索参数
        """
        self.search_kwargs.update(kwargs)
        
        # 如果更新了搜索类型，重新验证
        if "search_type" in kwargs:
            self.search_type = kwargs["search_type"]
            self._validate_search_config()
        
        logger.debug(f"更新搜索参数: {kwargs}")
    
    def __repr__(self) -> str:
        """返回检索器的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"vectorstore={self.vectorstore.__class__.__name__}, "
            f"search_type='{self.search_type}', "
            f"search_kwargs={self.search_kwargs})"
        )

