from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from dataclasses import dataclass, field

from pydantic import ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

from rag_factory.Retrieval.RetrieverBase import BaseRetriever, Document


def default_preprocessing_func(text: str) -> List[str]:
    """默认的文本预处理函数
    
    Args:
        text: 输入文本
        
    Returns:
        分词后的词语列表
    """
    return text.split()


def chinese_preprocessing_func(text: str) -> List[str]:
    """中文文本预处理函数
    
    Args:
        text: 输入的中文文本
        
    Returns:
        分词后的词语列表
    """
    try:
        import jieba
        return list(jieba.cut(text))
    except ImportError:
        logger.warning("jieba 未安装，使用默认分词方法。请安装: pip install jieba")
        return text.split()


class BM25Retriever(BaseRetriever):
    """BM25 检索器实现
    
    基于 BM25 算法的文档检索器。
    使用 rank_bm25 库实现高效的 BM25 搜索。
    
    注意：BM25 算法适用于相对静态的文档集合。虽然支持动态添加/删除文档，
    但每次操作都会重建整个索引，在大型文档集合上可能有性能问题。
    对于频繁更新的场景，建议使用 VectorStoreRetriever。
    
    Attributes:
        vectorizer: BM25 向量化器实例
        docs: 文档列表
        k: 返回的文档数量
        preprocess_func: 文本分词函数
        bm25_params: BM25 算法参数
    """
    
    vectorizer: Any = None
    """BM25 向量化器实例"""
    
    docs: List[Document] = Field(default_factory=list, repr=False)
    """文档列表"""
    
    k: int = 5
    """返回的文档数量，默认为 5"""
    
    preprocess_func: Callable[[str], List[str]] = Field(default=default_preprocessing_func)
    """文本预处理函数，默认使用空格分词"""
    
    bm25_params: Dict[str, Any] = Field(default_factory=dict)
    """BM25 算法参数"""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    def __init__(self, **kwargs):
        """初始化 BM25 检索器
        
        Args:
            vectorizer: BM25 向量化器
            docs: 文档列表
            k: 返回文档数量
            preprocess_func: 预处理函数
            bm25_params: BM25 参数
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        # 设置属性
        self.vectorizer = kwargs.get('vectorizer')
        self.docs = kwargs.get('docs', [])
        self.k = kwargs.get('k', 4)
        self.preprocess_func = kwargs.get('preprocess_func', default_preprocessing_func)
        self.bm25_params = kwargs.get('bm25_params', {})
        
        # 验证配置
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """验证配置参数
        
        Raises:
            ValueError: 如果配置无效
        """
        if self.k <= 0:
            raise ValueError(f"k 必须大于 0，当前值: {self.k}")
        
        if not callable(self.preprocess_func):
            raise ValueError("preprocess_func 必须是可调用的函数")
    
    @model_validator(mode="before")
    @classmethod
    def validate_params(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """验证参数（Pydantic 验证器）
        
        Args:
            values: 要验证的值
            
        Returns:
            验证后的值
        """
        k = values.get("k", 4)
        if k <= 0:
            raise ValueError(f"k 必须大于 0，当前值: {k}")
        
        return values
    
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[str]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        k: int = 4,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """从文本列表创建 BM25Retriever
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表，可选
            ids: ID列表，可选
            bm25_params: BM25 算法参数，可选
            preprocess_func: 预处理函数
            k: 返回文档数量
            **kwargs: 其他参数
            
        Returns:
            BM25Retriever 实例
            
        Raises:
            ImportError: 如果未安装 rank_bm25
            ValueError: 如果参数不匹配
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "未找到 rank_bm25 库，请安装: pip install rank_bm25"
            )
        
        # 转换为列表
        texts_list = list(texts)
        if not texts_list:
            raise ValueError("texts 不能为空")
        
        # 处理元数据和ID
        if metadatas is not None:
            metadatas_list = list(metadatas)
            if len(metadatas_list) != len(texts_list):
                raise ValueError(
                    f"metadatas 长度 ({len(metadatas_list)}) "
                    f"与 texts 长度 ({len(texts_list)}) 不匹配"
                )
        else:
            metadatas_list = [{} for _ in texts_list]
        
        if ids is not None:
            ids_list = list(ids)
            if len(ids_list) != len(texts_list):
                raise ValueError(
                    f"ids 长度 ({len(ids_list)}) "
                    f"与 texts 长度 ({len(texts_list)}) 不匹配"
                )
        else:
            ids_list = [None for _ in texts_list]
        
        # 预处理文本
        logger.info(f"正在预处理 {len(texts_list)} 个文本...")
        texts_processed = [preprocess_func(text) for text in texts_list]
        
        # 创建 BM25 向量化器
        bm25_params = bm25_params or {}
        logger.info(f"创建 BM25 向量化器，参数: {bm25_params}")
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        
        # 创建文档对象
        docs = []
        for text, metadata, doc_id in zip(texts_list, metadatas_list, ids_list):
            doc = Document(content=text, metadata=metadata, id=doc_id)
            docs.append(doc)
        
        logger.info(f"成功创建包含 {len(docs)} 个文档的 BM25Retriever")
        
        return cls(
            vectorizer=vectorizer,
            docs=docs,
            k=k,
            preprocess_func=preprocess_func,
            bm25_params=bm25_params,
            **kwargs
        )
    
    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        k: int = 4,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """从文档列表创建 BM25Retriever
        
        Args:
            documents: 文档列表
            bm25_params: BM25 算法参数，可选
            preprocess_func: 预处理函数
            k: 返回文档数量
            **kwargs: 其他参数
            
        Returns:
            BM25Retriever 实例
        """
        docs_list = list(documents)
        if not docs_list:
            raise ValueError("documents 不能为空")
        
        texts = [doc.content for doc in docs_list]
        metadatas = [doc.metadata for doc in docs_list]
        ids = [doc.id for doc in docs_list]
        
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            bm25_params=bm25_params,
            preprocess_func=preprocess_func,
            k=k,
            **kwargs,
        )
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """获取与查询相关的文档
        
        Args:
            query: 查询字符串
            **kwargs: 其他参数，可能包含 'k' 来覆盖默认的返回数量
            
        Returns:
            相关文档列表
            
        Raises:
            ValueError: 如果向量化器未初始化
        """
        if self.vectorizer is None:
            raise ValueError("BM25 向量化器未初始化")
        
        if not self.docs:
            logger.warning("文档列表为空，返回空结果")
            return []
        
        # 获取返回文档数量
        k = kwargs.get('k', self.k)
        k = min(k, len(self.docs))  # 确保不超过总文档数
        
        try:
            # 预处理查询
            processed_query = self.preprocess_func(query)
            logger.debug(f"预处理后的查询: {processed_query}")
            
            # 获取相关文档
            relevant_docs = self.vectorizer.get_top_n(
                processed_query, self.docs, n=k
            )
            
            logger.debug(f"找到 {len(relevant_docs)} 个相关文档")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"BM25 搜索时发生错误: {e}")
            raise
    
    def get_scores(self, query: str) -> List[float]:
        """获取查询对所有文档的 BM25 分数
        
        Args:
            query: 查询字符串
            
        Returns:
            所有文档的 BM25 分数列表
        """
        if self.vectorizer is None:
            raise ValueError("BM25 向量化器未初始化")
        
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        return scores.tolist()
    
    def get_top_k_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple[Document, float]]:
        """获取 top-k 文档及其分数
        
        Args:
            query: 查询字符串
            k: 返回文档数量，如果为 None 则使用实例的 k 值
            
        Returns:
            (文档, 分数) 元组列表
        """
        if self.vectorizer is None:
            raise ValueError("BM25 向量化器未初始化")
        
        if not self.docs:
            return []
        
        k = k or self.k
        k = min(k, len(self.docs))
        
        # 获取所有分数
        scores = self.get_scores(query)
        
        # 获取 top-k 索引
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]
        
        # 返回文档和分数
        results = []
        for idx in top_indices:
            results.append((self.docs[idx], scores[idx]))
        
        return results
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """添加新文档到检索器
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        对于频繁的文档更新操作，建议考虑使用 VectorStoreRetriever。
        
        Args:
            documents: 要添加的文档列表
            **kwargs: 其他参数
                rebuild_threshold: 文档数量阈值，超过此值会发出警告（默认1000）
            
        Returns:
            添加文档的ID列表
            
        Raises:
            ImportError: 如果未安装 rank_bm25
            RuntimeWarning: 如果文档数量超过建议阈值
        """
        if not documents:
            return []
        
        # 检查文档数量，发出性能警告
        rebuild_threshold = kwargs.get('rebuild_threshold', 1000)
        total_docs = len(self.docs) + len(documents)
        if total_docs > rebuild_threshold:
            import warnings
            warnings.warn(
                f"正在重建包含 {total_docs} 个文档的 BM25 索引，这可能很慢。"
                f"对于大型或频繁更新的文档集合，建议使用 VectorStoreRetriever。",
                RuntimeWarning,
                stacklevel=2
            )
        
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "未找到 rank_bm25 库，请安装: pip install rank_bm25"
            )
        
        # 添加文档到现有列表
        self.docs.extend(documents)
        
        # 重新构建 BM25 索引
        all_texts = [doc.content for doc in self.docs]
        texts_processed = [self.preprocess_func(text) for text in all_texts]
        
        self.vectorizer = BM25Okapi(texts_processed, **self.bm25_params)
        
        logger.info(f"添加了 {len(documents)} 个文档，重新构建了 BM25 索引")
        
        # 返回添加文档的ID（如果有的话）
        return [doc.id for doc in documents if doc.id is not None]
    
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """异步添加文档
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, self.add_documents, documents, **kwargs
            )
    
    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """删除文档
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        对于频繁的文档更新操作，建议考虑使用 VectorStoreRetriever。
        
        Args:
            ids: 要删除的文档ID列表，如果为None则删除所有文档
            **kwargs: 其他参数
                rebuild_threshold: 文档数量阈值，超过此值会发出警告（默认1000）
            
        Returns:
            删除是否成功
        """
        if ids is None:
            # 删除所有文档
            self.docs.clear()
            self.vectorizer = None
            logger.info("删除了所有文档")
            return True
        
        # 删除指定ID的文档
        original_count = len(self.docs)
        self.docs = [doc for doc in self.docs if doc.id not in ids]
        deleted_count = original_count - len(self.docs)
        
        if deleted_count > 0:
            # 检查文档数量，发出性能警告
            rebuild_threshold = kwargs.get('rebuild_threshold', 1000)
            if len(self.docs) > rebuild_threshold:
                import warnings
                warnings.warn(
                    f"正在重建包含 {len(self.docs)} 个文档的 BM25 索引，这可能很慢。"
                    f"对于大型或频繁更新的文档集合，建议使用 VectorStoreRetriever。",
                    RuntimeWarning,
                    stacklevel=2
                )
            
            # 重新构建索引
            if self.docs:
                try:
                    from rank_bm25 import BM25Okapi
                    all_texts = [doc.content for doc in self.docs]
                    texts_processed = [self.preprocess_func(text) for text in all_texts]
                    self.vectorizer = BM25Okapi(texts_processed, **self.bm25_params)
                except ImportError:
                    raise ImportError("未找到 rank_bm25 库")
            else:
                self.vectorizer = None
            
            logger.info(f"删除了 {deleted_count} 个文档，重新构建了 BM25 索引")
        
        return deleted_count > 0
    
    async def adelete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """异步删除文档
        
        警告：此操作会重新构建整个 BM25 索引，在大型文档集合上可能很慢。
        """
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor, self.delete_documents, ids, **kwargs
            )
    
    def get_document_count(self) -> int:
        """获取文档总数"""
        return len(self.docs)
    
    def get_bm25_info(self) -> Dict[str, Any]:
        """获取 BM25 检索器信息
        
        Returns:
            包含检索器信息的字典
        """
        info = {
            "document_count": len(self.docs),
            "k": self.k,
            "bm25_params": self.bm25_params,
            "preprocess_func": self.preprocess_func.__name__,
            "has_vectorizer": self.vectorizer is not None,
        }
        
        if self.vectorizer is not None:
            info.update({
                "vocab_size": len(self.vectorizer.idf),
                "average_doc_length": getattr(self.vectorizer, 'avgdl', 'N/A'),
            })
        
        return info
    
    def update_k(self, new_k: int) -> None:
        """更新返回文档数量
        
        Args:
            new_k: 新的文档返回数量
        """
        if new_k <= 0:
            raise ValueError(f"k 必须大于 0，当前值: {new_k}")
        
        self.k = new_k
        logger.debug(f"更新 k 值为: {new_k}")
    
    def get_name(self) -> str:
        """获取检索器名称"""
        return "BM25Retriever"
    
    def __repr__(self) -> str:
        """返回检索器的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"docs={len(self.docs)}, "
            f"k={self.k}, "
            f"preprocess_func={self.preprocess_func.__name__})"
        )
