from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Optional,
    TypeVar,
    Union,
    Sequence,
    Iterable,
    Iterator,
    Tuple,
)
from itertools import cycle
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
if TYPE_CHECKING:
    from collections.abc import Collection
    from rag_factory.Embed import Embeddings
    from ...Retrieval.Retriever.Retriever_VectorStore import VectorStoreRetriever

logger = logging.getLogger(__name__)



VST = TypeVar("VST", bound="VectorStore")


@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    document: Document
    score: float
    distance: float



class VectorStore(ABC):
    """向量数据库基类"""
    
    def __init__(self, **kwargs: Any):
        """初始化向量存储"""
        pass
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """添加文本到向量存储
        
        Args:
            texts: 要添加的文本迭代器
            metadatas: 可选的元数据列表
            ids: 可选的ID列表
            **kwargs: 其他参数
            
        Returns:
            添加文本的ID列表
            
        Raises:
            ValueError: 如果元数据数量与文本数量不匹配
            ValueError: 如果ID数量与文本数量不匹配
        """
        # 转换为文档格式并调用add_documents
        texts_: Sequence[str] = (
            texts if isinstance(texts, (list, tuple)) else list(texts)
        )
        
        if metadatas and len(metadatas) != len(texts_):
            msg = (
                "元数据数量必须与文本数量匹配。"
                f"得到 {len(metadatas)} 个元数据和 {len(texts_)} 个文本。"
            )
            raise ValueError(msg)
        
        metadatas_ = iter(metadatas) if metadatas else cycle([{}])
        ids_: Iterator[Optional[str]] = iter(ids) if ids else cycle([None])
        
        docs = [
            Document(id=id_, content=text, metadata=metadata_)
            for text, metadata_, id_ in zip(texts_, metadatas_, ids_)
        ]
        
        if ids is not None:
            kwargs["ids"] = ids
            
        return self.add_documents(docs, **kwargs)
    
    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """添加或更新文档到向量存储，已经是Document对象的列表
        
        Args:
            documents: 要添加的文档列表
            kwargs: 其他参数，如果包含ids且documents也包含ids，kwargs中的ids优先
            
        Returns:
            添加文本的ID列表
            
        Raises:
            ValueError: 如果ID数量与文档数量不匹配
        """
        # 如果没有提供ids，尝试从文档中获取
        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
            # 如果至少有一个有效ID，则使用ID
            if any(ids):
                kwargs["ids"] = ids
        
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)
    
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """异步添加文本到向量存储"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.add_texts, texts, metadatas, ids, **kwargs
        )
    
    async def aadd_documents(
        self, documents: list[Document], **kwargs: Any
    ) -> list[str]:
        """异步添加文档到向量存储"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.add_documents, documents, **kwargs
        )
    
    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """根据向量ID或其他条件删除
        
        Args:
            ids: 要删除的ID列表。如果为None，删除所有。默认为None
            **kwargs: 其他关键字参数
            
        Returns:
            Optional[bool]: 如果删除成功返回True，否则返回False，未实现返回None
        """
        msg = "delete方法必须由子类实现。"
        raise NotImplementedError(msg)
    
    async def adelete(
        self, ids: Optional[list[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """异步删除"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.delete, ids, **kwargs
        )
    
    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """根据ID获取文档
        
        Args:
            ids: 要检索的ID列表
            
        Returns:
            文档列表
        """
        msg = f"{self.__class__.__name__} 尚不支持 get_by_ids。"
        raise NotImplementedError(msg)
    
    async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """异步根据ID获取文档"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.get_by_ids, ids
        )
    
    def search(self, query: str, search_type: str, **kwargs: Any) -> list[Document]:
        """使用指定搜索类型返回与查询最相似的文档
        
        Args:
            query: 输入文本
            search_type: 要执行的搜索类型。可以是"similarity"、"mmr"或"similarity_score_threshold"
            **kwargs: 传递给搜索方法的参数
            
        Returns:
            与查询最相似的文档列表
            
        Raises:
            ValueError: 如果search_type不是允许的类型之一
        """
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        if search_type == "mmr":
            return self.max_marginal_relevance_search(query, **kwargs)
        
        msg = (
            f"search_type {search_type} 不被允许。期望的search_type是"
            "'similarity'、'similarity_score_threshold'或'mmr'。"
        )
        raise ValueError(msg)
    
    async def asearch(
        self, query: str, search_type: str, **kwargs: Any
    ) -> list[Document]:
        """异步搜索"""
        if search_type == "similarity":
            return await self.asimilarity_search(query, **kwargs)
        if search_type == "similarity_score_threshold":
            docs_and_similarities = await self.asimilarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        if search_type == "mmr":
            return await self.amax_marginal_relevance_search(query, **kwargs)
        
        msg = (
            f"search_type {search_type} 不被允许。期望的search_type是"
            "'similarity'、'similarity_score_threshold'或'mmr'。"
        )
        raise ValueError(msg)
    
    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """返回与查询最相似的文档
        
        Args:
            query: 输入文本
            k: 要返回的文档数量。默认为4
            **kwargs: 传递给搜索方法的参数
            
        Returns:
            与查询最相似的文档列表
        """
        pass
    
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """异步相似性搜索"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.similarity_search, query, k, **kwargs
        )
    
    @staticmethod
    def _euclidean_relevance_score_fn(distance: float) -> float:
        """返回[0, 1]范围内的相似性分数"""
        return 1.0 - distance / math.sqrt(2)
    
    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """将距离归一化为[0, 1]范围内的分数"""
        return 1.0 - distance
    
    @staticmethod
    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """将距离归一化为[0, 1]范围内的分数"""
        if distance > 0:
            return 1.0 - distance
        return -1.0 * distance
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """选择相关性评分函数
        
        正确的相关性函数可能因以下因素而异：
        - VectorStore使用的距离/相似性度量
        - 嵌入的尺度（OpenAI的是单位标准化的，许多其他的不是！）
        - 嵌入维度
        等等
        
        向量存储应该定义自己基于选择的相关性方法。
        """
        raise NotImplementedError
    
    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[Tuple[Document, float]]:
        """使用距离运行相似性搜索
        
        Args:
            *args: 传递给搜索方法的参数
            **kwargs: 传递给搜索方法的参数
            
        Returns:
            (文档, 相似性分数)的元组列表
        """
        raise NotImplementedError
    
    async def asimilarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> list[Tuple[Document, float]]:
        """异步使用距离运行相似性搜索"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.similarity_search_with_score, *args, **kwargs
        )
    
    def _similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Tuple[Document, float]]:
        """默认的带相关性分数的相似性搜索
        
        必要时在子类中修改。
        返回[0, 1]范围内的文档和相关性分数。
        
        0表示不相似，1表示最相似。
        
        Args:
            query: 输入文本
            k: 要返回的文档数量。默认为4
            **kwargs: 传递给相似性搜索的kwargs。应该包括：
                score_threshold: 可选，0到1之间的浮点值，用于过滤结果集
                
        Returns:
            (文档, 相似性分数)的元组列表
        """
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
    
    async def _asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Tuple[Document, float]]:
        """异步带相关性分数的相似性搜索"""
        relevance_score_fn = self._select_relevance_score_fn()
        docs_and_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [(doc, relevance_score_fn(score)) for doc, score in docs_and_scores]
    
    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Tuple[Document, float]]:
        """返回[0, 1]范围内的文档和相关性分数
        
        0表示不相似，1表示最相似。
        
        Args:
            query: 输入文本
            k: 要返回的文档数量。默认为4
            **kwargs: 传递给相似性搜索的kwargs。应该包括：
                score_threshold: 可选，0到1之间的浮点值，用于过滤结果集
                
        Returns:
            (文档, 相似性分数)的元组列表
        """
        score_threshold = kwargs.pop("score_threshold", None)
        
        docs_and_similarities = self._similarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                f"相关性分数必须在0和1之间，得到 {docs_and_similarities}",
                stacklevel=2,
            )
        
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "使用相关性分数阈值 %s 没有检索到相关文档",
                    score_threshold,
                )
        return docs_and_similarities
    
    async def asimilarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Tuple[Document, float]]:
        """异步返回[0, 1]范围内的文档和相关性分数"""
        score_threshold = kwargs.pop("score_threshold", None)
        
        docs_and_similarities = await self._asimilarity_search_with_relevance_scores(
            query, k=k, **kwargs
        )
        
        if any(
            similarity < 0.0 or similarity > 1.0
            for _, similarity in docs_and_similarities
        ):
            warnings.warn(
                f"相关性分数必须在0和1之间，得到 {docs_and_similarities}",
                stacklevel=2,
            )
        
        if score_threshold is not None:
            docs_and_similarities = [
                (doc, similarity)
                for doc, similarity in docs_and_similarities
                if similarity >= score_threshold
            ]
            if len(docs_and_similarities) == 0:
                logger.warning(
                    "使用相关性分数阈值 %s 没有检索到相关文档",
                    score_threshold,
                )
        return docs_and_similarities
    
    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """返回与嵌入向量最相似的文档
        
        Args:
            embedding: 要查找相似文档的嵌入
            k: 要返回的文档数量。默认为4
            **kwargs: 传递给搜索方法的参数
            
        Returns:
            与查询向量最相似的文档列表
        """
        raise NotImplementedError
    
    async def asimilarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """异步返回与嵌入向量最相似的文档"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.similarity_search_by_vector, embedding, k, **kwargs
        )
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """使用最大边际相关性返回选定的文档
        
        最大边际相关性优化与查询的相似性AND所选文档之间的多样性。
        
        Args:
            query: 要查找相似文档的文本
            k: 要返回的文档数量。默认为4
            fetch_k: 要获取传递给MMR算法的文档数量。默认为20
            lambda_mult: 0到1之间的数字，决定结果之间的多样性程度，
                0对应最大多样性，1对应最小多样性。默认为0.5
            **kwargs: 传递给搜索方法的参数
            
        Returns:
            通过最大边际相关性选择的文档列表
        """
        raise NotImplementedError
    
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """异步使用最大边际相关性返回选定的文档"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(),
            self.max_marginal_relevance_search,
            query,
            k,
            fetch_k,
            lambda_mult,
            **kwargs,
        )
    
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """使用最大边际相关性返回选定的文档"""
        raise NotImplementedError
    
    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """异步使用最大边际相关性返回选定的文档"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(),
            self.max_marginal_relevance_search_by_vector,
            embedding,
            k,
            fetch_k,
            lambda_mult,
            **kwargs,
        )
    
    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: "Embeddings",
        **kwargs: Any,
    ) -> "VectorStore":
        """从文档和嵌入返回初始化的VectorStore
        
        Args:
            documents: 要添加到向量存储的文档列表
            embedding: 要使用的嵌入函数
            kwargs: 其他关键字参数
            
        Returns:
            从文档和嵌入初始化的VectorStore
        """
        texts = [d.content for d in documents]
        metadatas = [d.metadata for d in documents]
        
        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
            # 如果至少有一个有效ID，则使用ID
            if any(ids):
                kwargs["ids"] = ids
        
        return cls.from_texts(texts, embedding, metadatas=metadatas, **kwargs)
    
    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: "Embeddings",
        **kwargs: Any,
    ) -> "VectorStore":
        """异步从文档和嵌入返回初始化的VectorStore"""
        texts = [d.content for d in documents]
        metadatas = [d.metadata for d in documents]
        
        if "ids" not in kwargs:
            ids = [doc.id for doc in documents]
            # 如果至少有一个有效ID，则使用ID
            if any(ids):
                kwargs["ids"] = ids
        
        return await cls.afrom_texts(texts, embedding, metadatas=metadatas, **kwargs)
    
    @classmethod
    @abstractmethod
    def from_texts(
        cls: type[VST],
        texts: list[str],
        embedding: "Embeddings",
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> VST:
        """从文本和嵌入返回初始化的VectorStore
        
        Args:
            texts: 要添加到向量存储的文本
            embedding: 要使用的嵌入函数
            metadatas: 与文本关联的可选元数据列表。默认为None
            ids: 与文本关联的可选ID列表
            kwargs: 其他关键字参数
            
        Returns:
            从文本和嵌入初始化的VectorStore
        """
        pass
    
    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: "Embeddings",
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "VectorStore":
        """异步从文本和嵌入返回初始化的VectorStore"""
        if ids is not None:
            kwargs["ids"] = ids
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), cls.from_texts, texts, embedding, metadatas, **kwargs
        )
    
    def _get_retriever_tags(self) -> list[str]:
        """获取检索器标签"""
        tags = [self.__class__.__name__]
        if hasattr(self, 'embeddings') and self.embeddings:
            tags.append(self.embeddings.__class__.__name__)
        return tags
    
    def as_retriever(self, **kwargs: Any) -> "VectorStoreRetriever":
        """从此VectorStore返回初始化的VectorStoreRetriever"""
        # 延迟导入以避免循环依赖
        from ...Retrieval.Retriever.Retriever_VectorStore import VectorStoreRetriever
        tags = kwargs.pop("tags", None) or [] + self._get_retriever_tags()
        return VectorStoreRetriever(vectorstore=self, tags=tags, **kwargs)




