# VectorStore/VectorStore_Faiss.py
import faiss
import pickle
import os
import uuid
import numpy as np
from typing import Any, Optional, Callable
from .VectorStoreBase import VectorStore, Document
from Embed import Embeddings
import asyncio
from concurrent.futures import ThreadPoolExecutor


def _mmr_select(
    docs_and_scores: list[tuple[Document, float]],
    embeddings: list[list[float]],
    query_embedding: list[float],
    k: int,
    lambda_mult: float = 0.5,
) -> list[Document]:
    """最大边际相关性选择算法"""
    if k >= len(docs_and_scores):
        return [doc for doc, _ in docs_and_scores]
    
    selected_indices = []
    selected_embeddings = []
    remaining_indices = list(range(len(docs_and_scores)))
    
    # 选择第一个文档（最相似的）
    first_idx = remaining_indices.pop(0)
    selected_indices.append(first_idx)
    selected_embeddings.append(embeddings[first_idx])
    
    # 选择剩余的k-1个文档
    for _ in range(k - 1):
        if not remaining_indices:
            break
            
        mmr_scores = []
        for idx in remaining_indices:
            # 计算与查询的相似性
            query_sim = np.dot(query_embedding, embeddings[idx])
            
            # 计算与已选择文档的最大相似性
            max_sim = 0
            for selected_emb in selected_embeddings:
                sim = np.dot(selected_emb, embeddings[idx])
                max_sim = max(max_sim, sim)
            
            # MMR分数
            mmr_score = lambda_mult * query_sim - (1 - lambda_mult) * max_sim
            mmr_scores.append((idx, mmr_score))
        
        # 选择MMR分数最高的文档
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        selected_embeddings.append(embeddings[best_idx])
        remaining_indices.remove(best_idx)
    
    return [docs_and_scores[idx][0] for idx in selected_indices]


class FaissVectorStore(VectorStore):
    """基于FAISS的向量存储实现"""
    
    def __init__(
        self, 
        embedding: Embeddings,
        index: Optional[faiss.Index] = None,
        index_type: str = "flat",
        metric: str = "cosine",
        normalize_L2: bool = False,
        **kwargs: Any
    ):
        """初始化FAISS向量存储
        
        Args:
            embedding: 嵌入函数
            index: 可选的现有FAISS索引
            index_type: 索引类型 ("flat", "ivf", "hnsw")
            metric: 距离度量 ("cosine", "l2", "ip")
            normalize_L2: 是否对向量进行L2归一化
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        
        self.embedding = embedding
        self.index_type = index_type
        self.metric = metric
        self.normalize_L2 = normalize_L2
        self.index = index
        
        # 存储文档和映射
        self.docstore: dict[str, Document] = {}
        self.index_to_docstore_id: dict[int, str] = {}
        
        # 如果没有提供索引，会在第一次添加文档时创建
        
    def _get_dimension(self) -> int:
        """获取嵌入维度"""
        if self.index is not None:
            return self.index.d
        
        # 通过嵌入一个测试文本来获取维度
        test_embedding = self.embedding.embed_query("test")
        return len(test_embedding)
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """创建FAISS索引"""
        if self.metric == "cosine":
            # 余弦相似度使用内积，需要归一化向量
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
        elif self.metric == "l2":
            if self.index_type == "flat":
                index = faiss.IndexFlatL2(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
        elif self.metric == "ip":
            if self.index_type == "flat":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            elif self.index_type == "hnsw":
                index = faiss.IndexHNSWFlat(dimension, 32)
                index.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
        else:
            raise ValueError(f"不支持的距离度量: {self.metric}")
            
        return index
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量"""
        if self.normalize_L2 or self.metric == "cosine":
            faiss.normalize_L2(vectors)
        return vectors
    
    def add_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """添加文本到向量存储"""
        if not texts:
            return []
        
        # 嵌入文本
        embeddings = self.embedding.embed_documents(texts)
        embeddings_np = np.array(embeddings).astype(np.float32)
        
        # 如果索引不存在，创建索引
        if self.index is None:
            dimension = embeddings_np.shape[1]
            self.index = self._create_index(dimension)
        
        # 归一化向量
        embeddings_np = self._normalize_vectors(embeddings_np)
        
        # 如果是IVF索引且未训练，则训练
        if (hasattr(self.index, 'is_trained') and 
            not self.index.is_trained and 
            len(embeddings) >= 100):
            self.index.train(embeddings_np)
        
        # 生成ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("ID数量必须与文本数量匹配")
        
        # 准备元数据
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("元数据数量必须与文本数量匹配")
        
        # 获取当前索引大小
        start_index = self.index.ntotal
        
        # 添加向量到索引
        self.index.add(embeddings_np)
        
        # 存储文档和映射
        for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
            doc = Document(content=text, metadata=metadata, id=doc_id)
            self.docstore[doc_id] = doc
            self.index_to_docstore_id[start_index + i] = doc_id
        
        return ids
    
    async def aadd_texts(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """异步添加文本"""
        return await asyncio.get_event_loop().run_in_executor(
            ThreadPoolExecutor(), self.add_texts, texts, metadatas, ids, **kwargs
        )
    
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """相似性搜索"""
        docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """带分数的相似性搜索"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 嵌入查询
        query_embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(query_embedding, k, **kwargs)
    
    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[Document]:
        """根据向量相似性搜索"""
        docs_and_scores = self.similarity_search_by_vector_with_score(embedding, k, **kwargs)
        return [doc for doc, _ in docs_and_scores]
    
    def similarity_search_by_vector_with_score(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """根据向量带分数的相似性搜索"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 准备查询向量
        query_vector = np.array([embedding]).astype(np.float32)
        query_vector = self._normalize_vectors(query_vector)
        
        # 搜索
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示无效结果
                continue
            
            doc_id = self.index_to_docstore_id[idx]
            doc = self.docstore[doc_id]
            results.append((doc, float(distance)))
        
        return results
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """最大边际相关性搜索"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 嵌入查询
        query_embedding = self.embedding.embed_query(query)
        
        # 获取fetch_k个候选文档
        docs_and_scores = self.similarity_search_by_vector_with_score(
            query_embedding, fetch_k, **kwargs
        )
        
        if not docs_and_scores:
            return []
        
        # 获取候选文档的嵌入
        candidate_embeddings = []
        for doc, _ in docs_and_scores:
            # 重新嵌入文档内容（实际应用中可能需要缓存）
            doc_embedding = self.embedding.embed_query(doc.content)
            candidate_embeddings.append(doc_embedding)
        
        # 归一化嵌入
        query_emb_norm = np.array(query_embedding)
        candidate_embs_norm = np.array(candidate_embeddings)
        
        if self.normalize_L2 or self.metric == "cosine":
            query_emb_norm = query_emb_norm / np.linalg.norm(query_emb_norm)
            candidate_embs_norm = candidate_embs_norm / np.linalg.norm(
                candidate_embs_norm, axis=1, keepdims=True
            )
        
        # MMR选择
        selected_docs = _mmr_select(
            docs_and_scores,
            candidate_embs_norm.tolist(),
            query_emb_norm.tolist(),
            k,
            lambda_mult,
        )
        
        return selected_docs
    
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """根据向量的最大边际相关性搜索"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 获取fetch_k个候选文档
        docs_and_scores = self.similarity_search_by_vector_with_score(
            embedding, fetch_k, **kwargs
        )
        
        if not docs_and_scores:
            return []
        
        # 获取候选文档的嵌入
        candidate_embeddings = []
        for doc, _ in docs_and_scores:
            doc_embedding = self.embedding.embed_query(doc.content)
            candidate_embeddings.append(doc_embedding)
        
        # 归一化嵌入
        query_emb_norm = np.array(embedding)
        candidate_embs_norm = np.array(candidate_embeddings)
        
        if self.normalize_L2 or self.metric == "cosine":
            query_emb_norm = query_emb_norm / np.linalg.norm(query_emb_norm)
            candidate_embs_norm = candidate_embs_norm / np.linalg.norm(
                candidate_embs_norm, axis=1, keepdims=True
            )
        
        # MMR选择
        selected_docs = _mmr_select(
            docs_and_scores,
            candidate_embs_norm.tolist(),
            query_emb_norm.tolist(),
            k,
            lambda_mult,
        )
        
        return selected_docs
    
    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        """删除文档（FAISS不支持直接删除，需要重建索引）"""
        if ids is None:
            # 删除所有
            self.docstore.clear()
            self.index_to_docstore_id.clear()
            if self.index is not None:
                self.index.reset()
            return True
        
        if not ids:
            return True
        
        # 检查要删除的ID是否存在
        for doc_id in ids:
            if doc_id not in self.docstore:
                return False
        
        # 获取要保留的文档
        remaining_docs = []
        remaining_texts = []
        remaining_metadatas = []
        remaining_ids = []
        
        for doc_id, doc in self.docstore.items():
            if doc_id not in ids:
                remaining_docs.append(doc)
                remaining_texts.append(doc.content)
                remaining_metadatas.append(doc.metadata)
                remaining_ids.append(doc_id)
        
        # 清空当前存储
        self.docstore.clear()
        self.index_to_docstore_id.clear()
        if self.index is not None:
            self.index.reset()
        
        # 重新添加保留的文档
        if remaining_texts:
            self.add_texts(remaining_texts, remaining_metadatas, ids=remaining_ids)
        
        return True
    
    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """根据ID获取文档"""
        return [self.docstore[doc_id] for doc_id in ids if doc_id in self.docstore]
    
    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """选择相关性评分函数"""
        if self.metric == "cosine" or self.normalize_L2:
            return self._cosine_relevance_score_fn
        elif self.metric == "l2":
            return self._euclidean_relevance_score_fn
        elif self.metric == "ip":
            return self._max_inner_product_relevance_score_fn
        else:
            raise ValueError(f"不支持的度量类型: {self.metric}")
    
    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """保存到本地文件夹"""
        os.makedirs(folder_path, exist_ok=True)
        
        # 保存FAISS索引
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(folder_path, f"{index_name}.faiss"))
        
        # 保存其他数据
        data = {
            "docstore": self.docstore,
            "index_to_docstore_id": self.index_to_docstore_id,
            "index_type": self.index_type,
            "metric": self.metric,
            "normalize_L2": self.normalize_L2,
        }
        
        with open(os.path.join(folder_path, f"{index_name}.pkl"), "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        index_name: str = "index",
        **kwargs: Any,
    ) -> "FaissVectorStore":
        """从本地文件夹加载"""
        # 加载其他数据
        with open(os.path.join(folder_path, f"{index_name}.pkl"), "rb") as f:
            data = pickle.load(f)
        
        # 加载FAISS索引
        index_path = os.path.join(folder_path, f"{index_name}.faiss")
        index = faiss.read_index(index_path) if os.path.exists(index_path) else None
        
        # 创建实例
        instance = cls(
            embedding=embeddings,
            index=index,
            index_type=data["index_type"],
            metric=data["metric"],
            normalize_L2=data["normalize_L2"],
            **kwargs,
        )
        
        instance.docstore = data["docstore"]
        instance.index_to_docstore_id = data["index_to_docstore_id"]
        
        return instance
    
    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "FaissVectorStore":
        """从文本创建FAISS向量存储"""
        faiss_vs = cls(embedding=embedding, **kwargs)
        faiss_vs.add_texts(texts, metadatas=metadatas, ids=ids)
        return faiss_vs
    
    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: Optional[list[dict]] = None,
        *,
        ids: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> "FaissVectorStore":
        """异步从文本创建FAISS向量存储"""
        faiss_vs = cls(embedding=embedding, **kwargs)
        await faiss_vs.aadd_texts(texts, metadatas=metadatas, ids=ids)
        return faiss_vs

