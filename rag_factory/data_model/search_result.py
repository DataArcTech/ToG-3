from __future__ import annotations

from dataclasses import dataclass

from .document import Document


@dataclass
class SearchResult:
    """向量搜索结果数据结构
    
    用于封装向量搜索的结果，包含文档和相关分数信息。
    
    Attributes:
        document (Document): 搜索到的文档
        score (float): 相似性分数
        distance (float): 距离分数
    """
    document: Document
    score: float
    distance: float