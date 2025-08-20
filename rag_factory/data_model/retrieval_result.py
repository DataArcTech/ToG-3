from dataclasses import dataclass

from .document import Document

@dataclass
class RetrievalResult:
    """检索结果的数据类
    
    用于封装向量搜索的结果，包含文档、相关分数信息与排名。
    
    Attributes:
        document (Document): 搜索到的文档
        score (float): 相似性分数
        rank (int): 排名
    """
    document: Document
    score: float
    rank: int = 0