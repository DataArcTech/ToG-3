from abc import ABC, abstractmethod
from ..Retrieval import Document
import warnings

class RerankerBase(ABC):
    """
    Reranker 基类，所有 Reranker 应该继承此类并实现 rerank 方法。
    不建议直接实例化本类。
    
    使用方法：
        class MyReranker(RerankerBase):
            def rerank(self, query: str, documents: list[str], **kwargs) -> list[float]:
                # 实现具体的重排序逻辑
                ...
    """
    def __init__(self):
        if type(self) is RerankerBase:
            warnings.warn("RerankerBase 是抽象基类，不应直接实例化。请继承并实现 rerank 方法。", UserWarning)

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], **kwargs) -> list[Document]:
        """
        Rerank the documents based on the query.
        需要子类实现。
        """
        warnings.warn("调用了未实现的 rerank 方法。请在子类中实现该方法。", UserWarning)
        raise NotImplementedError("子类必须实现 rerank 方法。")
