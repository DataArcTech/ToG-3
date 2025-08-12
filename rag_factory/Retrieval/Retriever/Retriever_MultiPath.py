from typing import List, Dict, Any, Optional, Union

from ..RetrieverBase import Document
from ..RetrieverBase import BaseRetriever
from ..utils.Fusion import FusionMethod, RRFusion, RetrievalResult


class MultiPathRetriever(BaseRetriever):
    """
    多路检索器
    
    该类实现了多路检索功能，可以同时使用多个检索器进行文档检索，
    并通过指定的融合方法将多个检索器的结果进行合并和排序。
    
    Attributes:
        retrievers (List[BaseRetriever]): 检索器列表，每个检索器需要实现retrieve方法
        fusion_method (FusionMethod): 融合方法，用于合并多个检索器的结果
        top_k_per_retriever (int): 每个检索器返回的结果数量
    """
    
    def __init__(self, 
                 retrievers: List[BaseRetriever],
                 fusion_method: Optional[FusionMethod] = None,
                 top_k_per_retriever: int = 50):
        """
        初始化多路检索器
        
        Args:
            retrievers (List[BaseRetriever]): 检索器列表，每个检索器需要实现retrieve方法
            fusion_method (Optional[FusionMethod]): 融合方法，默认为RRF (Reciprocal Rank Fusion)
            top_k_per_retriever (int): 每个检索器返回的结果数量，默认为50
        """
        self.retrievers = retrievers
        self.fusion_method = fusion_method or RRFusion()
        self.top_k_per_retriever = top_k_per_retriever
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        获取与查询相关的文档
        
        该方法会调用所有配置的检索器，获取每个检索器的检索结果，
        然后使用指定的融合方法将所有结果进行合并和排序。
        
        Args:
            query (str): 查询字符串
            **kwargs (Any): 其他参数，包括top_k等
            
        Returns:
            List[Document]: 融合后的相关文档列表，按相关性排序
            
        Note:
            - 每个检索器的结果会被转换为RetrievalResult格式
            - 支持多种输入格式：Document对象、字典格式、字符串等
            - 融合后的结果会将score和rank信息保存在Document的metadata中
        """
        top_k = kwargs.get('top_k', 10)
        
        # 从每个检索器获取结果
        all_results = []
        for retriever in self.retrievers:
            try:
                # 使用BaseRetriever的invoke方法
                documents = retriever.invoke(query, **{**kwargs, 'k': self.top_k_per_retriever})
                
                # 转换为RetrievalResult格式
                formatted_results = []
                for i, doc in enumerate(documents):
                    if isinstance(doc, Document):
                        # 如果是Document对象
                        retrieval_result = RetrievalResult(
                            document=doc,
                            score=getattr(doc, 'score', 1.0),
                            rank=i + 1
                        )
                    elif isinstance(doc, dict):
                        # 如果返回的是字典格式，需要转换为Document对象
                        content = doc.get('content', '')
                        metadata = doc.get('metadata', {})
                        doc_id = doc.get('id')
                        
                        document = Document(
                            content=content,
                            metadata=metadata,
                            id=doc_id
                        )
                        
                        retrieval_result = RetrievalResult(
                            document=document,
                            score=doc.get('score', 1.0),
                            rank=i + 1
                        )
                    else:
                        # 如果是字符串或其他格式，转换为Document对象
                        document = Document(
                            content=str(doc),
                            metadata={},
                            id=None
                        )
                        
                        retrieval_result = RetrievalResult(
                            document=document,
                            score=1.0,
                            rank=i + 1
                        )
                    formatted_results.append(retrieval_result)
                
                all_results.append(formatted_results)
                
            except Exception as e:
                print(f"检索器 {type(retriever).__name__} 执行失败: {e}")
                all_results.append([])
        
        # 使用融合方法合并结果
        if not all_results or all(len(results) == 0 for results in all_results):
            return []
        
        fused_results = self.fusion_method.fuse(all_results, top_k)
        
        # 转换回Document格式
        documents = []
        for result in fused_results:
            doc = result.document
            # 将score和rank添加到metadata中以便保留
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata['score'] = result.score
            doc.metadata['rank'] = result.rank
            documents.append(doc)
        
        return documents

    
    def add_retriever(self, retriever: BaseRetriever):
        """
        添加新的检索器到多路检索器中
        
        Args:
            retriever (BaseRetriever): 要添加的检索器实例
        """
        self.retrievers.append(retriever)
    
    def remove_retriever(self, name: str):
        """
        移除指定名称的检索器
        
        Args:
            name (str): 要移除的检索器的类名
            
        Note:
            该方法通过比较检索器的类名来识别要移除的检索器
        """
        for i, retriever in enumerate(self.retrievers):
            if hasattr(retriever, '__class__') and retriever.__class__.__name__ == name:
                self.retrievers.pop(i)
                break
    
    def set_fusion_method(self, fusion_method: FusionMethod):
        """
        设置融合方法
        
        Args:
            fusion_method (FusionMethod): 新的融合方法实例
        """
        self.fusion_method = fusion_method
