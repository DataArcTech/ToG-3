from abc import ABC, abstractmethod
from rag_factory.llm.base import LLMBase
from typing import Any
from rag_factory.data_model.document import Document


__all__ = ["GraphExtractorBase", "GraphExtractor"]



class GraphExtractorBase(ABC):
    """
    图提取器基类，定义了所有图提取器的通用接口和功能。
    
    子类需要实现：
    - _aextract: 异步提取单个文档的图结构
    - class_name: 返回类名的类方法
    
    提供的通用功能：
    - 并发控制
    - 批量处理
    - 进度显示
    - 同步/异步调用接口
    """
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        response_format: Any = None,
        max_concurrent: int = 100
    ) -> None:
        """
        初始化图提取器基类
        
        Args:
            llm: 大语言模型实例
            extract_prompt: 提取提示模板
            response_format: Pydantic 响应格式
            max_concurrent: 最大并发数
        """
        self.llm = llm
        self.extract_prompt = extract_prompt
        self.response_format = response_format
        self.max_concurrent = max_concurrent

    @abstractmethod
    async def _aextract(self, document: Document, semaphore) -> Document:
        """
        异步提取单个文档的图结构（抽象方法，子类必须实现）
        
        Args:
            document: 待处理的文档
            semaphore: 信号量，用于控制并发
            
        Returns:
            Document: 处理后的文档，metadata中包含提取的图结构
        """
        pass

    async def acall(self, documents: list[Document], show_progress: bool = False) -> list[Document]:
        """异步提取所有文档的图结构"""
        import asyncio
        
        # 创建信号量来控制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 创建任务列表
        tasks = [self._aextract(document, semaphore) for document in documents]
        
        if show_progress:
            print(f"开始从{len(documents)}个Chunk中提取图结构...")
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="提取图结构")
        else:
            results = await asyncio.gather(*tasks)
        
        if show_progress:
            print("图结构提取完成")
        
        return results

    def __call__(self, documents: list[Document], show_progress: bool = False) -> list[Document]:
        """同步接口：提取图结构"""
        import asyncio
        return asyncio.run(self.acall(documents, show_progress=show_progress))

    @classmethod
    @abstractmethod
    def class_name(cls) -> str:
        """返回类名（抽象方法，子类必须实现）"""
        pass
