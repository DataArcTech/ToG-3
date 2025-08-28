"""
多次提取实体、关系，避免遗漏。
多轮提取，后续清洗，避免出现错误。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Coroutine
import asyncio
import copy 

from rag_factory.llms.llm_base import LLMBase
from rag_factory.documents.schema import Document


class GraphExtractorBase(ABC):
    """
    图提取器基类，定义了所有图提取器的通用接口和功能。
    
    核心功能：
    - 多轮图结构提取
    - 可选的图结构清洗
    - 并发控制和批量处理
    - 同步/异步调用接口
    - 进度显示支持
    
    子类需要实现：
    - _aextract: 异步提取单个文档的图结构
    - _aclean: 异步清洗单个文档的图结构（可选）
    """
    
    def __init__(
        self,
        llm: LLMBase,
        max_concurrent: int = 100,
        enable_cleaning: bool = True,
        max_rounds: int = 3
    ) -> None:
        """
        初始化图提取器基类
        
        Args:
            llm: LLM实例
            max_concurrent: 最大并发数
            enable_cleaning: 是否启用清洗功能
            max_rounds: 最大提取轮次
        """
        self.llm = llm
        self.max_concurrent = max_concurrent
        self.enable_cleaning = enable_cleaning
        self.max_rounds = max_rounds


    @abstractmethod
    async def _aextract(
        self, 
        document: Document, 
        semaphore: asyncio.Semaphore, 
        history: Dict[str, List]
    ) -> Document:
        """
        异步单次抽取文档的图结构（抽象方法，子类必须实现）
        
        Args:
            document: 待处理的文档
            semaphore: 信号量，用于控制并发
            history: 抽取历史，包含已提取的实体和关系
            
        Returns:
            Document: 处理后的文档，metadata中包含【本轮】新提取的图结构
        """
        pass


    async def _aclean(
        self, 
        document: Document, 
        semaphore: asyncio.Semaphore
    ) -> Document:
        """
        异步清洗单个文档的图结构（可选实现，默认不做处理）
        
        子类可以重写此方法来实现自定义的清洗逻辑
        
        Args:
            document: 待清洗的文档（已包含提取的图结构）
            semaphore: 信号量，用于控制并发
            
        Returns:
            Document: 清洗后的文档
        """
        return document

    def is_extraction_complete(self, document: Document) -> bool:
        """
        判断当前抽取结果是否足够完整，用于提前终止多轮提取。默认返回True，子类可重写。
        
        Args:
            document: 当前处理的文档
            
        Returns:
            bool: 是否完成提取
        """
        return True

    async def _aprocess_document(
        self, 
        document: Document, 
        semaphore: asyncio.Semaphore
    ) -> Document:
        """
        处理单个文档：多轮提取，后续清洗（如果启用）
        """
        current_doc = copy.deepcopy(document)
        history = self._init_extraction_history(current_doc)
        
        # 多轮提取
        for _ in range(self.max_rounds):
            # _aextract应返回一个只包含本轮新结果的文档
            extracted_doc = await self._aextract(current_doc, semaphore, history)
            
            # 将新结果合并到历史记录中
            history = self._merge_extraction_history(extracted_doc, history)
            
            # 更新文档的元数据，以便下一轮可以看到完整的历史
            current_doc.metadata['entities'] = history['entities']
            current_doc.metadata['relations'] = history['relations']
            
            # 检查是否可以提前结束
            if self.is_extraction_complete(extracted_doc):
                break
        
        if self.enable_cleaning:
            cleaned_doc = await self._aclean(current_doc, semaphore)
            return cleaned_doc
        else:
            return current_doc

    def _init_extraction_history(self, document: Document) -> Dict[str, List]:
        """
        初始化抽取历史，从文档metadata中获取已抽取的实体和关系
        """
        if getattr(document, 'metadata', None) is None:
            document.metadata = {}
        metadata = document.metadata
        entities = metadata.get('entities', [])
        relations = metadata.get('relations', [])
        return {'entities': list(entities), 'relations': list(relations)}

    def _merge_extraction_history(
        self, 
        document: Document, 
        history: Dict[str, List]
    ) -> Dict[str, List]:
        """
        合并当前抽取结果与历史记录，自动去重。
        合并策略：对于ID相同的实体或关系，后提取的结果会覆盖旧的结果（Upsert）。
        """
        metadata = getattr(document, 'metadata', {})
        new_entities = metadata.get('entities', [])
        new_relations = metadata.get('relations', [])
        
        # 合并实体，使用字典来处理更新和去重
        entity_map = {e['id']: e for e in history['entities']}
        entity_map.update({e['id']: e for e in new_entities})
        
        # 合并关系
        relation_map = {r['id']: r for r in history['relations']}
        relation_map.update({r['id']: r for r in new_relations})
        
        return {
            'entities': list(entity_map.values()),
            'relations': list(relation_map.values())
        }

    async def _arun_tasks(
        self,
        tasks: List[Coroutine],
        description: str,
        show_progress: bool = False
    ) -> List[Any]:
        """
        一个通用的异步任务执行器，支持并发控制和tqdm进度条。
        """
        if show_progress:
            print(description)
            try:
                from tqdm.asyncio import tqdm_asyncio
                return await tqdm_asyncio.gather(*tasks, desc=description.split('...')[0])
            except ImportError:
                print("tqdm not installed. Please run `pip install tqdm` to show progress bars.")
                return await asyncio.gather(*tasks)
        else:
            return await asyncio.gather(*tasks)

    # ==================== 公共调用接口 ====================

    async def acall(
        self, 
        documents: List[Document], 
        show_progress: bool = False,
        enable_cleaning: Optional[bool] = None
    ) -> List[Document]:
        """
        异步提取和清洗所有文档的图结构
        """
        original_cleaning = self.enable_cleaning
        if enable_cleaning is not None:
            self.enable_cleaning = enable_cleaning
        
        try:    
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self._aprocess_document(doc, semaphore) for doc in documents]
            
            operation_desc = (
                f"开始从{len(documents)}个Chunk中提取并清洗图结构..." if self.enable_cleaning 
                else f"开始从{len(documents)}个Chunk中提取图结构..."
            )
            results = await self._arun_tasks(tasks, operation_desc, show_progress)
            
            if show_progress:
                completion_msg = (
                    "图结构提取和清洗完成" if self.enable_cleaning 
                    else "图结构提取完成"
                )
                print(completion_msg)
            return results
        finally:
            self.enable_cleaning = original_cleaning

    def __call__(
        self, 
        documents: List[Document], 
        show_progress: bool = False,
        enable_cleaning: Optional[bool] = None
    ) -> List[Document]:
        """同步接口：提取和清洗图结构"""
        return asyncio.run(
            self.acall(documents, show_progress=show_progress, enable_cleaning=enable_cleaning)
        )

    # ==================== 专用接口 ====================

    async def aextract_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """仅执行提取操作，跳过清洗"""
        return await self.acall(documents, show_progress=show_progress, enable_cleaning=False)

    def extract_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """同步接口：仅执行提取操作，跳过清洗"""
        return self(documents, show_progress=show_progress, enable_cleaning=False)

    async def aclean_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """
        仅执行清洗操作（假设文档已包含提取的图结构）
        """
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._aclean(doc, semaphore) for doc in documents]
        
        description = f"开始清洗{len(documents)}个文档的图结构..."
        results = await self._arun_tasks(tasks, description, show_progress)

        if show_progress:
            print("图结构清洗完成")
        return results

    def clean_only(
        self, 
        documents: List[Document], 
        show_progress: bool = False
    ) -> List[Document]:
        """同步接口：仅执行清洗操作"""
        return asyncio.run(self.aclean_only(documents, show_progress=show_progress))

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__