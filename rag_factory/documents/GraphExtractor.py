from abc import ABC, abstractmethod
from rag_factory.llms.llm_base import LLMBase
from typing import Callable, Dict, Set, Optional, Tuple
from datetime import datetime
import json
import os
import hashlib
from rag_factory.documents.Prompt import KG_TRIPLES_PROMPT, HYPERRAG_EXTRACTION_PROMPT
from rag_factory.documents.schema import Document
from rag_factory.Store.GraphStore.GraphNode import EntityNode, EventNode, MentionNode, Relation

KG_NODES_KEY = "entities"
KG_RELATIONS_KEY = "relations"
KG_EVENTS_KEY = "events"
KG_MENTIONS_KEY = "mentions"


__all__ = ["GraphExtractorBase", "GraphExtractor", "HyperRAGGraphExtractor", "IncrementalGraphProcessor", "GraphDataCache"]


class GraphDataCache:
    """
    核心功能：
    1. 本地文件缓存，避免重复计算
    2. 内存+磁盘双重缓存
    3. 增量更新支持
    4. 快速查询接口
    """
    
    def __init__(self, cache_dir: str = "./graph_cache"):
        """
        初始化图数据缓存
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = cache_dir
        self.entities_file = os.path.join(cache_dir, "entities.json")
        self.relations_file = os.path.join(cache_dir, "relations.json")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.processed_chunks_file = os.path.join(cache_dir, "processed_chunks.json")
        
        # 内存缓存
        self._entities_cache: Dict[str, dict] = {}  # name -> entity
        self._relations_cache: Dict[str, list] = {}  # (head, tail, type) -> relation
        self._processed_chunks: Set[str] = set()  # 已处理的chunk ID集合
        self._last_update_time: Optional[datetime] = None
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载现有缓存
        self._load_cache()
    
    def _load_cache(self):
        """从本地文件加载缓存"""
        try:
            # 加载元数据
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    last_update_str = metadata.get('last_update_time')
                    if last_update_str:
                        self._last_update_time = datetime.fromisoformat(last_update_str)
            
            # 加载实体缓存
            if os.path.exists(self.entities_file):
                with open(self.entities_file, 'r', encoding='utf-8') as f:
                    self._entities_cache = json.load(f)
            
            # 加载关系缓存
            if os.path.exists(self.relations_file):
                with open(self.relations_file, 'r', encoding='utf-8') as f:
                    relations_data = json.load(f)
                    # 转换为便于查询的格式
                    for rel in relations_data:
                        key = (rel.get('head_id', ''), rel.get('tail_id', ''), rel.get('label', ''))
                        if key not in self._relations_cache:
                            self._relations_cache[key] = []
                        self._relations_cache[key].append(rel)
            
            # 加载已处理的chunk信息
            if os.path.exists(self.processed_chunks_file):
                with open(self.processed_chunks_file, 'r', encoding='utf-8') as f:
                    processed_chunks_data = json.load(f)
                    self._processed_chunks = set(processed_chunks_data.get('processed_chunks', []))
            
            if self._entities_cache or self._relations_cache or self._processed_chunks:
                print(f"✅ 缓存加载完成：{len(self._entities_cache)}个实体，{len(self._relations_cache)}个关系类型，{len(self._processed_chunks)}个已处理chunk")
            
        except Exception as e:
            print(f"⚠️ 加载缓存失败：{e}")
    
    def _save_cache(self):
        """保存缓存到本地文件"""
        try:
            print(f"💾 正在保存缓存到 {self.cache_dir}...")
            print(f"  📊 实体数量: {len(self._entities_cache)}")
            print(f"  🔗 关系数量: {sum(len(rels) for rels in self._relations_cache.values())}")
            print(f"  📄 已处理chunk数量: {len(self._processed_chunks)}")
            
            # 保存实体
            with open(self.entities_file, 'w', encoding='utf-8') as f:
                json.dump(self._entities_cache, f, indent=2, ensure_ascii=False, default=str)
            print(f"  ✅ 实体文件已保存: {self.entities_file}")
            
            # 保存关系（转换为列表格式）
            relations_list = []
            for relations in self._relations_cache.values():
                relations_list.extend(relations)
            
            with open(self.relations_file, 'w', encoding='utf-8') as f:
                json.dump(relations_list, f, indent=2, ensure_ascii=False, default=str)
            print(f"  ✅ 关系文件已保存: {self.relations_file}")
            
            # 保存已处理的chunk信息
            processed_chunks_data = {
                'processed_chunks': list(self._processed_chunks),
                'chunk_count': len(self._processed_chunks),
                'last_update_time': self._last_update_time.isoformat() if self._last_update_time else None
            }
            
            with open(self.processed_chunks_file, 'w', encoding='utf-8') as f:
                json.dump(processed_chunks_data, f, indent=2, ensure_ascii=False)
            print(f"  ✅ 已处理chunk文件已保存: {self.processed_chunks_file}")
            
            # 保存元数据
            metadata = {
                'last_update_time': self._last_update_time.isoformat() if self._last_update_time else None,
                'entity_count': len(self._entities_cache),
                'relation_count': len(relations_list),
                'processed_chunk_count': len(self._processed_chunks),
                'save_time': datetime.now().isoformat()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"  ✅ 元数据文件已保存: {self.metadata_file}")
                
        except Exception as e:
            print(f"❌ 保存缓存失败：{e}")
            import traceback
            traceback.print_exc()
    
    def load_data(self, entities: list = None, relations: list = None):
        """
        手动加载数据到缓存
        
        Args:
            entities: 实体列表，每个实体应包含name字段
            relations: 关系列表，每个关系应包含head_id, tail_id, label字段
        """
        if entities:
            print(f"📥 加载 {len(entities)} 个实体到缓存...")
            for entity in entities:
                name = entity.get('name')
                if name:
                    entity['update_time'] = datetime.now().isoformat()
                    self._entities_cache[name] = entity
        
        if relations:
            print(f"📥 加载 {len(relations)} 个关系到缓存...")
            for relation in relations:
                head = relation.get('head_id', '')
                tail = relation.get('tail_id', '')
                rel_type = relation.get('label', '')
                
                key = (head, tail, rel_type)
                if key not in self._relations_cache:
                    self._relations_cache[key] = []
                
                relation['update_time'] = datetime.now().isoformat()
                self._relations_cache[key].append(relation)
        
        # 更新时间戳并保存
        self._last_update_time = datetime.now()
        self._save_cache()
        
        print(f"✅ 数据加载完成：总计 {len(self._entities_cache)} 个实体，{sum(len(rels) for rels in self._relations_cache.values())} 个关系")
    
    def get_entity(self, name: str) -> Optional[dict]:
        """获取实体"""
        return self._entities_cache.get(name)
    
    def get_all_entities(self) -> Dict[str, dict]:
        """获取所有实体"""
        return self._entities_cache.copy()
    
    def get_relations(self, head: str = None, tail: str = None, rel_type: str = None) -> list:
        """获取关系"""
        if head and tail and rel_type:
            # 精确查询
            key = (head, tail, rel_type)
            return self._relations_cache.get(key, [])
        
        # 模糊查询
        results = []
        for (h, t, r), relations in self._relations_cache.items():
            if (head is None or h == head) and \
               (tail is None or t == tail) and \
               (rel_type is None or r == rel_type):
                results.extend(relations)
        
        return results
    
    def get_all_relations(self) -> list:
        """获取所有关系"""
        relations = []
        for rel_list in self._relations_cache.values():
            relations.extend(rel_list)
        return relations
    
    def add_entity(self, entity: dict, save_immediately: bool = True):
        """添加实体到缓存"""
        name = entity.get('name')
        if name:
            entity['update_time'] = datetime.now().isoformat()
            self._entities_cache[name] = entity
            # 可选择立即保存到磁盘
            if save_immediately:
                self._save_cache()
    
    def add_relation(self, relation: dict, save_immediately: bool = True):
        """添加关系到缓存"""
        head = relation.get('head_id', '')
        tail = relation.get('tail_id', '')
        rel_type = relation.get('label', '')
        
        key = (head, tail, rel_type)
        if key not in self._relations_cache:
            self._relations_cache[key] = []
        
        # 避免重复
        for existing_rel in self._relations_cache[key]:
            if existing_rel.get('description') == relation.get('description'):
                return  # 已存在相同关系
        
        relation['update_time'] = datetime.now().isoformat()
        self._relations_cache[key].append(relation)
        # 可选择立即保存到磁盘
        if save_immediately:
            self._save_cache()
    
    def clear_cache(self):
        """清空缓存"""
        self._entities_cache.clear()
        self._relations_cache.clear()
        self._processed_chunks.clear()
        self._last_update_time = None
        self._save_cache()
        print("🗑️ 缓存已清空")
    
    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        total_relations = sum(len(rels) for rels in self._relations_cache.values())
        
        return {
            'entity_count': len(self._entities_cache),
            'relation_count': total_relations,
            'processed_chunk_count': len(self._processed_chunks),
            'last_update_time': self._last_update_time.isoformat() if self._last_update_time else None,
            'cache_size_mb': self._estimate_cache_size()
        }
    
    def _estimate_cache_size(self) -> float:
        """估算缓存大小（MB）"""
        try:
            total_size = 0
            for file_path in [self.entities_file, self.relations_file, self.metadata_file, self.processed_chunks_file]:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            return round(total_size / 1024 / 1024, 2)
        except:
            return 0.0
    
    def is_chunk_processed(self, chunk_id: str) -> bool:
        """
        检查指定chunk是否已经处理过
        
        Args:
            chunk_id: chunk的唯一标识符
            
        Returns:
            bool: 如果已处理返回True，否则返回False
        """
        return chunk_id in self._processed_chunks
    
    def mark_chunk_processed(self, chunk_id: str, save_immediately: bool = True):
        """
        标记chunk为已处理
        
        Args:
            chunk_id: chunk的唯一标识符
            save_immediately: 是否立即保存到磁盘
        """
        if chunk_id not in self._processed_chunks:
            self._processed_chunks.add(chunk_id)
            self._last_update_time = datetime.now()
            
            if save_immediately:
                self._save_cache()
    
    def mark_chunks_processed(self, chunk_ids: list[str], save_immediately: bool = True):
        """
        批量标记chunks为已处理
        
        Args:
            chunk_ids: chunk ID列表
            save_immediately: 是否立即保存到磁盘
        """
        new_chunks = [cid for cid in chunk_ids if cid not in self._processed_chunks]
        if new_chunks:
            self._processed_chunks.update(new_chunks)
            self._last_update_time = datetime.now()
            
            if save_immediately:
                self._save_cache()
            
            print(f"📄 标记了 {len(new_chunks)} 个新chunk为已处理")
    
    def get_processed_chunks(self) -> Set[str]:
        """
        获取所有已处理的chunk ID
        
        Returns:
            Set[str]: 已处理的chunk ID集合
        """
        return self._processed_chunks.copy()
    
    def filter_unprocessed_chunks(self, documents: list) -> list:
        """
        过滤出未处理的文档
        
        Args:
            documents: 文档列表，每个文档应包含metadata.chunk_id
            
        Returns:
            list: 未处理的文档列表
        """
        unprocessed_docs = []
        for doc in documents:
            chunk_id = None
            
            # 尝试从不同来源获取chunk_id
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                chunk_id = doc.metadata.get('chunk_id')
            elif hasattr(doc, 'chunk_id'):
                chunk_id = doc.chunk_id
            elif isinstance(doc, dict):
                chunk_id = doc.get('chunk_id') or doc.get('metadata', {}).get('chunk_id')
            
            if chunk_id and not self.is_chunk_processed(chunk_id):
                unprocessed_docs.append(doc)
            elif not chunk_id:
                # 如果没有chunk_id，则认为需要处理
                unprocessed_docs.append(doc)
        
        print(f"🔍 从 {len(documents)} 个文档中筛选出 {len(unprocessed_docs)} 个未处理的文档")
        return unprocessed_docs
    
    def remove_processed_chunk(self, chunk_id: str, save_immediately: bool = True):
        """
        移除已处理的chunk标记（用于重新处理）
        
        Args:
            chunk_id: chunk的唯一标识符
            save_immediately: 是否立即保存到磁盘
        """
        if chunk_id in self._processed_chunks:
            self._processed_chunks.remove(chunk_id)
            
            if save_immediately:
                self._save_cache()
            
            print(f"🔄 已移除chunk {chunk_id} 的处理标记，将重新处理")


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
        parse_fn: Callable = None,
        max_concurrent: int = 100,
        enable_incremental: bool = False,
        cache_dir: str = None
    ) -> None:
        """
        初始化图提取器基类
        
        Args:
            llm: 大语言模型实例
            extract_prompt: 提取提示模板
            parse_fn: 解析函数
            max_concurrent: 最大并发数
            enable_incremental: 是否启用增量处理
            cache_dir: 缓存目录路径，如果不提供则不启用持久化
        """
        self.llm = llm
        self.extract_prompt = extract_prompt
        self.parse_fn = parse_fn
        self.max_concurrent = max_concurrent
        self.enable_incremental = enable_incremental
        self.cache_dir = cache_dir
        
        # 使用GraphDataCache来处理缓存
        if self.cache_dir and self.enable_incremental:
            self.cache = GraphDataCache(cache_dir)
        else:
            self.cache = None

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

    def _filter_documents_for_incremental(self, documents: list[Document]) -> list[Document]:
        """
        过滤需要增量处理的文档
        
        Args:
            documents: 输入文档列表
            
        Returns:
            需要处理的文档列表
        """
        if not self.enable_incremental or not self.cache:
            return documents
        
        # 使用GraphDataCache的filter_unprocessed_chunks方法
        return self.cache.filter_unprocessed_chunks(documents)

    def _update_incremental_state(self, documents: list[Document]):
        """
        更新增量处理状态
        
        Args:
            documents: 已处理的文档列表
        """
        if not self.enable_incremental or not self.cache:
            return
            
        # 使用GraphDataCache标记chunks为已处理
        chunk_ids = []
        for doc in documents:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id:
                chunk_ids.append(chunk_id)
        
        if chunk_ids:
            self.cache.mark_chunks_processed(chunk_ids, save_immediately=False)

    async def acall(self, documents: list[Document], show_progress: bool = False) -> list[Document]:
        """异步提取所有文档的图结构"""
        import asyncio
        
        # 如果启用增量处理，先过滤文档
        docs_to_process = self._filter_documents_for_incremental(documents)
        
        if self.enable_incremental and show_progress:
            print(f"增量处理：从{len(documents)}个文档中筛选出{len(docs_to_process)}个需要处理")
        
        if not docs_to_process:
            if show_progress:
                print("无需处理新文档")
            return documents  # 返回原始文档列表
        
        # 创建信号量来控制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 创建任务列表
        tasks = [self._aextract(document, semaphore) for document in docs_to_process]
        
        if show_progress:
            print(f"开始从{len(docs_to_process)}个Chunk中提取图结构...")
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="提取图结构")
        else:
            results = await asyncio.gather(*tasks)
        
        # 更新增量处理状态
        self._update_incremental_state(results)
        
        # 保存缓存状态
        if self.enable_incremental and self.cache:
            self.cache._save_cache()
        
        if show_progress:
            print("图结构提取完成")
        
        # 如果是增量处理，需要合并结果
        if self.enable_incremental and len(docs_to_process) < len(documents):
            # 创建结果映射
            processed_map = {doc.metadata.get("chunk_id"): doc for doc in results}
            
            # 合并结果：保留原文档，更新已处理的部分
            final_results = []
            for original_doc in documents:
                chunk_id = original_doc.metadata.get("chunk_id")
                if chunk_id in processed_map:
                    final_results.append(processed_map[chunk_id])
                else:
                    final_results.append(original_doc)
            return final_results
        
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


class GraphExtractor(GraphExtractorBase):
    """
    GraphExtractor is a class that extracts triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.
    """
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        parse_fn: Callable = None,
        max_concurrent: int = 100,
        enable_incremental: bool = False,
        cache_dir: str = None
    ) -> None:
        extract_prompt = extract_prompt if extract_prompt is not None else KG_TRIPLES_PROMPT
        super().__init__(llm, extract_prompt, parse_fn, max_concurrent, enable_incremental, cache_dir)


    async def _aextract(self, document: Document, semaphore) -> dict:
        """从documents中异步提取实体、三元组"""
        async with semaphore:
            content = document.content
            if not content:
                return document
                
            try:
                prompt = self.extract_prompt.format(
                    text=content
                )
                messages = [
                    {"role": "user", "content": prompt}
                ]
                llm_response = await self.llm.achat(messages, response_format={"type": "json_object"})
                if self.parse_fn is None:
                    print("错误：parse_fn为None！")
                    entities, relationships = [], []
                else:
                    entities, relationships = self.parse_fn(llm_response)
                    print(f"GraphExtractor parse_fn返回: entities={len(entities)}, relationships={len(relationships)}")
            except Exception as e:
                print(f"提取三元组时出错: {e}")
                entities = []
                relationships = []
            
            # 获取已有实体、关系（从document中已提取三元组，需要更新entity、relation）
            existing_nodes = document.metadata.pop(KG_NODES_KEY, [])
            existing_relations = document.metadata.pop(KG_RELATIONS_KEY, [])
            entity_metadata = document.metadata.copy()

            # 构建entity
            for entity, entity_type, description in entities:
                entity_metadata["entity_description"] = description
                entity_metadata["source_chunk_id"] = document.metadata["chunk_id"]
                # 生成唯一的实体ID
                entity_id = f"Entity_{hash(entity) % 1000000:06d}"
                entity_node = EntityNode(
                    id_=entity_id,
                    name=entity, 
                    label=entity_type, 
                    metadatas=entity_metadata
                )
                existing_nodes.append(entity_node)

            relation_metadata = document.metadata.copy()

            # 构建relations
            for triple in relationships:
                head, tail, rel, description = triple
                relation_metadata["relationship_description"] = description
                rel_node = Relation(
                    label=rel,
                    head_id=head,
                    tail_id=tail,
                    metadatas=relation_metadata
                )
                existing_relations.append(rel_node)

            document.metadata[KG_NODES_KEY] = existing_nodes
            document.metadata[KG_RELATIONS_KEY] = existing_relations
            return document




    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"


class HyperRAGGraphExtractor(GraphExtractorBase):
    """
    HyperRAGGraphExtractor用于提取完整的图结构，包括事件、mention、实体和关系。
    
    使用专门的提示模板来提取层次化的图结构。
    """
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        parse_fn: Callable = None,
        max_concurrent: int = 100,
        enable_incremental: bool = False,
        cache_dir: str = None
    ) -> None:
        # 调用父类初始化
        extract_prompt = extract_prompt if extract_prompt is not None else HYPERRAG_EXTRACTION_PROMPT
        super().__init__(llm, extract_prompt, parse_fn, max_concurrent, enable_incremental, cache_dir)



    async def _aextract(self, document: Document, semaphore) -> Document:
        """从documents中异步提取完整的图结构"""
        async with semaphore:
            content = document.content
            if not content:
                return document
                
            try:
                chunk_id = document.metadata.get("chunk_id", f"chunk_{hash(content)}")
                
                prompt = self.extract_prompt.format(text=content)
                messages = [{"role": "user", "content": prompt}]
                llm_response = await self.llm.achat(messages, response_format={"type": "json_object"})
                print(f"HyperRAGGraphExtractor 提取结果: {llm_response}")
                if self.parse_fn is None:
                    print("错误：parse_fn为None！")
                    result = {"events": [], "mentions": [], "event_relations": [], "entity_relations": []}
                else:
                    result = self.parse_fn(llm_response, chunk_id)
                    print(f"HyperRAGGraphExtractor 解析结果: {len(result['events'])} 个事件, "
                          f"{len(result['mentions'])} 个mentions, "
                          f"{len(result.get('event_relations', []))} 个事件关系")
                    
            except Exception as e:
                print(f"提取图结构时出错: {e}")
                result = {"events": [], "mentions": [], "event_relations": [], "entity_relations": []}
            
            # 将结果存储到document metadata中
            document.metadata[KG_EVENTS_KEY] = result["events"]
            document.metadata[KG_MENTIONS_KEY] = result["mentions"]
            document.metadata["event_relations"] = result.get("event_relations", [])
            document.metadata["entity_relations"] = result.get("entity_relations", [])
            
            return document

    @classmethod
    def class_name(cls) -> str:
        return "HyperRAGGraphExtractor"




class IncrementalGraphProcessor:
    """
    增量图处理器 - 集成HyperRAG图结构处理和增量数据处理
    
    主要功能：
    1. 处理HyperRAG提取的完整图结构（events、mentions、entities、relations）
    2. 对mentions进行去重，构建最终的entities
    3. 检测新增和更新的实体/关系
    4. 合并新旧数据，避免重复
    5. 维护数据版本和时间戳
    6. 支持本地缓存机制
    """
    
    def __init__(self, cache: GraphDataCache = None, existing_entities: list = None, existing_relations: list = None):
        """
        初始化增量图处理器
        
        Args:
            cache: 图数据缓存实例（推荐）
            existing_entities: 已存在的实体列表（如果不使用缓存）
            existing_relations: 已存在的关系列表（如果不使用缓存）
        """
        self.cache = cache
        
        # 如果使用缓存，从缓存加载数据；否则使用传入的数据
        if self.cache:
            self.existing_entities = list(self.cache.get_all_entities().values())
            self.existing_relations = self.cache.get_all_relations()
        else:
            self.existing_entities = existing_entities or []
            self.existing_relations = existing_relations or []
        
        # 增量处理状态
        self.entity_name_map = {}  # 实体名称到实体对象的映射
        self.new_entities = []  # 新增实体
        self.updated_entities = []  # 更新实体
        self.new_relations = []  # 新增关系
        
        # HyperRAG处理状态
        self.all_mentions = []
        self.all_events = []
        self.all_event_relations = []
        self.all_entity_relations = []
        
        # 构建现有实体的映射
        self._build_entity_name_map()

    def process_documents(self, documents: list[Document]) -> dict:
        """
        处理所有文档，返回最终的图结构
        
        Args:
            documents: 已经通过HyperRAGGraphExtractor处理的文档列表
            
        Returns:
            dict: 包含最终entities, relations, events, event_relations的字典
        """
        # 收集所有mentions和events
        self._collect_mentions_and_events(documents)
        
        # 对mentions进行去重，构建entities
        entities = self._deduplicate_mentions_to_entities()
        
        # 构建基于事件的实体关系
        event_based_entity_relations = self._build_entity_relations(entities)
        
        # 整合直接提取的实体关系
        direct_entity_relations = self._consolidate_entity_relations()
        
        # 合并所有实体关系
        all_entity_relations = event_based_entity_relations + direct_entity_relations
        
        # 整合事件关系
        event_relations = self._consolidate_event_relations()
        
        return {
            "entities": entities,
            "entity_relations": all_entity_relations,
            "events": self.all_events,
            "event_relations": event_relations,
            "mentions": self.all_mentions  # 保留原始mentions用于调试
        }

    @classmethod
    def create_with_cache(cls, cache_dir: str = "./graph_cache", initial_entities: list = None, initial_relations: list = None):
        """
        使用缓存创建增量处理器实例（推荐方式）
        
        Args:
            cache_dir: 缓存目录
            initial_entities: 初始实体数据（可选）
            initial_relations: 初始关系数据（可选）
            
        Returns:
            IncrementalGraphProcessor: 带缓存的处理器实例
        """
        # 创建缓存实例
        cache = GraphDataCache(cache_dir)
        
        # 如果提供了初始数据且缓存为空，则加载初始数据
        if (initial_entities or initial_relations) and cache.get_cache_stats()['entity_count'] == 0:
            print("🔄 加载初始数据到缓存...")
            cache.load_data(initial_entities, initial_relations)
        
        # 创建处理器实例
        processor = cls(cache=cache)
        
        return processor
    
    def update_cache_with_results(self, incremental_result: dict):
        """
        将增量处理结果更新到缓存
        
        Args:
            incremental_result: 增量处理结果
        """
        if not self.cache:
            return
        
        print(f"💾 正在更新缓存...")
        entity_count = 0
        relation_count = 0
        
        # 更新新实体到缓存（批量，不立即保存）
        for entity in incremental_result.get('new_entities', []):
            if hasattr(entity, 'name'):
                entity_dict = {
                    'name': entity.name,
                    'type': getattr(entity, 'label', 'Unknown'),
                    'summary': getattr(entity, 'summary', ''),
                    'aliases': getattr(entity, 'aliases', []),
                }
            else:
                entity_dict = entity
            
            self.cache.add_entity(entity_dict, save_immediately=False)
            entity_count += 1
        
        # 更新修改的实体到缓存（批量，不立即保存）
        for entity in incremental_result.get('updated_entities', []):
            if hasattr(entity, 'name'):
                entity_dict = {
                    'name': entity.name,
                    'type': getattr(entity, 'label', 'Unknown'),
                    'summary': getattr(entity, 'summary', ''),
                    'aliases': getattr(entity, 'aliases', []),
                }
            else:
                entity_dict = entity
            
            self.cache.add_entity(entity_dict, save_immediately=False)
            entity_count += 1
        
        # 更新新关系到缓存（批量，不立即保存）
        for relation in incremental_result.get('new_relations', []):
            if hasattr(relation, 'head_id'):
                relation_dict = {
                    'head_id': relation.head_id,
                    'tail_id': relation.tail_id,
                    'label': relation.label,
                    'description': getattr(relation, 'metadatas', {}).get('description', '') if hasattr(relation, 'metadatas') else '',
                }
            else:
                relation_dict = relation
            
            self.cache.add_relation(relation_dict, save_immediately=False)
            relation_count += 1
        
        # 最后统一保存到磁盘
        if entity_count > 0 or relation_count > 0:
            self.cache._save_cache()
            print(f"✅ 缓存更新完成: {entity_count} 个实体, {relation_count} 个关系")
    
    def _build_entity_name_map(self):
        """构建实体名称到实体对象的映射"""
        for entity in self.existing_entities:
            if hasattr(entity, 'name'):
                self.entity_name_map[entity.name] = entity
            elif isinstance(entity, dict):
                name = entity.get('name')
                if name:
                    self.entity_name_map[name] = entity
    
    def process_documents_incremental(self, documents: list[Document]) -> dict:
        """
        增量处理文档，返回新增和更新的图结构
        
        Args:
            documents: 已经通过HyperRAGGraphExtractor处理的文档列表
            
        Returns:
            dict: 包含新增和更新的entities, relations, events的字典
        """
        # 如果使用缓存，先过滤出未处理的文档
        if self.cache:
            unprocessed_documents = self.cache.filter_unprocessed_chunks(documents)
            if len(unprocessed_documents) < len(documents):
                print(f"📊 缓存过滤：从 {len(documents)} 个文档中跳过了 {len(documents) - len(unprocessed_documents)} 个已处理的chunk")
        else:
            unprocessed_documents = documents
        
        # 如果没有需要处理的文档，返回空结果
        if not unprocessed_documents:
            print("✅ 所有文档都已处理过，跳过处理")
            return {
                "new_entities": [],
                "updated_entities": [],
                "new_relations": [],
                "all_entities": list(self.cache.get_all_entities().values()) if self.cache else [],
                "all_relations": self.cache.get_all_relations() if self.cache else [],
                "events": [],
                "event_relations": [],
                "mentions": []
            }
        
        # 先执行基础处理（仅处理未处理的文档）
        result = self.process_documents(unprocessed_documents)
        
        # 进行增量分析
        self._analyze_incremental_changes(result)
        
        # 构建返回结果
        incremental_result = {
            "new_entities": self.new_entities,
            "updated_entities": self.updated_entities,
            "new_relations": self.new_relations,
            "all_entities": result["entities"],
            "all_relations": result["entity_relations"],
            "events": result["events"],
            "event_relations": result["event_relations"],
            "mentions": result["mentions"]
        }
        
        # 如果使用缓存，更新缓存并标记chunk为已处理
        if self.cache:
            self.update_cache_with_results(incremental_result)
            
            # 标记所有处理过的chunk
            processed_chunk_ids = []
            for doc in unprocessed_documents:
                chunk_id = doc.metadata.get('chunk_id') if hasattr(doc, 'metadata') else None
                if chunk_id:
                    processed_chunk_ids.append(chunk_id)
            
            if processed_chunk_ids:
                self.cache.mark_chunks_processed(processed_chunk_ids, save_immediately=True)
        
        return incremental_result
    
    def _analyze_incremental_changes(self, processed_result: dict):
        """
        分析增量变化
        
        Args:
            processed_result: 处理后的图结构结果
        """
        current_entities = processed_result["entities"]
        current_relations = processed_result["entity_relations"]
        
        # 分析实体变化
        self._analyze_entity_changes(current_entities)
        
        # 分析关系变化
        self._analyze_relation_changes(current_relations)
    
    def _analyze_entity_changes(self, current_entities: list):
        """分析实体变化"""
        current_time = datetime.now()
        
        for entity in current_entities:
            entity_name = entity.name if hasattr(entity, 'name') else entity.get('name')
            
            if entity_name in self.entity_name_map:
                # 实体已存在，检查是否需要更新
                existing_entity = self.entity_name_map[entity_name]
                if self._entity_needs_update(existing_entity, entity):
                    # 合并实体信息
                    updated_entity = self._merge_entity_data(existing_entity, entity)
                    updated_entity.update_time = current_time
                    self.updated_entities.append(updated_entity)
            else:
                # 新实体
                if hasattr(entity, 'update_time'):
                    entity.update_time = current_time
                elif isinstance(entity, dict):
                    entity['update_time'] = current_time
                self.new_entities.append(entity)
                self.entity_name_map[entity_name] = entity
    
    def _analyze_relation_changes(self, current_relations: list):
        """分析关系变化"""
        # 构建现有关系的标识集合
        existing_rel_keys = set()
        for rel in self.existing_relations:
            if hasattr(rel, 'head_id') and hasattr(rel, 'tail_id') and hasattr(rel, 'label'):
                key = (rel.head_id, rel.tail_id, rel.label)
            elif isinstance(rel, dict):
                key = (rel.get('head_id'), rel.get('tail_id'), rel.get('label'))
            else:
                continue
            existing_rel_keys.add(key)
        
        # 检查新关系
        for rel in current_relations:
            if hasattr(rel, 'head_id') and hasattr(rel, 'tail_id') and hasattr(rel, 'label'):
                key = (rel.head_id, rel.tail_id, rel.label)
            elif isinstance(rel, dict):
                key = (rel.get('head_id'), rel.get('tail_id'), rel.get('label'))
            else:
                continue
                
            if key not in existing_rel_keys:
                self.new_relations.append(rel)
                existing_rel_keys.add(key)
    
    def _entity_needs_update(self, existing_entity, new_entity) -> bool:
        """
        检查实体是否需要更新
        
        Args:
            existing_entity: 现有实体
            new_entity: 新提取的实体
            
        Returns:
            bool: 是否需要更新
        """
        # 比较描述信息
        existing_desc = self._get_entity_description(existing_entity)
        new_desc = self._get_entity_description(new_entity)
        
        if new_desc and new_desc != existing_desc:
            return True
        
        # 比较别名
        existing_aliases = self._get_entity_aliases(existing_entity)
        new_aliases = self._get_entity_aliases(new_entity)
        
        if new_aliases and set(new_aliases) != set(existing_aliases):
            return True
        
        return False
    
    def _get_entity_description(self, entity) -> str:
        """获取实体描述"""
        if hasattr(entity, 'summary'):
            return entity.summary or ""
        elif hasattr(entity, 'description'):
            return entity.description or ""
        elif isinstance(entity, dict):
            return entity.get('summary', entity.get('description', ""))
        return ""
    
    def _get_entity_aliases(self, entity) -> list:
        """获取实体别名"""
        if hasattr(entity, 'aliases'):
            return entity.aliases or []
        elif isinstance(entity, dict):
            return entity.get('aliases', [])
        return []
    
    def _merge_entity_data(self, existing_entity, new_entity):
        """
        合并实体数据
        
        Args:
            existing_entity: 现有实体
            new_entity: 新实体数据
            
        Returns:
            合并后的实体
        """
        # 合并描述
        existing_desc = self._get_entity_description(existing_entity)
        new_desc = self._get_entity_description(new_entity)
        
        merged_desc = existing_desc
        if new_desc and new_desc not in existing_desc:
            merged_desc = f"{existing_desc}; {new_desc}" if existing_desc else new_desc
        
        # 合并别名
        existing_aliases = self._get_entity_aliases(existing_entity)
        new_aliases = self._get_entity_aliases(new_entity)
        merged_aliases = list(set(existing_aliases + new_aliases))
        
        # 创建合并后的实体
        if hasattr(existing_entity, 'name'):
            # EntityNode 对象
            existing_entity.summary = merged_desc
            existing_entity.aliases = merged_aliases
            return existing_entity
        else:
            # 字典格式
            merged_entity = existing_entity.copy()
            merged_entity['summary'] = merged_desc
            merged_entity['aliases'] = merged_aliases
            return merged_entity

    def _collect_mentions_and_events(self, documents: list[Document]):
        """收集所有文档的mentions和events"""
        self.all_mentions = []
        self.all_events = []
        self.all_event_relations = []
        self.all_entity_relations = []
        
        for doc in documents:
            # 收集mentions
            mentions = doc.metadata.get(KG_MENTIONS_KEY, [])
            for mention in mentions:
                mention['source_doc'] = doc.metadata.get('chunk_id', 'unknown')
                self.all_mentions.append(mention)
            
            # 收集events
            events = doc.metadata.get(KG_EVENTS_KEY, [])
            for event in events:
                event['source_doc'] = doc.metadata.get('chunk_id', 'unknown')
                self.all_events.append(event)
            
            # 收集event_relations
            event_relations = doc.metadata.get('event_relations', [])
            for rel in event_relations:
                rel['source_doc'] = doc.metadata.get('chunk_id', 'unknown')
                self.all_event_relations.append(rel)
            
            # 收集entity_relations
            entity_relations = doc.metadata.get('entity_relations', [])
            for rel in entity_relations:
                rel['source_doc'] = doc.metadata.get('chunk_id', 'unknown')
                self.all_entity_relations.append(rel)
    
    def _deduplicate_mentions_to_entities(self) -> list:
        """对mentions进行去重，构建最终的entities"""
        entity_groups = {}
        
        # 按entity_name分组mentions
        for mention in self.all_mentions:
            entity_name = mention.get('entity_name', '')
            if entity_name not in entity_groups:
                entity_groups[entity_name] = []
            entity_groups[entity_name].append(mention)
        
        entities = []
        for entity_name, mentions in entity_groups.items():
            # 合并同一实体的所有信息
            entity = self._merge_mentions_to_entity(entity_name, mentions)
            entities.append(entity)
        
        return entities
    
    def _merge_mentions_to_entity(self, entity_name: str, mentions: list) -> dict:
        """将同一实体的所有mentions合并为一个entity"""
        # 获取最常见的entity_type
        types = [m.get('entity_type', '') for m in mentions if m.get('entity_type')]
        entity_type = max(set(types), key=types.count) if types else "Unknown"
        
        # 合并描述
        descriptions = [m.get('entity_description', '') for m in mentions if m.get('entity_description')]
        entity_description = '; '.join(set(descriptions)) if descriptions else ""
        
        # 收集所有提及文本作为aliases
        aliases = list(set([m.get('text', '') for m in mentions if m.get('text')]))
        
        # 收集所有相关事件
        related_events = []
        for m in mentions:
            event_indices = m.get('event_indices', [])
            for event_idx in event_indices:
                # 找到对应的事件
                for event in self.all_events:
                    if event.get('id', '').endswith(f'_{event_idx}'):
                        related_events.append(event.get('id', ''))
                        break
        
        # 生成实体ID
        entity_id = f"Entity_{hash(entity_name) % 1000000:06d}"
        
        # 创建描述映射 - 使用chunk_id:description格式
        description_mapping = {}
        for mention in mentions:
            chunk_id = mention.get('source_doc', 'unknown_chunk')
            desc = mention.get('entity_description', '')
            if desc:
                if chunk_id in description_mapping:
                    # 如果同一个chunk有多个描述，用分号连接
                    description_mapping[chunk_id] += f"; {desc}"
                else:
                    description_mapping[chunk_id] = desc
        
        return EntityNode(
            id_=entity_id,
            name=entity_name,
            label=entity_type,
            aliases=aliases,
            description=description_mapping,
            summary=entity_description
        )
    
    def _build_entity_relations(self, entities: list) -> list:
        """构建实体之间的关系：基于事件参与关系 + 基于chunk共现关系"""
        relations = []
        
        # 1. 基于共同参与的事件建立实体关系
        for event in self.all_events:
            participants = event.get('participants', [])
            if len(participants) >= 2:
                # 为参与同一事件的实体建立关系
                for i in range(len(participants)):
                    for j in range(i + 1, len(participants)):
                        # 找到对应的实体
                        head_entity = next((e for e in entities if e.name == participants[i]), None)
                        tail_entity = next((e for e in entities if e.name == participants[j]), None)
                        
                        if head_entity and tail_entity:
                            relation = Relation(
                                label="共同参与事件",
                                head_id=head_entity.name,
                                tail_id=tail_entity.name,
                                metadatas={
                                    "description": f"在事件'{event.get('content', '')}'中共同参与",
                                    "event_id": event.get('id', ''),
                                    "event_type": event.get('type', ''),
                                    "source_doc": event.get('source_doc', ''),
                                    "relation_source": "event_based"
                                }
                            )
                            relations.append(relation)
        
        # 2. 基于chunk共现关系建立实体连接（适用于所有chunk，包括没有event的chunk）
        chunk_entities = {}  # chunk_id -> [entities]
        
        # 按chunk分组实体
        for entity in entities:
            # 从实体的description中获取所有相关的chunk_id
            if hasattr(entity, 'description') and isinstance(entity.description, dict):
                for chunk_id in entity.description.keys():
                    if chunk_id not in chunk_entities:
                        chunk_entities[chunk_id] = []
                    chunk_entities[chunk_id].append(entity)
        
        # 为每个chunk中共现的实体建立关系
        for chunk_id, chunk_entity_list in chunk_entities.items():
            if len(chunk_entity_list) >= 2:
                for i in range(len(chunk_entity_list)):
                    for j in range(i + 1, len(chunk_entity_list)):
                        head_entity = chunk_entity_list[i]
                        tail_entity = chunk_entity_list[j]
                        
                        # 检查是否已经有基于事件的关系，避免重复
                        existing_relation = any(
                            rel.head_id == head_entity.name and rel.tail_id == tail_entity.name
                            and rel.metadatas.get("relation_source") == "event_based"
                            for rel in relations
                        )
                        
                        if not existing_relation:
                            relation = Relation(
                                label="文档共现",
                                head_id=head_entity.name,
                                tail_id=tail_entity.name,
                                metadatas={
                                    "description": f"在文档chunk '{chunk_id}' 中共同出现",
                                    "source_doc": chunk_id,
                                    "relation_source": "cooccurrence_based"
                                }
                            )
                            relations.append(relation)
        
        return relations
    
    def _consolidate_entity_relations(self) -> list:
        """整合直接提取的实体关系"""
        relations = []
        
        # 去重实体关系
        seen = set()
        for rel in self.all_entity_relations:
            # 尝试不同的键名格式以保持兼容性
            head = rel.get('head', rel.get('head_entity', ''))
            tail = rel.get('tail', rel.get('tail_entity', ''))
            rel_type = rel.get('type', rel.get('relation_type', ''))
            
            key = (head, tail, rel_type)
            if key not in seen and head and tail and rel_type:
                seen.add(key)
                relation = Relation(
                    label=rel_type,
                    head_id=head,
                    tail_id=tail,
                    metadatas={
                        "description": rel.get('description', ''),
                        "source_doc": rel.get('source_doc', ''),
                        "relation_source": "direct_extraction"
                    }
                )
                relations.append(relation)
        
        return relations
    
    def _consolidate_event_relations(self) -> list:
        """整合事件关系"""
        # 去重事件关系
        unique_relations = []
        seen = set()
        
        for rel in self.all_event_relations:
            key = (rel.get('head_event', ''), rel.get('tail_event', ''), rel.get('relation_type', ''))
            if key not in seen:
                seen.add(key)
                unique_relations.append(rel)
        
        return unique_relations