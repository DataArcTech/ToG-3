from rag_factory.llms.llm_base import LLMBase
from typing import Callable
from rag_factory.documents.Prompt import KG_TRIPLES_PROMPT
from rag_factory.documents.schema import Document
from rag_factory.Store.GraphStore.GraphNode import EntityNode, Relation

KG_NODES_KEY = "entities"
KG_RELATIONS_KEY = "relations"


class GraphExtractor:
    """
    GraphExtractor is a class that extracts triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.
    """
    
    llm: LLMBase
    extract_prompt: str
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
    # llm_response_cache: LlmResponseCache
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        parse_fn: Callable = None,
        max_concurrent: int = 100
    ) -> None:
        self.llm = llm
        self.extract_prompt = extract_prompt if extract_prompt is not None else KG_TRIPLES_PROMPT
        self.parse_fn = parse_fn
        self.max_concurrent = max_concurrent


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
                entity_node = EntityNode(
                    name=entity, label=entity_type, metadatas=entity_metadata
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


    async def acall(self, documents: list[Document], show_progress: bool = False) -> list[Document]:
        """异步提取所有节点的实体、三元组"""
        import asyncio
        
        # 创建信号量来控制并发数量
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 创建任务列表
        tasks = [self._aextract(document, semaphore) for document in documents]
        
        if show_progress:
            print(f"开始从{len(documents)}个Chunk中提取实体、三元组...")
            from tqdm.asyncio import tqdm_asyncio
            results = await tqdm_asyncio.gather(*tasks, desc="提取实体、三元组")
        else:
            results = await asyncio.gather(*tasks)
        if show_progress:
            print("实体、三元组提取完成")
        return results
        

    def __call__(self, documents: list[Document], show_progress: bool = False) -> list[Document]:
        """提取实体、三元组"""
        import asyncio
        return asyncio.run(self.acall(documents, show_progress=show_progress))

    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"