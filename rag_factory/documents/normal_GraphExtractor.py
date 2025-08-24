from rag_factory.documents.Base_GraphExtractor import GraphExtractorBase
from rag_factory.documents.Prompt import KG_TRIPLES_PROMPT
from rag_factory.documents.pydantic_schema import GraphTriples
from rag_factory.documents.schema import Document
from rag_factory.Store.GraphStore.GraphNode import EntityNode, Relation
from rag_factory.llms import LLMBase
from typing import Any

KG_NODES_KEY = "entities"
KG_RELATIONS_KEY = "relations"

class GraphExtractor(GraphExtractorBase):
    """
    GraphExtractor is a class that extracts triples from a graph.

    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.
    """
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        response_format: Any = None,
        max_concurrent: int = 100
    ) -> None:
        extract_prompt = extract_prompt if extract_prompt is not None else KG_TRIPLES_PROMPT
        response_format = response_format if response_format is not None else GraphTriples
        super().__init__(llm, extract_prompt, response_format, max_concurrent)


    async def _aextract(self, document: Document, semaphore) -> Document:
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
                llm_response = await self.llm.aparse_chat(messages, response_format=self.response_format)

                # 直接使用 Pydantic 对象
                if llm_response is None:
                    print("错误：LLM 返回空响应！")
                    entities, relationships = [], []
                else:
                    # 从 Pydantic 对象中提取数据
                    entities = [(entity.name, entity.type, entity.description or "") for entity in llm_response.entities]
                    relationships = [(rel.head, rel.tail, rel.relation, rel.description or "") for rel in llm_response.relationships]
                    print(f"GraphExtractor 解析结果: entities={len(entities)}, relationships={len(relationships)}")
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