import asyncio
import os
from dataclasses import dataclass
from tkinter import DOTBOX
import neo4j
import logging
from typing import List, Dict, Any, Optional    

from rag_factory.Store.GraphStore.GraphNode import EntityNode, Relation, ChunkNode
from rag_factory.documents.schema import Document
from rag_factory.Embed import Embeddings


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)

@dataclass
class Neo4jGraphStore():

    def __init__(self,
        url: str,
        username:str,
        password:str,
        database:str,
        embedding:Embeddings,
        ):
        self._driver = None
        self._driver_lock = asyncio.Lock()
        self._driver: neo4j.AsyncDriver = neo4j.AsyncGraphDatabase.driver(
            url, auth=(username, password)
        )
        self.embedding = embedding

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        if self._driver:
            await self._driver.close()

    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=4, max=10),retry=retry_if_exception_type(neo4j_retry_errors))
    async def upsert_entity(self, entity: EntityNode):
        """
        Upsert a node in the Neo4j database.

        Args:
            entity: The entity node to upsert
        """
        label = entity.label.strip('"')
        name = entity.name.strip('"')
        properties = entity.metadatas or {}

        async def _do_upsert(tx: neo4j.AsyncManagedTransaction):
            query = f"""
            MERGE (n:`{entity.label}` {{name: $name}})
            SET n += $properties
            """
            await tx.run(query, name=name, properties=properties)
            print(
                f"Upserted node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            print(f"Error during upsert: {str(e)}")
            raise


    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=4, max=10),retry=retry_if_exception_type(neo4j_retry_errors))
    async def upsert_relation(
        self, relation: Relation
    ):
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge
        """
        head_id = relation.head_id.strip('"')
        tail_id = relation.tail_id.strip('"')
        relation_label = relation.label.strip('"')
        edge_properties = relation.metadatas

        async def _do_upsert_edge(tx: neo4j.AsyncManagedTransaction):
            query = f"""
            MATCH (head {{name: $head_id}})
            MATCH (tail {{name: $tail_id}})
            MERGE (head)-[r:`{relation_label}`]->(tail)
            SET r += $properties
            RETURN r
            """
            await tx.run(query, head_id=head_id, tail_id=tail_id, properties=edge_properties)
            print(f"Upserted edge from '{head_id}' to '{tail_id}' with properties: {edge_properties}")
        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert_edge)
        except Exception as e:
            print(f"Error during edge upsert: {str(e)}")
            raise



    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=4, max=10),retry=retry_if_exception_type(neo4j_retry_errors))
    async def upsert_node(self, doc_node: ChunkNode):
        """
        Upsert a document chunk node in the Neo4j database.

        The node is merged by property `name` using the chunk id, and stores
        `content` and any additional metadatas as properties.
        """
        label = doc_node.label.strip('"')
        name = doc_node.id
        def _sanitize_metadata(meta: dict) -> dict:
            if not meta:
                return {}
            clean = {}
            for k, v in meta.items():
                if k in ("entities", "relations"):
                    continue
                if isinstance(v, (str, int, float, bool)) or v is None:
                    clean[k] = v
                elif isinstance(v, list):
                    lv = [x for x in v if isinstance(x, (str, int, float, bool))]
                    if lv:
                        clean[k] = lv
                elif isinstance(v, dict):
                    dv = {ik: iv for ik, iv in v.items() if isinstance(iv, (str, int, float, bool))}
                    if dv:
                        clean[k] = dv
            return clean

        properties = {"content": doc_node.content}
        if doc_node.metadatas:
            properties.update(_sanitize_metadata(doc_node.metadatas))

        async def _do_upsert(tx: neo4j.AsyncManagedTransaction):
            query = f"""
            MERGE (n:`{label}` {{name: $name}})
            SET n += $properties
            """
            await tx.run(query, name=name, properties=properties)
            print(
                f"Upserted node with label '{label}' and properties: {properties}"
            )

        try:
            async with self._driver.session() as session:
                await session.execute_write(_do_upsert)
        except Exception as e:
            print(f"Error during upsert: {str(e)}")
            raise

    async def upsert_document(self, document: Document):
        """
        Upsert a document chunk as a node in the Neo4j database.

        Args:
            document: The document chunk to upsert as a node
        """
        try:
            # 创建文档chunk节点
            if "chunk_id" not in document.metadata:
                chunk_id = f"chunk_{hash(document.content)}"
            else:
                chunk_id = document.metadata["chunk_id"]
            # 过滤掉不适合写入 Neo4j 的复杂类型（例如包含 EntityNode/Relation 的列表）
            base_meta = (document.metadata.copy() if document.metadata else {})
            base_meta.pop('entities', None)
            base_meta.pop('relations', None)
            chunk_node = ChunkNode(
                content=document.content,
                id_=chunk_id,
                source=base_meta.get("file_name", "unknown"),
                label="text_chunk",
                metadatas={
                    "chunk_id": chunk_id,
                    **base_meta
                }
            )
            
            # 插入chunk节点
            await self.upsert_node(chunk_node)
            
            # 处理实体节点
            entities = document.metadata.get('entities', [])
            for entity in entities:
                await self.upsert_entity(entity)
                
                # 创建chunk与实体的关系
                chunk_entity_relation = Relation(
                    label='包含实体',
                    head_id=chunk_id,
                    tail_id=entity.name,
                    metadatas={'relationship_description': 'chunk包含该实体'}
                )
                await self.upsert_relation(chunk_entity_relation)
            
            # 处理关系
            relations = document.metadata.get('relations', [])
            for relation in relations:
                await self.upsert_relation(relation)
                
            print(f"成功插入chunk节点，包含 {len(entities)} 个实体和 {len(relations)} 个关系")
            
        except Exception as e:
            print(f"插入chunk时发生错误: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4j.exceptions.ServiceUnavailable,
                neo4j.exceptions.TransientError,
                neo4j.exceptions.WriteServiceUnavailable,
                neo4j.exceptions.ClientError,
            )
        ),
    )
    async def merge_node(self):
        """
        合并所有同名的实体节点，将label、description、source_chunk_id等属性合并为列表
        同时处理相关的关系，确保关系指向合并后的节点
        注意：不合并text_chunk类型的节点
        """
        try:
            async with self._driver.session() as session:
                # 第一步：找到所有需要合并的同名实体（排除text_chunk）
                find_duplicates_query = """
                MATCH (n)
                WHERE NOT 'text_chunk' IN labels(n) AND n.name IS NOT NULL
                WITH n.name as entity_name, collect(n) as nodes, labels(collect(n)[0]) as sample_labels
                WHERE size(nodes) > 1
                RETURN entity_name, nodes, sample_labels
                """
                
                result = await session.run(find_duplicates_query)
                duplicate_records = await result.data()
                
                merged_entity_count = 0
                
                for record in duplicate_records:
                    entity_name = record['entity_name']
                    nodes = record['nodes']
                    sample_labels = record['sample_labels']
                    
                    if len(nodes) <= 1:
                        continue
                    
                    # 收集所有节点的属性进行合并
                    merged_properties = await self._merge_node_properties(nodes, entity_name)
                    
                    # 在事务中执行合并操作
                    async def _merge_entity_transaction(tx: neo4j.AsyncManagedTransaction):
                        # 1. 获取所有指向这些重复节点的关系
                        get_incoming_relations_query = """
                        MATCH (source)-[r]->(target)
                        WHERE target.name = $entity_name AND NOT 'text_chunk' IN labels(target)
                        RETURN source.name as source_name, type(r) as rel_type, properties(r) as rel_props
                        """
                        incoming_result = await tx.run(get_incoming_relations_query, entity_name=entity_name)
                        incoming_relations = await incoming_result.data()
                        
                        # 2. 获取所有从这些重复节点出发的关系
                        get_outgoing_relations_query = """
                        MATCH (source)-[r]->(target)
                        WHERE source.name = $entity_name AND NOT 'text_chunk' IN labels(source)
                        RETURN target.name as target_name, type(r) as rel_type, properties(r) as rel_props
                        """
                        outgoing_result = await tx.run(get_outgoing_relations_query, entity_name=entity_name)
                        outgoing_relations = await outgoing_result.data()
                        
                        # 3. 删除所有重复的实体节点及其关系
                        delete_query = """
                        MATCH (n)
                        WHERE n.name = $entity_name AND NOT 'text_chunk' IN labels(n)
                        DETACH DELETE n
                        """
                        await tx.run(delete_query, entity_name=entity_name)
                        
                        # 4. 创建合并后的新节点，使用第一个节点的标签
                        primary_label = sample_labels[0] if sample_labels else "Entity"
                        create_query = f"""
                        CREATE (n:`{primary_label}`)
                        SET n = $properties
                        RETURN n
                        """
                        await tx.run(create_query, properties=merged_properties)
                        
                        # 5. 重建入边关系（合并同类型关系）
                        await self._rebuild_incoming_relations(tx, entity_name, incoming_relations)
                        
                        # 6. 重建出边关系（合并同类型关系）
                        await self._rebuild_outgoing_relations(tx, entity_name, outgoing_relations)
                    
                    await session.execute_write(_merge_entity_transaction)
                    merged_entity_count += 1
                    print(f"成功合并实体 '{entity_name}'，合并了 {len(nodes)} 个重复节点")
                
                print(f"实体合并完成，共处理了 {merged_entity_count} 个重复实体")
                
        except Exception as e:
            print(f"合并节点时发生错误: {str(e)}")
            raise

    async def _merge_node_properties(self, nodes: list, entity_name: str) -> dict:
        """
        合并多个节点的属性
        """
        merged_labels = set()
        merged_descriptions = set()
        merged_source_chunk_ids = set()
        merged_metadata = {}
        
        for node in nodes:
            # 收集label
            if 'label' in node and node['label']:
                if isinstance(node['label'], list):
                    merged_labels.update(node['label'])
                else:
                    merged_labels.add(node['label'])
            
            # 收集entity_description
            if 'entity_description' in node and node['entity_description']:
                if isinstance(node['entity_description'], list):
                    merged_descriptions.update(node['entity_description'])
                else:
                    merged_descriptions.add(node['entity_description'])
            
            # 收集source_chunk_id
            if 'source_chunk_id' in node and node['source_chunk_id']:
                if isinstance(node['source_chunk_id'], list):
                    merged_source_chunk_ids.update(node['source_chunk_id'])
                else:
                    merged_source_chunk_ids.add(node['source_chunk_id'])
            
            # 收集其他元数据，确保只存储简单类型
            for key, value in node.items():
                if key not in ['label', 'entity_description', 'source_chunk_id', 'name'] and value is not None:
                    # 只处理简单类型
                    if isinstance(value, (str, int, float, bool)):
                        if key not in merged_metadata:
                            merged_metadata[key] = value
                        elif merged_metadata[key] != value:
                            # 如果值不同，将其转换为列表（但确保列表元素都是简单类型）
                            if not isinstance(merged_metadata[key], list):
                                merged_metadata[key] = [merged_metadata[key]]
                            if value not in merged_metadata[key]:
                                merged_metadata[key].append(value)
                    elif isinstance(value, list):
                        # 如果是列表，只保留简单类型的元素
                        simple_values = [v for v in value if isinstance(v, (str, int, float, bool))]
                        if simple_values:
                            if key not in merged_metadata:
                                merged_metadata[key] = simple_values
                            else:
                                if not isinstance(merged_metadata[key], list):
                                    merged_metadata[key] = [merged_metadata[key]]
                                for v in simple_values:
                                    if v not in merged_metadata[key]:
                                        merged_metadata[key].append(v)
        
        # 构建合并后的属性
        merged_properties = {
            'name': entity_name,
            **merged_metadata
        }
        
        # 只有在有值的情况下才添加这些属性
        if merged_labels:
            merged_properties['label'] = list(merged_labels)
        if merged_descriptions:
            merged_properties['entity_description'] = list(merged_descriptions)
        if merged_source_chunk_ids:
            merged_properties['source_chunk_id'] = list(merged_source_chunk_ids)
        
        return merged_properties

    async def _rebuild_incoming_relations(self, tx: neo4j.AsyncManagedTransaction, target_name: str, relations: list):
        """
        重建指向目标节点的关系，合并同类型关系
        """
        # 按源节点和关系类型分组
        relation_groups = {}
        for rel in relations:
            key = (rel['source_name'], rel['rel_type'])
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(rel['rel_props'])
        
        # 为每组关系创建一个合并的关系
        for (source_name, rel_type), rel_props_list in relation_groups.items():
            merged_rel_props = await self._merge_relation_properties(rel_props_list)
            
            create_relation_query = f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            CREATE (source)-[r:`{rel_type}`]->(target)
            SET r = $properties
            """
            await tx.run(create_relation_query, 
                        source_name=source_name, 
                        target_name=target_name, 
                        properties=merged_rel_props)

    async def _rebuild_outgoing_relations(self, tx: neo4j.AsyncManagedTransaction, source_name: str, relations: list):
        """
        重建从源节点出发的关系，合并同类型关系
        """
        # 按目标节点和关系类型分组
        relation_groups = {}
        for rel in relations:
            key = (rel['target_name'], rel['rel_type'])
            if key not in relation_groups:
                relation_groups[key] = []
            relation_groups[key].append(rel['rel_props'])
        
        # 为每组关系创建一个合并的关系
        for (target_name, rel_type), rel_props_list in relation_groups.items():
            merged_rel_props = await self._merge_relation_properties(rel_props_list)
            
            create_relation_query = f"""
            MATCH (source {{name: $source_name}})
            MATCH (target {{name: $target_name}})
            CREATE (source)-[r:`{rel_type}`]->(target)
            SET r = $properties
            """
            await tx.run(create_relation_query, 
                        source_name=source_name, 
                        target_name=target_name, 
                        properties=merged_rel_props)

    async def _merge_relation_properties(self, rel_props_list: list) -> dict:
        """
        合并多个关系的属性
        """
        merged_descriptions = set()
        merged_source_chunk_ids = set()
        merged_metadata = {}
        
        for rel_props in rel_props_list:
            if not rel_props:
                continue
                
            # 收集relationship_description
            if 'relationship_description' in rel_props and rel_props['relationship_description']:
                if isinstance(rel_props['relationship_description'], list):
                    merged_descriptions.update(rel_props['relationship_description'])
                else:
                    merged_descriptions.add(rel_props['relationship_description'])
            
            # 收集source_chunk_id
            if 'source_chunk_id' in rel_props and rel_props['source_chunk_id']:
                if isinstance(rel_props['source_chunk_id'], list):
                    merged_source_chunk_ids.update(rel_props['source_chunk_id'])
                else:
                    merged_source_chunk_ids.add(rel_props['source_chunk_id'])
            
            # 收集其他元数据，确保只存储简单类型
            for key, value in rel_props.items():
                if key not in ['relationship_description', 'source_chunk_id', 'label'] and value is not None:
                    # 只处理简单类型
                    if isinstance(value, (str, int, float, bool)):
                        if key not in merged_metadata:
                            merged_metadata[key] = value
                        elif merged_metadata[key] != value:
                            # 如果值不同，将其转换为列表（但确保列表元素都是简单类型）
                            if not isinstance(merged_metadata[key], list):
                                merged_metadata[key] = [merged_metadata[key]]
                            if value not in merged_metadata[key]:
                                merged_metadata[key].append(value)
                    elif isinstance(value, list):
                        # 如果是列表，只保留简单类型的元素
                        simple_values = [v for v in value if isinstance(v, (str, int, float, bool))]
                        if simple_values:
                            if key not in merged_metadata:
                                merged_metadata[key] = simple_values
                            else:
                                if not isinstance(merged_metadata[key], list):
                                    merged_metadata[key] = [merged_metadata[key]]
                                for v in simple_values:
                                    if v not in merged_metadata[key]:
                                        merged_metadata[key].append(v)
        
        # 构建合并后的关系属性
        merged_rel_properties = {**merged_metadata}
        
        if merged_descriptions:
            merged_rel_properties['relationship_description'] = list(merged_descriptions)
        if merged_source_chunk_ids:
            merged_rel_properties['source_chunk_id'] = list(merged_source_chunk_ids)
        
        return merged_rel_properties

    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=4, max=10),retry=retry_if_exception_type(neo4j_retry_errors))
    async def cleanup_duplicate_relations(self):
        """
        清理重复的关系：合并具有相同源节点、目标节点和关系类型的关系
        """
        try:
            async with self._driver.session() as session:
                # 找到重复的关系
                find_duplicate_relations_query = """
                MATCH (source)-[r]->(target)
                WITH source.name as source_name, target.name as target_name, type(r) as rel_type, collect(r) as relations
                WHERE size(relations) > 1
                RETURN source_name, target_name, rel_type, relations
                """
                
                result = await session.run(find_duplicate_relations_query)
                duplicate_relations = await result.data()
                
                merged_relation_count = 0
                
                for record in duplicate_relations:
                    source_name = record['source_name']
                    target_name = record['target_name']
                    rel_type = record['rel_type']
                    relations = record['relations']
                    
                    if len(relations) <= 1:
                        continue
                    
                    # 合并关系属性
                    rel_props_list = [dict(rel) for rel in relations]
                    merged_rel_props = await self._merge_relation_properties(rel_props_list)
                    
                    async def _merge_relations_transaction(tx: neo4j.AsyncManagedTransaction):
                        # 删除所有重复关系
                        delete_rel_query = f"""
                        MATCH (source {{name: $source_name}})-[r:`{rel_type}`]->(target {{name: $target_name}})
                        DELETE r
                        """
                        await tx.run(delete_rel_query, 
                                   source_name=source_name, 
                                   target_name=target_name)
                        
                        # 创建合并后的新关系
                        create_rel_query = f"""
                        MATCH (source {{name: $source_name}})
                        MATCH (target {{name: $target_name}})
                        CREATE (source)-[r:`{rel_type}`]->(target)
                        SET r = $properties
                        """
                        await tx.run(create_rel_query, 
                                   source_name=source_name,
                                   target_name=target_name,
                                   properties=merged_rel_props)
                    
                    await session.execute_write(_merge_relations_transaction)
                    merged_relation_count += 1
                    print(f"成功合并关系 '{source_name}'-[{rel_type}]->'{target_name}'，合并了 {len(relations)} 个重复关系")
                
                print(f"关系合并完成，共处理了 {merged_relation_count} 个重复关系")
                
        except Exception as e:
            print(f"清理重复关系时发生错误: {str(e)}")
            raise

    async def get_entity_statistics(self):
        """
        获取图数据库的统计信息
        """
        try:
            async with self._driver.session() as session:
                # 统计各类型节点数量
                node_stats_query = """
                MATCH (n)
                RETURN labels(n) as node_labels, count(n) as count
                ORDER BY count DESC
                """
                result = await session.run(node_stats_query)
                node_stats = await result.data()
                
                # 统计关系数量
                rel_stats_query = """
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
                """
                result = await session.run(rel_stats_query)
                rel_stats = await result.data()
                
                # 检查是否还有重复节点
                duplicate_check_query = """
                MATCH (n)
                WHERE NOT 'text_chunk' IN labels(n) AND n.name IS NOT NULL
                WITH n.name as entity_name, count(n) as node_count
                WHERE node_count > 1
                RETURN entity_name, node_count
                ORDER BY node_count DESC
                """
                result = await session.run(duplicate_check_query)
                duplicates = await result.data()
                
                return {
                    'node_statistics': node_stats,
                    'relation_statistics': rel_stats,
                    'remaining_duplicates': duplicates
                }
                
        except Exception as e:
            print(f"获取统计信息时发生错误: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=4, max=10),retry=retry_if_exception_type(neo4j_retry_errors))
    async def vectorize_existing_nodes(self, batch_size: int = 50):
        """
        遍历数据库中所有节点，对缺少 embedding 的节点进行向量化补充。
        - 实体节点使用 `name`
        - text_chunk 节点使用 `content`
        """
        try:
            async with self._driver.session() as session:
                # 找到没有 embedding 的节点
                query = """
                MATCH (n)
                WHERE (NOT 'text_chunk' IN labels(n) AND n.name IS NOT NULL)
                   OR ('text_chunk' IN labels(n) AND n.content IS NOT NULL)
                AND n.embedding IS NULL
                RETURN id(n) as node_id, labels(n) as labels, n.name as name, n.content as content
                """
                result = await session.run(query)
                records = await result.data()

                print(f"需要补充 embedding 的节点数: {len(records)}")

                for i in range(0, len(records), batch_size):
                    batch = records[i:i+batch_size]
                    texts = []
                    node_ids = []

                    for record in batch:
                        node_id = record["node_id"]
                        labels = record["labels"]
                        text = record.get("name") if "text_chunk" not in labels else record.get("content")

                        if text:
                            texts.append(text)
                            node_ids.append(node_id)

                    if not texts:
                        continue

                    # 🔹 批量向量化
                    embeddings = []
                    for text in texts:
                        emb = self.embedding.embed_query(text)
                        embeddings.append(emb)

                    # 🔹 更新数据库
                    for node_id, emb in zip(node_ids, embeddings):
                        update_query = """
                        MATCH (n)
                        WHERE id(n) = $node_id
                        SET n.embedding = $embedding
                        """
                        await session.run(update_query, node_id=node_id, embedding=emb)

                    print(f"已更新 {len(batch)} 个节点的 embedding")

        except Exception as e:
            print(f"补充 embedding 时发生错误: {str(e)}")
            raise


    async def search(self, query: str, k: int = 5, search_type: str = "query") -> List[Dict[str, Any]]:
        query_embedding = self.embedding.embed_query(query)

        async with self._driver.session() as session:
            if search_type == "entity":
                # 实体检索逻辑保持不变
                cypher_entity = """
                MATCH (n)
                WHERE NOT 'text_chunk' IN labels(n) AND n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT $k
                RETURN n AS node, score
                """
                result = await session.run(cypher_entity, query_embedding=query_embedding, k=k)
                records = await result.data()

                results = []
                for record in records:
                    node = record["node"]
                    score = record["score"]
                    
                    # 移除 embedding 字段
                    if 'embedding' in node:
                        del node['embedding']

                    # cypher_rel = """
                    # MATCH (n {name: $name})-[r]-(m)
                    # RETURN type(r) AS rel_type, properties(r) AS rel_props, m AS neighbor
                    # """
                    cypher_rel = """
                    MATCH (n {name: $name})-[r]->(m)
                    RETURN type(r) AS rel_type, properties(r) AS rel_props, m AS neighbor
                    """
                    rel_result = await session.run(cypher_rel, name=node["name"])
                    rel_records = await rel_result.data()

                    relations = []
                    for r in rel_records:
                        neighbor = r["neighbor"]
                        # 移除邻居节点的 embedding 字段
                        if 'embedding' in neighbor:
                            del neighbor['embedding']
                        relations.append({"relation_type": r["rel_type"], "relation_properties": r["rel_props"], "neighbor": neighbor})

                    results.append({"node": node, "score": score, "relations": relations})
                return results

            else:
                # query 检索 text_chunk
                cypher_chunk = """
                MATCH (n:text_chunk)
                WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS score
                ORDER BY score DESC
                LIMIT $k
                RETURN n AS node, score
                """
                result = await session.run(cypher_chunk, query_embedding=query_embedding, k=k)
                records = await result.data()

                results = []
                for record in records:
                    chunk_node = record["node"]
                    score = record["score"]
                    
                    # 移除 embedding 字段
                    if 'embedding' in chunk_node:
                        del chunk_node['embedding']

                    # 查询该 chunk 包含的实体
                    cypher_entities = """
                    MATCH (chunk {name: $chunk_id})-[:包含实体]->(e)
                    RETURN e AS entity
                    """
                    entity_result = await session.run(cypher_entities, chunk_id=chunk_node["name"])
                    entity_records = await entity_result.data()

                    entities = []
                    for er in entity_records:
                        entity_node = er["entity"]
                        # 移除实体节点的 embedding 字段
                        if 'embedding' in entity_node:
                            del entity_node['embedding']
                            
                        # 查询实体关系
                        cypher_rel = """
                        MATCH (n {name: $name})-[r]-(m)
                        RETURN type(r) AS rel_type, properties(r) AS rel_props, m AS neighbor
                        """
                        rel_result = await session.run(cypher_rel, name=entity_node["name"])
                        rel_records = await rel_result.data()
                        
                        relations = []
                        for r in rel_records:
                            neighbor = r["neighbor"]
                            # 移除邻居节点的 embedding 字段
                            if 'embedding' in neighbor:
                                del neighbor['embedding']
                            relations.append({"relation_type": r["rel_type"], "relation_properties": r["rel_props"], "neighbor": neighbor})

                        entities.append({"node": entity_node, "relations": relations})

                    results.append({"chunk": chunk_node, "score": score, "entities": entities})

                return results
