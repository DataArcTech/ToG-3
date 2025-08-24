from typing import List, Dict, Any
from datetime import datetime

from rag_factory.Store.GraphStore.Base_Neo4j import GraphStoreBaseNeo4j
from rag_factory.Embed import Embeddings
from rag_factory.documents.schema import Document
from rag_factory.documents.pydantic_schema import PydanticUtils

# TODO: 批量写入，边提取边写入...
class HyperRAGNeo4jStore(GraphStoreBaseNeo4j):
    """
    HyperRAG专用的Neo4j存储类
    专门用于知识图谱的构建，处理events、mentions、entity_relations等数据
    支持chunk去重、实体合并、自动向量化等功能
    """
    
    def __init__(self, url: str, username: str, password: str, database: str, embedding: Embeddings):
        """
        初始化HyperRAG Neo4j存储
        
        Args:
            url: Neo4j数据库URL
            username: 用户名
            password: 密码
            database: 数据库名
            embedding: 嵌入模型
        """
        super().__init__(url, username, password, database, embedding)
    
    async def store_graph(self, documents: list[Document]) -> bool:
        return await self.store_hyperrag_graph(documents)
    
    async def store_hyperrag_graph(self, documents: list[Document]) -> bool:
        """
        存储HyperRAG图结构到Neo4j
        
        Args:
            documents: 文档列表(通过GraphExtractor提取的文档,包含图结构)
            
        Returns:
            bool: 存储是否成功
        """
        try:
            print("🚀 开始存储HyperRAG图结构到Neo4j...")
            
            if not documents:
                print("⚠️ 没有文档需要处理")
                return True
            
            # 过滤重复文档
            filtered_documents = await self._filter_duplicate_documents(documents)
            print(f"📊 过滤重复文档: 原始 {len(documents)} 个，过滤后 {len(filtered_documents)} 个")
            
            if not filtered_documents:
                print("⚠️ 所有文档都已存在，无需重复存储")
                return True
            
            # 创建约束和索引
            await self._create_constraints_and_indexes()

            # 提取所有数据
            all_chunks = []
            all_mentions = []
            all_events = []
            all_entity_relations = []
            all_event_relations = []
            
            for document in filtered_documents:
                # 生成chunk ID
                chunk_id = document.metadata.get("chunk_id")
                if not chunk_id:
                    chunk_id = self._generate_unique_id("chunk_", document.content)
                    document.metadata["chunk_id"] = chunk_id
                
                # 创建chunk
                chunk = {
                    "id_": chunk_id,
                    "content": document.content,
                    "source": document.metadata.get("source", "unknown"),
                    "create_time": datetime.now().isoformat()
                }
                all_chunks.append(chunk)
                
                # 提取mentions（实体）- 统一转换为字典格式
                mentions = document.metadata.get("mentions", [])
                for mention in mentions:
                    # 统一转换为字典格式
                    mention_dict = PydanticUtils.to_dict(mention)
                    
                    # 为每个mention生成唯一ID
                    entity_name = mention_dict.get('entity_name', '')
                    text = mention_dict.get('text', '')
                    mention_id = self._generate_unique_id("mention_", f"{entity_name}-{text}")
                    
                    # 添加存储所需字段
                    mention_dict.update({
                        "id_": mention_id,
                        "source_chunk": chunk_id,
                    })
                    all_mentions.append(mention_dict)
                
                # 提取events - 统一转换为字典格式
                events = document.metadata.get("events", [])
                for event in events:
                    # 统一转换为字典格式
                    event_dict = PydanticUtils.to_dict(event)
                    
                    # 为每个event生成唯一ID
                    content = event_dict.get("content", "")
                    event_id = self._generate_unique_id("event_", content)
                    
                    # 添加存储所需字段
                    event_dict.update({
                        "id_": event_id,
                        "source_chunk": chunk_id,
                    })
                    all_events.append(event_dict)
                
                # 提取实体关系 - 统一转换为字典格式
                entity_relations = document.metadata.get("entity_relations", [])
                for relation in entity_relations:
                    relation_dict = PydanticUtils.to_dict(relation)
                    relation_dict["source_chunk"] = chunk_id
                    all_entity_relations.append(relation_dict)
                
                # 提取事件关系 - 统一转换为字典格式
                event_relations = document.metadata.get("event_relations", [])
                for relation in event_relations:
                    relation_dict = PydanticUtils.to_dict(relation)
                    relation_dict["source_chunk"] = chunk_id
                    all_event_relations.append(relation_dict)
            
            # 存储数据
            print(f"📊 准备存储数据: {len(all_chunks)} chunks, {len(all_mentions)} mentions, "
                  f"{len(all_events)} events, {len(all_entity_relations)} entity relations, "
                  f"{len(all_event_relations)} event relations")
            
            # 1. 存储chunks
            if all_chunks:
                await self._store_chunks(all_chunks)
            
            # 2. 存储实体mentions（可能需要合并）
            if all_mentions:
                await self._store_mentions(all_mentions)
            
            # 3. 存储事件
            if all_events:
                await self._store_events(all_events)
            
            # 4. 存储实体关系
            if all_entity_relations:
                await self._store_entity_relations(all_entity_relations)
            
            # 5. 存储事件关系
            if all_event_relations:
                await self._store_event_relations(all_event_relations)
            
            # 6. 创建chunk-事件关系
            await self._create_chunk_event_relations(all_chunks, all_events)
            
            # 7. 创建chunk-实体关系
            await self._create_chunk_entity_relations(all_chunks, all_mentions)
            
            # 8. 创建事件-实体参与关系
            await self._create_event_mention_relations(all_events, all_mentions)
            
            # 9. 自动生成嵌入向量
            await self._generate_embeddings()
            
            # 9. 可选：使用APOC合并重复节点
            await self._merge_duplicate_entities()

            # 10. 事件消歧和同义事件挖掘
            await self._disambiguate_events_with_gds()


            print("✅ HyperRAG图结构存储完成")
            return True
            
        except Exception as e:
            print(f"❌ 存储HyperRAG图结构时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    async def _filter_duplicate_documents(self, documents: list[Document]) -> list[Document]:
        """
        过滤重复的文档
        
        Args:
            documents: 原始文档列表
            
        Returns:
            过滤后的文档列表
        """
        print("🔍 正在检查重复文档...")
        
        filtered_documents = []
        
        for document in documents:
            # 生成或获取chunk ID
            chunk_id = document.metadata.get("chunk_id")
            if not chunk_id:
                chunk_id = self._generate_unique_id("chunk_", document.content)
                document.metadata["chunk_id"] = chunk_id
            
            # 检查chunk是否已存在
            check_query = """
            MATCH (c:Chunk {id_: $chunk_id})
            RETURN c.id_ as id
            """
            
            try:
                result = await self._execute_query(check_query, {"chunk_id": chunk_id})
                records = await result.data()
                
                if not records:
                    # chunk不存在，添加到过滤后的列表
                    filtered_documents.append(document)
                else:
                    print(f"  ⏭️ 跳过重复chunk: {chunk_id}")
                    
            except Exception as e:
                print(f"  ⚠️ 检查chunk {chunk_id} 时出错: {e}")
                # 出错时保守处理，保留文档
                filtered_documents.append(document)
        
        return filtered_documents

    async def _create_constraints_and_indexes(self):
        """创建数据库约束和索引"""
        print("📋 创建数据库约束和索引...")
        
        constraints_and_indexes = [
            # Chunk约束和索引
            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id_ IS UNIQUE",
            "CREATE INDEX chunk_source_index IF NOT EXISTS FOR (c:Chunk) ON (c.source)",
            
            # Entity(Mention)约束和索引
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id_ IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            
            # 事件约束和索引
            "CREATE CONSTRAINT event_id_unique IF NOT EXISTS FOR (e:Event) REQUIRE e.id_ IS UNIQUE",
            "CREATE INDEX event_type_index IF NOT EXISTS FOR (e:Event) ON (e.type)",
            "CREATE INDEX event_source_index IF NOT EXISTS FOR (e:Event) ON (e.source_chunk)",
            
            # 事件集群索引
            "CREATE INDEX event_cluster_index IF NOT EXISTS FOR (e:Event) ON (e.cluster_id)",
            
            # 向量索引（如果支持）
            "CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS FOR (e:Entity) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
            "CREATE VECTOR INDEX event_embedding_index IF NOT EXISTS FOR (e:Event) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}"
        ]
        
        for statement in constraints_and_indexes:
            try:
                await self._execute_query(statement)
            except Exception as e:
                # 如果约束已存在或不支持向量索引，跳过
                if "already exists" in str(e) or "Unknown procedure" in str(e) or "Unsupported" in str(e):
                    continue
                print(f"⚠️ 创建约束/索引时警告: {e}")
    
    async def _store_chunks(self, chunks: List[Dict[str, Any]]):
        """存储Chunk节点"""
        print(f"📄 正在存储 {len(chunks)} 个Chunk...")
        
        for chunk in chunks:
            query = """
            MERGE (c:Chunk {id_: $id_})
            SET c.content = $content,
                c.source = $source,
                c.create_time = $create_time,
                c.update_time = datetime()
            RETURN c
            """
            
            await self._execute_query(query, {
                "id_": chunk["id_"],
                "content": chunk["content"],
                "source": chunk["source"],
                "create_time": chunk["create_time"]
            })
            print(f"  ✓ 存储Chunk: {chunk['id_']} (来源: {chunk['source']})")
    
    async def _store_mentions(self, mentions: List[Dict[str, Any]]):
        """存储实体mentions，支持同名实体的属性合并"""
        print(f"🏷️ 正在存储 {len(mentions)} 个实体mentions...")
        
        for mention in mentions:
            # 使用统一的属性获取方法
            entity_name = mention.get("entity_name", "")
            entity_type = mention.get("entity_type", "")
            entity_description = mention.get("entity_description", "")
            text = mention.get("text", "")
            source_chunk = mention.get("source_chunk", "")
            
            query = """
            MERGE (e:Entity {entity_name: $entity_name})
            ON CREATE SET 
                e.id_ = $id_,
                e.entity_type = $entity_type,
                e.entity_descriptions = [$entity_description],
                e.mention_texts = [$text],
                e.source_chunks = [$source_chunk],
                e.create_time = datetime(),
                e.update_time = datetime()
            ON MATCH SET
                e.entity_descriptions = CASE 
                    WHEN $entity_description IN e.entity_descriptions THEN e.entity_descriptions 
                    ELSE e.entity_descriptions + [$entity_description] 
                END,
                e.mention_texts = CASE 
                    WHEN $text IN e.mention_texts THEN e.mention_texts 
                    ELSE e.mention_texts + [$text] 
                END,
                e.source_chunks = CASE 
                    WHEN $source_chunk IN e.source_chunks THEN e.source_chunks 
                    ELSE e.source_chunks + [$source_chunk] 
                END,
                e.update_time = datetime()
            RETURN e
            """
            
            await self._execute_query(query, {
                "id_": mention["id_"],
                "entity_name": entity_name,
                "entity_type": entity_type,
                "entity_description": entity_description,
                "text": text,
                "source_chunk": source_chunk
            })
            print(f"  ✓ 存储实体: {entity_name} ({entity_type})")
    
    async def _store_events(self, events: List[Dict[str, Any]]):
        """存储事件节点"""
        print(f"📅 正在存储 {len(events)} 个事件...")
        
        for event in events:
            query = """
            MERGE (e:Event {id_: $id_})
            SET e.content = $content,
                e.type = $type,
                e.participants = $participants,
                e.source_chunk = $source_chunk,
                e.create_time = datetime(),
                e.update_time = datetime()
            RETURN e
            """
            
            await self._execute_query(query, {
                "id_": event["id_"],
                "content": event.get("content", ""),
                "type": event.get("type", ""),
                "participants": event.get("participants", []),
                "source_chunk": event.get("source_chunk", "")
            })
            print(f"  ✓ 存储事件: {event.get('content', '')[:50]}...")
    
    async def _store_entity_relations(self, relations: List[Dict[str, Any]]):
        """存储实体关系"""
        print(f"🔗 正在存储 {len(relations)} 个实体关系...")
        
        for relation in relations:
            head_entity = relation.get("head_entity", "")
            tail_entity = relation.get("tail_entity", "")
            relation_type = relation.get("relation_type", "")
            description = relation.get("description", "")
            source_chunk = relation.get("source_chunk", "")
            
            query = """
            MATCH (head:Entity {entity_name: $head_entity})
            MATCH (tail:Entity {entity_name: $tail_entity})
            MERGE (head)-[r:ENTITY_RELATION {type: $relation_type}]->(tail)
            SET r.description = $description,
                r.source_chunk = $source_chunk,
                r.create_time = datetime()
            RETURN r
            """
            
            await self._execute_query(query, {
                "head_entity": head_entity,
                "tail_entity": tail_entity,
                "relation_type": relation_type,
                "description": description,
                "source_chunk": source_chunk
            })
            print(f"  ✓ 存储实体关系: {head_entity} --[{relation_type}]--> {tail_entity}")
    
    async def _store_event_relations(self, relations: List[Dict[str, Any]]):
        """存储事件关系"""
        print(f"🔄 正在存储 {len(relations)} 个事件关系...")
        
        for relation in relations:
            # 支持两种引用方式：事件内容或事件ID
            head_event_content = relation.get("head_event_content")
            tail_event_content = relation.get("tail_event_content")
            
            relation_type = relation.get("relation_type", "")
            description = relation.get("description", "")
            source_chunk = relation.get("source_chunk", "")
            
            query = """
            MATCH (head:Event {content: $head_event_content})
            MATCH (tail:Event {content: $tail_event_content})
            MERGE (head)-[r:EVENT_RELATION {type: $relation_type}]->(tail)
            SET r.description = $description,
                r.source_chunk = $source_chunk,
                r.create_time = datetime()
            RETURN head.id_ as head_id, tail.id_ as tail_id
            """
            
            result = await self._execute_query(query, {
                "head_event_content": head_event_content,
                "tail_event_content": tail_event_content,
                "relation_type": relation_type,
                "description": description,
                "source_chunk": source_chunk
            })
            
            # 安全处理可能为 None 的内容
            if head_event_content:
                print_head = head_event_content[:30] + "..." if len(head_event_content) > 30 else head_event_content
            else:
                print_head = "None"
                
            if tail_event_content:
                print_tail = tail_event_content[:30] + "..." if len(tail_event_content) > 30 else tail_event_content
            else:
                print_tail = "None"
            
            rel_type_emoji = {
                "时序关系": "⏰",
                "因果关系": "🔗",
                "层级关系": "📊",
                "条件关系": "🔄"
            }.get(relation_type, "📎")
            
            print(f"  ✓ {rel_type_emoji} 存储事件关系: {print_head} --[{relation_type}]--> {print_tail}")
    
    async def _create_chunk_event_relations(self, chunks: List[Dict[str, Any]], events: List[Dict[str, Any]]):
        """创建chunk-事件关系"""
        print("📄 正在创建chunk-事件关系...")
        
        # 创建chunk_id到events的映射
        chunk_to_events = {}
        for event in events:
            source_chunk = event.get("source_chunk", "")
            if source_chunk:
                if source_chunk not in chunk_to_events:
                    chunk_to_events[source_chunk] = []
                chunk_to_events[source_chunk].append(event["id_"])
        
        for chunk in chunks:
            chunk_id = chunk["id_"]
            if chunk_id in chunk_to_events:
                for event_id in chunk_to_events[chunk_id]:
                    query = """
                    MATCH (chunk:Chunk {id_: $chunk_id})
                    MATCH (event:Event {id_: $event_id})
                    MERGE (chunk)-[:CONTAINS]->(event)
                    """
                    
                    await self._execute_query(query, {
                        "chunk_id": chunk_id,
                        "event_id": event_id
                    })
                    print(f"  ✓ Chunk {chunk_id} 包含事件 {event_id}")
    
    async def _create_chunk_entity_relations(self, chunks: List[Dict[str, Any]], mentions: List[Dict[str, Any]]):
        """创建chunk-实体关系"""
        print("🏷️ 正在创建chunk-实体关系...")
        
        # 创建chunk_id到mentions的映射
        chunk_to_entities = {}
        for mention in mentions:
            source_chunk = mention.get("source_chunk", "")
            entity_name = mention.get("entity_name", "")
            if source_chunk and entity_name:
                if source_chunk not in chunk_to_entities:
                    chunk_to_entities[source_chunk] = set()
                chunk_to_entities[source_chunk].add(entity_name)
        
        for chunk in chunks:
            chunk_id = chunk["id_"]
            if chunk_id in chunk_to_entities:
                for entity_name in chunk_to_entities[chunk_id]:
                    query = """
                    MATCH (chunk:Chunk {id_: $chunk_id})
                    MATCH (entity:Entity {entity_name: $entity_name})
                    MERGE (chunk)-[:MENTIONS]->(entity)
                    """
                    
                    await self._execute_query(query, {
                        "chunk_id": chunk_id,
                        "entity_name": entity_name
                    })
                    print(f"  ✓ Chunk {chunk_id} 提及实体 {entity_name}")
    
    async def _create_event_mention_relations(self, events: List[Dict[str, Any]], mentions: List[Dict[str, Any]]):
        """创建事件-实体参与关系"""
        print("👥 正在创建事件-实体参与关系...")
        
        for event in events:
            participants = event.get("participants", [])
            event_id = event["id_"]
            
            for participant in participants:
                # 查找对应的实体
                query = """
                MATCH (entity:Entity {entity_name: $participant})
                MATCH (event:Event {id_: $event_id})
                MERGE (entity)-[:PARTICIPATES_IN {role: "participant"}]->(event)
                """
                
                await self._execute_query(query, {
                    "participant": participant,
                    "event_id": event_id
                })
                print(f"  ✓ 👤 {participant} 参与事件 {event_id}")
    
    def _get_statistics_queries(self) -> Dict[str, str]:
        """获取统计查询语句"""
        return {
            "chunks": "MATCH (c:Chunk) RETURN count(c) as count",
            "entities": "MATCH (e:Entity) RETURN count(e) as count",
            "events": "MATCH (e:Event) RETURN count(e) as count",
            "entity_relations": "MATCH ()-[r:ENTITY_RELATION]->() RETURN count(r) as count",
            "event_relations": "MATCH ()-[r:EVENT_RELATION]->() RETURN count(r) as count",
            "similar_events": "MATCH ()-[r:SIMILAR_TO]->() RETURN count(r) as count",
            "participations": "MATCH ()-[r:PARTICIPATES_IN]->() RETURN count(r) as count",
            "contains_events": "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count",
            "mentions": "MATCH ()-[r:MENTIONS]->() RETURN count(r) as count",
            "event_clusters": "MATCH (e:Event) WHERE e.cluster_id IS NOT NULL RETURN count(DISTINCT e.cluster_id) as count",
            "chunks_with_embeddings": "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) as count",
            "entities_with_embeddings": "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) as count",
            "events_with_embeddings": "MATCH (e:Event) WHERE e.embedding IS NOT NULL RETURN count(e) as count"
        }

    def _get_delete_queries(self) -> Dict[str, List[str]]:
        """获取删除查询语句"""
        return {
            "all": [
                "MATCH (n) DETACH DELETE n",
            ],
            "entities": [
                "MATCH (e:Entity) DETACH DELETE e",
            ],
            "events": [
                "MATCH (e:Event) DETACH DELETE e",
            ],
            "relations": [
                "MATCH ()-[r:ENTITY_RELATION]->() DELETE r",
                "MATCH ()-[r:EVENT_RELATION]->() DELETE r",
                "MATCH ()-[r:SIMILAR_TO]->() DELETE r",
                "MATCH ()-[r:PARTICIPATES_IN]->() DELETE r",
                "MATCH ()-[r:CONTAINS]->() DELETE r",
                "MATCH ()-[r:MENTIONS]->() DELETE r",
            ]
        }

    async def _disambiguate_events_with_gds(self):
        """
        使用Neo4j GDS的KNN算法进行事件消歧，挖掘同义事件
        """
        print("🎯 开始事件消歧 - 使用GDS KNN挖掘同义事件...")

        try:
            # 1. 检查GDS是否可用
            gds_check_query = "RETURN gds.version() as version"
            result = await self._execute_query(gds_check_query)
            print("  ✅ Neo4j GDS插件可用")

            # 2. 创建事件嵌入向量的图投影
            projection_query = """
            CALL gds.graph.drop('event_similarity', false) YIELD graphName
            """
            try:
                await self._execute_query(projection_query)
                print("  📊 清理旧的图投影")
            except:
                pass  # 忽略不存在的图投影错误

            # 3. 创建新的图投影
            create_projection_query = """
            CALL gds.graph.project(
                'event_similarity',
                'Event',
                '*',
                {
                    nodeProperties: ['embedding']
                }
            )
            """
            await self._execute_query(create_projection_query)
            print("  📊 创建事件相似度图投影完成")

            # 4. 运行KNN算法计算相似事件
            knn_query = """
            CALL gds.knn.write(
                'event_similarity',
                {
                    topK: 10,
                    nodeProperties: ['embedding'],
                    writeRelationshipType: 'SIMILAR_TO',
                    writeProperty: 'similarity_score',
                    similarityCutoff: 0.9
                }
            )
            """
            await self._execute_query(knn_query)
            print("  🤖 KNN算法执行完成，创建相似事件关系")

            # 5. 处理已存在的事件关系，将相似度信息合并到现有关系中
            merge_similarity_query = """
            MATCH (e1:Event)-[r:EVENT_RELATION]->(e2:Event)
            MATCH (e1)-[s:SIMILAR_TO]-(e2)
            SET r.similarity_score = s.similarity_score,
                r.semantic_similarity = true,
                r.update_time = datetime()
            DELETE s
            """
            await self._execute_query(merge_similarity_query)
            print("  🔄 将相似度信息合并到现有事件关系中")

            # 6. 清理图投影
            cleanup_query = """
            CALL gds.graph.drop('event_similarity', false)
            """
            await self._execute_query(cleanup_query)
            print("  🧹 清理图投影完成")

        except Exception as e:
            print(f"  ⚠️ GDS事件消歧失败: {e}")

