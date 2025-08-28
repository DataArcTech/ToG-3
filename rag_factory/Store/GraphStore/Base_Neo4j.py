"""
Neo4j图存储基类
提供Neo4j图数据库的通用操作功能
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
import neo4j

from rag_factory.Embed import Embeddings
from rag_factory.documents.schema import Document

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging

logger = logging.getLogger(__name__)

neo4j_retry_errors = (
    neo4j.exceptions.ServiceUnavailable,
    neo4j.exceptions.TransientError,
    neo4j.exceptions.WriteServiceUnavailable,
    neo4j.exceptions.ClientError,
)


class GraphStoreBaseNeo4j(ABC):
    """
    Neo4j图存储基类
    提供Neo4j数据库连接、查询执行、约束管理等通用功能
    """
    
    def __init__(self, url: str, username: str, password: str, database: str, embedding: Optional[Embeddings] = None):
        """
        初始化Neo4j图存储基类
        
        Args:
            url: Neo4j数据库URL
            username: 用户名
            password: 密码
            database: 数据库名
            embedding: 嵌入模型（可选）
        """
        self._driver = None
        self._driver_lock = asyncio.Lock()
        self.database = database
        self.embedding = embedding
        
        try:
            self._driver: neo4j.AsyncDriver = neo4j.AsyncGraphDatabase.driver(
                url, auth=(username, password)
            )
            logger.info(f"✅ 成功连接到Neo4j数据库: {url}")
        except Exception as e:
            logger.error(f"❌ 初始化Neo4j连接失败: {e}")
            raise

    async def close(self):
        """关闭数据库连接"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        """异步上下文管理器退出方法"""
        if self._driver:
            await self._driver.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(neo4j_retry_errors))
    async def _execute_query(self, query: str, parameters: Dict[str, Any] = None):
        """
        执行Neo4j查询的通用方法，带重试机制
        
        Args:
            query: Cypher查询语句
            parameters: 查询参数
            
        Returns:
            查询结果
        """
        if parameters is None:
            parameters = {}
            
        async with self._driver.session(database=self.database) as session:
            return await session.run(query, **parameters)

    def _generate_unique_id(self, prefix: str, content: str) -> str:
        """
        生成唯一ID
        
        Args:
            prefix: ID前缀 (如 "chunk_", "event_", "entity_")
            content: 用于生成hash的内容
            
        Returns:
            唯一ID字符串
        """
        hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        return f"{prefix}{hash_value}"

    async def filter_existing_chunks(self, documents: List[Document]) -> List[Document]:
        """
        过滤已存在的chunk，返回未处理过的chunk
        
        Args:
            documents: 待检查的文档列表
            
        Returns:
            未在Neo4j中存在的文档列表
        """
        logger.info(f"🔍 正在检查 {len(documents)} 个chunk是否已存在...")
        
        # 为文档生成chunk ID
        chunk_ids = []
        doc_to_chunk_id = {}

        # 如果文档中没有chunk_id，则生成chunk_id
        for doc in documents:
            if "chunk_id" not in doc.metadata:
                chunk_content = doc.content.strip()
                chunk_id = self._generate_unique_id("chunk_", chunk_content)
                doc.metadata["chunk_id"] = chunk_id
                chunk_ids.append(chunk_id)
            else:
                chunk_id = doc.metadata["chunk_id"]
            doc_to_chunk_id[chunk_id] = doc
        
        # 查询Neo4j中已存在的chunk ID
        existing_chunks = set()
        if chunk_ids:
            query = """
            MATCH (c:Chunk)
            WHERE c.id_ IN $chunk_ids
            RETURN c.id_ as chunk_id
            """
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, chunk_ids=chunk_ids)
                async for record in result:
                    existing_chunks.add(record["chunk_id"])
        
        # 筛选出不存在的文档
        new_documents = []
        for chunk_id, doc in doc_to_chunk_id.items():
            if chunk_id not in existing_chunks:
                new_documents.append(doc)
            else:
                logger.info(f"  ⚠️ 跳过已存在的chunk: {chunk_id}")
        
        logger.info(f"  ✅ 发现 {len(new_documents)} 个新chunk，已跳过 {len(existing_chunks)} 个重复chunk")
        return new_documents


    async def _generate_embeddings(self):
        """自动为没有embedding的节点生成嵌入向量"""
        if not self.embedding:
            logger.error("⚠️ 未提供嵌入模型，跳过向量生成")
            return
            
        logger.info("🧠 正在自动生成缺失的嵌入向量...")
        
        # 先获取总数用于进度显示
        async def get_total_count(node_type, condition="embedding IS NULL"):
            count_query = f"MATCH (n:{node_type}) WHERE n.{condition} RETURN count(n) as total"
            async with self._driver.session(database=self.database) as session:
                result = await session.run(count_query)
                record = await result.single()
                return record["total"] if record else 0
        
        # 处理Chunks
        total_chunks = await get_total_count("Chunk")
        if total_chunks > 0:
            logger.info(f"  📊 发现 {total_chunks} 个chunk需要生成嵌入向量")
            await self._process_chunk_embeddings(total_chunks)
        else:
            logger.info("  ✅ 所有chunk已有嵌入向量")
        
        # 处理Entities  
        total_entities = await get_total_count("Entity")
        if total_entities > 0:
            logger.info(f"  📊 发现 {total_entities} 个实体需要生成嵌入向量")
            await self._process_entity_embeddings(total_entities)
        else:
            logger.info("  ✅ 所有实体已有嵌入向量")
        
        # 处理Events
        total_events = await get_total_count("Event")
        if total_events > 0:
            logger.info(f"  📊 发现 {total_events} 个事件需要生成嵌入向量")
            await self._process_event_embeddings(total_events)
        else:
            logger.info("  ✅ 所有事件已有嵌入向量")

    async def _process_chunk_embeddings(self, total_count):
        """处理chunk嵌入向量生成"""
        batch_size = 100
        processed = 0
        
        while processed < total_count:
            # 每次重新查询确保获取最新的未处理数据
            query = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NULL
            RETURN c.id_ as id_, c.content as content
            LIMIT $limit
            """
            
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {"limit": batch_size})
                records = await result.data()  # 使用data()方法获取所有记录
                
                if not records:
                    break  # 没有更多需要处理的数据
                
                chunks_to_embed = []
                chunk_texts = []
                
                for record in records:
                    chunk_id = record["id_"]
                    content = record["content"] or ""
                    
                    if content.strip():  # 跳过空内容
                        chunk_texts.append(content)
                        chunks_to_embed.append(chunk_id)
                
                if chunks_to_embed:
                    logger.info(f"    🧠 处理chunk {processed + 1}-{processed + len(chunks_to_embed)}/{total_count}")
                    embeddings = self.embedding.embed_documents(chunk_texts)
                    
                    # 批量更新
                    update_query = """
                    UNWIND $updates as update
                    MATCH (c:Chunk {id_: update.id_})
                    SET c.embedding = update.embedding
                    """
                    
                    updates = [
                        {"id_": chunk_id, "embedding": embedding}
                        for chunk_id, embedding in zip(chunks_to_embed, embeddings)
                    ]
                    
                    await self._execute_query(update_query, {"updates": updates})
                    processed += len(chunks_to_embed)
                else:
                    # 如果这批记录都是空内容，标记为已处理以避免无限循环
                    empty_updates = []
                    for record in records:
                        chunk_id = record["id_"]
                        content = record["content"] or ""
                        if not content.strip():
                            empty_updates.append(chunk_id)
                    
                    if empty_updates:
                        # 为空内容的chunk设置空的embedding或标记
                        empty_query = """
                        UNWIND $ids as id_
                        MATCH (c:Chunk {id_: id_})
                        SET c.embedding = []
                        """
                        await self._execute_query(empty_query, {"ids": empty_updates})
                        processed += len(empty_updates)

    async def _process_entity_embeddings(self, total_count):
        """处理实体嵌入向量生成"""
        batch_size = 100
        processed = 0
        
        while processed < total_count:
            query = """
            MATCH (e:Entity)
            WHERE e.embedding IS NULL
            RETURN e.id_ as id_, e.entity_name as name, e.entity_descriptions as descriptions
            LIMIT $limit
            """
            
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {"limit": batch_size})
                records = await result.data()
                
                if not records:
                    break
                
                entities_to_embed = []
                entity_texts = []
                
                for record in records:
                    entity_id = record["id_"]
                    entity_name = record["name"] or ""
                    descriptions = record["descriptions"] or []
                    
                    text = f"{entity_name}: {' '.join(descriptions)}"
                    entity_texts.append(text)
                    entities_to_embed.append(entity_id)
                
                if entities_to_embed:
                    logger.info(f"    🧠 处理实体 {processed + 1}-{processed + len(entities_to_embed)}/{total_count}")
                    embeddings = self.embedding.embed_documents(entity_texts)
                    
                    update_query = """
                    UNWIND $updates as update
                    MATCH (e:Entity {id_: update.id_})
                    SET e.embedding = update.embedding
                    """
                    
                    updates = [
                        {"id_": entity_id, "embedding": embedding}
                        for entity_id, embedding in zip(entities_to_embed, embeddings)
                    ]
                    
                    await self._execute_query(update_query, {"updates": updates})
                    processed += len(entities_to_embed)

    async def _process_event_embeddings(self, total_count):
        """处理事件嵌入向量生成"""
        batch_size = 100
        processed = 0
        
        while processed < total_count:
            query = """
            MATCH (e:Event)
            WHERE e.embedding IS NULL
            RETURN e.id_ as id_, e.content as content
            LIMIT $limit
            """
            
            async with self._driver.session(database=self.database) as session:
                result = await session.run(query, {"limit": batch_size})
                records = await result.data()
                
                if not records:
                    break
                
                events_to_embed = []
                event_texts = []
                
                for record in records:
                    event_id = record["id_"]
                    content = record["content"] or ""
                    
                    if content.strip():  # 跳过空内容
                        event_texts.append(content)
                        events_to_embed.append(event_id)
                
                if events_to_embed:
                    logger.info(f"    🧠 处理事件 {processed + 1}-{processed + len(events_to_embed)}/{total_count}")
                    embeddings = self.embedding.embed_documents(event_texts)
                    
                    update_query = """
                    UNWIND $updates as update
                    MATCH (e:Event {id_: update.id_})
                    SET e.embedding = update.embedding
                    """
                    
                    updates = [
                        {"id_": event_id, "embedding": embedding}
                        for event_id, embedding in zip(events_to_embed, embeddings)
                    ]
                    
                    await self._execute_query(update_query, {"updates": updates})
                    processed += len(events_to_embed)
                else:
                    # 处理空内容的事件
                    empty_updates = []
                    for record in records:
                        event_id = record["id_"]
                        content = record["content"] or ""
                        if not content.strip():
                            empty_updates.append(event_id)
                    
                    if empty_updates:
                        empty_query = """
                        UNWIND $ids as id_
                        MATCH (e:Event {id_: id_})
                        SET e.embedding = []
                        """
                        await self._execute_query(empty_query, {"ids": empty_updates})
                        processed += len(empty_updates)

    async def _merge_duplicate_entities(self):
        """使用Louvain算法基于实体名称相似度进行社区检测和合并"""
        logger.info("🔄 正在使用Louvain算法进行实体聚类合并...")
        
        try:
            # 1. 检查GDS库是否可用
            if not await self._check_gds_availability():
                logger.warning("  ⚠️ GDS库不可用，回退到基础合并方式")
                await self._fallback_name_based_merge()
                return
            
            # 2. 创建基于实体名称相似度的图投影
            graph_name, index_to_node_id = await self._create_similarity_graph()
            if not graph_name:
                logger.warning("  ⚠️ 图投影创建失败，回退到基础合并方式")
                await self._fallback_name_based_merge()
                return
            
            # 3. 使用Louvain算法检测社区
            clusters = await self._detect_entity_clusters(graph_name, index_to_node_id)
            if not clusters:
                logger.warning("  ⚠️ 聚类检测失败或无聚类结果，回退到基础合并方式")
                await self._cleanup_resources(graph_name)
                await self._fallback_name_based_merge()
                return
            
            # 4. 合并同社区实体
            merged_count = await self._merge_clusters(clusters)
            
            # 5. 清理资源
            await self._cleanup_resources(graph_name)
            
            logger.info(f"  ✅ Louvain聚类合并完成，共合并 {merged_count} 个重复实体")
            
        except Exception as e:
            logger.error(f"  ⚠️ Louvain聚类合并失败: {e}")
            # 添加异常详细信息
            import traceback
            logger.error(f"  详细错误信息: {traceback.format_exc()}")
            
            # 清理可能残留的资源
            try:
                await self._cleanup_resources("entity_similarity_graph")
                await self._cleanup_similarity_relationships()
            except:
                pass
            
            # 回退到基础合并方式
            await self._fallback_name_based_merge()

    async def _check_gds_availability(self) -> bool:
        """检查Graph Data Science库是否可用"""
        try:
            # 首先尝试直接检查GDS版本
            check_query = "RETURN gds.version() as version"
            async with self._driver.session(database=self.database) as session:
                result = await session.run(check_query)
                record = await result.single()
                if record:
                    version = record['version']
                    logger.info(f"  ✅ GDS库可用，版本: {version}")
                    
                    # 尝试检查Neo4j版本兼容性
                    try:
                        # 尝试不同的系统过程名称
                        neo4j_version_queries = [
                            "CALL dbms.components() YIELD versions, name WHERE name = 'Neo4j Kernel' RETURN versions[0] as version",
                            "CALL dbms.components() YIELD versions, name WHERE name = 'Neo4j Kernel' RETURN versions as version",
                            "RETURN '5.0.0' as version"  # 默认版本
                        ]
                        
                        neo4j_version = None
                        for query in neo4j_version_queries:
                            try:
                                neo4j_result = await session.run(query)
                                neo4j_record = await neo4j_result.single()
                                if neo4j_record:
                                    neo4j_version = neo4j_record['version']
                                    if isinstance(neo4j_version, list):
                                        neo4j_version = neo4j_version[0]
                                    break
                            except:
                                continue
                        
                        if neo4j_version:
                            logger.info(f"  ℹ️ Neo4j版本: {neo4j_version}")
                            
                            # 检查版本兼容性
                            if not self._check_version_compatibility(neo4j_version, version):
                                logger.warning(f"  ⚠️ Neo4j版本 {neo4j_version} 与GDS版本 {version} 可能不兼容")
                                return False
                    except Exception as e:
                        logger.warning(f"  ⚠️ Neo4j版本检查失败: {e}")
                    
                    return True
                else:
                    logger.warning("  ⚠️ GDS库未安装或不可用")
                    return False
                    
        except Exception as e:
            logger.warning(f"  ⚠️ GDS库检查失败: {e}")
        
        return False
    
    def _check_version_compatibility(self, neo4j_version: str, gds_version: str) -> bool:
        """检查Neo4j和GDS版本兼容性"""
        try:
            # 提取主版本号
            neo4j_major = int(neo4j_version.split('.')[0])
            gds_major = int(gds_version.split('.')[0])
            
            # 基本兼容性检查
            if neo4j_major >= 5 and gds_major >= 2:
                return True
            elif neo4j_major >= 4 and gds_major >= 1:
                return True
            
            return False
        except:
            # 如果版本解析失败，保守地返回False
            return False

    async def _create_similarity_graph(self) -> str:
        """创建基于实体名称相似度的图投影"""
        logger.info("  🔧 创建实体相似度图...")
        
        graph_name = "entity_similarity_graph"
        
        # 清理可能存在的旧图
        await self._cleanup_resources(graph_name)
        
        # 获取所有实体的embedding
        entities_query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NOT NULL
        RETURN elementId(e) as node_id,
               e.entity_name as name,
               e.embedding as embedding
        """
        
        # 创建节点ID到索引的映射
        node_id_to_index = {}
        index_to_node_id = {}
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(entities_query)
            entities = await result.data()
        
        if len(entities) < 2:
            return None
        
        # 计算实体间的embedding余弦相似度并创建SIMILAR关系
        similarity_threshold = 0.95  # 相似度阈值
        relationships_created = 0
        
        # 导入向量计算库
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # 准备embedding数据
        embeddings = []
        entity_map = {}
        
        for i, entity in enumerate(entities):
            embedding = entity['embedding']
            if embedding:
                embeddings.append(embedding)
                entity_map[i] = entity['node_id']
                node_id_to_index[entity['node_id']] = i
                index_to_node_id[i] = entity['node_id']
        
        if len(embeddings) < 2:
            logger.info("  ℹ️ 有效embedding数量不足，跳过相似度计算")
            return None
        
        # 转换为numpy数组并计算余弦相似度矩阵
        try:
            embeddings_array = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings_array)
            
            # 创建相似度关系
            async with self._driver.session(database=self.database) as session:
                for i in range(len(embeddings)):
                    for j in range(i + 1, len(embeddings)):
                        similarity_score = similarity_matrix[i][j]
                        
                        if similarity_score >= similarity_threshold:
                            create_relation_query = """
                            MATCH (e1:Entity), (e2:Entity)
                            WHERE elementId(e1) = $node_id1 AND elementId(e2) = $node_id2
                            CREATE (e1)-[:SIMILAR {similarity: $similarity}]->(e2)
                            """
                            
                            await session.run(create_relation_query, {
                                'node_id1': entity_map[i],
                                'node_id2': entity_map[j],
                                'similarity': float(similarity_score)
                            })
                            relationships_created += 1
                            
                            # 添加调试信息
                            if relationships_created <= 5:  # 只显示前5个关系
                                logger.info(f"  🔗 创建相似度关系: {entities[i]['name']} -> {entities[j]['name']} (相似度: {similarity_score:.3f})")
            
            logger.info(f"  ✅ 创建了 {relationships_created} 个相似度关系")
            
        except Exception as e:
            logger.warning(f"  ⚠️ embedding相似度计算失败: {e}")
            relationships_created = 0
        
        # 如果没有创建任何关系，说明没有相似实体
        if relationships_created == 0:
            logger.info("  ℹ️ 没有发现相似实体，跳过图投影创建")
            return None
        
        # 创建图投影 - 使用旧版本语法
        try:
            create_projection_query = f"""
            CALL gds.graph.project(
                '{graph_name}',
                'Entity',
                'SIMILAR'
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """
            
            async with self._driver.session(database=self.database) as session:
                result = await session.run(create_projection_query)
                records = await result.data()
                if records:
                    logger.info(f"  ✅ 旧版本图投影创建成功: {records[0]['nodeCount']} 个节点, {records[0]['relationshipCount']} 个关系")
                    return graph_name, index_to_node_id
                    
        except Exception as e:
            logger.error(f"  ❌ 图投影创建失败: {e}")
            # 清理已创建的关系
            await self._cleanup_similarity_relationships()
            return None, {}
        
        return None, {}

    async def _cleanup_similarity_relationships(self):
        """清理SIMILAR关系"""
        try:
            cleanup_query = "MATCH ()-[r:SIMILAR]->() DELETE r"
            async with self._driver.session(database=self.database) as session:
                await session.run(cleanup_query)
        except Exception as e:
            logger.warning(f"  ⚠️ 清理SIMILAR关系失败: {e}")

    async def _detect_entity_clusters(self, graph_name: str, index_to_node_id: dict = None) -> dict:
        """使用Louvain算法检测实体聚类"""
        if not graph_name:
            logger.info("  ℹ️ 没有图投影，跳过聚类检测")
            return {}
            
        # 使用Louvain算法检测实体聚类
        
        try:
            louvain_query = f"""
            CALL gds.louvain.stream(
                '{graph_name}',
                {{
                    maxIterations: 10
                }}
            )
            YIELD nodeId, communityId
            WITH communityId, collect(nodeId) as nodeIds
            WHERE size(nodeIds) > 1
            RETURN communityId, nodeIds
            ORDER BY size(nodeIds) DESC
            """
            
            async with self._driver.session(database=self.database) as session:
                result = await session.run(louvain_query)
                louvain_records = await result.data()
                
        except Exception as e:
            logger.error(f"  ❌ Louvain算法失败: {e}")
            return {}
        
        # 获取聚类详情
        clusters = {}
        for record in louvain_records:
            community_id = record['communityId']
            node_ids = record['nodeIds']
            
            # 将图投影的索引ID转换为实际的节点ID
            actual_node_ids = []
            if index_to_node_id:
                for node_id in node_ids:
                    if node_id in index_to_node_id:
                        actual_node_ids.append(index_to_node_id[node_id])
            else:
                # 如果没有映射，直接使用原始ID
                actual_node_ids = [str(nid) for nid in node_ids]
            
            if not actual_node_ids:
                continue
            
            # 获取实体详细信息
            entities_query = """
            MATCH (e:Entity)
            WHERE elementId(e) IN $node_ids
            RETURN elementId(e) as node_id,
                   e.entity_name as name,
                   coalesce(e.entity_descriptions, []) as descriptions,
                   coalesce(e.mention_texts, []) as mentions,
                   coalesce(e.source_chunks, []) as sources
            """
            
            async with self._driver.session(database=self.database) as session:
                entities_result = await session.run(entities_query, {'node_ids': actual_node_ids})
                entities_records = await entities_result.data()
                entities = [dict(record) for record in entities_records]
            
            clusters[community_id] = entities
        
        logger.info(f"  ✅ 检测到 {len(clusters)} 个需要合并的聚类")
        
        # 添加调试信息
        for cluster_id, entities in clusters.items():
            entity_names = [e.get('name', '') for e in entities]
            logger.info(f"  🔍 聚类 {cluster_id}: {entity_names}")
        
        return clusters

    async def _merge_clusters(self, clusters: dict) -> int:
        """合并同社区的实体"""
        if not clusters:
            logger.info("  ℹ️ 没有需要合并的聚类")
            return 0
            
        logger.info("  🔄 开始合并实体聚类...")
        
        total_merged = 0
        
        for cluster_id, entities in clusters.items():
            if len(entities) < 2:
                continue
            
            # 选择信息最丰富的实体作为主实体
            primary_entity = max(entities, key=lambda e: (
                len(e.get('descriptions', [])) +
                len(e.get('mentions', [])) +
                len(e.get('sources', []))
            ))
            
            other_entities = [e for e in entities if e['node_id'] != primary_entity['node_id']]
            
            # 收集所有属性
            all_descriptions = set(primary_entity.get('descriptions', []))
            all_mentions = set(primary_entity.get('mentions', []))
            all_sources = set(primary_entity.get('sources', []))
            
            for entity in other_entities:
                all_descriptions.update(entity.get('descriptions', []))
                all_mentions.update(entity.get('mentions', []))
                all_sources.update(entity.get('sources', []))
            
            # 执行合并，使用elementId()替代id()，并迁移关系
            other_node_ids = [entity['node_id'] for entity in other_entities]
            
            merge_query = """
            // 找到主实体和其他实体
            MATCH (primary:Entity) WHERE elementId(primary) = $primary_id
            MATCH (other:Entity) WHERE elementId(other) IN $other_ids
            
            // 迁移其他实体的所有关系到主实体
            WITH primary, collect(other) as others
            UNWIND others as other
            
            // 处理出向关系
            OPTIONAL MATCH (other)-[r]->(target)
            WHERE NOT target:Entity OR elementId(target) <> elementId(primary)
            WITH primary, other, collect({rel: r, target: target}) as outRels
            
            // 处理入向关系  
            OPTIONAL MATCH (source)-[r]->(other)
            WHERE NOT source:Entity OR elementId(source) <> elementId(primary)
            WITH primary, other, outRels, collect({rel: r, source: source}) as inRels
            
            // 创建出向关系
            UNWIND outRels as outRel
            WITH primary, other, inRels, outRel
            WHERE outRel.rel IS NOT NULL
            CALL apoc.create.relationship(
                primary, 
                type(outRel.rel), 
                properties(outRel.rel), 
                outRel.target
            ) YIELD rel as newOutRel
            
            // 创建入向关系
            WITH primary, other, inRels
            UNWIND inRels as inRel
            WITH primary, other, inRel
            WHERE inRel.rel IS NOT NULL
            CALL apoc.create.relationship(
                inRel.source, 
                type(inRel.rel), 
                properties(inRel.rel), 
                primary
            ) YIELD rel as newInRel
            
            // 更新主实体属性
            WITH primary, other
            SET primary.entity_descriptions = $descriptions,
                primary.mention_texts = $mentions,
                primary.source_chunks = $sources,
                primary.update_time = datetime()
            
            // 删除其他实体
            DETACH DELETE other
            
            RETURN 1 as merged_count
            """
            
            # 由于上述查询较复杂，使用简化版本
            simplified_merge_query = """
            // 1. 找到主实体和其他实体
            MATCH (primary:Entity) WHERE elementId(primary) = $primary_id
            WITH primary
            MATCH (other:Entity) WHERE elementId(other) IN $other_ids
            
            // 2. 迁移其他实体的出向关系
            OPTIONAL MATCH (other)-[r]->(target)
            WHERE NOT (primary)-[]->(target) OR NOT target:Entity
            WITH primary, other, r, target
            WHERE r IS NOT NULL AND target IS NOT NULL
            CREATE (primary)-[newR]->(target)
            SET newR = properties(r)
            WITH primary, other, count(r) as out_count
            
            // 3. 迁移其他实体的入向关系  
            OPTIONAL MATCH (source)-[r]->(other)
            WHERE NOT (source)-[]->(primary) OR NOT source:Entity
            WITH primary, other, out_count, r, source
            WHERE r IS NOT NULL AND source IS NOT NULL
            CREATE (source)-[newR]->(primary)
            SET newR = properties(r)
            WITH primary, other, out_count, count(r) as in_count
            
            // 4. 更新主实体属性并删除其他实体
            SET primary.entity_descriptions = $descriptions,
                primary.mention_texts = $mentions,
                primary.source_chunks = $sources,
                primary.update_time = datetime()
            
            DETACH DELETE other
            
            RETURN 1 as merged_count
            """
            
            async with self._driver.session(database=self.database) as session:
                try:
                    result = await session.run(merge_query, {
                        'primary_id': primary_entity['node_id'],
                        'other_ids': other_node_ids,
                        'descriptions': list(all_descriptions),
                        'mentions': list(all_mentions),
                        'sources': list(all_sources)
                    })
                    merge_records = await result.data()
                    merged_count = len(merge_records)
                    total_merged += merged_count
                    
                    entity_names = [e.get('name', '') for e in entities]
                    logger.info(f"  🔄 合并聚类 {cluster_id} ({len(entities)}个): {entity_names} -> 合并了 {merged_count} 个实体")
                    
                except Exception as e:
                    logger.error(f"  ❌ 合并聚类 {cluster_id} 失败: {e}")
                    # 回退到简单合并方式
                    simple_merge_query = """
                    // 更新主实体
                    MATCH (primary:Entity) WHERE elementId(primary) = $primary_id
                    SET primary.entity_descriptions = $descriptions,
                        primary.mention_texts = $mentions,
                        primary.source_chunks = $sources,
                        primary.update_time = datetime()
                    
                    // 删除其他实体
                    WITH primary
                    MATCH (other:Entity) WHERE elementId(other) IN $other_ids
                    DETACH DELETE other
                    
                    RETURN size($other_ids) as merged_count
                    """
                    
                    result = await session.run(simple_merge_query, {
                        'primary_id': primary_entity['node_id'],
                        'other_ids': other_node_ids,
                        'descriptions': list(all_descriptions),
                        'mentions': list(all_mentions),
                        'sources': list(all_sources)
                    })
                    merge_records = await result.data()
                    merged_count = merge_records[0]['merged_count'] if merge_records else 0
                    total_merged += merged_count
                    
                    entity_names = [e.get('name', '') for e in entities]
                    logger.info(f"  🔄 简单合并聚类 {cluster_id} ({len(entities)}个): {entity_names} -> 合并了 {merged_count} 个实体")
        
        return total_merged

    async def _cleanup_resources(self, graph_name: str):
        """清理临时资源"""
        try:
            # 删除GDS图投影
            if graph_name:
                drop_query = f"CALL gds.graph.drop('{graph_name}', false)"
                async with self._driver.session(database=self.database) as session:
                    try:
                        result = await session.run(drop_query)
                        await result.consume()
                    except Exception as e:
                        # 忽略图不存在的错误
                        if "Graph with name" not in str(e) and "does not exist" not in str(e):
                            raise e
            
            # 删除临时相似度关系
            cleanup_query = """
            MATCH ()-[r:SIMILAR]-()
            DELETE r
            """
            
            async with self._driver.session(database=self.database) as session:
                result = await session.run(cleanup_query)
                await result.consume()
            
            logger.info(f"  🧹 资源清理完成")
            
        except Exception as e:
            logger.warning(f"  ⚠️ 资源清理失败: {e}")

    async def _fallback_name_based_merge(self):
        """回退到基础的名称匹配合并方式"""
        logger.info("  🔄 使用基础名称匹配进行实体合并...")
        
        merge_query = """
        MATCH (e1:Entity), (e2:Entity) 
        WHERE toLower(e1.entity_name) = toLower(e2.entity_name)
        AND elementId(e1) > elementId(e2)
        WITH e1, e2, 
             coalesce(e1.entity_descriptions, []) + coalesce(e2.entity_descriptions, []) as merged_desc,
             coalesce(e1.mention_texts, []) + coalesce(e2.mention_texts, []) as merged_mentions,
             coalesce(e1.source_chunks, []) + coalesce(e2.source_chunks, []) as merged_sources
        
        SET e2.entity_descriptions = merged_desc,
            e2.mention_texts = merged_mentions,
            e2.source_chunks = merged_sources,
            e2.update_time = datetime()
        
        DETACH DELETE e1
        
        RETURN count(e1) as merged_count
        """
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(merge_query)
            records = await result.data()
        
        merged_count = records[0]['merged_count'] if records else 0
        logger.info(f"  ✅ 基础合并完成，合并了 {merged_count} 个重复实体")

    async def get_graph_statistics(self) -> Dict[str, int]:
        """获取图统计信息"""
        queries = self._get_statistics_queries()
        
        statistics = {}
        for stat_name, query in queries.items():
            try:
                async with self._driver.session(database=self.database) as session:
                    result = await session.run(query)
                    records = await result.data()
                    if records:
                        statistics[stat_name] = records[0]["count"]
                    else:
                        statistics[stat_name] = 0
            except Exception as e:
                logger.error(f"⚠️ 获取统计信息 {stat_name} 时出错: {e}")
                statistics[stat_name] = 0
        
        return statistics

    async def delete_graph_data(self, delete_type: str = "all"):
        """
        删除图数据
        
        Args:
            delete_type: 删除类型 ("all", "entities", "events", "relations")
        """
        logger.info(f"🗑️ 正在删除图数据: {delete_type}")
        
        delete_queries = self._get_delete_queries()
        
        if delete_type not in delete_queries:
            raise ValueError(f"不支持的删除类型: {delete_type}")
        
        queries = delete_queries[delete_type]
        for query in queries:
            try:
                await self._execute_query(query)
                logger.info(f"  ✓ 执行删除查询: {query}")
            except Exception as e:
                logger.error(f"  ❌ 删除查询失败: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试数据库连接
            async with self._driver.session(database=self.database) as session:
                result = await session.run("RETURN 1 as test")
                records = await result.data()
                if not records or records[0]["test"] != 1:
                    raise Exception("数据库连接测试失败")
            
            # 获取基本统计信息
            stats = await self.get_graph_statistics()
            
            return {
                "status": "healthy",
                "database": self.database,
                "statistics": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    # =============================================================================
    # 抽象方法 - 子类必须实现
    # =============================================================================
    
    @abstractmethod
    async def store_graph(self, documents: List[Document]) -> bool:
        """
        存储图结构到Neo4j（抽象方法，子类必须实现）
        
        Args:
            documents: 文档列表
            
        Returns:
            bool: 存储是否成功
        """
        pass

    @abstractmethod
    async def _create_constraints_and_indexes(self):
        """创建数据库约束和索引（抽象方法，子类必须实现）"""
        pass

    @abstractmethod
    def _get_statistics_queries(self) -> Dict[str, str]:
        """获取统计查询语句（抽象方法，子类必须实现）"""
        pass

    @abstractmethod
    def _get_delete_queries(self) -> Dict[str, List[str]]:
        """获取删除查询语句（抽象方法，子类必须实现）"""
        pass
