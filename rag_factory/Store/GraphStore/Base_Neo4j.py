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
            print(f"✅ 成功连接到Neo4j数据库: {url}")
        except Exception as e:
            print(f"❌ 初始化Neo4j连接失败: {e}")
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
        print(f"🔍 正在检查 {len(documents)} 个chunk是否已存在...")
        
        # 为文档生成chunk ID
        chunk_ids = []
        doc_to_chunk_id = {}
        
        for doc in documents:
            # 根据内容生成唯一的chunk ID
            chunk_content = doc.content.strip()
            chunk_id = self._generate_unique_id("chunk_", chunk_content)
            chunk_ids.append(chunk_id)
            doc_to_chunk_id[chunk_id] = doc
            # 将chunk_id保存到文档metadata中
            doc.metadata["chunk_id"] = chunk_id
        
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
                print(f"  ⚠️ 跳过已存在的chunk: {chunk_id}")
        
        print(f"  ✅ 发现 {len(new_documents)} 个新chunk，已跳过 {len(existing_chunks)} 个重复chunk")
        return new_documents

    async def _generate_embeddings(self):
        """自动为没有embedding的节点生成嵌入向量"""
        if not self.embedding:
            print("⚠️ 未提供嵌入模型，跳过向量生成")
            return
            
        print("🧠 正在自动生成缺失的嵌入向量...")
        
        # 为chunk生成嵌入
        chunk_query = """
        MATCH (c:Chunk)
        WHERE c.embedding IS NULL
        RETURN c.id_ as id_, c.content as content
        LIMIT 100
        """
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(chunk_query)
            chunks_to_embed = []
            chunk_texts = []
            
            async for record in result:
                chunk_id = record["id_"]
                content = record["content"] or ""
                
                chunk_texts.append(content)
                chunks_to_embed.append(chunk_id)
            
            if chunk_texts:
                print(f"  🧠 为 {len(chunk_texts)} 个chunk生成嵌入向量...")
                embeddings = self.embedding.embed_documents(chunk_texts)
                
                for chunk_id, embedding in zip(chunks_to_embed, embeddings):
                    update_query = """
                    MATCH (c:Chunk {id_: $id_})
                    SET c.embedding = $embedding
                    """
                    await self._execute_query(update_query, {"id_": chunk_id, "embedding": embedding})
                
                print(f"  ✅ 完成 {len(embeddings)} 个chunk嵌入向量生成")
        
        # 为实体生成嵌入
        entity_query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NULL
        RETURN e.id_ as id_, e.entity_name as name, e.entity_descriptions as descriptions
        LIMIT 100
        """
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(entity_query)
            entities_to_embed = []
            entity_texts = []
            
            async for record in result:
                entity_id = record["id_"]
                entity_name = record["name"]
                descriptions = record["descriptions"] or []
                
                # 构建用于嵌入的文本
                text = f"{entity_name}: {' '.join(descriptions)}"
                entity_texts.append(text)
                entities_to_embed.append(entity_id)
            
            if entity_texts:
                print(f"  🧠 为 {len(entity_texts)} 个实体生成嵌入向量...")
                embeddings = self.embedding.embed_documents(entity_texts)
                
                for entity_id, embedding in zip(entities_to_embed, embeddings):
                    update_query = """
                    MATCH (e:Entity {id_: $id_})
                    SET e.embedding = $embedding
                    """
                    await self._execute_query(update_query, {"id_": entity_id, "embedding": embedding})
                
                print(f"  ✅ 完成 {len(embeddings)} 个实体嵌入向量生成")
        
        # 为事件生成嵌入
        event_query = """
        MATCH (e:Event)
        WHERE e.embedding IS NULL
        RETURN e.id_ as id_, e.content as content
        LIMIT 100
        """
        
        async with self._driver.session(database=self.database) as session:
            result = await session.run(event_query)
            events_to_embed = []
            event_texts = []
            
            async for record in result:
                event_id = record["id_"]
                content = record["content"] or ""
                
                event_texts.append(content)
                events_to_embed.append(event_id)
            
            if event_texts:
                print(f"  🧠 为 {len(event_texts)} 个事件生成嵌入向量...")
                embeddings = self.embedding.embed_documents(event_texts)
                
                for event_id, embedding in zip(events_to_embed, embeddings):
                    update_query = """
                    MATCH (e:Event {id_: $id_})
                    SET e.embedding = $embedding
                    """
                    await self._execute_query(update_query, {"id_": event_id, "embedding": embedding})
                
                print(f"  ✅ 完成 {len(embeddings)} 个事件嵌入向量生成")

    async def _merge_duplicate_entities(self):
        """使用APOC合并可能重复的实体节点（可选功能）"""
        print("🔄 正在使用APOC合并重复实体...")
        
        try:
            # 检查APOC是否可用
            apoc_check_query = "RETURN apoc.version() as version"
            await self._execute_query(apoc_check_query)
            print("  ✅ APOC插件可用，开始合并重复实体")
            
            # 查找同名实体并合并
            merge_query = """
            CALL apoc.periodic.iterate(
                "MATCH (e1:Entity), (e2:Entity) 
                 WHERE e1.entity_name = e2.entity_name AND id(e1) > id(e2) 
                 RETURN e1, e2",
                "CALL apoc.refactor.mergeNodes([e1, e2], {
                    properties: {
                        entity_descriptions: 'combine',
                        mention_texts: 'combine',
                        source_chunks: 'combine',
                        update_time: 'overwrite'
                    }
                }) YIELD node RETURN node",
                {batchSize: 10, parallel: false}
            )
            """
            
            await self._execute_query(merge_query)
            print("  ✅ 完成实体合并")
            
        except Exception as e:
            print(f"  ⚠️ APOC合并功能不可用或失败: {e}")
            print("  💡 建议安装APOC插件以获得更好的实体合并功能")

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
                print(f"⚠️ 获取统计信息 {stat_name} 时出错: {e}")
                statistics[stat_name] = 0
        
        return statistics

    async def delete_graph_data(self, delete_type: str = "all"):
        """
        删除图数据
        
        Args:
            delete_type: 删除类型 ("all", "entities", "events", "relations")
        """
        print(f"🗑️ 正在删除图数据: {delete_type}")
        
        delete_queries = self._get_delete_queries()
        
        if delete_type not in delete_queries:
            raise ValueError(f"不支持的删除类型: {delete_type}")
        
        queries = delete_queries[delete_type]
        for query in queries:
            try:
                await self._execute_query(query)
                print(f"  ✓ 执行删除查询: {query}")
            except Exception as e:
                print(f"  ❌ 删除查询失败: {e}")

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
