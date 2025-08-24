import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import logging
import re

from rag_factory.Store.GraphStore.event_graphrag_neo4j import HyperRAGNeo4jStore
from rag_factory.Embed import Embeddings
from rag_factory.llms.llm_base import LLMBase

logger = logging.getLogger(__name__)

@dataclass
class SeedNode:
    """种子节点"""
    id_: str
    name: str
    type: str  # 'entity' or 'event'
    score: float
    source: str  # 'extracted' or 'linked'

@dataclass
class PPRResult:
    """PageRank结果"""
    node_scores: Dict[str, float]
    chunk_scores: Dict[str, float]
    traversal_path: List[str]

@dataclass
class RetrievalContext:
    """检索上下文"""
    chunks: List[Dict[str, Any]]
    chunk_scores: List[float]
    evidence_sources: List[str]
    seed_nodes: List[SeedNode]
    ppr_result: PPRResult

@dataclass
class GenerationResult:
    """生成结果"""
    answer: str
    evidence_chunks: List[str]
    citations: List[str]
    confidence: float
    retrieval_context: RetrievalContext

class EventGraphRetriever:
    """
    实现完整的5步流程：
    1. 查询分析与意图理解
    2. 种子节点识别
    3. 个性化PageRank图遍历
    4. Chunk分数计算与排序
    5. 上下文构建与答案生成
    """
    
    def __init__(self,
                 graph_store: HyperRAGNeo4jStore,
                 embedding_model: Embeddings,
                 llm_model: LLMBase,
                 max_seed_nodes: int = 10,
                 ppr_iterations: int = 20,
                 ppr_damping: float = 0.85,
                 top_k_chunks: int = 5,
                 similarity_threshold: float = 0.7,
                 enable_fallback: bool = True):
        """
        初始化在线检索系统
        
        Args:
            graph_store: 图存储实例
            embedding_model: 嵌入模型
            llm_model: 大语言模型
            max_seed_nodes: 最大种子节点数
            ppr_iterations: PageRank迭代次数
            ppr_damping: PageRank阻尼因子
            top_k_chunks: 返回的chunk数量
            similarity_threshold: 相似度阈值
            enable_fallback: 是否启用兜底机制
        """
        self.graph_store = graph_store
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.max_seed_nodes = max_seed_nodes
        self.ppr_iterations = ppr_iterations
        self.ppr_damping = ppr_damping
        self.top_k_chunks = top_k_chunks
        self.similarity_threshold = similarity_threshold
        self.enable_fallback = enable_fallback
        
        # 缓存数据
        self.entity_data = []
        self.entity_embeddings = np.array([])
        self.event_data = []
        self.event_embeddings = np.array([])
        self.chunk_data = []
        self.chunk_embeddings = np.array([])
        
        # Graph adjacency for PPR
        self.graph_adjacency = {}
        
    async def initialize(self):
        """初始化系统，加载图数据"""
        print("🔄 正在初始化检索系统...")
        
        # 加载实体数据
        await self._load_entity_data()
        
        # 加载事件数据
        await self._load_event_data()
        
        # 加载chunk数据
        await self._load_chunk_data()
        
        # 构建图邻接表
        await self._build_graph_adjacency()
        
        print("✅ 检索系统初始化完成")
    
    async def retrieve_and_generate(self, query: str) -> GenerationResult:
        """
        执行完整的检索与生成流程
        
        Args:
            query: 用户查询
            
        Returns:
            GenerationResult: 生成结果
        """
        print(f"🔍 开始处理查询: {query}")
        
        # Step 1: 查询分析与意图理解
        extracted_entities, extracted_events = await self._step1_query_analysis(query)
        
        # Step 2: 种子节点识别
        seed_nodes = await self._step2_seed_identification(
            extracted_entities, extracted_events, query
        )
        
        # 如果没有找到种子节点，使用兜底机制
        if not seed_nodes and self.enable_fallback:
            print("⚠️ 未找到种子节点，启用稠密召回兜底机制")
            return await self._fallback_dense_retrieval(query)
        
        # Step 3: 个性化PageRank图遍历
        ppr_result = await self._step3_personalized_pagerank(seed_nodes)
        
        # Step 4: Chunk分数计算与排序
        ranked_chunks = await self._step4_chunk_scoring(ppr_result)
        
        # Step 5: 上下文构建与答案生成
        generation_result = await self._step5_answer_generation(
            query, ranked_chunks, seed_nodes, ppr_result
        )
        
        return generation_result
    
    async def _step1_query_analysis(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Step 1: 查询分析与意图理解
        
        从查询中提取实体和事件
        """
        print("📋 Step 1: 查询分析与意图理解")
        
        # 实体提取prompt
        entity_prompt = f"""
请从以下查询中提取所有重要的实体名称。只返回实体列表，用逗号分隔。

查询: {query}

实体列表:
"""
        
        # 事件提取prompt
        event_prompt = f"""
请从以下查询中提取所有事件或动作。只返回事件列表，用逗号分隔。

查询: {query}

事件列表:
"""
        
        # 调用LLM提取实体和事件
        try:
            entity_response = await self.llm_model.achat([{"role": "user", "content": entity_prompt}])
            event_response = await self.llm_model.achat([{"role": "user", "content": event_prompt}])
            
            # 解析响应
            entities = [e.strip() for e in entity_response.split(',') if e.strip()]
            events = [e.strip() for e in event_response.split(',') if e.strip()]
            
            print(f"  提取到实体: {entities}")
            print(f"  提取到事件: {events}")
            
            return entities, events
            
        except Exception as e:
            print(f"⚠️ LLM提取失败: {e}")
            return [], []
    
    async def _step2_seed_identification(self, 
                                       entities: List[str], 
                                       events: List[str], 
                                       query: str) -> List[SeedNode]:
        """
        Step 2: 种子节点识别
        
        将实体和事件映射到知识图谱中的节点
        """
        print("🎯 Step 2: 种子节点识别")
        
        seed_nodes = []
        
        # 2.1 实体链接
        for entity in entities:
            linked_nodes = await self._link_entity(entity)
            seed_nodes.extend(linked_nodes)
        
        # 2.2 事件链接
        for event in events:
            linked_nodes = await self._link_event(event)
            seed_nodes.extend(linked_nodes)
        
        # 2.3 兜底逻辑：从事件中提取参与实体
        if not seed_nodes and events:
            print("  ⚠️ 直接链接失败，从事件中提取参与实体")
            for event in events:
                participant_entities = await self._extract_event_participants(event)
                for entity in participant_entities:
                    linked_nodes = await self._link_entity(entity)
                    seed_nodes.extend(linked_nodes)
        
        # 2.4 最终兜底：向量相似度匹配
        if not seed_nodes:
            print("  ⚠️ 实体链接失败，使用向量相似度匹配")
            seed_nodes = await self._vector_based_linking(query)
        
        # 去重并限制数量
        seen_ids = set()
        unique_seeds = []
        for node in seed_nodes:
            if node.id_ not in seen_ids:
                seen_ids.add(node.id_)
                unique_seeds.append(node)
                
        result_seeds = unique_seeds[:self.max_seed_nodes]
        
        print(f"  识别到 {len(result_seeds)} 个种子节点")
        for node in result_seeds:
            print(f"    - {node.name} ({node.type}, 分数: {node.score:.3f})")
        
        return result_seeds
    
    async def _step3_personalized_pagerank(self, seed_nodes: List[SeedNode]) -> PPRResult:
        """
        Step 3: 个性化PageRank图遍历
        
        从种子节点向外扩散，发现相关的实体和事件
        """
        print("🌐 Step 3: 个性化PageRank图遍历")
        
        if not seed_nodes:
            return PPRResult({}, {}, [])
        
        # 初始化节点分数
        node_scores = defaultdict(float)
        seed_scores = {}
        
        # 设置种子节点初始分数
        total_seed_score = sum(node.score for node in seed_nodes)
        for node in seed_nodes:
            normalized_score = node.score / total_seed_score if total_seed_score > 0 else 1.0 / len(seed_nodes)
            seed_scores[node.id_] = normalized_score
            node_scores[node.id_] = normalized_score
        
        traversal_path = []
        
        # PageRank迭代
        for iteration in range(self.ppr_iterations):
            new_scores = defaultdict(float)
            
            # 遍历所有节点
            for node_id, score in node_scores.items():
                if node_id in self.graph_adjacency:
                    neighbors = self.graph_adjacency[node_id]
                    if neighbors:
                        # 将分数平均分配给邻居
                        neighbor_score = score * self.ppr_damping / len(neighbors)
                        for neighbor_id in neighbors:
                            new_scores[neighbor_id] += neighbor_score
                
                # Teleport回种子节点
                if node_id in seed_scores:
                    new_scores[node_id] += score * (1 - self.ppr_damping) * seed_scores[node_id]
            
            # 更新分数
            node_scores = new_scores
            
            # 记录遍历路径
            if iteration % 5 == 0:
                top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                traversal_path.append(f"Iteration {iteration}: {[f'{nid}({score:.3f})' for nid, score in top_nodes]}")
        
        # 计算chunk分数
        chunk_scores = await self._compute_chunk_scores_from_nodes(node_scores)
        
        print(f"  PageRank完成，发现 {len(node_scores)} 个相关节点")
        print(f"  计算出 {len(chunk_scores)} 个chunk分数")
        
        return PPRResult(dict(node_scores), chunk_scores, traversal_path)
    
    async def _step4_chunk_scoring(self, ppr_result: PPRResult) -> List[Dict[str, Any]]:
        """
        Step 4: Chunk分数计算与排序
        
        将节点分数映射到原始chunk，得到与查询相关的文本证据
        """
        print("📊 Step 4: Chunk分数计算与排序")
        
        # 获取所有chunk及其分数
        chunk_scores = ppr_result.chunk_scores
        
        # 查询chunk详细信息
        ranked_chunks = []
        for chunk_id, score in sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True):
            chunk_info = await self._get_chunk_info(chunk_id)
            if chunk_info:
                chunk_info["ppr_score"] = score
                ranked_chunks.append(chunk_info)
        
        print(f"  排序完成，top-{min(len(ranked_chunks), self.top_k_chunks)} chunks:")
        for i, chunk in enumerate(ranked_chunks[:self.top_k_chunks]):
            print(f"    {i+1}. [分数: {chunk['ppr_score']:.3f}] {chunk['content'][:100]}...")
        
        return ranked_chunks[:self.top_k_chunks]
    
    async def _step5_answer_generation(self, 
                                     query: str, 
                                     ranked_chunks: List[Dict[str, Any]], 
                                     seed_nodes: List[SeedNode], 
                                     ppr_result: PPRResult) -> GenerationResult:
        """
        Step 5: 上下文构建与答案生成
        
        基于高分chunk生成最终答案
        """
        print("🤖 Step 5: 上下文构建与答案生成")
        
        if not ranked_chunks:
            return GenerationResult(
                answer="抱歉，我无法在知识库中找到相关信息来回答您的问题。",
                evidence_chunks=[],
                citations=[],
                confidence=0.0,
                retrieval_context=RetrievalContext([], [], [], seed_nodes, ppr_result)
            )
        
        # 构建上下文
        context_chunks = ranked_chunks[:self.top_k_chunks]
        context_text = ""
        citations = []
        
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n[证据{i+1}]: {chunk['content']}\n"
            citations.append(f"证据{i+1}")
        
        # 构建生成prompt
        generation_prompt = f"""请根据以下背景信息，回答用户的问题。请确保：
1. 回答基于提供的证据
2. 在回答中引用相关证据
3. 如果信息不足，请明确说明

[背景信息]
{context_text}

[用户问题]
{query}

[你的回答]
"""
        
        try:
            # 生成答案
            response = await self.llm_model.achat([{"role": "user", "content": generation_prompt}])
            
            # 计算置信度（基于chunk分数和数量）
            confidence = min(1.0, sum(chunk.get("ppr_score", 0) for chunk in context_chunks) / len(context_chunks))
            
            print(f"  答案生成完成，置信度: {confidence:.3f}")
            
            return GenerationResult(
                answer=response,
                evidence_chunks=[chunk["content"] for chunk in context_chunks],
                citations=citations,
                confidence=confidence,
                retrieval_context=RetrievalContext(
                    context_chunks,
                    [chunk.get("ppr_score", 0) for chunk in context_chunks],
                    citations,
                    seed_nodes,
                    ppr_result
                )
            )
            
        except Exception as e:
            print(f"⚠️ 答案生成失败: {e}")
            return GenerationResult(
                answer="抱歉，在生成答案时遇到了问题。",
                evidence_chunks=[chunk["content"] for chunk in context_chunks],
                citations=citations,
                confidence=0.0,
                retrieval_context=RetrievalContext(context_chunks, [], citations, seed_nodes, ppr_result)
            )
    
    async def _fallback_dense_retrieval(self, query: str) -> GenerationResult:
        """兜底机制：稠密向量检索"""
        print("🔄 执行稠密向量检索兜底机制")
        
        if len(self.chunk_embeddings) == 0:
            return GenerationResult(
                answer="抱歉，系统暂时无法处理您的查询。",
                evidence_chunks=[],
                citations=[],
                confidence=0.0,
                retrieval_context=RetrievalContext([], [], [], [], PPRResult({}, {}, []))
            )
        
        # 编码查询
        query_embedding = self.embedding_model.embed_documents([query])[0]
        
        # 计算相似度
        similarities = np.dot(self.chunk_embeddings, query_embedding)
        
        # 获取top-k
        top_indices = np.argsort(similarities)[-self.top_k_chunks:][::-1]
        
        top_chunks = []
        for idx in top_indices:
            chunk = self.chunk_data[idx].copy()
            chunk["similarity"] = float(similarities[idx])
            top_chunks.append(chunk)
        
        # 生成答案
        return await self._step5_answer_generation(
            query, top_chunks, [], PPRResult({}, {}, ["Dense retrieval fallback"])
        )
    
    # === 辅助方法 ===
    
    async def _load_entity_data(self):
        """加载实体数据"""
        query = """
        MATCH (e:Entity) 
        WHERE e.embedding IS NOT NULL
        RETURN e.id_ as id_, e.entity_name as name, e.entity_type as type, 
               e.entity_descriptions as descriptions, e.embedding as embedding
        """
        
        async with self.graph_store._driver.session(database=self.graph_store.database) as session:
            result = await session.run(query)
            records = await result.data()
            
            entities = []
            embeddings = []
            
            for record in records:
                entity_data = {
                    "id_": record["id_"],
                    "name": record["name"],
                    "type": record["type"],
                    "descriptions": record["descriptions"] or []
                }
                entities.append(entity_data)
                embeddings.append(record["embedding"])
            
            self.entity_data = entities
            self.entity_embeddings = np.array(embeddings) if embeddings else np.array([])
            print(f"  📊 加载了 {len(entities)} 个实体")
    
    async def _load_event_data(self):
        """加载事件数据"""
        query = """
        MATCH (e:Event) 
        WHERE e.embedding IS NOT NULL
        RETURN e.id_ as id_, e.content as content, e.type as type,
               e.participants as participants, e.embedding as embedding
        """
        
        async with self.graph_store._driver.session(database=self.graph_store.database) as session:
            result = await session.run(query)
            records = await result.data()
            
            events = []
            embeddings = []
            
            for record in records:
                event_data = {
                    "id_": record["id_"],
                    "content": record["content"],
                    "type": record["type"],
                    "participants": record["participants"] or []
                }
                events.append(event_data)
                embeddings.append(record["embedding"])
            
            self.event_data = events
            self.event_embeddings = np.array(embeddings) if embeddings else np.array([])
            print(f"  📊 加载了 {len(events)} 个事件")
    
    async def _load_chunk_data(self):
        """加载chunk数据"""
        query = """
        MATCH (c:Chunk) 
        WHERE c.embedding IS NOT NULL
        RETURN c.id_ as id_, c.content as content, c.source as source, c.embedding as embedding
        """
        
        async with self.graph_store._driver.session(database=self.graph_store.database) as session:
            result = await session.run(query)
            records = await result.data()
            
            chunks = []
            embeddings = []
            
            for record in records:
                chunk_data = {
                    "id_": record["id_"],
                    "content": record["content"],
                    "source": record["source"]
                }
                chunks.append(chunk_data)
                embeddings.append(record["embedding"])
            
            self.chunk_data = chunks
            self.chunk_embeddings = np.array(embeddings) if embeddings else np.array([])
            print(f"  📊 加载了 {len(chunks)} 个chunk")
    
    async def _build_graph_adjacency(self):
        """构建图邻接表用于PageRank"""
        print("  🔗 构建图邻接表...")
        
        queries = [
            # 实体-实体关系
            "MATCH (a:Entity)-[:ENTITY_RELATION]->(b:Entity) RETURN a.id_ as src, b.id_ as dst",
            # 实体-事件关系
            "MATCH (a:Entity)-[:PARTICIPATES_IN]->(b:Event) RETURN a.id_ as src, b.id_ as dst",
            "MATCH (a:Event)<-[:PARTICIPATES_IN]-(b:Entity) RETURN a.id_ as src, b.id_ as dst",
            # 事件-事件关系
            "MATCH (a:Event)-[:EVENT_RELATION]->(b:Event) RETURN a.id_ as src, b.id_ as dst",
            # chunk关系
            "MATCH (a:Chunk)-[:CONTAINS]->(b:Event) RETURN a.id_ as src, b.id_ as dst",
            "MATCH (a:Chunk)-[:MENTIONS]->(b:Entity) RETURN a.id_ as src, b.id_ as dst",
        ]
        
        adjacency = defaultdict(set)
        
        for query in queries:
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query)
                records = await result.data()
                
                for record in records:
                    src = record["src"]
                    dst = record["dst"]
                    adjacency[src].add(dst)
                    adjacency[dst].add(src)  # 无向图
        
        # 转换为普通dict
        self.graph_adjacency = {k: list(v) for k, v in adjacency.items()}
        
        total_edges = sum(len(neighbors) for neighbors in self.graph_adjacency.values()) // 2
        print(f"    构建完成: {len(self.graph_adjacency)} 个节点, {total_edges} 条边")
    
    async def _link_entity(self, entity_name: str) -> List[SeedNode]:
        """链接实体到图谱节点"""
        seed_nodes = []
        
        # 精确匹配
        for entity in self.entity_data:
            if entity["name"].lower() == entity_name.lower():
                seed_nodes.append(SeedNode(
                    id_=entity["id_"],
                    name=entity["name"],
                    type="entity",
                    score=1.0,
                    source="exact_match"
                ))
        
        # 如果没有精确匹配，使用向量相似度
        if not seed_nodes and len(self.entity_embeddings) > 0:
            entity_embedding = self.embedding_model.embed_documents([entity_name])[0]
            similarities = np.dot(self.entity_embeddings, entity_embedding)
            
            for i, sim in enumerate(similarities):
                if sim >= self.similarity_threshold:
                    entity = self.entity_data[i]
                    seed_nodes.append(SeedNode(
                        id_=entity["id_"],
                        name=entity["name"],
                        type="entity",
                        score=float(sim),
                        source="vector_match"
                    ))
        
        return seed_nodes
    
    async def _link_event(self, event_text: str) -> List[SeedNode]:
        """链接事件到图谱节点"""
        seed_nodes = []
        
        # 向量相似度匹配
        if len(self.event_embeddings) > 0:
            event_embedding = self.embedding_model.embed_documents([event_text])[0]
            similarities = np.dot(self.event_embeddings, event_embedding)
            
            for i, sim in enumerate(similarities):
                if sim >= self.similarity_threshold:
                    event = self.event_data[i]
                    seed_nodes.append(SeedNode(
                        id_=event["id_"],
                        name=event["content"],
                        type="event",
                        score=float(sim),
                        source="vector_match"
                    ))
        
        return seed_nodes
    
    async def _extract_event_participants(self, event_text: str) -> List[str]:
        """从Neo4j中获取事件的参与者实体"""
        # 通过向量相似度找到最相似的事件，然后获取其参与者
        if len(self.event_embeddings) == 0:
            return []
        
        # 编码事件文本
        event_embedding = self.embedding_model.embed_documents([event_text])[0]
        
        # 计算相似度
        similarities = np.dot(self.event_embeddings, event_embedding)
        
        # 找到最相似的事件
        most_similar_idx = np.argmax(similarities)
        most_similar_score = similarities[most_similar_idx]
        
        if most_similar_score < self.similarity_threshold:
            return []
        
        # 获取最相似事件的参与者
        similar_event = self.event_data[most_similar_idx]
        participants = similar_event.get("participants", [])
        
        # 查询这些参与者的详细信息
        if participants:
            query = """
            MATCH (e:Entity)
            WHERE e.entity_name IN $participants
            RETURN e.entity_name as name
            """
            
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query, participants=participants)
                records = await result.data()
                
                entity_names = [record["name"] for record in records]
                return entity_names
        
        return []
    
    async def _vector_based_linking(self, query: str) -> List[SeedNode]:
        """基于向量相似度的链接"""
        seed_nodes = []
        
        query_embedding = self.embedding_model.embed_documents([query])[0]
        
        # 搜索相似实体
        if len(self.entity_embeddings) > 0:
            similarities = np.dot(self.entity_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-5:][::-1]
            
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    entity = self.entity_data[idx]
                    seed_nodes.append(SeedNode(
                        id_=entity["id_"],
                        name=entity["name"],
                        type="entity",
                        score=float(similarities[idx]),
                        source="query_vector_match"
                    ))
        
        # 搜索相似事件
        if len(self.event_embeddings) > 0:
            similarities = np.dot(self.event_embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-5:][::-1]
            
            for idx in top_indices:
                if similarities[idx] >= self.similarity_threshold:
                    event = self.event_data[idx]
                    seed_nodes.append(SeedNode(
                        id_=event["id_"],
                        name=event["content"],
                        type="event",
                        score=float(similarities[idx]),
                        source="query_vector_match"
                    ))
        
        return seed_nodes
    
    async def _compute_chunk_scores_from_nodes(self, node_scores: Dict[str, float]) -> Dict[str, float]:
        """从节点分数计算chunk分数"""
        chunk_scores = defaultdict(float)
        
        # 查询chunk与节点的关系
        queries = [
            # chunk包含的事件
            """
            MATCH (c:Chunk)-[:CONTAINS]->(e:Event)
            WHERE e.id_ IN $node_ids
            RETURN c.id_ as chunk_id, e.id_ as node_id
            """,
            # chunk提及的实体
            """
            MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
            WHERE e.id_ IN $node_ids
            RETURN c.id_ as chunk_id, e.id_ as node_id
            """
        ]
        
        node_ids = list(node_scores.keys())
        
        for query in queries:
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query, node_ids=node_ids)
                records = await result.data()
                
                for record in records:
                    chunk_id = record["chunk_id"]
                    node_id = record["node_id"]
                    if node_id in node_scores:
                        chunk_scores[chunk_id] += node_scores[node_id]
        
        return dict(chunk_scores)
    
    async def _get_chunk_info(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取chunk详细信息"""
        for chunk in self.chunk_data:
            if chunk["id_"] == chunk_id:
                return chunk.copy()
        return None

# 使用示例
class OnlineRetrievalExample:
    """在线检索系统使用示例"""
    
    @staticmethod
    async def run_example():
        """运行示例"""
        # 这里需要根据实际情况提供store和models
        # store = HyperRAGNeo4jStore(...)
        # embedding_model = SomeEmbeddingModel(...)
        # llm_model = SomeLLMModel(...)
        
        # system = OnlineRetrievalSystem(store, embedding_model, llm_model)
        # await system.initialize()
        
        # 示例查询
        queries = [
            "苹果公司发布Vision Pro后，哪些供应商的股价受到了影响？",
            "张伟在华星科技公司的工作情况如何？",
            "AI技术的最新发展趋势是什么？"
        ]
        
        # for query in queries:
        #     print(f"\n{'='*50}")
        #     result = await system.retrieve_and_generate(query)
        #     print(f"查询: {query}")
        #     print(f"答案: {result.answer}")
        #     print(f"置信度: {result.confidence:.3f}")
        #     print(f"证据数量: {len(result.evidence_chunks)}")
        
        print("示例代码已准备完毕，请根据实际环境配置相关组件")

if __name__ == "__main__":
    asyncio.run(OnlineRetrievalExample.run_example())
