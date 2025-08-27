import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Union

from collections import defaultdict
import logging

from rag_factory.store.graph.event_graphrag import HyperRAGNeo4jStore
from rag_factory.embeddings.base import Embeddings
from rag_factory.llm.base import LLMBase
from rag_factory.data_model.graph import GenerationResult, RetrievalContext, PPRResult, RetrievalItem, SeedNode

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)



class QueryPreference(BaseModel):
    """查询偏好"""
    chunk_weight: float = Field(default=0.5, description="chunk权重", ge=0, le=1)
    event_weight: float = Field(default=0.5, description="event权重", ge=0, le=1)
    
    @classmethod
    def model_validate(cls, v):
        """确保chunk_weight + event_weight = 1"""
        if isinstance(v, dict) and abs(v.get('chunk_weight', 0.5) + v.get('event_weight', 0.5) - 1.0) > 0.01:
            raise ValueError("chunk_weight和event_weight之和必须为1")
        return super().model_validate(v)


class QueryAnalysisResult(BaseModel):
    """查询分析结果"""
    extracted_entities: List[str] = Field(default_factory=list, description="提取的实体")
    extracted_events: List[str] = Field(default_factory=list, description="提取的事件")
    query_preference: QueryPreference = Field(default_factory=QueryPreference, description="查询偏好")


# ===== 主要类定义 =====

class EventGraphRetriever:
    """基于事件图谱的检索器"""
    
    # 常量定义
    DEFAULT_SIMILARITY_THRESHOLD = 0.8
    DEFAULT_PPR_DAMPING = 0.85
    DEFAULT_CONVERGENCE_TOLERANCE = 1e-6
    
    # 权重配置
    SOURCE_WEIGHTS = {
        "exact_match": 1.0,
        "vector_match": 0.8,
        "query_vector_match": 0.6,
        "extracted": 0.9
    }
    
    TYPE_WEIGHTS = {
        "entity": 1.0,
        "event": 1.1  # 事件通常更重要
    }
    
    def __init__(self,
                 graph_store: HyperRAGNeo4jStore,
                 embedding_model: Embeddings,
                 llm_model: LLMBase,
                 max_seed_nodes: int = 10,
                 ppr_iterations: int = 50,
                 ppr_damping: float = DEFAULT_PPR_DAMPING,
                 top_k_items: int = 10,
                 similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                 enable_fallback: bool = True,
                 chunk_event_balance: float = 0.5,
                 convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE):
        """
        初始化检索系统
        
        Args:
            chunk_event_balance: 0.0偏向chunk，1.0偏向event，0.5平衡
            convergence_tolerance: PageRank收敛容忍度
        """
        # 核心组件
        self.graph_store = graph_store
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # 配置参数
        self.max_seed_nodes = max_seed_nodes
        self.ppr_iterations = ppr_iterations
        self.ppr_damping = ppr_damping
        self.top_k_items = top_k_items
        self.similarity_threshold = similarity_threshold
        self.enable_fallback = enable_fallback
        self.chunk_event_balance = chunk_event_balance
        self.convergence_tolerance = convergence_tolerance
        
        # 数据缓存
        self._reset_cache()
        
        # 初始化标志
        self.is_initialized = False
    
    def _reset_cache(self):
        """重置缓存数据"""
        self.entity_data = []
        self.entity_embeddings = np.array([])
        self.event_data = []
        self.event_embeddings = np.array([])
        self.chunk_data = []
        self.chunk_embeddings = np.array([])
        self.graph_adjacency = {}
        self.node_types = {}
    
    # ===== 初始化方法 =====
    
    async def initialize(self):
        """初始化系统，加载图数据"""
        if self.is_initialized:
            logger.info("系统已初始化，跳过")
            return
            
        logger.info("🔄 正在初始化检索系统...")
        
        try:
            await self._load_entity_data()
            await self._load_event_data() 
            await self._load_chunk_data()
            await self._build_directed_graph()
            
            self.is_initialized = True
            logger.info("✅ 检索系统初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            self._reset_cache()
            raise
    
    async def _load_entity_data(self):
        """加载实体数据"""
        query = """
        MATCH (e:Entity) 
        WHERE e.embedding IS NOT NULL
        RETURN e.id_ as id_, e.entity_name as name, e.entity_type as type, 
               e.entity_descriptions as descriptions, e.embedding as embedding
        """
        
        try:
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query)
                records = await result.data()
                
                entities = []
                embeddings = []
                
                for record in records:
                    entity_data = {
                        "id_": record["id_"],
                        "name": record["name"] or "",
                        "type": record["type"] or "unknown",
                        "descriptions": record["descriptions"] or []
                    }
                    entities.append(entity_data)
                    embeddings.append(record["embedding"])
                
                self.entity_data = entities
                self.entity_embeddings = np.array(embeddings) if embeddings else np.array([])
                logger.info(f"  📊 加载了 {len(entities)} 个实体")
                
        except Exception as e:
            logger.error(f"加载实体数据失败: {e}")
            raise
    
    async def _load_event_data(self):
        """加载事件数据"""
        query = """
        MATCH (e:Event) 
        WHERE e.embedding IS NOT NULL
        RETURN e.id_ as id_, e.content as content, e.type as type,
               e.participants as participants, e.embedding as embedding
        """
        
        try:
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query)
                records = await result.data()
                
                events = []
                embeddings = []
                
                for record in records:
                    event_data = {
                        "id_": record["id_"],
                        "content": record["content"] or "",
                        "type": record["type"] or "unknown",
                        "participants": record["participants"] or []
                    }
                    events.append(event_data)
                    embeddings.append(record["embedding"])
                
                self.event_data = events
                self.event_embeddings = np.array(embeddings) if embeddings else np.array([])
                logger.info(f"  📊 加载了 {len(events)} 个事件")
                
        except Exception as e:
            logger.error(f"加载事件数据失败: {e}")
            raise
    
    async def _load_chunk_data(self):
        """加载chunk数据"""
        query = """
        MATCH (c:Chunk) 
        WHERE c.embedding IS NOT NULL
        RETURN c.id_ as id_, c.content as content, c.source as source, c.embedding as embedding
        """
        
        try:
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query)
                records = await result.data()
                
                chunks = []
                embeddings = []
                
                for record in records:
                    chunk_data = {
                        "id_": record["id_"],
                        "content": record["content"] or "",
                        "source": record["source"] or "unknown"
                    }
                    chunks.append(chunk_data)
                    embeddings.append(record["embedding"])
                
                self.chunk_data = chunks
                self.chunk_embeddings = np.array(embeddings) if embeddings else np.array([])
                logger.info(f"  📊 加载了 {len(chunks)} 个chunk")
                
        except Exception as e:
            logger.error(f"加载chunk数据失败: {e}")
            raise
    
    async def _build_directed_graph(self):
        """构建有向图邻接表"""
        logger.info("  🔗 构建有向图邻接表...")
        
        # 定义有向关系
        # directed_relations = [
        #     # 实体参与事件（单向：实体 -> 事件）
        #     ("MATCH (e:Entity)-[:PARTICIPATES_IN]->(v:Event) RETURN e.id_ as src, v.id_ as dst", True),
        #     # chunk包含事件（单向：chunk -> 事件）  
        #     ("MATCH (c:Chunk)-[:CONTAINS]->(e:Event) RETURN c.id_ as src, e.id_ as dst", True),
        #     # chunk提及实体（单向：chunk -> 实体）
        #     ("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c.id_ as src, e.id_ as dst", True),
        #     # 实体间关系（双向）
        #     ("MATCH (a:Entity)-[:ENTITY_RELATION]->(b:Entity) RETURN a.id_ as src, b.id_ as dst", False),
        #     # 事件间关系（双向）
        #     ("MATCH (a:Event)-[:EVENT_RELATION]->(b:Event) RETURN a.id_ as src, b.id_ as dst", False),
        # ]

        directed_relations = [
            # 实体参与事件（单向：实体 -> 事件）
            ("MATCH (e:Entity)-[:PARTICIPATES_IN]->(v:Event) RETURN e.id_ as src, v.id_ as dst", False),
            # chunk包含事件（单向：chunk -> 事件）  
            ("MATCH (c:Chunk)-[:CONTAINS]->(e:Event) RETURN c.id_ as src, e.id_ as dst", False),
            # chunk提及实体（单向：chunk -> 实体）
            ("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) RETURN c.id_ as src, e.id_ as dst", False),
            # 实体间关系（双向）
            ("MATCH (a:Entity)-[:ENTITY_RELATION]->(b:Entity) RETURN a.id_ as src, b.id_ as dst", True),
            # 事件间关系（双向）
            ("MATCH (a:Event)-[:EVENT_RELATION]->(b:Event) RETURN a.id_ as src, b.id_ as dst", True),
        ]
        
        try:
            adjacency = defaultdict(set)
            
            for query, is_directed in directed_relations:
                async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                    result = await session.run(query)
                    records = await result.data()
                    
                    for record in records:
                        src = record["src"]
                        dst = record["dst"]
                        if src and dst:  # 确保ID有效
                            adjacency[src].add(dst)
                            
                            # 双向关系
                            if not is_directed:
                                adjacency[dst].add(src)
            
            # 构建节点类型映射
            await self._build_node_type_mapping()
            
            # 确保所有节点都在邻接表中
            all_nodes = set(self.node_types.keys())
            for node_id in all_nodes:
                if node_id not in adjacency:
                    adjacency[node_id] = set()
            
            # 转换为普通dict
            self.graph_adjacency = {k: list(v) for k, v in adjacency.items()}
            
            total_edges = sum(len(neighbors) for neighbors in self.graph_adjacency.values())
            logger.info(f"    构建完成: {len(self.graph_adjacency)} 个节点, {total_edges} 条有向边")
            
        except Exception as e:
            logger.error(f"构建图结构失败: {e}")
            raise
    
    async def _build_node_type_mapping(self):
        """构建节点类型映射"""
        queries = [
            ("MATCH (e:Entity) RETURN e.id_ as id_, 'entity' as type", "entity"),
            ("MATCH (e:Event) RETURN e.id_ as id_, 'event' as type", "event"), 
            ("MATCH (c:Chunk) RETURN c.id_ as id_, 'chunk' as type", "chunk")
        ]
        
        for query, node_type in queries:
            async with self.graph_store._driver.session(database=self.graph_store.database) as session:
                result = await session.run(query)
                records = await result.data()
                
                for record in records:
                    node_id = record["id_"]
                    if node_id:  # 确保ID有效
                        self.node_types[node_id] = node_type
    

    # ===== 主要检索流程 =====
    
    async def retrieve_and_generate(self, query: str) -> GenerationResult:
        """执行完整的检索与生成流程"""
        if not self.is_initialized:
            await self.initialize()
            
        logger.info(f"🔍 开始处理查询: {query}")
        
        try:
            # Step 1: 查询分析与意图理解
            entities, events, preference = await self._step1_query_analysis(query)
            
            # Step 2: 种子节点识别
            seed_nodes = await self._step2_seed_identification(entities, events, query)
            
            if not seed_nodes and self.enable_fallback:
                logger.warning("⚠️ 未找到种子节点，启用稠密召回兜底机制")
                return await self._fallback_dense_retrieval(query)
            
            # Step 3: 个性化PageRank
            ppr_result = await self._step3_personalized_pagerank(seed_nodes)
            
            # Step 4: 选择chunk或event
            ranked_items = await self._step4_item_selection(
                ppr_result, query, preference
            )
            
            # Step 5: 上下文构建与答案生成
            generation_result = await self._step5_answer_generation(
                query, ranked_items, seed_nodes, ppr_result
            )
            
            return generation_result
            
        except Exception as e:
            logger.error(f"检索生成过程失败: {e}")
            # 返回错误结果而不是抛出异常
            return GenerationResult(
                answer=f"处理查询时遇到错误: {str(e)}",
                evidence_items=[],
                citations=[],
                confidence=0.0,
                retrieval_context=RetrievalContext([], [], PPRResult({}, {}, [], {}))
            )
    
    async def _step1_query_analysis(self, query: str) -> Tuple[List[str], List[str], QueryPreference]:
        """Step 1: 查询分析与意图理解"""
        logger.info("📋 Step 1: 查询分析与意图理解")
        
        prompt = self._build_query_analysis_prompt(query)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self.llm_model.aparse_chat(messages, QueryAnalysisResult)
            
            entities = response.extracted_entities
            events = response.extracted_events
            preference = response.query_preference
            
            logger.info(f"  提取到实体: {entities}")
            logger.info(f"  提取到事件: {events}")
            logger.info(f"  查询偏好: chunk={preference.chunk_weight:.2f}, event={preference.event_weight:.2f}")
            
            return entities, events, preference
            
        except Exception as e:
            logger.warning(f"⚠️ 查询分析失败: {e}，使用默认配置")
            return [], [], QueryPreference()
    
    def _build_query_analysis_prompt(self, query: str) -> str:
        """构建查询分析的prompt"""
        return f"""
你是一个专业的行测资料分析专家。请分析用户查询并提取关键信息：

1. **提取实体 (extracted_entities)**：
   - 识别行测资料分析中的专业术语和概念
   - 例如：'同比增长率', '环比增长率', '比重', '平均数', '基期', '现期'等
   - 忽略具体的公司名、产品名等背景信息

2. **提取事件 (extracted_events)**：
   - 识别查询对应的行测分析题型或解题行为
   - 例如：'计算增长量', '计算增长率', '比较增长快慢', '计算比重'等

3. **设定查询偏好 (query_preference)**：
   - 如果查询需要具体数据计算，设置更高的chunk_weight（0.7-0.8）
   - 如果查询询问概念或方法，设置更高的event_weight（0.7-0.8）
   - 标准计算题可设置均等权重（各0.5）

查询: {query}

请严格按照JSON格式返回结果。
"""
    
    async def _step2_seed_identification(self, 
                                       entities: List[str], 
                                       events: List[str], 
                                       query: str) -> List[SeedNode]:
        """Step 2: 种子节点识别"""
        logger.info("🎯 Step 2: 种子节点识别")
        
        seed_nodes = []
        
        try:
            # 实体链接
            for entity in entities:
                linked_nodes = await self._link_entity(entity)
                seed_nodes.extend(linked_nodes)
            
            # 事件链接
            for event in events:
                linked_nodes = await self._link_event(event)
                seed_nodes.extend(linked_nodes)
            
            # 向量检索兜底
            if not seed_nodes:
                logger.info("  ⚠️ 使用向量检索兜底")
                return []
            
            # 去重并限制数量
            unique_seeds = self._deduplicate_seed_nodes(seed_nodes)
            result_seeds = unique_seeds[:self.max_seed_nodes]
            
            logger.info(f"  识别到 {len(result_seeds)} 个种子节点")
            for node in result_seeds:
                logger.info(f"    - {node.name} ({node.type}, 分数: {node.score:.3f})")
            
            return result_seeds
            
        except Exception as e:
            logger.error(f"种子节点识别失败: {e}")
            return []
    
    def _deduplicate_seed_nodes(self, seed_nodes: List[SeedNode]) -> List[SeedNode]:
        """去重种子节点，保留最高分的"""
        seen_ids = {}
        for node in seed_nodes:
            if node.id_ not in seen_ids or node.score > seen_ids[node.id_].score:
                seen_ids[node.id_] = node
        return list(seen_ids.values())
    
    
    async def _step3_personalized_pagerank(self, seed_nodes: List[SeedNode]) -> PPRResult:
        """Step 3: 个性化PageRank"""
        logger.info("🌐 Step 3: 个性化PageRank")
        
        if not seed_nodes:
            return PPRResult({}, {}, [], {"converged": False, "total_nodes": 0})
        
        try:
            # 计算种子节点权重
            seed_weights = self._compute_seed_weights(seed_nodes)
            
            # 初始化所有节点分数
            all_nodes = set(self.graph_adjacency.keys())
            for neighbors in self.graph_adjacency.values():
                all_nodes.update(neighbors)
            
            # 运行PageRank
            node_scores, convergence_info = self._run_pagerank(all_nodes, seed_weights)
            
            # 计算item得分
            item_scores = await self._compute_item_scores_from_nodes(node_scores)
            
            # 构建遍历路径（用于调试）
            traversal_path = self._build_traversal_path(node_scores, convergence_info)
            
            logger.info(f"  PageRank完成，计算出 {len(item_scores)} 个item分数")
            
            return PPRResult(node_scores, item_scores, traversal_path, convergence_info)
            
        except Exception as e:
            logger.error(f"PageRank计算失败: {e}")
            return PPRResult({}, {}, [], {"converged": False, "error": str(e)})
    
    def _compute_seed_weights(self, seed_nodes: List[SeedNode]) -> Dict[str, float]:
        """计算种子节点权重"""
        weights = {}
        
        # 计算加权分数
        for node in seed_nodes:
            base_weight = node.score
            source_mult = self.SOURCE_WEIGHTS.get(node.source, 0.5)
            type_mult = self.TYPE_WEIGHTS.get(node.type, 0.5)
            
            final_weight = base_weight * source_mult * type_mult
            weights[node.id_] = final_weight
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # 均匀分布兜底
            uniform_weight = 1.0 / len(seed_nodes)
            weights = {node.id_: uniform_weight for node in seed_nodes}
        
        return weights
    
    def _run_pagerank(self, 
                     all_nodes: Set[str], 
                     seed_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """运行PageRank算法"""
        # 初始分数：只有种子节点有分数
        node_scores = {node_id: 0.0 for node_id in all_nodes}
        for seed_id, weight in seed_weights.items():
            if seed_id in node_scores:
                node_scores[seed_id] = weight
        
        prev_scores = None
        converged_at = -1
        
        for iteration in range(self.ppr_iterations):
            new_scores = self._pagerank_iteration(node_scores, seed_weights)
            
            # 收敛检测
            if prev_scores and self._has_converged(new_scores, prev_scores):
                converged_at = iteration
                logger.info(f"  PageRank在第{iteration}轮收敛")
                break
            
            prev_scores = node_scores.copy()
            node_scores = new_scores
        
        if converged_at == -1:
            logger.info(f"  PageRank未收敛，完成{self.ppr_iterations}轮迭代")
        
        convergence_info = {
            "converged": converged_at != -1,
            "converged_at": converged_at,
            "final_iteration": self.ppr_iterations if converged_at == -1 else converged_at,
            "total_nodes": len(all_nodes),
            "total_score": sum(node_scores.values())
        }
        
        return node_scores, convergence_info
    
    def _pagerank_iteration(self, 
                          node_scores: Dict[str, float], 
                          seed_weights: Dict[str, float]) -> Dict[str, float]:
        """PageRank单次迭代"""
        new_scores = defaultdict(float)
        
        # 1. 计算teleport分数（重新分配到种子节点）
        total_score = sum(node_scores.values())
        teleport_mass = total_score * (1 - self.ppr_damping)
        
        for seed_id, seed_weight in seed_weights.items():
            new_scores[seed_id] += teleport_mass * seed_weight
        
        # 2. 计算传播分数（从每个节点传播到其邻居）
        for node_id, score in node_scores.items():
            if node_id in self.graph_adjacency and score > 0:
                out_neighbors = self.graph_adjacency[node_id]
                if out_neighbors:
                    propagate_score = score * self.ppr_damping / len(out_neighbors)
                    for neighbor_id in out_neighbors:
                        new_scores[neighbor_id] += propagate_score
        
        # 3. 确保所有节点都存在于结果中
        for node_id in node_scores.keys():
            if node_id not in new_scores:
                new_scores[node_id] = 0.0
        
        return dict(new_scores)
    
    def _has_converged(self, 
                      current_scores: Dict[str, float], 
                      prev_scores: Dict[str, float]) -> bool:
        """检测PageRank是否收敛"""
        total_diff = sum(
            abs(current_scores[node_id] - prev_scores.get(node_id, 0.0))
            for node_id in current_scores
        )
        return total_diff < self.convergence_tolerance
    
    def _build_traversal_path(self, 
                            node_scores: Dict[str, float], 
                            convergence_info: Dict[str, Any]) -> List[str]:
        """构建遍历路径用于调试"""
        top_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        path_info = f"Top nodes: {[(nid[:8], f'{score:.4f}') for nid, score in top_nodes]}"
        convergence_status = "收敛" if convergence_info.get("converged", False) else "未收敛"
        
        return [f"{convergence_status} - {path_info}"]
    
    async def _compute_item_scores_from_nodes(self, node_scores: Dict[str, float]) -> Dict[str, float]:
        """从节点分数计算item（chunk和event）分数"""
        item_scores = defaultdict(float)
        
        # 直接分数：event节点和chunk节点
        for node_id, score in node_scores.items():
            node_type = self.node_types.get(node_id)
            if node_type in ["event", "chunk"]:
                item_scores[node_id] = score
        
        # 间接分数：通过关系传播
        await self._propagate_scores_to_items(node_scores, item_scores)
        
        return dict(item_scores)
    
    async def _propagate_scores_to_items(self, 
                                       node_scores: Dict[str, float], 
                                       item_scores: Dict[str, float]):
        """将实体分数传播到相关的chunk和event"""
        
        # 获取有分数的实体ID
        entity_ids = [
            node_id for node_id, score in node_scores.items() 
            if self.node_types.get(node_id) == "entity" and score > 0
        ]
        
        if not entity_ids:
            return
        
        try:
            # 实体分数传播到相关事件
            await self._propagate_entity_to_events(entity_ids, node_scores, item_scores)
            
            # 实体分数传播到相关chunk
            await self._propagate_entity_to_chunks(entity_ids, node_scores, item_scores)
            
        except Exception as e:
            logger.error(f"分数传播失败: {e}")
    
    async def _propagate_entity_to_events(self, 
                                        entity_ids: List[str], 
                                        node_scores: Dict[str, float], 
                                        item_scores: Dict[str, float]):
        """实体分数传播到事件"""
        query = """
        MATCH (e:Entity)-[:PARTICIPATES_IN]->(v:Event)
        WHERE e.id_ IN $entity_ids
        RETURN e.id_ as entity_id, v.id_ as event_id
        """
        
        async with self.graph_store._driver.session(database=self.graph_store.database) as session:
            result = await session.run(query, entity_ids=entity_ids)
            records = await result.data()
            
            for record in records:
                entity_id = record["entity_id"]
                event_id = record["event_id"]
                if entity_id in node_scores:
                    # 实体分数的一部分传播给事件
                    item_scores[event_id] += node_scores[entity_id] * 0.5
    
    async def _propagate_entity_to_chunks(self, 
                                        entity_ids: List[str], 
                                        node_scores: Dict[str, float], 
                                        item_scores: Dict[str, float]):
        """实体分数传播到chunk"""
        query = """
        MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
        WHERE e.id_ IN $entity_ids  
        RETURN e.id_ as entity_id, c.id_ as chunk_id
        """
        
        async with self.graph_store._driver.session(database=self.graph_store.database) as session:
            result = await session.run(query, entity_ids=entity_ids)
            records = await result.data()
            
            for record in records:
                entity_id = record["entity_id"]
                chunk_id = record["chunk_id"]
                if entity_id in node_scores:
                    # 实体分数的一部分传播给chunk
                    item_scores[chunk_id] += node_scores[entity_id] * 0.3
    
    async def _step4_item_selection(self, 
                                              ppr_result: PPRResult, 
                                              query: str,
                                              preference: QueryPreference) -> List[RetrievalItem]:
        """Step 4: 选择chunk或event"""
        logger.info("🎯 Step 4: 选择检索项")
        
        item_scores = ppr_result.item_scores
        if not item_scores:
            return []
        
        try:
            # 获取所有候选项
            candidate_items = await self._get_candidate_items(item_scores, preference)
            
            # 选择top-k，保持chunk/event平衡
            selected_items = self._balance_chunk_event_selection(candidate_items)
            
            logger.info(f"  选择了 {len(selected_items)} 个检索项:")
            for i, item in enumerate(selected_items):
                logger.info(f"    {i+1}. [{item.type}] 分数: {item.score:.4f} - {item.content[:80]}...")
            
            return selected_items
            
        except Exception as e:
            logger.error(f"选择检索项失败: {e}")
            return []
    
    async def _get_candidate_items(self, 
                                 item_scores: Dict[str, float], 
                                 preference: QueryPreference) -> List[RetrievalItem]:
        """获取候选检索项"""
        candidate_items = []
        
        for item_id, score in item_scores.items():
            item_details = await self._get_item_details(item_id)
            if item_details:
                # 根据查询偏好调整分数
                adjusted_score = self._adjust_score_by_preference(
                    score, item_details["type"], preference
                )
                
                candidate_items.append(RetrievalItem(
                    id_=item_id,
                    content=item_details["content"],
                    type=item_details["type"],
                    score=adjusted_score,
                    source="pagerank",
                    metadata=item_details.get("metadata", {})
                ))
        
        # 按调整后分数排序
        candidate_items.sort(key=lambda x: x.score, reverse=True)
        return candidate_items
    
    def _adjust_score_by_preference(self, 
                                  base_score: float, 
                                  item_type: str, 
                                  preference: QueryPreference) -> float:
        """根据查询偏好调整得分"""
        if item_type == "chunk":
            return base_score * preference.chunk_weight
        elif item_type == "event":
            return base_score * preference.event_weight
        else:
            return base_score
    
    def _balance_chunk_event_selection(self, candidate_items: List[RetrievalItem]) -> List[RetrievalItem]:
        """平衡选择chunk和event，确保多样性"""
        if not candidate_items:
            return []
        
        # 按类型分组
        chunks = [item for item in candidate_items if item.type == "chunk"]
        events = [item for item in candidate_items if item.type == "event"]
        
        # 计算每种类型的目标数量
        total_chunks = len(chunks)
        total_events = len(events)
        
        if total_chunks == 0:
            return events[:self.top_k_items]
        elif total_events == 0:
            return chunks[:self.top_k_items]
        
        # 基于平衡因子分配
        target_chunks = max(1, int(self.top_k_items * (1 - self.chunk_event_balance)))
        target_events = self.top_k_items - target_chunks
        
        # 确保不超过可用数量
        actual_chunks = min(target_chunks, total_chunks)
        actual_events = min(target_events, total_events)
        
        # 如果一种类型不够，从另一种类型补充
        if actual_chunks < target_chunks:
            actual_events = min(self.top_k_items - actual_chunks, total_events)
        if actual_events < target_events:
            actual_chunks = min(self.top_k_items - actual_events, total_chunks)
        
        # 选择最高分的项
        selected_chunks = chunks[:actual_chunks]
        selected_events = events[:actual_events]
        
        # 合并并按分数重新排序
        selected_items = selected_chunks + selected_events
        selected_items.sort(key=lambda x: x.score, reverse=True)
        
        return selected_items
    
    async def _get_item_details(self, item_id: str) -> Optional[Dict[str, Any]]:
        """获取item（chunk或event）的详细信息"""
        node_type = self.node_types.get(item_id)
        
        if node_type == "chunk":
            for chunk in self.chunk_data:
                if chunk["id_"] == item_id:
                    return {
                        "content": chunk["content"],
                        "type": "chunk",
                        "metadata": {"source": chunk.get("source", "")}
                    }
        elif node_type == "event":
            for event in self.event_data:
                if event["id_"] == item_id:
                    return {
                        "content": event["content"],
                        "type": "event", 
                        "metadata": {
                            "participants": event.get("participants", []),
                            "event_type": event.get("type", "")
                        }
                    }
        
        return None
    
    async def _step5_answer_generation(self, 
                                     query: str, 
                                     ranked_items: List[RetrievalItem], 
                                     seed_nodes: List[SeedNode], 
                                     ppr_result: PPRResult) -> GenerationResult:
        """Step 5: 上下文构建与答案生成"""
        logger.info("🤖 Step 5: 上下文构建与答案生成")
        
        if not ranked_items:
            return self._create_empty_result(seed_nodes, ppr_result)
        
        try:
            # 构建上下文
            context_text, citations = self._build_generation_context(ranked_items)
            
            # 构建生成prompt
            generation_prompt = self._build_generation_prompt(query, context_text)
            
            # 生成答案
            response = await self.llm_model.achat([{"role": "user", "content": generation_prompt}])
            
            # 计算置信度
            confidence = self._calculate_confidence(ranked_items, ppr_result)
            
            logger.info(f"  答案生成完成，置信度: {confidence:.3f}")
            
            return GenerationResult(
                answer=response,
                evidence_items=ranked_items,
                citations=citations,
                confidence=confidence,
                retrieval_context=RetrievalContext(ranked_items, seed_nodes, ppr_result)
            )
            
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            return GenerationResult(
                answer="抱歉，在生成答案时遇到了问题。",
                evidence_items=ranked_items,
                citations=[],
                confidence=0.0,
                retrieval_context=RetrievalContext(ranked_items, seed_nodes, ppr_result)
            )
    
    def _create_empty_result(self, seed_nodes: List[SeedNode], ppr_result: PPRResult) -> GenerationResult:
        """创建空结果"""
        return GenerationResult(
            answer="抱歉，我无法在知识库中找到相关信息来回答您的问题。",
            evidence_items=[],
            citations=[],
            confidence=0.0,
            retrieval_context=RetrievalContext([], seed_nodes, ppr_result)
        )
    
    def _build_generation_context(self, ranked_items: List[RetrievalItem]) -> Tuple[str, List[str]]:
        """构建生成上下文"""
        context_text = ""
        citations = []
        
        chunk_items = [item for item in ranked_items if item.type == "chunk"]
        event_items = [item for item in ranked_items if item.type == "event"]
        
        # 添加chunk证据
        if chunk_items:
            context_text += "\n【文档证据】\n"
            for i, item in enumerate(chunk_items):
                context_text += f"[文档{i+1}]: {item.content}\n\n"
                citations.append(f"文档{i+1}")
        
        # 添加event证据
        if event_items:
            context_text += "\n【事件信息】\n"
            for i, item in enumerate(event_items):
                participants = item.metadata.get("participants", [])
                participant_str = f"(参与者: {', '.join(participants)})" if participants else ""
                context_text += f"[事件{i+1}]: {item.content} {participant_str}\n\n"
                citations.append(f"事件{i+1}")
        
        return context_text, citations
    
    def _build_generation_prompt(self, query: str, context_text: str) -> str:
        """构建生成prompt"""
        return f"""请根据以下背景信息，回答用户的问题。请确保：
1. 回答基于提供的证据
2. 在回答中引用相关证据（如[文档1]、[事件1]等）
3. 如果信息不足，请明确说明
4. 区分文档信息和事件信息的来源

{context_text}

【用户问题】
{query}

【你的回答】
"""
    
    def _calculate_confidence(self, 
                            items: List[RetrievalItem], 
                            ppr_result: PPRResult) -> float:
        """计算答案置信度"""
        if not items:
            return 0.0
        
        # 基于多个因素计算置信度
        factors = {
            "avg_score": np.mean([item.score for item in items]),
            "top_score": max(item.score for item in items),
            "diversity": len(set(item.type for item in items)) / 2.0,
            "convergence": 1.0 if ppr_result.convergence_info.get("converged", False) else 0.8,
            "coverage": min(1.0, len(items) / self.top_k_items)
        }
        
        # 加权计算置信度
        weights = {
            "avg_score": 0.3,
            "top_score": 0.25,
            "diversity": 0.15,
            "convergence": 0.15,
            "coverage": 0.15
        }
        
        confidence = sum(factors[k] * weights[k] for k in factors.keys())
        return min(1.0, max(0.0, confidence))
    
    # ===== 兜底机制 =====
    
    async def _fallback_dense_retrieval(self, query: str) -> GenerationResult:
        """兜底机制：稠密向量检索"""
        logger.info("🔄 执行稠密向量检索兜底机制")
        
        if len(self.chunk_embeddings) == 0:
            return self._create_empty_result([], PPRResult({}, {}, [], {}))
        
        try:
            # 编码查询
            query_embedding = self.embedding_model.embed_documents([query])[0]
            
            # 获取top items
            top_items = await self._get_dense_retrieval_items(query_embedding)
            
            # 生成答案
            return await self._step5_answer_generation(
                query, top_items, [], 
                PPRResult({}, {}, ["Dense retrieval fallback"], {"converged": False})
            )
            
        except Exception as e:
            logger.error(f"稠密检索兜底失败: {e}")
            return self._create_empty_result([], PPRResult({}, {}, [], {}))
    
    async def _get_dense_retrieval_items(self, query_embedding: np.ndarray) -> List[RetrievalItem]:
        """获取稠密检索的items"""
        top_items = []
        
        # 计算与chunk的相似度
        if len(self.chunk_embeddings) > 0:
            chunk_similarities = np.dot(self.chunk_embeddings, query_embedding)
            top_chunk_indices = np.argsort(chunk_similarities)[-(self.top_k_items//2):][::-1]
            
            for idx in top_chunk_indices:
                if idx < len(self.chunk_data):
                    chunk = self.chunk_data[idx]
                    top_items.append(RetrievalItem(
                        id_=chunk["id_"],
                        content=chunk["content"],
                        type="chunk",
                        score=float(chunk_similarities[idx]),
                        source="dense_retrieval",
                        metadata={"source": chunk.get("source", "")}
                    ))
        
        # 计算与event的相似度
        if len(self.event_embeddings) > 0:
            event_similarities = np.dot(self.event_embeddings, query_embedding)
            top_event_indices = np.argsort(event_similarities)[-(self.top_k_items//2):][::-1]
            
            for idx in top_event_indices:
                if idx < len(self.event_data):
                    event = self.event_data[idx]
                    top_items.append(RetrievalItem(
                        id_=event["id_"],
                        content=event["content"],
                        type="event",
                        score=float(event_similarities[idx]),
                        source="dense_retrieval",
                        metadata={"participants": event.get("participants", [])}
                    ))
        
        # 按分数排序并限制数量
        top_items.sort(key=lambda x: x.score, reverse=True)
        return top_items[:self.top_k_items]
    
    # ===== 实体和事件链接方法 =====
    
    async def _link_entity(self, entity_name: str) -> List[SeedNode]:
        """链接实体到图谱节点"""
        seed_nodes = []
        
        try:
            # 精确匹配
            exact_matches = self._find_exact_entity_matches(entity_name)
            seed_nodes.extend(exact_matches)
            
            # 向量相似度匹配
            if not seed_nodes and len(self.entity_embeddings) > 0:
                vector_matches = await self._find_vector_entity_matches(entity_name)
                seed_nodes.extend(vector_matches)
                
        except Exception as e:
            logger.error(f"实体链接失败 {entity_name}: {e}")
        
        return seed_nodes
    
    def _find_exact_entity_matches(self, entity_name: str) -> List[SeedNode]:
        """精确匹配实体"""
        matches = []
        for entity in self.entity_data:
            if entity["name"].lower() == entity_name.lower():
                matches.append(SeedNode(
                    id_=entity["id_"],
                    name=entity["name"],
                    type="entity",
                    score=1.0,
                    source="exact_match"
                ))
        return matches
    
    async def _find_vector_entity_matches(self, entity_name: str) -> List[SeedNode]:
        """向量相似度匹配实体"""
        matches = []
        
        try:
            entity_embedding = self.embedding_model.embed_documents([entity_name])[0]
            similarities = np.dot(self.entity_embeddings, entity_embedding)
            
            for i, sim in enumerate(similarities):
                if sim >= self.similarity_threshold and i < len(self.entity_data):
                    entity = self.entity_data[i]
                    matches.append(SeedNode(
                        id_=entity["id_"],
                        name=entity["name"],
                        type="entity",
                        score=float(sim),
                        source="vector_match"
                    ))
                    
        except Exception as e:
            logger.error(f"向量匹配实体失败: {e}")
        
        return matches
    
    async def _link_event(self, event_text: str) -> List[SeedNode]:
        """链接事件到图谱节点"""
        seed_nodes = []
        
        try:
            # 向量相似度匹配（事件通常没有精确匹配）
            if len(self.event_embeddings) > 0:
                vector_matches = await self._find_vector_event_matches(event_text)
                seed_nodes.extend(vector_matches)
                
        except Exception as e:
            logger.error(f"事件链接失败 {event_text}: {e}")
        
        return seed_nodes
    
    async def _find_vector_event_matches(self, event_text: str) -> List[SeedNode]:
        """向量相似度匹配事件"""
        matches = []
        
        try:
            event_embedding = self.embedding_model.embed_documents([event_text])[0]
            similarities = np.dot(self.event_embeddings, event_embedding)
            
            for i, sim in enumerate(similarities):
                if sim >= self.similarity_threshold and i < len(self.event_data):
                    event = self.event_data[i]
                    matches.append(SeedNode(
                        id_=event["id_"],
                        name=event["content"],
                        type="event",
                        score=float(sim),
                        source="vector_match"
                    ))
                    
        except Exception as e:
            logger.error(f"向量匹配事件失败: {e}")
        
        return matches
    
    # ===== 工具方法 =====
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "initialized": self.is_initialized,
            "entities": len(self.entity_data),
            "events": len(self.event_data),
            "chunks": len(self.chunk_data),
            "graph_nodes": len(self.node_types),
            "graph_edges": sum(len(neighbors) for neighbors in self.graph_adjacency.values()),
            "config": {
                "max_seed_nodes": self.max_seed_nodes,
                "top_k_items": self.top_k_items,
                "similarity_threshold": self.similarity_threshold,
                "chunk_event_balance": self.chunk_event_balance
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            if not self.is_initialized:
                return {"status": "not_initialized"}
            
            # 检查数据完整性
            checks = {
                "entity_data_loaded": len(self.entity_data) > 0,
                "entity_embeddings_loaded": len(self.entity_embeddings) > 0,
                "event_data_loaded": len(self.event_data) > 0,
                "chunk_data_loaded": len(self.chunk_data) > 0,
                "graph_built": len(self.graph_adjacency) > 0,
                "node_types_mapped": len(self.node_types) > 0
            }
            
            all_passed = all(checks.values())
            
            return {
                "status": "healthy" if all_passed else "unhealthy",
                "checks": checks,
                "stats": self.get_stats()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }