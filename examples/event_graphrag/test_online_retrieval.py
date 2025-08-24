#!/usr/bin/env python3
"""
在线检索与生成系统测试
使用真实模型进行图检索和问答测试
"""

import asyncio
import json
import os
import traceback
from typing import List, Dict, Any
from dataclasses import dataclass

# 导入RAG-Factory组件
from rag_factory.llms.openai_llm import OpenAILLM
from rag_factory.Embed import HuggingFaceEmbeddings
from rag_factory.Store.GraphStore.event_graphrag_neo4j import HyperRAGNeo4jStore
from rag_factory.Retrieval.GraphRetriever.Event_GraphRetriever import EventGraphRetriever


@dataclass
class TestConfig:
    """测试配置类"""
    # Neo4j数据库配置
    neo4j_url: str = "bolt://localhost:7681"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "12345678"
    neo4j_database: str = "neo4j"
    
    # 嵌入模型配置
    embedding_model_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B"
    embedding_device: str = "cuda:0"
    
    # LLM配置
    llm_model_name: str = "gpt-4o-mini"
    llm_api_key: str = "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"
    llm_base_url: str = "https://api.gptsapi.net/v1"
    
    # 检索系统配置
    max_seed_nodes: int = 8
    ppr_iterations: int = 15
    ppr_damping: float = 0.85
    top_k_chunks: int = 3
    similarity_threshold: float = 0.6
    enable_fallback: bool = True


class ModelInitializer:
    """模型初始化器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
    
    def create_embedding_model(self) -> HuggingFaceEmbeddings:
        """创建嵌入模型"""
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_path,
            model_kwargs={'device': self.config.embedding_device}
        )
    
    def create_llm_model(self) -> OpenAILLM:
        """创建LLM模型"""
        return OpenAILLM(
            model_name=self.config.llm_model_name,
            api_key=self.config.llm_api_key,
            base_url=self.config.llm_base_url
        )


class GraphStoreManager:
    """图存储管理器"""
    
    def __init__(self, config: TestConfig, embedding_model: HuggingFaceEmbeddings):
        self.config = config
        self.embedding_model = embedding_model
        self.store = None
    
    async def initialize(self) -> HyperRAGNeo4jStore:
        """初始化图存储"""
        self.store = HyperRAGNeo4jStore(
            url=self.config.neo4j_url,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password,
            database=self.config.neo4j_database,
            embedding=self.embedding_model
        )
        return self.store
    
    async def check_graph_status(self) -> Dict[str, Any]:
        """检查图状态"""
        if not self.store:
            raise RuntimeError("图存储未初始化")
        
        stats = await self.store.get_graph_statistics()
        print(f"当前图统计: {stats}")
        
        if stats.get('chunks', 0) == 0:
            raise RuntimeError("图数据库为空，请先运行 test_hyperrag_store.py 来填充数据")
        
        return stats
    
    async def close(self):
        """关闭图存储连接"""
        if self.store:
            await self.store.close()


class RetrievalSystemManager:
    """检索系统管理器"""
    
    def __init__(self, config: TestConfig, store: HyperRAGNeo4jStore, 
                 embedding_model: HuggingFaceEmbeddings, llm_model: OpenAILLM):
        self.config = config
        self.store = store
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.retrieval_system = None
    
    async def initialize(self) -> EventGraphRetriever:
        """初始化检索系统"""
        self.retrieval_system = EventGraphRetriever(
            graph_store=self.store,
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            max_seed_nodes=self.config.max_seed_nodes,
            ppr_iterations=self.config.ppr_iterations,
            ppr_damping=self.config.ppr_damping,
            top_k_chunks=self.config.top_k_chunks,
            similarity_threshold=self.config.similarity_threshold,
            enable_fallback=self.config.enable_fallback
        )
        
        await self.retrieval_system.initialize()
        return self.retrieval_system


class QueryTester:
    """查询测试器"""
    
    def __init__(self, retrieval_system: EventGraphRetriever):
        self.retrieval_system = retrieval_system
    
    async def test_single_query(self, query: str, query_id: int) -> bool:
        """测试单个查询"""
        print(f"\n--- 测试查询 {query_id}: {query} ---")
        
        try:
            # 执行完整的检索与生成流程
            result = await self.retrieval_system.retrieve_and_generate(query)
            
            # 显示生成结果
            self._display_generation_result(result)
            
            # 显示检索过程详情
            self._display_retrieval_process(result)
            
            return True
            
        except Exception as e:
            print(f"❌ 查询处理失败: {e}")
            traceback.print_exc()
            return False
    
    def _display_generation_result(self, result):
        """显示生成结果"""
        print(f"\n📝 生成结果:")
        print(f"答案: {result.answer}")
        print(f"置信度: {result.confidence:.3f}")
        print(f"证据数量: {len(result.evidence_chunks)}")
    
    def _display_retrieval_process(self, result):
        """显示检索过程"""
        print(f"\n🔍 检索过程:")
        
        # 显示种子节点
        if result.retrieval_context.seed_nodes:
            print("种子节点:")
            for node in result.retrieval_context.seed_nodes:
                print(f"  - {node.name} ({node.type}, 分数: {node.score:.3f})")
        
        # 显示证据内容
        if result.evidence_chunks:
            print("证据内容:")
            for j, chunk in enumerate(result.evidence_chunks[:2]):
                print(f"  证据{j+1}: {chunk[:150]}...")


class OnlineRetrievalTester:
    """在线检索测试器主类"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.model_initializer = ModelInitializer(self.config)
        self.store_manager = None
        self.retrieval_manager = None
        self.query_tester = None
    
    async def run_tests(self, test_queries: List[str]):
        """运行测试"""
        print("=== 在线检索与生成系统测试 ===")
        
        try:
            # 1. 初始化模型
            await self._initialize_models()
            
            # 2. 初始化图存储
            await self._initialize_graph_store()
            
            # 3. 初始化检索系统
            await self._initialize_retrieval_system()
            
            # 4. 执行查询测试
            await self._execute_query_tests(test_queries)
            
            print("\n✅ 测试完成")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            traceback.print_exc()
        
        finally:
            # 5. 清理资源
            await self._cleanup()
    
    async def _initialize_models(self):
        """初始化模型"""
        print("\n=== 初始化模型 ===")
        self.embedding_model = self.model_initializer.create_embedding_model()
        self.llm_model = self.model_initializer.create_llm_model()
        print("✅ 模型初始化完成")
    
    async def _initialize_graph_store(self):
        """初始化图存储"""
        print("\n=== 初始化图存储 ===")
        self.store_manager = GraphStoreManager(self.config, self.embedding_model)
        self.store = await self.store_manager.initialize()
        await self.store_manager.check_graph_status()
        print("✅ 图存储初始化完成")
    
    async def _initialize_retrieval_system(self):
        """初始化检索系统"""
        print("\n=== 初始化在线检索系统 ===")
        self.retrieval_manager = RetrievalSystemManager(
            self.config, self.store, self.embedding_model, self.llm_model
        )
        self.retrieval_system = await self.retrieval_manager.initialize()
        self.query_tester = QueryTester(self.retrieval_system)
        print("✅ 检索系统初始化完成")
    
    async def _execute_query_tests(self, test_queries: List[str]):
        """执行查询测试"""
        print("\n=== 执行检索与生成测试 ===")
        
        success_count = 0
        for i, query in enumerate(test_queries, 1):
            if await self.query_tester.test_single_query(query, i):
                success_count += 1
        
        print(f"\n📊 测试统计: {success_count}/{len(test_queries)} 个查询成功")
    
    async def _cleanup(self):
        """清理资源"""
        if self.store_manager:
            await self.store_manager.close()


async def main():
    """主函数"""
    print("🚀 开始在线检索与生成系统测试")
    
    # 测试配置
    config = TestConfig()
    
    # 测试查询
    test_queries = [
        "资料分析主要测查报考者的什么能力？"
    ]
    
    # 创建测试器并运行测试
    tester = OnlineRetrievalTester(config)
    await tester.run_tests(test_queries)


if __name__ == "__main__":
    asyncio.run(main())
