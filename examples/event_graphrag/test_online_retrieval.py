#!/usr/bin/env python3
"""
检索与生成系统测试
使用真实模型进行图检索和问答测试
"""

import asyncio
import json
import os
import sys

rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)
import traceback
from typing import List, Dict, Any
from dataclasses import dataclass

# 导入RAG-Factory组件
from rag_factory.llms.openai import OpenAILLM
from rag_factory.embeddings.huggingface import HuggingFaceEmbeddings
from rag_factory.store.graph_store.event_graphrag_neo4j import HyperRAGNeo4jStore
from rag_factory.retrieval.graph.event_graph_retriever import EventGraphRetriever


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
    llm_model_name: str = "gpt-5-mini"
    llm_api_key: str = "xxxx"
    llm_base_url: str = "https://api.gptsapi.net/v1"
    
    # 检索系统配置
    max_seed_nodes: int = 8
    ppr_iterations: int = 30
    ppr_damping: float = 0.85
    top_k_chunks: int = 10
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
            top_k_items=self.config.top_k_chunks,
            similarity_threshold=self.config.similarity_threshold,
            enable_fallback=self.config.enable_fallback
        )
        
        await self.retrieval_system.initialize()
        return self.retrieval_system


class QueryTester:
    """查询测试器"""
    
    def __init__(self, retrieval_system: EventGraphRetriever):
        self.retrieval_system = retrieval_system
    
    async def test_single_query(self, query: str, query_id: int, expected_answer: str = "", expected_explanation: str = "") -> Dict[str, Any]:
        """测试单个查询"""
        print(f"\n--- 测试查询 {query_id}: {query} ---")
        
        result_data = {
            "query_id": query_id,
            "query": query,
            "expected_answer": expected_answer,
            "expected_explanation": expected_explanation,
            "success": False,
            "generated_answer": "",
            "confidence": 0.0,
            "evidence_count": 0,
            "seed_nodes": [],
            "evidence_items": [],
            "error": None
        }
        
        try:
            # 执行完整的检索与生成流程
            result = await self.retrieval_system.retrieve_and_generate(query)
            
            # 填充结果数据
            result_data.update({
                "success": True,
                "generated_answer": result.answer,
                "confidence": result.confidence,
                "evidence_count": len(result.evidence_items),
                "seed_nodes": [
                    {
                        "name": node.name,
                        "type": node.type,
                        "score": node.score
                    } for node in result.retrieval_context.seed_nodes
                ] if result.retrieval_context.seed_nodes else [],
                "evidence_items": [
                    {
                        "content": item.content,  # 保存完整内容，不截断
                        "score": getattr(item, 'score', 0.0),
                        "metadata": getattr(item, 'metadata', {}),
                        "id": getattr(item, 'id', ''),
                        "type": getattr(item, 'type', ''),
                        "source": getattr(item, 'source', '')
                    } for item in result.evidence_items
                ]
            })
            
            # 显示生成结果
            self._display_generation_result(result)
            
            # 显示检索过程详情
            self._display_retrieval_process(result)
            
        except Exception as e:
            print(f"❌ 查询处理失败: {e}")
            result_data["error"] = str(e)
            traceback.print_exc()
        
        return result_data
    
    def _display_generation_result(self, result):
        """显示生成结果"""
        print(f"\n📝 生成结果:")
        print(f"答案: {result.answer}")
        print(f"置信度: {result.confidence:.3f}")
        print(f"证据数量: {len(result.evidence_items)}")
    
    def _display_retrieval_process(self, result):
        """显示检索过程"""
        print(f"\n🔍 检索过程:")
        
        # 显示种子节点
        if result.retrieval_context.seed_nodes:
            print("种子节点:")
            for node in result.retrieval_context.seed_nodes:
                print(f"  - {node.name} ({node.type}, 分数: {node.score:.3f})")
        
        # 显示证据内容（控制台显示时截断，但保存完整数据）
        if result.evidence_items:
            print("证据内容:")
            for j, item in enumerate(result.evidence_items[:3]):  # 只显示前3个证据
                display_content = item.content[:150] + "..." if len(item.content) > 150 else item.content
                print(f"  证据{j+1}: {display_content}")
                if j == 2 and len(result.evidence_items) > 3:
                    print(f"  ... 还有 {len(result.evidence_items) - 3} 个证据")
                    break


class OnlineRetrievalTester:
    """在线检索测试器主类"""
    
    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.model_initializer = ModelInitializer(self.config)
        self.store_manager = None
        self.retrieval_manager = None
        self.query_tester = None
        self.test_results = []  # 存储所有测试结果
    
    async def run_tests(self, test_queries: List[str]):
        """运行测试"""
        print("=== 在线检索与生成系统测试 ===")
        
        # 初始化期望答案列表
        expected_answers = [""] * len(test_queries)
        expected_explanations = [""] * len(test_queries)
        
        try:
            # 1. 初始化模型
            await self._initialize_models()
            
            # 2. 初始化图存储
            await self._initialize_graph_store()
            
            # 3. 初始化检索系统
            await self._initialize_retrieval_system()
            
            # 4. 执行查询测试
            await self._execute_query_tests(test_queries)
            
            # 5. 保存结果
            await self._save_results()
            
            print("\n✅ 测试完成")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            traceback.print_exc()
        
        finally:
            # 6. 清理资源
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
            result_data = await self.query_tester.test_single_query(
                query, i
            )
            
            # 收集结果
            self.test_results.append(result_data)
            
            if result_data["success"]:
                success_count += 1
                print(f"✅ 查询 {i} 成功")
            else:
                print(f"❌ 查询 {i} 失败: {result_data['error']}")
        
        print(f"\n📊 测试统计: {success_count}/{len(test_queries)} 个查询成功")
    
    async def _save_results(self):
        """保存测试结果"""
        print("\n=== 保存测试结果 ===")
        
        # 创建结果目录
        results_dir = "/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/event_graphrag/results_1"
        os.makedirs(results_dir, exist_ok=True)
        
        # 生成时间戳
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_results_file = os.path.join(results_dir, f"detailed_results_{timestamp}.json")
        with open(detailed_results_file, "w", encoding="utf-8") as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2)
        
        # 保存简化结果（用于分析）
        simplified_results = []
        for result in self.test_results:
            simplified_result = {
                "query_id": result["query_id"],
                "query": result["query"],
                "expected_answer": result["expected_answer"],
                "generated_answer": result["generated_answer"],
                "confidence": result["confidence"],
                "evidence_count": result["evidence_count"],
                "success": result["success"],
                "error": result["error"],
                "seed_nodes_count": len(result["seed_nodes"]),
                "evidence_summary": [
                    {
                        "content_preview": item["content"][:100] + "..." if len(item["content"]) > 100 else item["content"],
                        "score": item["score"],
                        "type": item.get("type", ""),
                        "source": item.get("source", "")
                    } for item in result["evidence_items"]
                ]
            }
            simplified_results.append(simplified_result)
        
        simplified_results_file = os.path.join(results_dir, f"simplified_results_{timestamp}.json")
        with open(simplified_results_file, "w", encoding="utf-8") as f:
            json.dump(simplified_results, f, ensure_ascii=False, indent=2)
        
        # 保存完整的检索文档信息
        full_evidence_file = os.path.join(results_dir, f"full_evidence_{timestamp}.json")
        full_evidence_data = []
        for result in self.test_results:
            evidence_data = {
                "query_id": result["query_id"],
                "query": result["query"],
                "generated_answer": result["generated_answer"],
                "success": result["success"],
                "seed_nodes": result["seed_nodes"],
                "evidence_items": result["evidence_items"]  # 包含完整的证据内容
            }
            full_evidence_data.append(evidence_data)
        
        with open(full_evidence_file, "w", encoding="utf-8") as f:
            json.dump(full_evidence_data, f, ensure_ascii=False, indent=2)
        
        # 生成统计报告
        success_count = sum(1 for r in self.test_results if r["success"])
        avg_confidence = sum(r["confidence"] for r in self.test_results if r["success"]) / max(success_count, 1)
        avg_evidence_count = sum(r["evidence_count"] for r in self.test_results if r["success"]) / max(success_count, 1)
        
        report = {
            "timestamp": timestamp,
            "total_queries": len(self.test_results),
            "successful_queries": success_count,
            "success_rate": success_count / len(self.test_results),
            "average_confidence": avg_confidence,
            "average_evidence_count": avg_evidence_count,
            "failed_queries": [
                {
                    "query_id": r["query_id"],
                    "query": r["query"],
                    "error": r["error"]
                } for r in self.test_results if not r["success"]
            ]
        }
        
        report_file = os.path.join(results_dir, f"test_report_{timestamp}.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到:")
        print(f"  详细结果: {detailed_results_file}")
        print(f"  简化结果: {simplified_results_file}")
        print(f"  完整证据: {full_evidence_file}")
        print(f"  测试报告: {report_file}")
        print(f"📊 成功率: {success_count}/{len(self.test_results)} ({success_count/len(self.test_results)*100:.1f}%)")
        print(f"📊 平均置信度: {avg_confidence:.3f}")
        print(f"📊 平均证据数量: {avg_evidence_count:.1f}")
    
    async def _cleanup(self):
        """清理资源"""
        if self.store_manager:
            await self.store_manager.close()


async def main():
    """主函数"""
    print("🚀 开始在线检索与生成系统测试")
    
    # 测试配置
    config = TestConfig()

    # 直接使用硬编码的测试查询列表
    test_queries = ["1. 请输出资料分析中高频考点的核心公式。",
"2. 请输出与比重相关的内容，包括知识点（含题型特征、核心公式、题型分类、解题思路等）、例题及讲解、点拨及误区等。",  
"3. 统计术语中，现期量与基期量的定义是什么？如何区分？",
"4. 增长量与增长率的计算公式分别是什么？增幅、增速与增长率的关系是什么？",
"5. 平均增长率（复合增长率）与平均增长量的计算公式是什么？",
"6. 拉动增长率的计算公式是什么？如何理解其含义？",
"7. 同比与环比的区别是什么？请举例说明。",
"8. 百分数与百分点的区别是什么？增长率之间的比较常用哪种表述？",
"9. 比重、增长贡献率的计算公式分别是什么？",
"10. 一般增长率的题型特征是什么？计算公式有哪些？",
"11. 混合增长率的题型特征是什么？解题的口诀和线段法的原理是什么？",
"12. 间隔增长率的题型特征是什么？计算公式是什么？",
"13. 年均增长率的题型特征是什么？比较大小时可采用什么技巧？",
"14. 资料分析中常见的陷阱有哪些？如何避开？",
"15. 解题时可利用哪些工具辅助？"]
    
    print(f"📊 加载了 {len(test_queries)} 个测试查询")

    # 创建测试器并运行测试
    tester = OnlineRetrievalTester(config)
    await tester.run_tests(test_queries)

def test_file_structure():
    """测试文件结构是否正确"""
    print("🔍 检查文件结构...")
    
    # 检查测试文件是否存在
    test_file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/event_graphrag/资料分析题目_例题.json"
    if os.path.exists(test_file_path):
        print(f"✅ 测试文件存在: {test_file_path}")
        
        # 读取并显示文件内容结构
        with open(test_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(f"📊 文件包含 {len(data)} 个测试项")
            if len(data) > 0:
                print(f"📝 第一个测试项结构: {list(data[0].keys())}")
    else:
        print(f"❌ 测试文件不存在: {test_file_path}")
    
    # 检查结果目录
    results_dir = "/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/event_graphrag/results_2"
    if os.path.exists(results_dir):
        print(f"✅ 结果目录存在: {results_dir}")
    else:
        print(f"📁 结果目录不存在，将在运行时创建: {results_dir}")


if __name__ == "__main__":
    test_file_structure()
    print("\n" + "="*50)
    asyncio.run(main())



