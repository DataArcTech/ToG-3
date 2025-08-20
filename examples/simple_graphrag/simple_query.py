# import sys
# import os
# import json

# # 添加 RAG-Factory 目录到 Python 路径
# rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
# sys.path.insert(0, rag_factory_path)


# from rag_factory.documents.GraphExtractor import GraphExtractor
# from rag_factory.documents.Prompt import KG_TRIPLES_PROMPT, ENTITY_EXTRACT_PROMPT
# from rag_factory.documents.schema import Document
# from rag_factory.llms import OpenAILLM
# from rag_factory.Store.GraphStore.graphrag_neo4j import Neo4jGraphStore
# from rag_factory.documents.parse_fn import parse_extraction_result, parse_entity_extraction_result
# from rag_factory.Store.GraphStore.GraphNode import EntityNode
# from rag_factory.Embed import HuggingFaceEmbeddings

# llm = OpenAILLM(
#     model_name="gpt-5-mini",
#     api_key="sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2",
#     base_url="https://api.gptsapi.net/v1",
# )

# storage = Neo4jGraphStore(
#         url="bolt://localhost:7680",
#         username="neo4j",
#         password="12345678",
#         database="neo4j",
#         embedding=HuggingFaceEmbeddings(
#             model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
#             model_kwargs={'device': 'cuda:0'}
#         )
#     )

# extractor = GraphExtractor(
#     llm=llm,
#     extract_prompt=ENTITY_EXTRACT_PROMPT,
#     parse_fn=parse_entity_extraction_result,
# )

# async def query_kg(query: str, k: int = 1, search_type: str = "query"):
#     result = await storage.search(query, k=k, search_type=search_type)
#     return result

# if __name__ == "__main__":
#     import asyncio
#     input_query = "2024年上半年，文化企业实现营业收入64961亿元，比上年同期增长7.5%，则现期是（ ），现期量为（ ）亿元；基期是（ ），基期量为（ ）亿元。"

#     entities = asyncio.run(extractor.acall([Document(content=input_query, metadata={"file_name": "input_query", "chunk_id": "1"})]))
#     # print(entities)
#     entities_list = []
#     for entity in entities[0].metadata['entities']:
#         entities_list.append(entity.name)
#     print(entities_list)
#     for entity in entities_list:
#         result = asyncio.run(query_kg(entity, k=3, search_type="entity"))
#         # print(result)
#         # doc_list = []
#         # for res in result:
#         #     doc_list.append(res['chunk'])
#         # print(doc_list)
#         for res in result:
#             print(f"节点: {res['node']}")
#             print(f"相似度得分: {res['score']}")
#             print("关系:")
#             for rel in res["relations"]:
#                 print(f"head: -> {rel['relation_type']} -> {rel['neighbor']['name']}, description: {rel['relation_properties']['relationship_description']}")
#             print("-------------------")
#     # result = asyncio.run(query_kg("【例1】** 2024年7月份，全国工业生产者出厂价格同比下降0.8%，环比下降0.2%，则2024年7月份，全国工业生产者出厂价格与（ ）相比下降了0.8%，与（ ）相比下降了0.2%。"))
#     # print("=== chunk 检索结果 ===")
#     # for res in result:
#     #     print(f"Chunk: {res['chunk']}")
#     #     print(f"相似度得分: {res['score']}")
#     #     print("包含的实体:")
#     #     for entity in res["entities"]:
#     #         print(f"  实体节点: {entity['node']}")
#     #     #     print("  关系:")
#     #     #     for rel in entity["relations"]:
#     #     #         print(f"    类型: {rel['relation_type']}, 属性: {rel['relation_properties']}, 邻居: {rel['neighbor']}")
#     #     print("-------------------")


#     # result = asyncio.run(query_kg("同比", k=3, search_type="entity"))
#     # for res in result:
#     #     print(f"节点: {res['node']}")
#     #     print(f"相似度得分: {res['score']}")
#     #     print("关系:")
#     #     for rel in res["relations"]:
#     #         print(f"head:同比 -> {rel['relation_type']} -> {rel['neighbor']['name']}, description: {rel['relation_properties']['relationship_description']}")
#     #         # print(f"  类型: {rel['relation_type']}, 属性: {rel['relation_properties']}, 邻居: {rel['neighbor']}")
#     #     print("-------------------")









import sys
import os
import json
import asyncio
from typing import List, Dict, Any, Optional

# 添加 RAG-Factory 目录到 Python 路径
rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)

from rag_factory.documents.GraphExtractor import GraphExtractor
from rag_factory.documents.Prompt import KG_TRIPLES_PROMPT, ENTITY_EXTRACT_PROMPT
from rag_factory.documents.schema import Document
from rag_factory.llms import OpenAILLM
from rag_factory.Store.GraphStore.graphrag_neo4j import Neo4jGraphStore
from rag_factory.documents.parse_fn import parse_extraction_result, parse_entity_extraction_result
from rag_factory.Store.GraphStore.GraphNode import EntityNode
from rag_factory.Embed import HuggingFaceEmbeddings

class KnowledgeGraphRAG:
    def __init__(self):
        """初始化知识图谱RAG系统"""
        self.llm = OpenAILLM(
            model_name="gpt-5-mini",
            api_key="xxx",
            base_url="xxx",
        )

        self.storage = Neo4jGraphStore(
            url="bolt://localhost:7680",
            username="neo4j",
            password="12345678",
            database="neo4j",
            embedding=HuggingFaceEmbeddings(
                model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
                model_kwargs={'device': 'cuda:0'}
            )
        )

        self.extractor = GraphExtractor(
            llm=self.llm,
            extract_prompt=ENTITY_EXTRACT_PROMPT,
            parse_fn=parse_entity_extraction_result,
        )

    async def extract_entities(self, text: str) -> List[str]:
        """从文本中提取实体"""
        try:
            entities = await self.extractor.acall([
                Document(content=text, metadata={"file_name": "query", "chunk_id": "1"})
            ])
            
            entities_list = []
            if entities and len(entities) > 0 and 'entities' in entities[0].metadata:
                for entity in entities[0].metadata['entities']:
                    entities_list.append(entity.name)
            
            return entities_list
        except Exception as e:
            print(f"实体提取出错: {e}")
            return []

    async def search_entities(self, entities: List[str], k: int = 3) -> Dict[str, List[Dict]]:
        """搜索实体相关信息"""
        entity_results = {}
        
        for entity in entities:
            try:
                result = await self.storage.search(entity, k=k, search_type="entity")
                entity_results[entity] = result
            except Exception as e:
                print(f"搜索实体 '{entity}' 时出错: {e}")
                entity_results[entity] = []
        
        return entity_results

    async def search_query(self, query: str, k: int = 5) -> List[Dict]:
        """基于查询进行语义搜索"""
        try:
            result = await self.storage.search(query, k=k, search_type="query")
            return result
        except Exception as e:
            print(f"查询搜索出错: {e}")
            return []

    def format_entity_info(self, entity: str, entity_data: List[Dict]) -> str:
        """格式化实体信息"""
        if not entity_data:
            return f"实体 '{entity}': 未找到相关信息\n"
        
        info = f"实体 '{entity}':\n"
        for i, data in enumerate(entity_data):
            info += f"  节点 {i+1}: {data['node']}\n"
            info += f"  相似度: {data['score']:.4f}\n"
            
            if data.get('relations'):
                info += "  关系:\n"
                for rel in data['relations']:
                    neighbor_name = rel['neighbor'].get('name', '未知')
                    relation_type = rel.get('relation_type', '未知关系')
                    description = rel.get('relation_properties', {}).get('relationship_description', '无描述')
                    info += f"    -> {relation_type} -> {neighbor_name}: {description}\n"
            
            info += "\n"
        
        return info

    def format_chunk_info(self, chunk_data: List[Dict]) -> str:
        """格式化文档块信息"""
        if not chunk_data:
            return "未找到相关文档块\n"
        
        info = "相关文档块:\n"
        for i, data in enumerate(chunk_data):
            info += f"  块 {i+1}: {data.get('chunk', '无内容')}\n"
            info += f"  相似度: {data.get('score', 0):.4f}\n"
            
            if data.get('entities'):
                info += "  包含实体:\n"
                for entity in data['entities']:
                    info += f"    - {entity.get('node', '未知实体')}\n"
            
            info += "\n"
        
        return info

    def build_context(self, entity_results: Dict[str, List[Dict]], chunk_results: List[Dict]) -> str:
        """构建上下文信息"""
        context = "=== 知识图谱检索结果 ===\n\n"
        
        # 添加实体信息
        context += "== 实体信息 ==\n"
        for entity, data in entity_results.items():
            context += self.format_entity_info(entity, data)
        
        # 添加文档块信息
        context += "== 文档块信息 ==\n"
        context += self.format_chunk_info(chunk_results)
        
        return context

    async def generate_response(self, query: str, context: str) -> str:
        """使用LLM生成回答"""
        prompt = f"""基于以下知识图谱检索到的信息回答问题：

问题: {query}

检索到的知识信息:
{context}

请根据检索到的信息，准确回答问题。如果信息不足以回答问题，请说明需要更多信息。
特别注意：
1. 对于数学计算题，请明确指出现期、基期的定义和数值
2. 引用具体的数据和关系
3. 保持回答的准确性和逻辑性

回答:"""

        try:
            messages = [
                {"role": "system", "content": "你是一个面向 **行测（行政职业能力测验）** 的 **知识问答助手**。"},
                {"role": "user", "content": prompt}
            ]
            response = await self.llm.achat(messages)
            return response
        except Exception as e:
            print(f"生成回答时出错: {e}")
            return "抱歉，无法生成回答。"

    async def rag_query(self, query: str, entity_k: int = 3, chunk_k: int = 5) -> Dict[str, Any]:
        """完整的RAG查询流程"""
        print(f"处理查询: {query}\n")
        
        # 1. 实体提取
        print("1. 提取实体...")
        entities = await self.extract_entities(query)
        print(f"提取到的实体: {entities}\n")
        
        # 2. 实体搜索
        print("2. 搜索实体信息...")
        entity_results = await self.search_entities(entities, k=entity_k)
        
        # 3. 查询搜索
        print("3. 进行语义搜索...")
        chunk_results = await self.search_query(query, k=chunk_k)
        
        # 4. 构建上下文
        context = self.build_context(entity_results, chunk_results)
        
        # 5. 生成回答
        print("4. 生成回答...")
        response = await self.generate_response(query, context)
        
        return {
            "query": query,
            "entities": entities,
            "entity_results": entity_results,
            "chunk_results": chunk_results,
            "context": context,
            "response": response
        }

    def print_detailed_results(self, results: Dict[str, Any]):
        """打印详细结果"""
        print("=" * 80)
        print("RAG 查询结果")
        print("=" * 80)
        
        print(f"\n查询: {results['query']}")
        print(f"\n提取的实体: {results['entities']}")
        
        print("\n=== 实体检索详情 ===")
        for entity, data in results['entity_results'].items():
            print(f"\n实体: {entity}")
            if not data:
                print("  未找到相关信息")
                continue
                
            for i, item in enumerate(data):
                print(f"  结果 {i+1}:")
                print(f"    节点: {item['node']}")
                print(f"    相似度: {item['score']:.4f}")
                
                if item.get('relations'):
                    print("    关系:")
                    for rel in item['relations']:
                        neighbor_name = rel['neighbor'].get('name', '未知')
                        relation_type = rel.get('relation_type', '未知关系')
                        description = rel.get('relation_properties', {}).get('relationship_description', '无描述')
                        print(f"      -> {relation_type} -> {neighbor_name}: {description}")
        
        print("\n=== 文档块检索详情 ===")
        if not results['chunk_results']:
            print("未找到相关文档块")
        else:
            for i, item in enumerate(results['chunk_results']):
                print(f"\n文档块 {i+1}:")
                print(f"  内容: {item.get('chunk', '无内容')}")
                print(f"  相似度: {item.get('score', 0):.4f}")
                
                if item.get('entities'):
                    print("  包含实体:")
                    for entity in item['entities']:
                        print(f"    - {entity.get('node', '未知实体')}")
        
        print(f"\n=== 最终回答 ===")
        print(results['response'])

    async def batch_query(self, queries: List[str], **kwargs) -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        for query in queries:
            result = await self.rag_query(query, **kwargs)
            results.append(result)
        return results

# # 使用示例
# async def main():
#     # 初始化RAG系统
#     rag_system = KnowledgeGraphRAG()
    
#     # 测试查询
#     test_queries = [
#         "2024年上半年，文化企业实现营业收入64961亿元，比上年同期增长7.5%，则现期是（ ），现期量为（ ）亿元；基期是（ ），基期量为（ ）亿元。",
#         "2024年7月份，全国工业生产者出厂价格同比下降0.8%，环比下降0.2%，则2024年7月份，全国工业生产者出厂价格与（ ）相比下降了0.8%，与（ ）相比下降了0.2%。"
#     ]
    
#     for query in test_queries:
#         print("\n" + "="*100)
#         print(f"处理查询: {query}")
#         print("="*100)
        
#         # 执行RAG查询
#         result = await rag_system.rag_query(query, entity_k=3, chunk_k=5)
        
#         # 打印详细结果
#         rag_system.print_detailed_results(result)
        
#         # 也可以只打印关键信息
#         print("\n=== 简要总结 ===")
#         print(f"提取实体: {len(result['entities'])} 个")
#         print(f"找到实体相关节点: {sum(len(data) for data in result['entity_results'].values())} 个")
#         print(f"找到相关文档块: {len(result['chunk_results'])} 个")





# 高级功能类
class AdvancedKGRAG(KnowledgeGraphRAG):
    """扩展的知识图谱RAG系统，包含更多高级功能"""
    
    async def get_entity_neighborhood(self, entity: str, hops: int = 2) -> Dict[str, Any]:
        """获取实体的多跳邻居信息"""
        try:
            result = await self.storage.search(entity, k=10, search_type="entity")
            
            neighborhood = {
                "center_entity": entity,
                "direct_neighbors": [],
                "relations_summary": {}
            }
            
            for item in result:
                if item.get('relations'):
                    for rel in item['relations']:
                        neighbor = rel['neighbor'].get('name', '未知')
                        relation_type = rel.get('relation_type', '未知关系')
                        
                        neighborhood["direct_neighbors"].append({
                            "neighbor": neighbor,
                            "relation": relation_type,
                            "description": rel.get('relation_properties', {}).get('relationship_description', '')
                        })
                        
                        # 统计关系类型
                        if relation_type not in neighborhood["relations_summary"]:
                            neighborhood["relations_summary"][relation_type] = 0
                        neighborhood["relations_summary"][relation_type] += 1
            
            return neighborhood
        except Exception as e:
            print(f"获取实体邻域信息出错: {e}")
            return {"center_entity": entity, "direct_neighbors": [], "relations_summary": {}}

    async def multi_hop_reasoning(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """多跳推理"""
        print(f"开始多跳推理，最大跳数: {max_hops}")
        
        # 1. 初始实体提取
        initial_entities = await self.extract_entities(query)
        print(f"初始实体: {initial_entities}")
        
        all_entities = set(initial_entities)
        hop_results = {}
        
        # 2. 多跳扩展
        current_entities = initial_entities
        for hop in range(max_hops):
            print(f"\n第 {hop + 1} 跳:")
            hop_entities = set()
            
            for entity in current_entities:
                neighborhood = await self.get_entity_neighborhood(entity, hops=1)
                hop_results[f"{entity}_hop_{hop}"] = neighborhood
                
                # 收集邻居实体
                for neighbor_info in neighborhood["direct_neighbors"]:
                    hop_entities.add(neighbor_info["neighbor"])
            
            # 更新实体集合（排除已处理的）
            new_entities = hop_entities - all_entities
            all_entities.update(new_entities)
            current_entities = list(new_entities)
            
            print(f"  新发现实体: {len(new_entities)} 个")
            
            if not new_entities:
                print(f"  第 {hop + 1} 跳未发现新实体，停止扩展")
                break
        
        return {
            "query": query,
            "initial_entities": initial_entities,
            "all_discovered_entities": list(all_entities),
            "hop_results": hop_results,
            "total_hops": hop + 1
        }

    async def semantic_similarity_ranking(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[Dict]:
        """基于语义相似度对候选结果排序"""
        sorted_candidates = sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)
        return sorted_candidates[:top_k]

    async def enhanced_rag_query(self, query: str, entity_k: int = 3, chunk_k: int = 5, 
                                enable_multi_hop: bool = True, max_hops: int = 2) -> Dict[str, Any]:
        """增强的RAG查询"""
        print(f"开始增强RAG查询: {query}\n")
        
        # 基础RAG查询
        basic_result = await self.rag_query(query, entity_k, chunk_k)
        
        enhanced_result = basic_result.copy()
        
        # 多跳推理（可选）
        if enable_multi_hop and basic_result['entities']:
            print("\n执行多跳推理...")
            multi_hop_result = await self.multi_hop_reasoning(query, max_hops)
            enhanced_result['multi_hop_analysis'] = multi_hop_result
        
        # 结果融合和重排序
        all_results = basic_result['chunk_results'].copy()
        if enhanced_result.get('multi_hop_analysis'):
            # 这里可以添加基于多跳结果的额外搜索
            pass
        
        # 重排序
        enhanced_result['ranked_results'] = await self.semantic_similarity_ranking(
            query, all_results, top_k=min(10, len(all_results))
        )
        
        return enhanced_result

def create_rag_prompts():
    """创建RAG相关的提示词模板"""
    prompts = {
        "answer_prompt": """基于以下知识图谱检索到的信息回答问题：

问题: {query}

检索到的知识信息:
{context}

请根据检索到的信息，准确回答问题。如果信息不足以回答问题，请说明需要更多信息。
特别注意：
1. 对于数学计算题，请明确指出现期、基期的定义和数值
2. 引用具体的数据和关系
3. 保持回答的准确性和逻辑性
4. 如果涉及计算，请给出具体的计算过程

回答:""",
        
        "entity_analysis_prompt": """分析以下实体在知识图谱中的信息：

实体: {entity}
相关信息: {entity_info}

请总结该实体的关键特征、相关概念和重要关系。""",
        
        "multi_hop_summary_prompt": """基于多跳推理结果，总结实体间的复杂关系：

查询: {query}
多跳分析结果: {multi_hop_results}

请总结发现的关键关系路径和推理链。"""
    }
    return prompts

# 主函数
async def main_advanced():
    """高级功能演示"""
    # 创建增强RAG系统
    advanced_rag = AdvancedKGRAG()
    
    # 测试查询
    query = """【['2022年,深圳全市绿化覆盖总面积达到101385.6公顷,比上年增长513.4公顷;其中建成区绿化覆盖面积', '41457.5公顷,比上年增长345.2公顷;建成区绿化覆盖率43.09%,比上年增长0.09个百分点;全市绿地面积', '98270.2公顷,比上年增长1.45%;其中建成区绿地面积36613.1公顷,比上年增长\n3.30%;建成区绿地率比上年增长0.98个百分点;全市公园绿地面积22219.0公顷,比上年\n增长222.5公顷,按常住人口计算人均公园绿地面积比上年增长1.13%。', '2022年全市公园已达到1260个,公园总面积38209.9公顷,分别是2003年的8.6倍和\n5.6倍;全市公园中,有自然公园37个,比上年增加4个;城市公园191个,比上年增加4\n个;社区公园1032个,比上年增加14个。三级公园体系的建成,基本实现了市民出门500\n米可达社区公园,2公里可达城市综合圈,5公里可达自然公园的目标。全市公园绿化活动场\n地服务半径覆盖的居住用地面积达到21018.2公顷,较上年增长280.8公顷,公园绿化活动场\n地服务半径覆盖率达到90.87%,与上年持平。', '2022年,全市绿道长度3119公里,比上年增长9.70%,增速较上年收窄5.78个百分点,\n万人拥有绿道长度 1.77公里,比上年增长9.94%;绿道密度(全市绿道长度与全市面积之\n比)1.56公里/平方公里,居广东省首位。深圳市建成区绿化覆盖率、公园绿化活动场地服务\n半径覆盖率、万人拥有绿道长度3项指标已达到国家生态园林城市标准。']
下列统计指标中，其 2022 年同比增长率最高的是（）
A.全市绿化覆盖总面积 B.全市公园绿地面积 C.全市公园总数 D.全市居住用地面积"
"""
    # 执行增强RAG查询
    result = await advanced_rag.enhanced_rag_query(
        query, 
        entity_k=3, 
        chunk_k=5, 
        enable_multi_hop=True, 
        max_hops=2
    )
    
    # 打印结果
    advanced_rag.print_detailed_results(result)
    
    if result.get('multi_hop_analysis'):
        print("\n=== 多跳推理结果 ===")
        multi_hop = result['multi_hop_analysis']
        print(f"发现总实体数: {len(multi_hop['all_discovered_entities'])}")
        print(f"推理跳数: {multi_hop['total_hops']}")

if __name__ == "__main__":
    # 运行基础示例
    # print("运行基础RAG示例...")
    # asyncio.run(main())
    
    print("\n" + "="*100)
    
    # 运行高级示例
    asyncio.run(main_advanced())