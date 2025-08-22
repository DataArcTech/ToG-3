import sys
import os
import json
import asyncio
from traceback import print_tb
from typing import List, Dict, Any, Optional, Set, Tuple
import numpy as np

# 添加 RAG-Factory 目录到 Python 路径
rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)

from rag_factory.documents.GraphExtractor import GraphExtractor
from rag_factory.documents.Prompt import ENTITY_EXTRACT_PROMPT
from rag_factory.documents.schema import Document
from rag_factory.llms import OpenAILLM
from rag_factory.Store.GraphStore.graphrag_neo4j import Neo4jGraphStore
from rag_factory.documents.parse_fn import parse_entity_extraction_result
from rag_factory.Embed import HuggingFaceEmbeddings

class KnowledgeGraphRAG:
    def __init__(self):
        self.llm = OpenAILLM(
            model_name="gpt-5-mini",
            api_key="sk-xxxx", # 请替换为你的api key
            base_url="https://api.gptsapi.net/v1",
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
        
        # ToG-2 参数
        self.max_depth = 3  # D: 最大迭代深度
        self.exploration_width = 3  # W: 探索宽度
        self.context_k = 10  # K: 上下文数量
        self.relation_threshold = 0.2  # 关系筛选阈值
        self.decay_alpha = 0.5  # 指数衰减参数

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

    async def topic_prune(self, query: str, extracted_entities: List[str]) -> List[str]:
        """Topic Prune: 使用LLM评估并选择与问题最相关的实体 (ToG-2)"""
        if not extracted_entities:
            return []
        
        if len(extracted_entities) <= self.exploration_width:
            return extracted_entities
        
        try:
            prompt = f"""你是一个知识图谱专家。给定一个问题和从中提取的实体列表，请评估每个实体与问题的相关性并选择最相关的{self.exploration_width}个实体作为起始探索点。

问题: {query}

提取的实体: {', '.join(extracted_entities)}

请按照相关性从高到低排序，并选择最相关的{self.exploration_width}个实体。考虑以下因素：
1. 实体是否是问题的核心概念
2. 实体是否可能包含答案的关键信息
3. 实体是否适合作为图遍历的起点

请按照以下格式输出：
选择的实体: [实体1, 实体2, 实体3]
理由: 简要说明选择理由"""

            messages = [
                {"role": "system", "content": "你是一个专业的知识图谱分析专家，擅长识别与问题最相关的关键实体。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.achat(messages)
            
            # 解析响应，提取选择的实体
            lines = response.strip().split('\n')
            selected_entities = []
            
            for line in lines:
                if line.startswith('选择的实体:'):
                    # 提取方括号内的内容
                    import re
                    match = re.search(r'\[(.*?)\]', line)
                    if match:
                        entities_str = match.group(1)
                        selected_entities = [e.strip() for e in entities_str.split(',')]
                        break
            
            # 确保选择的实体都在原始列表中
            valid_entities = []
            for entity in selected_entities:
                if entity in extracted_entities:
                    valid_entities.append(entity)
            
            # 如果解析失败，返回前几个实体
            if not valid_entities:
                valid_entities = extracted_entities[:self.exploration_width]
            
            print(f"Topic Prune: {len(extracted_entities)} -> {len(valid_entities)}")
            print(f"选择的起始实体: {valid_entities}")
            
            return valid_entities
            
        except Exception as e:
            print(f"Topic Prune过程中出错: {e}")
            # 出错时返回前几个实体
            return extracted_entities[:self.exploration_width]




    async def get_entity_chunks(self, entity: str) -> List[str]:
        """获取实体相关的所有chunks"""
        try:
            # 搜索实体相关的chunks
            results = await self.storage.search(entity, k=10, search_type="query")
            chunks = []
            for result in results:
                if 'chunk' in result:
                    chunk_content = result['chunk']
                    # 确保chunk是字符串
                    if isinstance(chunk_content, dict):
                        # 如果是字典，尝试提取文本内容
                        if 'content' in chunk_content:
                            chunk_content = chunk_content['content']
                        elif 'text' in chunk_content:
                            chunk_content = chunk_content['text']
                        else:
                            # 如果无法提取，跳过这个chunk
                            continue
                    elif not isinstance(chunk_content, str):
                        # 如果不是字符串，跳过
                        continue
                    
                    chunks.append(chunk_content)
            return chunks
        except Exception as e:
            print(f"获取实体chunks时出错: {e}")
            return []
    
    async def get_entity_neighbors(self, entity: str) -> List[Dict[str, Any]]:
        """获取实体的所有邻居，包含边的详细信息"""
        try:
            # 搜索实体
            results = await self.storage.search(entity, k=5, search_type="entity")
            neighbors = []
            seen_neighbors = set()
            
            for result in results:
                if 'relations' in result:
                    for relation in result['relations']:
                        neighbor_name = relation.get('neighbor', {}).get('name', '')
                        if neighbor_name and neighbor_name != entity and neighbor_name not in seen_neighbors:
                            neighbor_info = {
                                'name': neighbor_name,
                                'relation_type': relation.get('relation_type', '未知关系'),
                                'relation_description': relation.get('relation_properties', {}).get('relationship_description', '无描述'),
                                'source_entity': entity
                            }
                            neighbors.append(neighbor_info)
                            seen_neighbors.add(neighbor_name)
            
            return neighbors
        except Exception as e:
            print(f"获取实体邻居时出错: {e}")
            return []

    async def relation_prune(self, query: str, entity: str, relations: List[Dict[str, Any]], 
                           current_clues: str = "") -> List[Dict[str, Any]]:
        """Relation Prune: 使用LLM评估并筛选最有用的关系 (ToG-2)"""
        if not relations:
            return []
        
        try:
            # 构建关系描述
            relation_descriptions = []
            for i, rel in enumerate(relations):
                rel_type = rel.get('relation_type', '未知关系')
                rel_desc = rel.get('relation_description', '无描述')
                neighbor_name = rel.get('name', '未知实体')
                
                desc = f"{i+1}. 关系: {entity} -[{rel_type}]-> {neighbor_name}"
                if rel_desc != '无描述':
                    desc += f" (描述: {rel_desc})"
                relation_descriptions.append(desc)
            
            prompt = f"""你是一个知识图谱专家。给定一个问题、当前实体和它的所有邻接关系，请评估每个关系对回答问题的有用性，并给出0-10的评分（10分最有用）。

问题: {query}
当前实体: {entity}
{"当前线索: " + current_clues if current_clues else ""}

可选关系:
{chr(10).join(relation_descriptions)}

请为每个关系评分，考虑以下因素：
1. 关系是否可能通向包含答案的实体
2. 关系的语义是否与问题相关
3. 邻居实体是否可能有用

请按照以下格式输出（每行一个关系的评分）：
1: [评分] - [简要理由]
2: [评分] - [简要理由]
...

只选择评分 >= {int(self.relation_threshold * 10)} 的关系。"""

            messages = [
                {"role": "system", "content": "你是一个专业的知识图谱分析专家，擅长评估关系对问题的相关性。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.achat(messages)
            
            # 解析响应
            lines = response.strip().split('\n')
            selected_relations = []
            
            for line in lines:
                if ':' in line:
                    try:
                        # 解析 "数字: 评分 - 理由" 格式
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            rel_idx = int(parts[0].strip()) - 1  # 转换为0-based索引
                            content = parts[1].strip()
                            
                            # 提取评分
                            if ' -' in content:
                                score_part = content.split(' -')[0].strip()
                            else:
                                score_part = content.strip()
                            
                            try:
                                score = float(score_part)
                                
                                # 检查评分是否达到阈值
                                if score >= self.relation_threshold * 10 and 0 <= rel_idx < len(relations):
                                    rel_copy = relations[rel_idx].copy()
                                    rel_copy['llm_score'] = score
                                    selected_relations.append(rel_copy)
                            except ValueError:
                                continue
                    except (ValueError, IndexError):
                        continue
            
            # 按评分排序
            selected_relations.sort(key=lambda x: x.get('llm_score', 0), reverse=True)
            
            print(f"Relation Prune for '{entity}': {len(relations)} -> {len(selected_relations)}")
            for rel in selected_relations[:3]:  # 只打印前3个
                print(f"  - {rel['name']}: {rel['relation_type']} (LLM评分: {rel.get('llm_score', 0):.1f})")
            
            return selected_relations
            
        except Exception as e:
            print(f"Relation Prune过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回所有关系
            return relations

    async def retrieve_chunks_from_topic_entities(self, query: str, topic_entities: List[str], k: int = 5) -> List[Dict[str, Any]]:
        """从topic entities关联的chunks中进行向量检索"""
        try:
            print(f"从{len(topic_entities)}个topic entities关联的chunks中检索...")
            
            # 收集所有topic entities关联的chunks
            all_entity_chunks = []
            entity_chunk_map = {}  # 记录chunk属于哪个entity
            
            for entity in topic_entities:
                chunks = await self.get_entity_chunks(entity)
                print(f"  实体'{entity}': 找到{len(chunks)}个chunks")
                
                for chunk in chunks:
                    all_entity_chunks.append(chunk)
                    entity_chunk_map[chunk] = entity
            
            if not all_entity_chunks:
                print("❌ 未找到任何关联的chunks")
                return []
            
            print(f"总共收集到{len(all_entity_chunks)}个chunks，开始向量检索...")
            
            # 对query进行embedding
            query_embedding = await self.storage.embedding.aembed_query(query)
            query_vec = np.array(query_embedding)
            
            # 计算每个chunk与query的相似度
            chunk_scores = []
            for chunk in all_entity_chunks:
                try:
                    chunk_embedding = await self.storage.embedding.aembed_query(chunk)
                    chunk_vec = np.array(chunk_embedding)
                    
                    # 计算余弦相似度
                    similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
                    
                    chunk_scores.append({
                        'chunk': chunk,
                        'entity': entity_chunk_map[chunk],
                        'similarity_score': max(0.0, similarity)
                    })
                except Exception as e:
                    print(f"计算chunk相似度时出错: {e}")
                    continue
            
            # 按相似度排序并返回top-k
            chunk_scores.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_chunks = chunk_scores[:k]
            
            print(f"✅ 检索完成，返回top-{len(top_chunks)}个最相关的chunks")
            for i, chunk_info in enumerate(top_chunks[:3]):  # 显示前3个
                print(f"  {i+1}. 来自实体'{chunk_info['entity']}' (相似度: {chunk_info['similarity_score']:.4f})")
                print(f"     {chunk_info['chunk'][:100]}...")
            
            return top_chunks
            
        except Exception as e:
            print(f"从topic entities检索chunks时出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def entity_guided_context_retrieval(self, query: str, candidate_entity: str, 
                                             triple_path: List[Tuple[str, str]], k: int = None) -> List[Dict[str, Any]]:
        """Entity-guided Context Retrieval: 结合triple路径的上下文检索 (ToG-2)"""
        if k is None:
            k = self.context_k
        
        try:
            # 构建triple路径的文本描述
            path_description = ""
            if triple_path:
                path_parts = []
                for source_entity, relation in triple_path:
                    path_parts.append(f"{source_entity} -[{relation}]->")
                path_parts.append(candidate_entity)
                path_description = " ".join(path_parts)
            
            # 获取候选实体相关的文档chunks
            entity_chunks = await self.get_entity_chunks(candidate_entity)
            
            if not entity_chunks:
                return []
            
            # 为每个chunk计算增强的相关性分数
            enhanced_chunks = []
            for chunk in entity_chunks:
                try:
                    # 如果有路径描述，将其附加到chunk后面来计算相似度
                    if path_description:
                        enhanced_text = f"{path_description}: {chunk}"
                    else:
                        enhanced_text = chunk
                    
                    # 计算query与增强文本的相似度
                    query_embedding = await self.storage.embedding.aembed_query(query)
                    enhanced_embedding = await self.storage.embedding.aembed_query(enhanced_text)
                    
                    query_vec = np.array(query_embedding)
                    enhanced_vec = np.array(enhanced_embedding)
                    
                    # 计算余弦相似度
                    similarity = np.dot(query_vec, enhanced_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(enhanced_vec))
                    
                    enhanced_chunks.append({
                        'chunk': chunk,
                        'entity': candidate_entity,
                        'triple_path': triple_path,
                        'path_description': path_description,
                        'similarity_score': max(0.0, similarity)
                    })
                    
                except Exception as e:
                    print(f"计算chunk相似度时出错: {e}")
                    # 如果出错，给一个默认分数
                    enhanced_chunks.append({
                        'chunk': chunk,
                        'entity': candidate_entity,
                        'triple_path': triple_path,
                        'path_description': path_description,
                        'similarity_score': 0.0
                    })
            
            # 按相似度排序并返回top-k
            enhanced_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_chunks = enhanced_chunks[:k]
            
            print(f"Entity-guided Context Retrieval for '{candidate_entity}': {len(enhanced_chunks)} -> {len(top_chunks)}")
            if top_chunks:
                print(f"  最高分数: {top_chunks[0]['similarity_score']:.4f}")
            
            return top_chunks
            
        except Exception as e:
            print(f"Entity-guided Context Retrieval过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def context_based_entity_prune(self, query: str, candidate_entities: List[Dict[str, Any]], 
                                       triple_paths: Dict[str, List[Tuple[str, str]]]) -> List[Dict[str, Any]]:
        """Context-based Entity Prune: 基于上下文质量评分候选实体 (ToG-2)"""
        try:
            scored_entities = []
            
            for entity_info in candidate_entities:
                entity_name = entity_info['name']
                triple_path = triple_paths.get(entity_name, [])
                
                # 获取实体的上下文
                contexts = await self.entity_guided_context_retrieval(query, entity_name, triple_path, k=5)
                
                if not contexts:
                    entity_info['context_score'] = 0.0
                    scored_entities.append(entity_info)
                    continue
                
                # 使用指数衰减加权计算总分
                total_score = 0.0
                for i, ctx in enumerate(contexts):
                    weight = np.exp(-self.decay_alpha * i)  # 指数衰减权重
                    total_score += ctx['similarity_score'] * weight
                
                entity_info['context_score'] = total_score
                entity_info['top_contexts'] = contexts[:3]  # 保存前3个最佳上下文
                scored_entities.append(entity_info)
            
            # 按上下文分数排序
            scored_entities.sort(key=lambda x: x['context_score'], reverse=True)
            
            # 选择top-W个实体
            top_entities = scored_entities[:self.exploration_width]
            
            print(f"Context-based Entity Prune: {len(candidate_entities)} -> {len(top_entities)}")
            for entity in top_entities:
                print(f"  - {entity['name']}: 上下文分数 {entity['context_score']:.4f}")
            
            return top_entities
            
        except Exception as e:
            print(f"Context-based Entity Prune过程中出错: {e}")
            import traceback
            traceback.print_exc()
            # 出错时返回原始列表的前W个
            return candidate_entities[:self.exploration_width]

    async def hybrid_knowledge_reasoning(self, query: str, previous_clues: str, triple_paths: List[List[Tuple[str, str]]], 
                                       top_entities: List[Dict[str, Any]], contexts: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """混合知识推理: 基于图谱路径和上下文进行推理判断 (ToG-2)"""
        try:
            # 构建知识描述
            knowledge_parts = []
            
            # 1. 上一轮线索
            if previous_clues:
                knowledge_parts.append(f"前轮线索: {previous_clues}")
            
            # 2. Triple路径信息
            if triple_paths:
                path_descriptions = []
                for i, path in enumerate(triple_paths[:3]):  # 只显示前3条路径
                    if path:
                        path_str = ""
                        for source, relation in path:
                            path_str += f"{source} -[{relation}]-> "
                        path_str = path_str.rstrip(" -> ")
                        path_descriptions.append(f"路径{i+1}: {path_str}")
                
                if path_descriptions:
                    knowledge_parts.append("图谱路径信息:\n" + "\n".join(path_descriptions))
            
            # 3. Top实体及其上下文
            if top_entities:
                entity_contexts = []
                for entity in top_entities:
                    entity_name = entity['name']
                    entity_contexts.append(f"实体: {entity_name} (上下文分数: {entity.get('context_score', 0):.3f})")
                    
                    # 添加实体的上下文信息
                    if 'top_contexts' in entity:
                        for ctx in entity['top_contexts'][:2]:  # 每个实体最多2个上下文
                            chunk = ctx['chunk'][:200] + "..." if len(ctx['chunk']) > 200 else ctx['chunk']
                            entity_contexts.append(f"  - {chunk}")
                
                if entity_contexts:
                    knowledge_parts.append("相关实体及上下文:\n" + "\n".join(entity_contexts))
            
            # 4. 额外的上下文
            if contexts:
                context_descriptions = []
                for ctx in contexts[:5]:  # 最多5个上下文
                    chunk = ctx['chunk'] if isinstance(ctx, dict) and 'chunk' in ctx else str(ctx)
                    chunk_preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                    score = ctx.get('similarity_score', 0) if isinstance(ctx, dict) else 0
                    context_descriptions.append(f"- {chunk_preview} (分数: {score:.3f})")
                
                if context_descriptions:
                    knowledge_parts.append("额外相关上下文:\n" + "\n".join(context_descriptions))
            
            # 构建推理提示
            knowledge_summary = "\n\n".join(knowledge_parts) if knowledge_parts else "暂无相关知识"
            
            prompt = f"""你是一个专业的知识推理专家。基于提供的知识信息，判断是否足够回答问题，并按要求输出。

问题: {query}

当前知识信息:
{knowledge_summary}

请仔细分析以上知识，然后按照以下格式输出：

判断: [Yes/No]

如果判断为Yes，请输出:
答案: [你的完整答案，将关键实体用{{实体名}}包围]

如果判断为No，请输出:
线索: {{总结当前有用的线索和发现，用于指导下一轮搜索}}

注意：
1. 只有在知识充分且能给出准确答案时才判断为Yes
2. 线索应该简洁明了，突出对下一轮搜索有帮助的信息
3. 实体名必须用双大括号包围
"""

            messages = [
                {"role": "system", "content": "你是一个专业的知识推理和分析专家，擅长基于多源知识进行综合判断。"},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm.achat(messages)
            
            # 解析响应
            lines = response.strip().split('\n')
            is_sufficient = False
            result_content = ""
            
            for i, line in enumerate(lines):
                if line.startswith('判断:'):
                    is_sufficient = 'Yes' in line or 'yes' in line or '是' in line
                elif line.startswith('答案:') and is_sufficient:
                    result_content = line.split(':', 1)[1].strip()
                    # 如果答案跨多行，继续收集
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(('线索:', '判断:')):
                            result_content += " " + lines[j].strip()
                        else:
                            break
                    break
                elif line.startswith('线索:') and not is_sufficient:
                    result_content = line.split(':', 1)[1].strip()
                    # 如果线索跨多行，继续收集
                    for j in range(i+1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(('答案:', '判断:')):
                            result_content += " " + lines[j].strip()
                        else:
                            break
                    break
            
            print(f"混合知识推理结果: {'足够' if is_sufficient else '不足够'}")
            if is_sufficient:
                print(f"答案: {result_content[:100]}...")
            else:
                print(f"线索: {result_content[:100]}...")
            
            return is_sufficient, result_content
            
        except Exception as e:
            print(f"混合知识推理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False, "推理过程出错"


    async def tog2_multi_hop_reasoning(self, query: str) -> Dict[str, Any]: 
        try:
            # === 1. 初始化阶段 ===
            print("=" * 50)
            print("📝 初始化阶段")
            print("=" * 50)
            
            # 1.1 实体提取
            print("1.1 提取实体...")
            extracted_entities = await self.extract_entities(query)
            print(f"提取到的实体: {extracted_entities}")
            
            if not extracted_entities:
                print("❌ 未能提取到实体，退出")
                return {"query": query, "entities": [], "response": "无法从问题中提取到有效实体", "hops": 0}
            
            # 1.2 Topic Prune
            print("\n1.2 Topic Prune...")
            topic_entities = await self.topic_prune(query, extracted_entities)
            
            # 1.3 从topic entities关联的chunks中检索
            print("\n1.3 从topic entities关联的chunks中检索...")
            initial_contexts = await self.retrieve_chunks_from_topic_entities(
                query, topic_entities, k=self.context_k
            )
            
            # 1.4 初始推理
            print("\n1.4 初始推理...")
            is_sufficient, result = await self.hybrid_knowledge_reasoning(
                query, "", [], 
                [{"name": e, "context_score": 1.0} for e in topic_entities], 
                initial_contexts
            )
            
            if is_sufficient:
                print("✅ 初始阶段已找到答案!")
                return {
                    "query": query,
                    "entities": extracted_entities,
                    "topic_entities": topic_entities,
                    "response": result,
                    "hops": 0,
                    "found_answer": True,
                    "reasoning_type": "initial"
                }
            
            print(f"❌ 初始信息不足，进入多跳探索")
            current_clues = result
            
            # === 2. 多跳探索阶段 ===
            print("\n" + "=" * 50)
            print("🔍 多跳探索阶段")
            print("=" * 50)
            
            current_topic_entities = topic_entities
            all_triple_paths = []
            search_history = []
            
            for hop in range(1, self.max_depth + 1):
                print(f"\n🏃‍♂️ 第 {hop} 跳探索")
                print("-" * 30)
                
                print(f"当前实体: {current_topic_entities}")
                print(f"当前线索: {current_clues[:100]}...")
                
                # 2.1 Relation Discovery & Prune
                print(f"\n2.1 关系发现与筛选...")
                all_candidate_entities = []
                hop_triple_paths = {}
                
                for entity in current_topic_entities:
                    # 获取实体的所有关系
                    neighbors = await self.get_entity_neighbors(entity)
                    
                    if not neighbors:
                        continue
                    
                    # Relation Prune
                    selected_relations = await self.relation_prune(query, entity, neighbors, current_clues)
                    
                    # Entity Discovery
                    for rel in selected_relations:
                        candidate_entity = rel['name']
                        relation_type = rel['relation_type']
                        
                        # 构建triple路径
                        base_path = hop_triple_paths.get(entity, [])
                        new_path = base_path + [(entity, relation_type)]
                        hop_triple_paths[candidate_entity] = new_path
                        
                        # 添加到候选实体
                        rel_copy = rel.copy()
                        rel_copy['source_entity'] = entity
                        all_candidate_entities.append(rel_copy)
                
                if not all_candidate_entities:
                    print(f"❌ 第 {hop} 跳未找到候选实体，停止探索")
                    break
                
                print(f"找到 {len(all_candidate_entities)} 个候选实体")
                
                # 2.2 Context-based Entity Prune
                print(f"\n2.2 基于上下文的实体筛选...")
                top_entities = await self.context_based_entity_prune(
                    query, all_candidate_entities, hop_triple_paths
                )
                
                # 收集当前跳的triple路径
                current_paths = []
                for entity in top_entities:
                    path = hop_triple_paths.get(entity['name'], [])
                    if path:
                        current_paths.append(path)
                
                all_triple_paths.extend(current_paths)
                
                # 2.3 混合知识推理
                print(f"\n2.3 混合知识推理...")
                
                # 收集所有相关上下文
                all_contexts = []
                for entity in top_entities:
                    if 'top_contexts' in entity:
                        all_contexts.extend(entity['top_contexts'])
                
                is_sufficient, result = await self.hybrid_knowledge_reasoning(
                    query, current_clues, current_paths, top_entities, all_contexts
                )
                
                # 记录搜索历史
                search_history.append({
                    "hop": hop,
                    "topic_entities": current_topic_entities.copy(),
                    "candidate_entities": len(all_candidate_entities),
                    "selected_entities": [e['name'] for e in top_entities],
                    "triple_paths": current_paths,
                    "is_sufficient": is_sufficient,
                    "clues": current_clues
                })
                
                if is_sufficient:
                    print(f"✅ 第 {hop} 跳找到答案!")
                    return {
                        "query": query,
                        "entities": extracted_entities,
                        "topic_entities": topic_entities,
                        "response": result,
                        "hops": hop,
                        "found_answer": True,
                        "reasoning_type": "multi_hop",
                        "search_history": search_history,
                        "final_entities": top_entities,
                        "triple_paths": all_triple_paths
                    }
                
                # 更新下一轮的topic entities和clues
                current_topic_entities = [e['name'] for e in top_entities]
                current_clues = result
                
                print(f"❌ 第 {hop} 跳信息仍不足够，继续下一跳")
            
            # === 3. 达到最大深度 ===
            print(f"\n⏰ 达到最大深度 {self.max_depth}，生成最佳答案")
            
            # 使用所有收集的信息生成最终答案
            final_contexts = []
            final_entities = []
            
            for history in search_history:
                for entity_name in history['selected_entities']:
                    contexts = await self.entity_guided_context_retrieval(
                        query, entity_name, [], k=2
                    )
                    final_contexts.extend(contexts)
                    final_entities.append({"name": entity_name, "context_score": 1.0})
            
            # 最终推理
            _, final_result = await self.hybrid_knowledge_reasoning(
                query, current_clues, all_triple_paths, final_entities, final_contexts
            )
            
            return {
                "query": query,
                "entities": extracted_entities,
                "topic_entities": topic_entities,
                "response": final_result,
                "hops": self.max_depth,
                "found_answer": False,
                "reasoning_type": "exhaustive",
                "search_history": search_history,
                "final_entities": final_entities,
                "triple_paths": all_triple_paths
            }
            
        except Exception as e:
            print(f"❌ ToG-2推理过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {
                "query": query,
                "entities": [],
                "response": f"推理过程出错: {str(e)}",
                "hops": 0,
                "found_answer": False,
                "error": str(e)
            }
    

async def demo_tog2():
    rag = KnowledgeGraphRAG()
    # 测试查询
    query = "2012—2021年全国羊肉产量年度变化情况（万吨）2021年，全国羊肉产量同比增长率约为（ ）。A. 2.4%\nB. 3.4%\nC. 4.4%\nD. 5.4%"
    
    
    try:
        # 使用ToG-2推理
        result = await rag.tog2_multi_hop_reasoning(query)

        print(f"result: {result}")
        

    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()




if __name__ == "__main__":

    asyncio.run(demo_tog2())
