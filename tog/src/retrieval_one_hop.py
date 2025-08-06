import os
import json
import re
from typing import List
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain.schema import Document

# 需要确保这些模块存在，否则需要替换实现
try:
    from gptqa import label_entity
    from utils import *
    from Qwen3_Reranker import Qwen3Reranker
except ImportError as e:
    print(f"Warning: Could not import {e.name}. Please ensure the module is available.")

# OpenAI API 配置
from openai import OpenAI
client = OpenAI(api_key="sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2", base_url="https://api.gptsapi.net/v1")

# 批次处理配置
BATCH_SIZE = 8  # 并行线程数，可根据需求调整

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B"):
        self.model = SentenceTransformer(model_name_or_path, device="cuda:1")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True, prompt_name="document", batch_size=1).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

class RAGPipeline:
    def __init__(self):
        # 初始化各种模型和集合
        self.model = SentenceTransformerEmbeddings()
        self.topic_sentence_dict = {}
        self.all_ids_to_filter = []
        
        # 加载向量数据库
        vector_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/vector_data"
        entity_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/entity_vector"
        
        self.client = chromadb.PersistentClient(path=vector_path)
        self.entity_client = chromadb.PersistentClient(path=entity_path)
        self.collection = self.client.get_collection(name="edu_chunks")
        self.entity_collection = self.entity_client.get_collection(name="all_entity")

        # 初始化重排序模型
        reranker_model_path = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B"
        self.reranker_model = Qwen3Reranker(
            model_name_or_path=reranker_model_path,
            max_length=2048,
            instruction="Given the user query, retrieval the relevant passages",
            device_id="cuda:7"
        )

    def extract_entity(self, results):
        """从检索结果中提取实体信息"""
        try:
            documents = results.get("documents", [])[0] if results.get("documents") else []
            distances = results.get("distances", [])[0] if results.get("distances") else []
            
            entities = []
            for doc, dist in zip(documents, distances):
                # 假设文档格式为 "entity_name|entity_id"
                if "|" in doc:
                    entity_name, entity_id = doc.split("|", 1)
                    entities.append([(entity_name, entity_id, dist)])
                else:
                    entities.append([(doc, "unknown", dist)])
            return entities
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def query_link_to_graph(self, topic_entity_name, n_results=5):
        """查询实体与图谱的链接"""
        try:
            topic_entity_embedding = self.model.embed_query(topic_entity_name)    
            topic_entities_result = retrieve_from_graph(self.entity_collection, topic_entity_embedding, n_results)

            topic_entity_in_graph = self.extract_entity(topic_entities_result)

            threshold = 0.2 
            filtered_topic_entity_in_graph = [[item for item in sublist if item[2] <= threshold] for sublist in topic_entity_in_graph]

            normalized_entities = [(entity, ids) for entity_list in filtered_topic_entity_in_graph for entity, ids, _ in entity_list]

            if len(normalized_entities) == 0:
                return topic_entity_name, None, None

            topic_entities_graph_info = get_graph_info(topic_entities_result, filtered_topic_entity_in_graph)
            
            return topic_entities_graph_info, filtered_topic_entity_in_graph
        except Exception as e:
            print(f"Error in query_link_to_graph: {e}")
            return None, None

    def targeted_retrieve(self, query_embedding, topic_entities_name, topic_entities_graph_info):
        """定向检索相关文档"""
        try:
            ids_to_filter_dict = build_ids_to_filter(topic_entities_name, topic_entities_graph_info)

            ids_to_filter = set()
            for ids in ids_to_filter_dict.values():
                ids_to_filter.update(ids)
            ids_to_filter = list(ids_to_filter)
            
            topic_documents = retrieve_from_graph(self.collection, query_embedding, n_results=50, ids_to_filter=ids_to_filter)
            topic_documents_list = topic_documents.get("documents", [])[0] if topic_documents.get("documents") else []
            
            return topic_documents_list
        except Exception as e:
            print(f"Error in Targeted_Retrieve: {e}")
            return []

    def simple_entity_extraction(self, query, language="Chinese"):
        """简单的实体提取实现（当gptqa模块不可用时）"""
        # 使用正则表达式提取中文实体
        import jieba
        words = jieba.lcut(query)
        # 过滤掉停用词和标点符号
        entities = [word for word in words if len(word) > 1 and word.isalpha()]
        return [{"entity_text": entity} for entity in entities[:5]]  # 限制实体数量

    def inference_main(self, query, query_embedding):
        """主要的推理函数"""
        try:
            # 实体提取
            try:
                labeled_entities = label_entity([query], language="Chinese", openai_model_name="gpt-4.1-mini")
                topic_entities_name = [entity['entity_text'] for entity in labeled_entities[0]['entity_list']]
            except:
                # 如果gptqa模块不可用，使用简单的实体提取
                entities = self.simple_entity_extraction(query)
                topic_entities_name = [entity['entity_text'] for entity in entities]
            
            if len(topic_entities_name) == 0:  
                return []

            topic_entities_graph_info, filtered_topic_entity_in_graph = self.query_link_to_graph(topic_entities_name)
            
            if topic_entities_graph_info is None:
                return []
                
            topic_documents_list = self.targeted_retrieve(query_embedding, topic_entities_name, topic_entities_graph_info)

            return topic_documents_list
        except Exception as e:
            print(f"Error in inference_main: {e}")
            return []

    def rerank_documents(self, query: str, documents: List[str], batch_size: int = 10) -> List[str]:
        """重新排序文档"""
        try:
            scored_docs = []
            num_docs = len(documents)
            for start_idx in range(0, num_docs, batch_size):
                end_idx = min(start_idx + batch_size, num_docs)
                batch_docs = documents[start_idx:end_idx]
                pairs = [(query, doc) for doc in batch_docs]
                scores = self.reranker_model.compute_scores(pairs)
                scored_docs.extend(zip(batch_docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in scored_docs]
        except Exception as e:
            print(f"Error in rerank_documents: {e}")
            return documents  # 返回原始文档列表

    def reciprocal_rank_fusion(self, doc_lists, k=60, weight=60):
        """
        简易版 RRF，融合多个文档列表，根据 reciprocal rank 打分
        """
        try:
            scores = defaultdict(float)
            doc_map = {}

            for doc_list in doc_lists:
                for rank, doc in enumerate(doc_list[:k]):
                    scores[doc] += 1.0 / (weight + rank)
                    doc_map[doc] = doc
            
            sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [doc_map[doc_id] for doc_id, _ in sorted_docs]
        except Exception as e:
            print(f"Error in reciprocal_rank_fusion: {e}")
            return doc_lists[0] if doc_lists else []

    def rerank_invoke(self, query, n_results=10):
        """完整的检索和重排序流程"""
        try:
            # 获取查询向量
            query_embedding = self.model.embed_query(query)
            
            # 定向检索
            topic_documents_list = self.inference_main(query, query_embedding)
            
            # 嵌入式检索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=50,
                include=["embeddings", "documents", "distances", "metadatas"]
            )
            
            embedding_doc_list = results.get("documents", [])[0] if results.get("documents") else []
            
            # 融合检索结果
            reciprocal_rank_fusion_docs = self.reciprocal_rank_fusion([topic_documents_list, embedding_doc_list])
            
            # 重排序
            rerank_doc_list = self.rerank_documents(query, reciprocal_rank_fusion_docs, batch_size=10)
            rerank_doc_list = rerank_doc_list[:n_results]  # 限制返回的文档数量
            
            # 转换为Document格式
            documents = [Document(page_content=doc) for doc in rerank_doc_list]
            
            return documents
        except Exception as e:
            print(f"Error in rerank_invoke: {e}")
            return []

    def answer(self, query, docs, max_docs=5):
        """调用OpenAI API回答问题"""
        try:
            # 限制上下文文档数量
            context_docs = [doc.page_content for doc in docs[:max_docs]]
            context = "\n\n".join([f"文档{i+1}: {doc}" for i, doc in enumerate(context_docs)])
            
            prompt = f"""
请根据以下检索材料回答用户问题。回答必须有依据，严禁凭空猜测。
任务要求：
1. 先分析题干，明确题目类型（单选、多选）；
2. 请判断每个选项：
    a. 分析选项本身是否科学、准确、逻辑合理；
    b. 判断该选项是否贴合题干要求，语义是否匹配；
3. 仅当选项本身正确且符合题意，才判定为"正确选项"。
4. 明确指出每个选项是"正确"还是"错误"，并说明原因。
5. 若题目类型为单选，请只回答一个选项，若题目类型为多选，请回答多个正确选项。
6. 回答需简洁有力，结论明确。
7. 需要使用<answer>标签包裹答案部分。例如 <answer>AC</answer>
文档内容：
{context}

用户问题：
{query}

答复：
"""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手，基于提供的文档内容回答用户问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return "抱歉，调用OpenAI API时发生错误。"



def load_processed_questions(output_path):
    """加载已处理的问题"""
    processed = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                processed.add(item.get("question", ""))
    except FileNotFoundError:
        pass
    return processed

def extract_answer(text):
    """从 llm_answer 中提取 <answer> 标签内容"""
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else ""
    return answer

def process_item(item, pipeline):
    """处理单个问题项"""
    try:
        question = item.get("question", "")
        options = item.get("options", [])
        answer = item.get("answer", "")
        explanation = item.get("explanation", "")
        extension = item.get("extension", "")
        
        if not question or not options:
            return None

      
        query = f"问题：{question}\n答案选项: {options}"
        docs = pipeline.rerank_invoke(query, 10)

        # 一次性回答所有选项
        llm_answer = pipeline.answer(query, docs)

        # 新增：提取 answer 与 explanation
        parsed_answer = extract_answer(llm_answer)

        result = {
            "question": question,
            "options": options,
            "answer": answer,        
            "parsed_answer": parsed_answer,
            "llm_answer": llm_answer,
            "retrieved_docs": [doc.page_content for doc in docs],
            "explanation": explanation,
            "extension": extension
        }

        return result
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

def chunks(lst, n):
    """将列表 lst 分割成每组 n 个的小列表"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    """主函数"""
    # 文件路径配置
    input_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/datasets/200_que.jsonl"
    output_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/batch_tog_results_4o-mini.jsonl"
    
    # 初始化RAG管道
    print("正在初始化RAG管道...")
    pipeline = RAGPipeline()
    
    # 加载已处理的问题
    processed_questions = load_processed_questions(output_path)
    print(f"已处理问题数量: {len(processed_questions)}")

    # 加载未处理的数据项
    all_items = []
    num_quest = 200
    with open(input_path, "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            if index >= num_quest:
                break
            item = json.loads(line)
            if item.get("question", "") not in processed_questions:
                all_items.append(item)
    
    print(f"待处理问题数量: {len(all_items)}")

    # 按 BATCH_SIZE 分批并并行处理
    for batch_items in tqdm(chunks(all_items, BATCH_SIZE), 
                           total=(len(all_items) + BATCH_SIZE - 1) // BATCH_SIZE, 
                           desc="Processing batches"):
        results = []
        with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
            futures = {executor.submit(process_item, item, pipeline): item for item in batch_items}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"处理批次时发生错误: {e}")

        # 每批处理完成立即写入文件
        if results:
            with open(output_path, "a", encoding="utf-8") as out_f:
                for result in results:
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"已保存 {len(results)} 个结果到文件")

    print("批次处理完成！")

if __name__ == "__main__":
    main()