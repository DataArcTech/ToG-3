from rag_flow_example_commentary import TCL_RAG
from examples.simple_graphrag.simple_query import KnowledgeGraphRAG
from examples.event_graphrag.test_online_retrieval import OnlineRetrievalTester, TestConfig
import yaml
import json
import os
from dataclasses import dataclass
import time
import asyncio
from rag_factory.Retrieval import Document
import traceback

# 加载配置文件
with open('/finance_ML/liuyingqi/RAG-Factory/examples/TCL_rag/config_commentary.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

llm_config = config['llm']
embedding_config = config['embedding']
reranker_config = config['reranker']
bm25_retriever_config = config['bm25']
retriever_config = config['retriever']
vector_store_config = config['store']
rewrite_config = config['rewrite']

import re

def extract_json_from_markdown(text):
    """从Markdown文本中提取JSON内容"""
    # 匹配 ```json ... ``` 或 ``` ... ``` 中的内容
    json_match = re.search(r'["\']*(?:json)?\s*(\{.*\})\s*["\']*', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    
    # 如果没有找到代码块，尝试直接查找JSON对象
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    
    return None

@dataclass
class Example:
    question_id: str
    question_content: str
    materials: list
    answer: str

@dataclass
class Example_query:
    question_content: str
    materials: list
    rewritten_query: str
    answer: str

    def to_dict(self):
        return {
            "question_content": self.question_content,
            "rewritten_query": self.rewritten_query,
            "materials": self.materials,
            "answer": self.answer
        }

async def rewrite_query(rag, example_data, file_path):
    example_query_list = []
    for idx, example in enumerate(example_data):
        query = example.question_content
        answer = example.answer
        materials = example.materials
        rewritten_query = await asyncio.to_thread(rag.rewrite_query, query)
        example_query = Example_query(question_content = query,
                        materials = materials,
                        rewritten_query = rewritten_query,
                        answer = answer)
        # print("example_query: ", example_query.to_dict())
        example_query_list.append(example_query)
    if os.path.exists(file_path):
        os.remove(file_path)                    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps([example.to_dict() for example in example_query_list], ensure_ascii=False, indent=2) + "\n")
            

async def run_eval():
    rag = TCL_RAG(llm_config=llm_config, 
                embedding_config=embedding_config, 
                reranker_config=reranker_config, 
                retriever_config=retriever_config, 
                vector_store_config=vector_store_config,
                bm25_retriever_config=bm25_retriever_config)
    
    example_data_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/example/申论真题2020-2024.json"
    example_data = []
    with open(example_data_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    for example in examples:
        example_data.append(Example(
            question_id=example['question_id'], 
            question_content=example['question_content'], 
            materials=example['materials'], 
            answer=example['answer']))
    print("example_data_num: ", len(example_data))

    file_path = rewrite_config['path']
    try:
        await rewrite_query(rag, example_data, file_path)
    except Exception as e:
        print("完整堆栈信息:")
        traceback.print_exc()  # 直接打印错误堆栈

if __name__ == "__main__":
    asyncio.run(run_eval())
        
        