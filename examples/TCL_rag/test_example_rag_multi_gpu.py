from rag_flow_example import TCL_RAG
from examples.simple_graphrag.simple_query import KnowledgeGraphRAG
from examples.event_graphrag.test_online_retrieval import OnlineRetrievalTester, TestConfig
import yaml
import json
import os
import copy
from dataclasses import dataclass
import time
import asyncio
from rag_factory.Retrieval import Document
import re
from prompt import ANALYZE_RAG_PROMPT

def extract_json_from_markdown(text):
    """从Markdown文本中提取JSON内容"""
    print("解析json文本:", text)
    json_match = re.search(r'["\']*(?:json)?\s*(\{.*\})\s*["\']*', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    
    return None

@dataclass
class Example_query:
    question_number: str
    question_text: str
    rewritten_query: str
    knowledge: list
    answer: str
    explanation: str

@dataclass
class Example_result:
    query_id: str
    rag_spend_time: float
    rag_answer: dict
    rag_retrieval: list
    rag_is_correct: bool
    
    def to_dict(self):
        return {
            "query_id": self.query_id,
            "rag_spend_time": self.rag_spend_time,
            "rag_answer": self.rag_answer,
            "rag_retrieval": self.rag_retrieval,
            "rag_is_correct": self.rag_is_correct
        }

class MultiGPUConfigManager:
    """多GPU配置管理器"""
    
    def __init__(self, base_config_path: str):
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
    
    def _load_base_config(self) -> dict:
        """加载基础配置文件"""
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def create_gpu_config(self, gpu_id: int, test_round: int) -> dict:
        """为指定GPU和测试轮次创建配置"""
        config = copy.deepcopy(self.base_config)
        
        # 更新embedding设备
        config['embedding']['model_kwargs']['device'] = f"cuda:{gpu_id}"
        
        # 更新reranker设备
        if 'reranker' in config:
            config['reranker']['device_id'] = f"cuda:{gpu_id}"
        
        return config
    
    def create_file_path(self, gpu_id: int, test_round: int, base_path: str) -> str:
        """创建对应的文件路径"""
        dir_path = os.path.dirname(base_path)
        file_name = os.path.basename(base_path)
        
        name_without_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]
        
        new_file_name = f"{name_without_ext}_gpu{gpu_id}_round{test_round}{ext}"
        return os.path.join(dir_path, new_file_name)

async def write_result(rag, example_data, file_path, gpu_id, test_round, query_id_list, correct_count, total_count, error_file_name):
    """写入结果到指定文件"""
    score_threshold = 0.5
    example_result = []
    
    print(f"🚀 GPU {gpu_id} 第 {test_round} 轮行测RAG测试开始，结果保存到: {file_path}")
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU {gpu_id} 第 {test_round} 轮行测RAG测试结果" + "\n")
        f.write("="*80 + "\n")
    
    for example in example_data:
        query_id = example.question_number
        if query_id not in query_id_list:
            continue
        query = example.question_text
        answer = example.answer
        explanation = example.explanation
        knowledge = example.knowledge
        rewritten_query = example.rewritten_query
        
        # RAG检索和生成
        rag_start_time = time.time()
        result = await asyncio.to_thread(rag.invoke, rewritten_query, k=20)
        result = await asyncio.to_thread(rag.rerank, rewritten_query, result, k=10)
        docs = [doc for doc, score in result if score > score_threshold]
        rag_answer = await asyncio.to_thread(rag.rag_answer, query, docs)
        rag_end_time = time.time()
        rag_spend_time = rag_end_time - rag_start_time
        
        rag_answer = extract_json_from_markdown(rag_answer)
        rag_answer_dict = json.loads(rag_answer) if isinstance(rag_answer, str) else rag_answer
        
        # 统计正确性
        if not rag_answer_dict['answer'] == answer:
            error_file_name[query_id].append(file_path.split("/")[-1])
        else:
            correct_count[query_id] += 1
        total_count[query_id] += 1
        
        # 保存结果
        example_result.append(
            Example_result(
                query_id=query_id,
                rag_spend_time=rag_spend_time,
                rag_answer=rag_answer_dict,
                rag_retrieval=[{'doc': doc.to_dict(), 'score': score} for doc, score in result if score > score_threshold],
                rag_is_correct=rag_answer_dict['answer'] == answer
            )
        )
    
    # 写入结果文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps([example.to_dict() for example in example_result], ensure_ascii=False, indent=2) + "\n")
    
    print(f"✅ GPU {gpu_id} 第 {test_round} 轮行测RAG测试完成，结果已保存")

async def run_single_gpu_test(gpu_id: int, test_round: int, config_manager: MultiGPUConfigManager, 
                             example_data: list, base_file_path: str, query_id_list: list, correct_count: dict, total_count: dict, error_file_name: dict,
                             rag: TCL_RAG):
    """在单个GPU上运行一轮测试"""
    try:
        # 创建文件路径
        file_path = config_manager.create_file_path(gpu_id, test_round, base_file_path)
        
        # 运行测试（使用已初始化的rag）
        await write_result(rag, example_data, file_path, gpu_id, test_round, query_id_list, correct_count, total_count, error_file_name)
        
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 第 {test_round} 轮行测RAG测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_multi_gpu_tests(gpu_ids: list, test_rounds: int, config_path: str, 
                             example_data_path: str, base_result_path: str, query_id_list: list, result_path: str):
    """运行多GPU多轮测试"""
    print(f"🚀 开始行测RAG多GPU多轮测试")
    print(f"GPU设备: {gpu_ids}")
    print(f"测试轮次: {test_rounds}")
    print(f"配置文件: {config_path}")
    print(f"示例数据: {example_data_path}")
    print(f"结果基础路径: {base_result_path}")
    print("="*80)
    
    # 初始化配置管理器
    config_manager = MultiGPUConfigManager(config_path)
    
    # 加载示例数据
    example_data = []
    with open(example_data_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
        for example in examples:
            example_data.append(
                Example_query(
                    question_number=example['question_number'],
                    question_text=example['question_text'],
                    rewritten_query=example['rewritten_query'],
                    knowledge=example['knowledge'],
                    answer=example['answer'],
                    explanation=example['explanation']
                )
            )
    
    print(f"📊 加载了 {len(example_data)} 个行测测试示例")

    # 对每一道题的测试结果进行保存，计算结果的正确率，并记录每一道题目的错误结果保存在哪个json文件中
    correct_count = {query_id: 0 for query_id in query_id_list}
    total_count = {query_id: 0 for query_id in query_id_list}
    error_file_name = {query_id: [] for query_id in query_id_list}
    
    # 为每个GPU初始化一次RAG系统
    gpu_rag_systems = {}
    
    print("🚀 为每个GPU初始化RAG系统...")
    for gpu_id in gpu_ids:
        try:
            # 创建GPU特定配置
            gpu_config = config_manager.create_gpu_config(gpu_id, 1)  # 使用第一轮配置
            
            # 初始化RAG系统
            rag = TCL_RAG(
                llm_config=gpu_config['llm'],
                embedding_config=gpu_config['embedding'],
                reranker_config=gpu_config['reranker'],
                retriever_config=gpu_config['retriever'],
                vector_store_config=gpu_config['store'],
                bm25_retriever_config=gpu_config['bm25']
            )
            
            gpu_rag_systems[gpu_id] = rag
            
            print(f"✅ GPU {gpu_id} 初始化完成")
            
        except Exception as e:
            print(f"❌ GPU {gpu_id} 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建任务列表
    tasks = []
    for gpu_id in gpu_ids:
        if gpu_id not in gpu_rag_systems:
            continue
        for round_num in range(1, test_rounds + 1):
            task = run_single_gpu_test(
                gpu_id, round_num, config_manager, example_data, base_result_path, 
                query_id_list, correct_count, total_count, error_file_name, gpu_rag_systems[gpu_id]
            )
            tasks.append(task)
    
    # 并发执行所有任务
    print(f"🚀 开始执行 {len(tasks)} 个测试任务...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计结果
    success_count = sum(1 for r in results if r is True)
    failed_count = len(results) - success_count

    print("\n" + "="*80)
    print(f"📊 行测RAG测试完成统计:")
    print(f"总任务数: {len(tasks)}")
    print(f"成功任务: {success_count}")
    print(f"失败任务: {failed_count}")
    print(f"任务成功率: {success_count/len(tasks)*100:.1f}%")
    print("\n" + "="*80)
    # 打印题目正确率
    print(f"题目正确率:")
    for query_id in correct_count:
        print(f"第{query_id}题的正确率: {correct_count[query_id]/total_count[query_id]*100:.1f}%    正确数: {correct_count[query_id]}    总数: {total_count[query_id]}\n错题所在文件: {str(error_file_name[query_id])}\n")
    print(f"总正确率: {sum(correct_count.values())/sum(total_count.values())*100:.1f}%")
    print("\n" + "="*80)
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"prompt: {ANALYZE_RAG_PROMPT}\n")
        f.write("\n" + "="*40 + "\n")
        f.write(f"总正确率: {sum(correct_count.values())/sum(total_count.values())*100:.1f}%" + "\n")
        for query_id in correct_count:
            f.write(f"第{query_id}题的正确率: {correct_count[query_id]/total_count[query_id]*100:.1f}%    正确数: {correct_count[query_id]}    总数: {total_count[query_id]}\n错题所在文件: {str(error_file_name[query_id])}\n")
        f.write("\n" + "="*80 + "\n")
    if failed_count > 0:
        print(f"\n❌ 失败的任务:")
        for i, result in enumerate(results):
            if result is not True:
                print(f"  任务 {i+1}: {result}")
    
    return results

async def main():
    """主函数"""
    # 配置参数
    gpu_ids = [0, 2, 3, 4, 5, 6, 7]  # 要使用的GPU设备ID
    test_rounds = 3  # 每个GPU的测试轮次
    # query_id_list = [1, 6, 7, 10, 14, 17, 18, 20, 21] # 要测试的题目编号
    query_id_list = [i for i in range(1, 28)] # 全部题目编号
    config_path = "/finance_ML/liuyingqi/RAG-Factory/examples/TCL_rag/config_rag.yaml"
    example_data_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/result/rewrite_query.json"
    base_result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/rag_answer_test/answer_by_rag_multi_gpu.json"
    result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/rag_answer_test/4.1_result.txt"
    # 运行多GPU多轮测试
    await run_multi_gpu_tests(gpu_ids, test_rounds, config_path, example_data_path, base_result_path, query_id_list, result_path)

if __name__ == "__main__":
    asyncio.run(main())


