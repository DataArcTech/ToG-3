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

def extract_json_from_markdown(text):
    """从Markdown文本中提取JSON内容"""
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
class graph_result:
    query_id: str
    graph_spend_time: float
    graph_answer: dict
    graph_retrieval: list
    graph_result: list
    graph_is_correct: bool

@dataclass
class rag_result:
    query_id: str
    rag_spend_time: float
    rag_answer: dict
    rag_retrieval: list
    rag_is_correct: bool

@dataclass
class llm_result:
    query_id: str
    llm_spend_time: float
    llm_answer: dict
    llm_is_correct: bool

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

async def write_result(rag, example_data, graph_path, rag_path, llm_path, file_path, gpu_id, test_round, query_id_list, correct_count, total_count, error_file_name):
    """写入结果到指定文件"""
    rag_scores = []
    llm_scores = []
    graph_scores = []
    rag_is_correct = []
    llm_is_correct = []
    graph_is_correct = []
    rag_time = []
    llm_time = []
    graph_time = []
    score_threshold = 0.5
    graph_score_threshold = 0.5
    graph_result = []
    rag_result = []
    llm_result = []
    
    # 读取各GPU的结果文件
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_result = json.load(f)
        graph_result = {result["query_id"]: result for result in graph_result}
    with open(rag_path, 'r', encoding='utf-8') as f:
        rag_result = json.load(f)
        rag_result = {result["query_id"]: result for result in rag_result}
    with open(llm_path, 'r', encoding='utf-8') as f:
        llm_result = json.load(f)
        llm_result = {result["query_id"]: result for result in llm_result}

    print(f"🚀 GPU {gpu_id} 第 {test_round} 轮行测对比测试开始，结果保存到: {file_path}")
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU {gpu_id} 第 {test_round} 轮行测对比测试结果" + "\n")
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
        
        graph_answer = graph_result[query_id]['graph_answer']
        rag_answer = rag_result[query_id]['rag_answer']
        llm_answer = llm_result[query_id]['llm_answer']

        # 知识点匹配评估
        match_knowledge = await asyncio.to_thread(rag.match_knowledge, graph_answer['knowledge'], rag_answer['knowledge'], llm_answer['knowledge'], knowledge)
        match_knowledge = extract_json_from_markdown(match_knowledge)

        # 统计正确性
        if graph_answer['answer'] == answer:
            graph_is_correct.append(query)
        if rag_answer['answer'] == answer:
            rag_is_correct.append(query)
        if llm_answer['answer'] == answer:
            llm_is_correct.append(query)
        
        # 统计知识点命中率
        graph_scores.append(match_knowledge['graph_hit_rate'])
        rag_scores.append(match_knowledge['rag_hit_rate'])
        llm_scores.append(match_knowledge['llm_hit_rate'])
        
        # 统计时间
        graph_time.append(graph_result[query_id]['graph_spend_time'])
        rag_time.append(rag_result[query_id]['rag_spend_time'])
        llm_time.append(llm_result[query_id]['llm_spend_time'])
        
        # 统计正确性
        if not graph_answer['answer'] == answer:
            error_file_name[query_id].append(file_path.split("/")[-1])
        else:
            correct_count[query_id] += 1
        total_count[query_id] += 1
        
        # 写入详细结果
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write("query_id: " + str(query_id) + "\n")
            f.write("query: \n" + query + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("参考答案: \n" + answer + "\n\n")
            f.write("参考解析: \n" + explanation + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("graph_spend_time: " + str(graph_result[query_id]['graph_spend_time']) + "\n\n")
            f.write("answer by graph: \n" + json.dumps(graph_answer, ensure_ascii=False, indent=2) + "\n\n")
            f.write(f"\n提取的实体: {graph_result[query_id]['seed_nodes']}" + "\n\n")
            f.write("\n=== 实体检索详情 ===" + "\n")
            f.write("graph+rag检索结果：\n")
            f.write(json.dumps(graph_result[query_id]['graph_retrieval'], ensure_ascii=False, indent=2) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("rewritten_query: \n" + rewritten_query + "\n\n")
            f.write("rag_spend_time: " + str(rag_result[query_id]['rag_spend_time']) + "\n\n")
            f.write("answer by rag: \n" + json.dumps(rag_answer, ensure_ascii=False, indent=2) + "\n\n")
            f.write("retrieved materials: \n")
            f.write(json.dumps(rag_result[query_id]['rag_retrieval'], ensure_ascii=False, indent=2) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("llm_spend_time: " + str(llm_result[query_id]['llm_spend_time']) + "\n\n")
            f.write("answer by llm: \n" + json.dumps(llm_answer, ensure_ascii=False, indent=2) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("知识点: " + str(knowledge) + "\n\n")
            f.write("graph知识点: " + str(graph_answer['knowledge'])  + "\n\n")
            f.write("rag知识点: " + str(rag_answer['knowledge']) + "\n\n")
            f.write("llm知识点: " + str(llm_answer['knowledge']) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("graph命中率: " + str(match_knowledge['graph_hit_rate']) + "%" + "\n\n")
            f.write("rag命中率: " + str(match_knowledge['rag_hit_rate']) + "%" + "\n\n")
            f.write("llm命中率: " + str(match_knowledge['llm_hit_rate']) + "%" + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("graph答案: ")
            if graph_answer['answer'] == answer:
                f.write("正确" + "\n\n")
            else:
                f.write("错误" + "\n\n")
            f.write("rag答案: ")
            if rag_answer['answer'] == answer:
                f.write("正确" + "\n\n")
            else:
                f.write("错误" + "\n\n")
            f.write("llm答案: ")
            if llm_answer['answer'] == answer:
                f.write("正确" + "\n\n")
            else:
                f.write("错误" + "\n\n")
            f.write("-"*100 + "\n\n")

    # 写入统计结果
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write("题目总数: " + str(len(rag_scores)) + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("graph平均命中率: " + str(round(sum(graph_scores)/len(graph_scores), 2)) + "%" + "\n")
        f.write("graph正确率: " + str(round(len(graph_is_correct)/len(graph_scores)*100, 2)) + "%" + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("rag平均命中率: " + str(round(sum(rag_scores)/len(rag_scores), 2)) + "%" + "\n")
        f.write("rag正确率: " + str(round(len(rag_is_correct)/len(rag_scores)*100, 2)) + "%" + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("llm平均命中率: " + str(round(sum(llm_scores)/len(llm_scores), 2)) + "%" + "\n")
        f.write("llm正确率: " + str(round(len(llm_is_correct)/len(llm_scores)*100, 2)) + "%" + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("graph时间: " + str(round(sum(graph_time)/len(graph_time), 2)) + "秒" + "\n")
        f.write("rag时间: " + str(round(sum(rag_time)/len(rag_time), 2)) + "秒" + "\n")
        f.write("llm时间: " + str(round(sum(llm_time)/len(llm_time), 2)) + "秒" + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
    
    print(f"✅ GPU {gpu_id} 第 {test_round} 轮行测对比测试完成，结果已保存")

async def run_single_gpu_test(gpu_id: int, test_round: int, config_manager: MultiGPUConfigManager, 
                             example_data: list, graph_path: str, rag_path: str, llm_path: str, base_file_path: str,
                             query_id_list: list, correct_count: dict, total_count: dict, error_file_name: dict,
                             rag: TCL_RAG):
    """在单个GPU上运行一轮测试"""
    try:
        # 创建文件路径
        file_path = config_manager.create_file_path(gpu_id, test_round, base_file_path)
        
        # 运行测试（使用已初始化的rag）
        await write_result(rag, example_data, graph_path, rag_path, llm_path, file_path, gpu_id, test_round, query_id_list, correct_count, total_count, error_file_name)
        
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 第 {test_round} 轮行测对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_multi_gpu_tests(gpu_ids: list, test_rounds: int, config_path: str, 
                             example_data_path: str, graph_path: str, rag_path: str, llm_path: str, base_result_path: str, query_id_list: list):
    """运行多GPU多轮测试"""
    print(f"🚀 开始行测对比多GPU多轮测试")
    print(f"GPU设备: {gpu_ids}")
    print(f"测试轮次: {test_rounds}")
    print(f"配置文件: {config_path}")
    print(f"示例数据: {example_data_path}")
    print(f"图检索结果: {graph_path}")
    print(f"RAG结果: {rag_path}")
    print(f"LLM结果: {llm_path}")
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
                gpu_id, round_num, config_manager, example_data, graph_path, rag_path, llm_path, base_result_path,
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
    print(f"📊 行测对比测试完成统计:")
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
    if failed_count > 0:
        print(f"\n❌ 失败的任务:")
        for i, result in enumerate(results):
            if result is not True:
                print(f"  任务 {i+1}: {result}")
    
    return results

async def main():
    """主函数"""
    # 配置参数
    gpu_ids = [0, 1, 2, 3, 4, 6, 7]  # 要使用的GPU设备ID
    test_rounds = 10  # 每个GPU的测试轮次
    query_id_list = [1, 6, 7, 10, 14, 17, 18, 20, 21] # 要测试的题目编号
    config_path = "/finance_ML/liuyingqi/RAG-Factory/examples/TCL_rag/config_llm.yaml"
    example_data_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/result/rewrite_query.json"
    graph_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/graph_answer_test/answer_by_graph_multi_gpu.json"
    rag_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/rag_answer_test/answer_by_rag_multi_gpu.json"
    llm_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/llm_answer_test/answer_by_llm_multi_gpu.json"
    base_result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/match_test/answer_by_match_multi_gpu.txt"
    result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/match_test/4.1_result.txt"
    # 运行多GPU多轮测试
    await run_multi_gpu_tests(gpu_ids, test_rounds, config_path, example_data_path, graph_path, rag_path, llm_path, base_result_path, query_id_list, result_path)

if __name__ == "__main__":
    asyncio.run(main())


