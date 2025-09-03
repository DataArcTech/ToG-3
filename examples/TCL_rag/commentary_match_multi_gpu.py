from rag_flow_example_commentary import TCL_RAG
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
import itertools

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
    question_content: str
    materials: str
    answer: str

@dataclass
class Example_result:
    query_id: str
    graph_spend_time: float
    seed_nodes: list
    graph_retrieval: list
    graph_answer: str

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

async def write_result(rag, example_data, graph_path, rag_path, llm_path, file_path, gpu_id, test_round, graph_scores: dict, rag_scores: dict, llm_scores: dict, graph_win: dict, rag_win: dict, llm_win: dict):
    """写入结果到指定文件"""
    rag_single_scores = []
    graph_single_scores = []
    llm_single_scores = []
    rag_single_win = []
    graph_single_win = []
    llm_single_win = []
    rag_time = []
    llm_time = []
    graph_time = []
    rag_score_threshold = 0.3
    graph_score_threshold = 0.1
    graph_result = []
    rag_result = []
    llm_result = []
    graph_path = graph_path.replace("answer_by_graph_commentary_multi_gpu.json", f"answer_by_graph_commentary_multi_gpu_gpu{gpu_id}_round{test_round}.json")
    rag_path = rag_path.replace("answer_by_rag_commentary_multi_gpu.json", f"answer_by_rag_commentary_multi_gpu_gpu{gpu_id}_round{test_round}.json")
    llm_path = llm_path.replace("answer_by_llm_commentary_multi_gpu.json", f"answer_by_llm_commentary_multi_gpu_gpu{gpu_id}_round{test_round}.json")

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

    print(f"🚀 GPU {gpu_id} 第 {test_round} 轮申论对比测试开始，结果保存到: {file_path}")
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU {gpu_id} 第 {test_round} 轮申论对比测试结果" + "\n")
        f.write("="*80 + "\n")
    
    for idx, example in enumerate(example_data):
        query_id = idx
        query = example.question_content
        answer = example.answer
        materials = example.materials
        
        graph_answer = graph_result[query_id]['graph_answer']
        rag_answer = rag_result[query_id]['rag_answer']
        llm_answer = llm_result[query_id]['llm_answer']

        # 答案质量评估
        judge = await asyncio.to_thread(rag.judge_answer, query, materials, graph_answer, rag_answer, llm_answer, answer)
        judge_json = extract_json_from_markdown(judge)

        # 统计得分
        rag_single_scores.append(judge_json['rag_score'])
        graph_single_scores.append(judge_json['graph_score'])
        llm_single_scores.append(judge_json['llm_score'])
        graph_scores[query_id].append(judge_json['graph_score'])
        rag_scores[query_id].append(judge_json['rag_score'])
        llm_scores[query_id].append(judge_json['llm_score'])
        
        # 统计获胜情况
        if judge_json['recommend'] == 'graph':
            graph_single_win.append(query)
            graph_win[query_id] += 1
        elif judge_json['recommend'] == 'rag':
            rag_single_win.append(query)
            rag_win[query_id] += 1
        else:
            llm_single_win.append(query)
            llm_win[query_id] += 1
        
        # 统计时间
        graph_time.append(graph_result[query_id]['graph_spend_time'])
        rag_time.append(rag_result[query_id]['rag_spend_time'])
        llm_time.append(llm_result[query_id]['llm_spend_time'])

        # 写入详细结果
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write("query_id: " + str(query_id) + "\n")
            f.write("query: \n" + query + "\n\n")
            f.write("materials: \n" + str(materials) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("参考答案: \n" + answer + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("graph_spend_time: " + str(graph_result[query_id]['graph_spend_time']) + "\n\n")
            f.write("answer by graph: \n" + json.dumps(graph_answer, ensure_ascii=False, indent=2) + "\n\n")
            f.write(f"\n提取的实体: {graph_result[query_id]['seed_nodes']}" + "\n\n")
            f.write("\n=== 实体检索详情 ===" + "\n")
            f.write("graph+rag检索结果：\n")
            f.write(json.dumps(graph_result[query_id]['graph_retrieval'], ensure_ascii=False, indent=2) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("rag_spend_time: " + str(rag_result[query_id]['rag_spend_time']) + "\n\n")
            f.write("answer by rag: \n" + json.dumps(rag_answer, ensure_ascii=False, indent=2) + "\n\n")
            f.write("retrieved materials: \n")
            f.write(json.dumps(rag_result[query_id]['rag_retrieval'], ensure_ascii=False, indent=2) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("llm_spend_time: " + str(llm_result[query_id]['llm_spend_time']) + "\n\n")
            f.write("answer by llm: \n" + json.dumps(llm_answer, ensure_ascii=False, indent=2) + "\n\n")
            f.write("-"*50 + "\n\n")
            f.write("judge: \n" + judge + "\n\n")
            f.write("-"*100 + "\n\n")

    # 写入统计结果
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write("题目总数: " + str(len(rag_scores)) + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("graph平均得分: " + str(round(sum(graph_single_scores)/len(graph_single_scores), 2)) + "\n")
        f.write("graph获胜数量: " + str(len(graph_single_win)) + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("rag平均得分: " + str(round(sum(rag_single_scores)/len(rag_single_scores), 2)) + "\n")
        f.write("rag获胜数量: " + str(len(rag_single_win)) + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("llm平均得分: " + str(round(sum(llm_single_scores)/len(llm_single_scores), 2)) + "\n")
        f.write("llm获胜数量: " + str(len(llm_single_win)) + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
        f.write("graph时间: " + str(round(sum(graph_time)/len(graph_time), 2)) + "秒" + "\n")
        f.write("rag时间: " + str(round(sum(rag_time)/len(rag_time), 2)) + "秒" + "\n")
        f.write("llm时间: " + str(round(sum(llm_time)/len(llm_time), 2)) + "秒" + "\n")
        f.write("------------------------------------------------------------------------------------------------" + "\n")
    
    print(f"✅ GPU {gpu_id} 第 {test_round} 轮申论对比测试完成，结果已保存")

async def run_single_gpu_test(gpu_id: int, test_round: int, config_manager: MultiGPUConfigManager, 
                             example_data: list, graph_path: str, rag_path: str, llm_path: str, base_file_path: str,
                             rag: TCL_RAG, graph_scores: dict, rag_scores: dict, llm_scores: dict, graph_win: dict, rag_win: dict, llm_win: dict):
    """在单个GPU上运行一轮测试"""
    try:
        # 创建文件路径
        file_path = config_manager.create_file_path(gpu_id, test_round, base_file_path)
        
        # 运行测试（使用已初始化的rag）
        await write_result(rag, example_data, graph_path, rag_path, llm_path, file_path, gpu_id, test_round, graph_scores, rag_scores, llm_scores, graph_win, rag_win, llm_win)
        
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 第 {test_round} 轮申论对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_multi_gpu_tests(gpu_ids: list, test_rounds: int, config_path: str, 
                             example_data_path: str, graph_path: str, rag_path: str, llm_path: str, base_result_path: str, query_id_count: int, result_path: str):
    """运行多GPU多轮测试"""
    print(f"🚀 开始申论对比多GPU多轮测试")
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
                    question_content=example['question_content'],
                    materials=example['materials'],
                    answer=example['answer']
                )
            )
    
    print(f"📊 加载了 {len(example_data)} 个申论测试示例")
    
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
    
    # 统计每一道题目不同模型的平均得分和获胜次数
    graph_scores = {query_id: [] for query_id in range(query_id_count)}
    rag_scores = {query_id: [] for query_id in range(query_id_count)}
    llm_scores = {query_id: [] for query_id in range(query_id_count)}
    graph_win = {query_id: 0 for query_id in range(query_id_count)}
    rag_win = {query_id: 0 for query_id in range(query_id_count)}
    llm_win = {query_id: 0 for query_id in range(query_id_count)}

    # 创建任务列表
    tasks = []
    for gpu_id in gpu_ids:
        if gpu_id not in gpu_rag_systems:
            continue
        for round_num in range(1, test_rounds + 1):
            task = run_single_gpu_test(
                gpu_id, round_num, config_manager, example_data[:query_id_count], graph_path, rag_path, llm_path, base_result_path,
                gpu_rag_systems[gpu_id], graph_scores, rag_scores, llm_scores, graph_win, rag_win, llm_win
            )
            tasks.append(task)
    
    # 并发执行所有任务
    print(f"🚀 开始执行 {len(tasks)} 个测试任务...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计结果
    success_count = sum(1 for r in results if r is True)
    failed_count = len(results) - success_count

    # 记录graph/rag/llm的平均得分和获胜次数以及每道题各个模型的得分和获胜次数
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"总轮次: {test_rounds}" + "\n")
        f.write(f"总题目数: {query_id_count}" + "\n")
        f.write(f"总GPU数: {len(gpu_ids)}" + "\n")
        f.write(f"graph平均得分: {round(sum(itertools.chain(*graph_scores.values()))/sum(len(lst) for lst in graph_scores.values()), 2)}   总分: {sum(itertools.chain(*graph_scores.values()))}  次数: {sum(len(lst) for lst in graph_scores.values())}" + "\n")
        f.write(f"rag平均得分: {round(sum(itertools.chain(*rag_scores.values()))/sum(len(lst) for lst in rag_scores.values()), 2)}   总分: {sum(itertools.chain(*rag_scores.values()))}  次数: {sum(len(lst) for lst in rag_scores.values())}" + "\n")
        f.write(f"llm平均得分: {round(sum(itertools.chain(*llm_scores.values()))/sum(len(lst) for lst in llm_scores.values()), 2)}   总分: {sum(itertools.chain(*llm_scores.values()))}  次数: {sum(len(lst) for lst in llm_scores.values())}" + "\n")
        f.write(f"graph获胜次数: {sum(graph_win.values())}" + "\n")
        f.write(f"rag获胜次数: {sum(rag_win.values())}" + "\n")
        f.write(f"llm获胜次数: {sum(llm_win.values())}" + "\n")
        f.write("="*80 + "\n")
        f.write("每道题各个模型的得分和获胜次数:" + "\n")
        for query_id in range(query_id_count):
            f.write(f"题目{query_id}的得分和获胜次数:" + "\n")
            f.write(f"graph得分: {round(sum(graph_scores[query_id])/len(graph_scores[query_id]), 2)}" + "\n")
            f.write(f"graph获胜次数: {graph_win[query_id]}" + "\n")
            f.write(f"rag得分: {round(sum(rag_scores[query_id])/len(rag_scores[query_id]), 2)}" + "\n")
            f.write(f"rag获胜次数: {rag_win[query_id]}" + "\n")
            f.write(f"llm得分: {round(sum(llm_scores[query_id])/len(llm_scores[query_id]), 2)}" + "\n")
            f.write(f"llm获胜次数: {llm_win[query_id]}" + "\n")
            f.write("-"*40 + "\n")
        f.write("="*80 + "\n")


    print("\n" + "="*80)
    print(f"📊 申论对比测试完成统计:")
    print(f"总任务数: {len(tasks)}")
    print(f"成功任务: {success_count}")
    print(f"失败任务: {failed_count}")
    print(f"任务成功率: {success_count/len(tasks)*100:.1f}%")
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
    test_rounds = 1  # 每个GPU的测试轮次
    query_id_count = 17 # 要测试前多少道题目
    config_path = "/finance_ML/liuyingqi/RAG-Factory/examples/TCL_rag/config_commentary_llm.yaml"
    example_data_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/example/申论真题2020-2024.json"
    graph_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/commentary_test/graph_answer/answer_by_graph_commentary_multi_gpu.json"
    rag_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/commentary_test/rag_answer/answer_by_rag_commentary_multi_gpu.json"
    llm_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/commentary_test/llm_answer/answer_by_llm_commentary_multi_gpu.json"
    base_result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/commentary_test/judge_result/answer_by_judge_commentary_multi_gpu.txt"
    result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/commentary_test/judge_result/4.1_result.txt"
    
    # 运行多GPU多轮测试
    await run_multi_gpu_tests(gpu_ids, test_rounds, config_path, example_data_path, graph_path, rag_path, llm_path, base_result_path, query_id_count, result_path)

if __name__ == "__main__":
    asyncio.run(main())


