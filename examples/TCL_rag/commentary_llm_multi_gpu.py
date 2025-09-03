from rag_flow_example_commentary import TCL_RAG
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
    question_content: str
    materials: str
    answer: str

@dataclass
class Example_result:
    query_id: str
    llm_spend_time: float
    llm_answer: str
    
    def to_dict(self):
        return {
            "query_id": self.query_id,
            "llm_spend_time": self.llm_spend_time,
            "llm_answer": self.llm_answer
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

async def write_result(rag, example_data, file_path, gpu_id, test_round):
    """写入结果到指定文件"""
    example_result = []
    
    print(f"🚀 GPU {gpu_id} 第 {test_round} 轮申论LLM测试开始，结果保存到: {file_path}")
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU {gpu_id} 第 {test_round} 轮申论LLM测试结果" + "\n")
        f.write("="*80 + "\n")
    
    for idx, example in enumerate(example_data):
        query_id = idx
        query = example.question_content
        answer = example.answer
        materials = example.materials
        
        # LLM直接回答
        llm_start_time = time.time()
        llm_answer = await asyncio.to_thread(rag.llm_answer, query, materials)
        llm_end_time = time.time()
        llm_spend_time = llm_end_time - llm_start_time
        
        # 统计正确性（这里需要根据实际情况调整判断逻辑）
        # 由于申论题目没有标准答案，这里暂时跳过正确性统计  
        
        # 保存结果
        example_result.append(
            Example_result(
                query_id=query_id,
                llm_spend_time=llm_spend_time,
                llm_answer=llm_answer
            )
        )
    
    # 写入结果文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps([example.to_dict() for example in example_result], ensure_ascii=False, indent=2) + "\n")
    
    print(f"✅ GPU {gpu_id} 第 {test_round} 轮申论LLM测试完成，结果已保存")

async def run_single_gpu_test(gpu_id: int, test_round: int, config_manager: MultiGPUConfigManager, 
                             example_data: list, base_file_path: str,
                             rag: TCL_RAG):
    """在单个GPU上运行一轮测试"""
    try:
        # 创建文件路径
        file_path = config_manager.create_file_path(gpu_id, test_round, base_file_path)
        
        # 运行测试（使用已初始化的rag）
        await write_result(rag, example_data, file_path, gpu_id, test_round)
        
        return True
        
    except Exception as e:
        print(f"❌ GPU {gpu_id} 第 {test_round} 轮申论LLM测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_multi_gpu_tests(gpu_ids: list, test_rounds: int, config_path: str, 
                             example_data_path: str, base_result_path: str, query_id_count: int):
    """运行多GPU多轮测试"""
    print(f"🚀 开始申论LLM多GPU多轮测试")
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
    
    # 创建任务列表
    tasks = []
    for gpu_id in gpu_ids:
        if gpu_id not in gpu_rag_systems:
            continue
        for round_num in range(1, test_rounds + 1):
            task = run_single_gpu_test(
                gpu_id, round_num, config_manager, example_data[:query_id_count], base_result_path, 
                gpu_rag_systems[gpu_id]
            )
            tasks.append(task)
    
    # 并发执行所有任务
    print(f"🚀 开始执行 {len(tasks)} 个测试任务...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计结果
    success_count = sum(1 for r in results if r is True)
    failed_count = len(results) - success_count

    print("\n" + "="*80)
    print(f"📊 申论LLM测试完成统计:")
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
    base_result_path = "/data/FinAi_Mapping_Knowledge/liuyingqi/commentary_test/llm_answer/answer_by_llm_commentary_multi_gpu.json"
    
    # 运行多GPU多轮测试
    await run_multi_gpu_tests(gpu_ids, test_rounds, config_path, example_data_path, base_result_path, query_id_count)

if __name__ == "__main__":
    asyncio.run(main())
