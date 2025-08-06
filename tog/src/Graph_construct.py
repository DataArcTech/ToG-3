# 用于构件图的各种函数
# TODO 所有的graph[node.name] 要检查一下 还有 return self 的问题

import argparse
import torch
from tqdm import tqdm
import json
import os
import torch.multiprocessing as mp
import time
import logging
from gptqa import label_entity
from Graph_define import Node, Position, Entity


import re

def detect_language(text_list):
    chinese_chars = re.compile(r'[\u4e00-\u9fff]')
    english_chars = re.compile(r'[a-zA-Z]')

    chinese_count = 0
    english_count = 0

    for text in text_list:
        chinese_count += len(chinese_chars.findall(text[0]))
        english_count += len(english_chars.findall(text[0]))

    total = chinese_count + english_count
    if total == 0:
        return "Chinese"  # 默认回退

    english_ratio = english_count / total
    return "English" if english_ratio > 2/3 else "Chinese"



class Graph:
    def __init__(self, input_data):
        """
        从文件列表中读取数据
        
        args:
            input_data: jsonl文件路径
        """
        self.graph = dict()
        self.edges = dict()
        
        self.input_data = input_data
    
    def assign_ids(self):
        """为每个节点分配唯一的 ID"""
        current_id = 0
        for node_name, node in self.graph.items():
            node.id = current_id
            current_id += 1

    def construct_graph_batch_parallel(self, get_entity: callable, get_entity_method="gpt", model=None, tokenizer=None, num_models=8, batch_size=30, article_child=False, gpu_id=None, 
                                        result_path=None, article_id=0):
        """
        并行处理输入数据中的段落或句子，构建图，支持批处理。
        """
        if gpu_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cuda")
            torch.cuda.set_device(device)
            model.model.to(device)
        
        # 检查文件夹是否存在，如不存在创建
        if result_path:
            os.makedirs(result_path, exist_ok=True)

        all_batches = []
        total_sentences = 0  # 统计总的句子数量
        data, positions = self._load_data(self.input_data) 
        batch = []
        for chunk_content, pos in zip(data, positions):
            chunk_content: str
            # for sentence_id, sentence in enumerate(paragraph):
            batch.append((chunk_content, (pos["art_id"], pos["par_id"], pos["sen_id"])))
            total_sentences += 1

            # 将句子批次化
            if len(batch) == batch_size:
                all_batches.append(batch)
                batch = []

        if batch: # 处理剩下的句子
            all_batches.append(batch)

            # article_id += 1


        # 按批次分割成 num_models 份
        split_batches = [all_batches[i::num_models] for i in range(num_models)]
        
        mp.set_start_method('spawn', force=True)
        # print(f"all_batches[0]: {all_batches[0]}")
        # print(f"all_batches[1]: {all_batches[1]}")
        # 使用 tqdm 显示进度条
        with tqdm(total=total_sentences, desc="句子处理进度", unit="句子") as pbar:
            with mp.Pool(num_models) as pool:
                results = []
                for i in range(num_models):
                    result = pool.apply_async(
                        self._process_batches,
                        (split_batches[i], get_entity, get_entity_method, model, tokenizer, article_child),
                        callback=lambda res: pbar.update(sum(len(batch) for batch in split_batches[i]))  # 每处理完一个batch更新句子数量的进度
                    )
                    results.append(result)

                combined_graph = {}
                for result in results:
                    partial_graph = result.get()
                    for name, node in partial_graph.items():
                        if name in combined_graph:
                            existing_node = combined_graph[name]
                            existing_node.type.update(node.type)
                            existing_node.description.append(node.description)
                            existing_node.positions.update(node.positions)
                            existing_node.sen_childs.update(node.sen_childs - existing_node.sen_childs)
                            existing_node.par_childs.update(node.par_childs - existing_node.par_childs)
                            existing_node.art_childs.update(node.art_childs - existing_node.art_childs)
                        else:
                            combined_graph[name] = node

        self.graph = combined_graph
        self.assign_ids()

        return combined_graph


    def _load_data(self, file_path):
        """
        从文件列表中读取数据
        """
        article_data = []
        postions = []
        for file in file_path:
            with open(file, "r") as f:
                chunks_data = json.load(f)
                for chunk in chunks_data:
                    article_data.append(chunk["chunk_content"])
                    postions.append(chunk["position"])

        print(f"len(data): {len(article_data)}")

        return article_data, postions

    
    
    def _process_batches(self, batch_list, get_entity, get_entity_method, model, tokenizer, article_child):
        """
        处理给定的句子批次列表，进行NER并构建图，支持批处理。
        """
        graph = dict()
        for batch in batch_list:
            language = detect_language(batch)
            self._process_batch(batch, get_entity, get_entity_method, model, tokenizer, graph, article_child, language=language)

        return graph


    def _process_batch(self, batch, get_entity, get_entity_method, model, tokenizer, graph, article_child, language):
        """
        批处理一组句子并更新图，支持不同choices下的NER结果合并。
        
        args:
            batch: 一个批次的句子，每个元素是 (sentence, (article_id, paragraph_id, sentence_id)) 的形式。
            get_entity: callable，NER函数，用于从句子中提取实体。
            model: 对应get_entity的模型。
            tokenizer: 对应的tokenizer。
            graph: 当前图的字典。
            article_child: 是否记录同一篇文章里的实体关系。
        """
        print(f"开始get_entity，batch size: {len(batch)}")
        if get_entity_method == "gpt":
            result_list = get_entity(batch, language=language)
            result_list: dict
        else:
            result_list = get_entity(model, tokenizer, batch, language=language)
        print(f"get_entity完成，result_list: {result_list}")
        # 先保存一下result_dict
        time_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        try:
            with open(f"/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/temp/temp_dict/{time_str}.json", "w", encoding="utf-8") as f:
                json.dump(result_list, f, ensure_ascii=False, indent=4)
        except:
            pass

        for article_id, article_dict in result_list.items():
            article_entitys = set()

            for paragraph_id, paragraph_dict in article_dict.items():
                paragraph_entitys = set()

                for sentence_id, sentence_entitys in paragraph_dict.items():
                    for entity in sentence_entitys:
                        if entity.name not in graph:
                            graph[entity.name] = Node.form_entity(entity)
                        node = graph[entity.name]
                        node.refresh_sen_child(sentence_entitys)
                        node.refresh_position(Position.form_id(article_id, paragraph_id, sentence_id))
                        node.refresh_description(entity.description)
                    paragraph_entitys.update(sentence_entitys)

                for entity in paragraph_entitys:
                    graph[entity.name].refresh_par_child(paragraph_entitys)

                if article_child:
                    article_entitys.update(paragraph_entitys)

            if article_child:
                for entity in article_entitys:
                    graph[entity.name].refresh_art_child(article_entitys)


        # 先保存一次
        try:
            self.save_graph("/data/FinAi_Mapping_Knowledge/SynKG/SynFin/graph_data/temp/temp_graph.jsonl")
            self.save_edge("/data/FinAi_Mapping_Knowledge/SynKG/SynFin/graph_data/temp/temp_edge.jsonl")
        except:
            pass


    def save_graph(self, data_dir="/data/FinAi_Mapping_Knowledge/zhangliyu/tog3/data/graph_data/graph.jsonl"):
        '''
        把图序列化保存一下
        '''
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        file = open(data_dir, "w", encoding="utf-8")
        for node_name in self.graph:
            node:Node = self.graph[node_name]
            node = node.to_dict()
            file.write(json.dumps(node, ensure_ascii=False)+"\n")
        file.close()

    def save_edge(self, data_dir="/data/FinAi_Mapping_Knowledge/zhangliyu/tog3/data/graph_data/edge.jsonl", weights=[5,3,1]):
        '''
        保存成一行是一个边，计算权重:
        如果两个节点在同一句子中，权重增加 weights[0]。
        如果在同一段落中但不同句子中，权重增加 weights[1]。
        如果在同一篇文章中但不同段落中，权重增加 weights[2]。
        '''

        def get_edge_weight_sen(positions_1, positions_2, weights):
            '''
            计算两个节点直接的权重，除以positions_1出现的次数，做归一化
            args:
                weights:[sen_weight, par_weight, art_weight]
            '''
            weight = 0
            for position_1 in positions_1:
                position_1:Position
                for position_2 in positions_2:
                    position_2:Position
                    if position_1 == position_2:
                        weight += weights[0]
                    elif position_1.article_id == position_2.article_id and position_1.paragraph_id == position_2.paragraph_id:
                        weight += weights[1]
                    elif position_1.article_id == position_2.article_id:
                        weight += weights[2]
            return weight
        
        def get_edge_weight_par(positions_1, positions_2, weights):
            '''
            计算两个节点直接的权重，除以positions_1出现的次数，做归一化
            args:
                weights:[sen_weight, par_weight, art_weight]
            '''
            weight = 0
            for position_1 in positions_1:
                position_1:Position
                for position_2 in positions_2:
                    position_2:Position
                    if position_1.article_id == position_2.article_id and position_1.paragraph_id == position_2.paragraph_id:
                        weight += weights[1]
                    elif position_1.article_id == position_2.article_id:
                        weight += weights[2]
            return weight
        
        def get_edge_weight_art(positions_1, positions_2, weights):
            '''
            计算两个节点直接的权重，除以positions_1出现的次数，做归一化
            args:
                weights:[sen_weight, par_weight, art_weight]
            '''
            weight = 0
            for position_1 in positions_1:
                position_1:Position
                for position_2 in positions_2:
                    position_2:Position
                    if position_1.article_id == position_2.article_id:
                        weight += weights[2]
            return weight
        
        # edges = list()
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        file = open(data_dir, "w", encoding="utf-8")
        for node_1_name in tqdm(self.graph):
            node_1 = self.graph[node_1_name]
            node_1:Node
            node_1_type = "_".join(node_1.type).strip()

            for node_2_name in node_1.sen_childs:
                if node_1_name < node_2_name:
                    # 出现过了不再计算 可以通过有序性来做
                    continue
                node_2 = self.graph[node_2_name]
                node_2:Node
                node_2_type = "_".join(node_2.type).strip()
                weight = get_edge_weight_sen(node_1.positions, node_2.positions, weights)
                edge_1 = {"node1":node_1_name, "node1_type":node_1_type, "node2":node_2_name, "node2_type":node_2_type, "weight":weight, "normlize_weight":weight / len(node_1.positions), "nums":len(node_1.positions)}
                edge_2 = {"node1":node_2_name, "node1_type":node_2_type, "node2":node_1_name, "node2_type":node_1_type, "weight":weight, "normlize_weight":weight / len(node_2.positions), "nums":len(node_2.positions)}
                file.write(json.dumps(edge_1, ensure_ascii=False)+"\n")
                file.write(json.dumps(edge_2, ensure_ascii=False)+"\n")

            for node_2_name in node_1.par_childs:
                if node_1_name < node_2_name:
                    # 出现过了不再计算 可以通过有序性来做
                    continue
                node_2 = self.graph[node_2_name]
                node_2:Node
                node_2_type = "_".join(node_2.type).strip()
                weight = get_edge_weight_par(node_1.positions, node_2.positions, weights)
                edge_1 = {"node1":node_1_name, "node1_type":node_1_type, "node2":node_2_name, "node2_type":node_2_type, "weight":weight, "normlize_weight":weight / len(node_1.positions), "nums":len(node_1.positions)}
                edge_2 = {"node1":node_2_name, "node1_type":node_2_type, "node2":node_1_name, "node2_type":node_1_type, "weight":weight, "normlize_weight":weight / len(node_2.positions), "nums":len(node_2.positions)}
                file.write(json.dumps(edge_1, ensure_ascii=False)+"\n")
                file.write(json.dumps(edge_2, ensure_ascii=False)+"\n")
            
        file.close()





def get_entity_GPT_batch(batch, gpu_id=None, openai_model_name="gpt-4.1-mini", language="Chinese"):
    """
    使用 GPTQA 从输入的句子批量获取实体。
    Args:
        model: 未使用，仅保留一致性
        tokenizer: 未使用，仅保留一致性
        batch: 一个包含句子和位置信息的列表，格式为 (sentence, (article_id, paragraph_id, sentence_id))。
        gpu_id: 未使用，仅保留一致性
        openai_model_name: GPT 模型名称
    Returns:
        返回一个字典，包含提取的实体，结构为 {article_id: {paragraph_id: {sentence_id: [Entity...]}}}
    """
    sentences = [sentence for sentence, _ in batch]
    
    try:
        # 引入tqdm进度条
        with tqdm(total=len(sentences), desc="GPT标注实体进度", unit="句子") as pbar:
            labeled_entities = label_entity(sentences, language=language, openai_model_name=openai_model_name)
            print(f"label_entity的output: {labeled_entities}")
            
            return_dict = {}
            for labeled_data, (sentence, (article_id, paragraph_id, sentence_id)) in zip(labeled_entities, batch):
                entities = [Entity(name=entity["entity_text"], type=entity["entity_type"], description=entity["entity_description"]) for entity in labeled_data["entity_list"]]
                
                if article_id not in return_dict:
                    return_dict[article_id] = {}
                if paragraph_id not in return_dict[article_id]:
                    return_dict[article_id][paragraph_id] = {}
                
                return_dict[article_id][paragraph_id][sentence_id] = entities

                # 每处理一个句子更新一次进度条
                pbar.update(1)

        return return_dict
    
    except Exception as e:
        # 捕获异常并记录到日志
        logging.error(f"Error processing batch: {batch}, Error: {str(e)}")
        return {}



def merge_entity_results(result_list, new_result_list, rename_type):
    """
    Merge new_result_list into result_list while keeping the previous entity_type
    if an entity has already been extracted. Also remove duplicates in the 'entitys' list.
    If an entity_type belongs to choices[1:], rename it to choices[1][0].
    """

    # 如果rename_type的长度为1，直接返回，不进行实体类型替换
    if len(rename_type) == 1:
        return result_list

    # 如果rename_type包含多个嵌套数组，执行替换逻辑
    rename_choices = set([etype for sublist in rename_type[1:] for etype in sublist])
    rename_to = rename_type[1][0]  # Rename entity_type to this value if it matches choices[1:] 

    for article_id, article_data in new_result_list.items():
        if article_id not in result_list:
            result_list[article_id] = article_data
            continue

        for paragraph_id, paragraph_data in article_data.items():
            if paragraph_id not in result_list[article_id]:
                result_list[article_id][paragraph_id] = paragraph_data
                continue

            for sentence_id, sentence_entities in paragraph_data.items():
                existing_entities = {entity.name for entity in result_list[article_id][paragraph_id][sentence_id]}

                for new_entity in sentence_entities:
                    # 如果实体类型属于choices[1:]，进行类型替换
                    if new_entity.type in rename_choices:
                        new_entity.type = rename_to  # Rename entity type

                    if new_entity.name not in existing_entities:
                        result_list[article_id][paragraph_id][sentence_id].append(new_entity)

    return result_list



if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Graph construction script")
    parser.add_argument("--gpu_id", type=int, default=None, help="GPU ID to use")
    parser.add_argument("--num_models", type=int, default=15, help="Number of concurrent models to use")

    args = parser.parse_args()

    gpu_id = args.gpu_id
    num_models = args.num_models


    input_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/new_test_data"

    input_data = []
    for filename in os.listdir(input_file):
        if filename.endswith(".json"):
            input_data.append(os.path.join(input_file, filename))

    print(input_data)

    # input_data = ["/data/FinAi_Mapping_Knowledge/chenmingzhen/tog3_backend/medical_tog3/chunks/2e9449fa-1d4d-3752-b872-d7dcf9626444.jsonl"]
    # input_data = ["/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/parse_datasets/tree_chunked_data/2023版马克思主义基本原理(目录版).json",""]

    result_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_data_test"


    graph = Graph(input_data)


    new_graph = graph.construct_graph_batch_parallel(model=None, tokenizer=None, 
                                        get_entity=get_entity_GPT_batch, get_entity_method="gpt",
                                        article_child=True,
                                        num_models=60, gpu_id=gpu_id,
                                        result_path=result_path,
                                        article_id=0
                                        )


    print("构建图用时", time.time()-start_time)
    graph.save_graph(result_path+"/graph_with_new_type_11.jsonl")
    print("保存图用时", time.time()-start_time)
    graph.save_edge(result_path+"/edge_with_new_type_11.jsonl")
    print("总用时", time.time()-start_time)

