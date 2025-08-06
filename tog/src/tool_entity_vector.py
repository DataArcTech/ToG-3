"""
构建实体的embedding
"""

import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import json
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
from langchain_core.embeddings import Embeddings

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B"):
        self.model = SentenceTransformer(model_name_or_path, device="cuda:1")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True, prompt_name="Document", batch_size=1).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)

def read_graph_data(graph_data_path):
    '''
    读取原始图数据
    '''
    with open(graph_data_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]

def create_entity_info(graphs):
    '''
    对每个实体创建相应的embedding，以及实体对应的id，实体名字
    '''

    entity_names = [graph["name"] for graph in graphs]
    entity_ids = [graph["id"] for graph in graphs]

    embeddings_list = []
    batch_size = 1000
    

    model = SentenceTransformerEmbeddings("/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B")


    for i in range(0, len(entity_names), batch_size):
        sentences = entity_names[i:i+batch_size]
        embeddings_batch = model.embed_documents(sentences)
        embeddings_list.extend(embeddings_batch)

    return embeddings_list, entity_names, entity_ids


def create_edge_info(entity_ids, entity_names, edge_data_path):  
    result = {name: {'entity_id': str(entity_ids[i]), 'edges': [], 'edges_ids': [], 'weight': [], 'normlize_weight': []}
              for i, name in enumerate(entity_names)}
    name_to_id = dict(zip(entity_names, entity_ids))

    with open(edge_data_path, 'r') as file:  
        for line in file:  
            edge = json.loads(line)  
            node1, node2 = edge['node1'], edge['node2']
  
            def update_node_info(node, target_node, edge_info):  
                if target_node not in result[node]['edges']:  
                    result[node]['edges'].append(target_node)
                    result[node]['edges_ids'].append(str(name_to_id[target_node]))
                    result[node]['weight'].append(str(edge_info['weight']))  
                    result[node]['normlize_weight'].append(str(edge_info['normlize_weight']))
  
            if node1 in result: update_node_info(node1, node2, edge)
            if node2 in result: update_node_info(node2, node1, edge)
  
    for info in result.values():  
        for key in ['edges', 'weight', 'normlize_weight', 'edges_ids']:
            info[key] = '<delimiter>'.join(info[key])
  
    return result


 
## NOTE conan
def save_to_vectordb(embeddings, entity_names, entity_ids, edges_info, graphs, vectordb_path):
    client = chromadb.PersistentClient(path=vectordb_path)
    collection = client.create_collection(
        name="all_entity", 
        metadata={
            "hnsw:space": "cosine",
            "hnsw:batch_size": 100000,
            "hnsw:sync_threshold": 100000
            })

    metadatas = []
    for name, graph, edge_name in zip(entity_names, graphs, edges_info):
        positions_str = "<delimiter>".join([f"{pos['art_id']}-{pos['par_id']}-{pos['sen_id']}" for pos in graph["positions"]])
        metadata = {
            "name": name,
            "positions": positions_str,
            'entity_id': edges_info[edge_name]['entity_id'],
            'edges': edges_info[edge_name]['edges'],
            'weight': edges_info[edge_name]['weight'],
            'normlize_weight': edges_info[edge_name]['normlize_weight'],
            'edges_id': edges_info[edge_name]['edges_ids']
        }
        metadatas.append(metadata)
    
    entity_ids_str = [str(id_) for id_ in entity_ids]
    batch_size = 500
    for i in range(0, len(embeddings), batch_size):
        # Convert embeddings to a NumPy array if it's not one already
        embeddings_batch = np.array(embeddings[i:i+batch_size]).tolist()  # This step ensures it's in the correct format
        collection.add(
            embeddings=embeddings_batch,
            documents=entity_names[i:i+batch_size],
            ids=entity_ids_str[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

        
def construct_entity_vector(graph_data_path, edge_data_path, vectordb_path):
    graphs = read_graph_data(graph_data_path)
    embeddings, entity_names, entity_ids = create_entity_info(graphs)
    edges_info = create_edge_info(entity_ids, entity_names, edge_data_path)
    save_to_vectordb(embeddings, entity_names, entity_ids, edges_info, graphs, vectordb_path)
    return embeddings, entity_names

if __name__ == "__main__":

    graph_data_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA.jsonl"
    edge_data_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_edge_gpt4.1_mini_new_type_with_QA.jsonl"

    vectordb_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/entity_vector"

    embeddings, entity_names = construct_entity_vector(graph_data_path, edge_data_path, vectordb_path)
    print(f"Vectorized {len(entity_names)} entities and saved to ChromaDB.")
