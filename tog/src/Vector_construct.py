import argparse
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import json
import os
import chromadb
import re
import time
import multiprocessing as mp
import stat
import subprocess
import logging
from pypinyin import lazy_pinyin as pinyin
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


# 设置日志记录
logging.basicConfig(
    filename='vector_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

def read_chunked_data(file_path):
    """
    读取新格式的分块数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_embeddings_from_chunks(batch_file_paths, batch_size, model_path):
    """
    从分块数据计算嵌入向量
    """
    model = SentenceTransformerEmbeddings(model_path)
    
    chunk_contents = []  # 保存chunk内容
    ids = []
    metadata = []

    # 遍历每个文件
    for file_path in tqdm(batch_file_paths, desc="Processing files"):
        chunks_data = read_chunked_data(file_path)
        
        for chunk_item in chunks_data:
            chunk_content = chunk_item.get("chunk_content", "")
            position = chunk_item.get("position", {})
            
            if not chunk_content.strip():
                continue  # 跳过空内容
            
            # 提取位置信息
            art_id = position.get("art_id", 0)
            par_id = position.get("par_id", 0)
            sen_id = position.get("sen_id", 0)
            
            # 构建唯一ID
            unique_id = f"{art_id}-{par_id}-{sen_id}"
            
            chunk_contents.append(chunk_content)
            ids.append(unique_id)
            
            # 构建元数据
            meta = {
                "position": unique_id,
                "position_p": f"{art_id}-{par_id}",
                "article_id": art_id,
                "paragraph_id": par_id,
                "sentence_id": sen_id,
                "book_name": chunk_item.get("book_name", ""),
                "title_lv1": chunk_item.get("title_lv1", ""),
                "title_lv2": chunk_item.get("title_lv2", ""),
                "title_lv3": chunk_item.get("title_lv3", ""),
                "title_lv4": chunk_item.get("title_lv4", ""),
                "title_n": chunk_item.get("title_n", ""),
                "table": chunk_item.get("table", ""),
                "file_path": file_path  # 添加文件路径信息
            }
            metadata.append(meta)
    
    # 批量处理嵌入
    batch_embeddings = []
    for i in tqdm(range(0, len(chunk_contents), batch_size), desc="Computing embeddings"):
        batch_contents = chunk_contents[i:i+batch_size]
        batch_ids_subset = ids[i:i+batch_size]
        batch_metadata_subset = metadata[i:i+batch_size]
        
        # 计算嵌入
        embeddings = model.embed_documents(batch_contents)
        
        # 记录日志
        logging.info(f"Processed batch {i//batch_size + 1}, size: {len(batch_contents)}")
        logging.info(f"Sample metadata: {batch_metadata_subset[0] if batch_metadata_subset else 'None'}")
        
        batch_embeddings.append((embeddings, batch_contents, batch_ids_subset, batch_metadata_subset))
    
    return batch_embeddings

def construct_vectordb_from_chunks(file_paths, num_workers=4, batch_size=32, model_path=None, vectordb_info=None):
    """
    从分块数据构建向量数据库
    """
    start_time = time.time()
    
    # 将文件路径分割给不同的worker
    chunk_size = len(file_paths) // num_workers
    if chunk_size == 0:
        chunk_size = 1
    
    file_chunks = []
    for i in range(0, len(file_paths), chunk_size):
        file_chunks.append(file_paths[i:i + chunk_size])
    
    logging.info(f"File chunks: {file_chunks}")
    
    # 多进程处理
    pool = mp.Pool(processes=num_workers)
    results = []
    
    for i, chunk in enumerate(file_chunks):
        result = pool.apply_async(compute_embeddings_from_chunks, args=(chunk, batch_size, model_path))
        results.append(result)
        print(f"第{i}个进程开始")
    
    pool.close()
    pool.join()
    
    print(f"所有进程结束，耗时{time.time()-start_time}")
    
    # 在主进程中初始化 ChromaDB
    client = chromadb.PersistentClient(path=vectordb_info["db_path"])
    collection = client.get_or_create_collection(
        name=vectordb_info["name"],
        metadata={
            "hnsw:space": "cosine",
            "hnsw:batch_size": 100000,
            "hnsw:sync_threshold": 100000
        } 
    )
    
    # 从所有worker收集嵌入并插入到数据库中
    print("开始插入数据")
    total_inserted = 0
    for result in results:
        batch_embeddings = result.get()
        for embeddings, chunk_contents, batch_ids_subset, batch_metadata_subset in batch_embeddings:
            logging.info(f"当前add id: {batch_ids_subset[:5]}...")  # 只记录前5个ID
            
            collection.add(
                embeddings=embeddings, 
                documents=chunk_contents,
                ids=batch_ids_subset, 
                metadatas=batch_metadata_subset
            )
            total_inserted += len(batch_ids_subset)
    
    print(f"数据插入完成，共插入{total_inserted}条记录，总耗时{time.time()-start_time}")
    return collection

def collect_chunk_files(input_path, limit=None):
    """
    收集分块数据文件
    """
    file_paths = []
    
    if os.path.isfile(input_path) and input_path.endswith('.json'):
        # 如果是单个文件
        file_paths = [input_path]
    elif os.path.isdir(input_path):
        # 如果是文件夹，递归查找所有json文件
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.json'):
                    file_paths.append(os.path.join(root, file))
    else:
        raise ValueError("input_path应该是json文件或包含json文件的文件夹")
    
    if limit:
        file_paths = file_paths[:limit]
    
    return file_paths

# def get_best_gpus(num_gpus=1):
#     result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], 
#                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#     gpu_memory = []
#     for line in result.stdout.strip().split('\n'):
#         index, free_mem = line.split(',')
#         gpu_memory.append((int(index), int(free_mem)))  
#     best_gpus = [gpu[0] for gpu in sorted(gpu_memory, key=lambda x: x[1], reverse=True)[:num_gpus]]
#     return best_gpus

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, get_best_gpus()))
    # print(f"Setting CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--limit", type=int, required=False, help="Limit on the number of files to process")
    parser.add_argument("--model_path", type=str, default="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", help="Path to the embedding model")
    parser.add_argument("--input_folder", type=str, default="/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/parse_datasets/tree_chunked_data", help="Input folder containing chunked data")
    parser.add_argument("--output_folder", type=str, default="/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB", help="Output folder for vector database")

    args = parser.parse_args()
    
    # 创建输出文件夹
    if "vector_data" not in args.output_folder:
        output_folder = os.path.join(args.output_folder, "vector_data")
    else:
        output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    
    # 收集输入文件
    input_files = collect_chunk_files(args.input_folder, limit=args.limit)
    print(f"找到{len(input_files)}个文件进行处理")
    print(f"示例文件: {input_files[:3]}")

    # ChromaDB配置
    vectordb_info = {
        "db_path": output_folder,
        "name": "edu_chunks"
    }
    
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 构建向量数据库
    collection = construct_vectordb_from_chunks(
        input_files, 
        num_workers=args.num_workers, 
        batch_size=args.batch_size, 
        model_path=args.model_path, 
        vectordb_info=vectordb_info
    )
    
    print("向量数据库构建完成！")
    print(f"数据库路径: {output_folder}")
    print(f"集合名称: {vectordb_info['name']}")
    
    # 验证数据库
    print(f"数据库中的文档数量: {collection.count()}")