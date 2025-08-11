import sys
import os

rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)

import json
from rag_factory.Retrieval import Document
from rag_factory.Retrieval import RetrieverRegistry

import yaml


def load_data(jsonl_path: str):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        docs = []
        for item in data:
            content = item.get("full_content", "")
            metadata = {"title": item.get("original_title", "")}
            docs.append(Document(content=content, metadata=metadata))
        return docs

def chinese_preprocessing_func(text: str) -> str:
    import jieba
    return " ".join(jieba.cut(text))

if __name__ == "__main__":
    docs = load_data("/data/FinAi_Mapping_Knowledge/chenmingzhen/tog3_backend/TCL/syn_table_data/data_all_clearn_short_chunk_with_caption_desc.json")
    with open("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/bm25/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    bm25_retriever = RetrieverRegistry.create(**config["retriever"])
    bm25_retriever = bm25_retriever.from_documents(documents=docs, preprocess_func=chinese_preprocessing_func, k=config["retriever"]["k"])

    print(bm25_retriever.invoke("什么是TCL？"))