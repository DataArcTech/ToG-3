import sys
import os

# 添加 RAG-Factory 目录到 Python 路径
rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)

from rag_factory.Store import VectorStoreRegistry
from rag_factory.Embed import EmbeddingRegistry
import yaml
from rag_factory.Retrieval import Document
import json


with open("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/faiss_construct/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

store_config = config["store"]
embedding_config = config["embedding"]
dataset_config = config["dataset"]["data_path"]
embedding = EmbeddingRegistry.create(**embedding_config)
store = VectorStoreRegistry.create(**store_config, embedding=embedding)


if __name__ == "__main__":

    # 读取数据
    with open(dataset_config, "r", encoding="utf-8") as f:
        docs = []
        data = json.load(f)
        for item in data:
            full_content = item.get("full_content", "")
            metadata = {
                "title": item.get("original_filename"),
            }

            docs.append(Document(content=full_content, metadata=metadata))

    # 创建向量库
    vectorstore = store.from_documents(docs, embedding=embedding)

    # 保存到本地
    vectorstore.save_local(store_config["folder_path"])