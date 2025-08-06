import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from typing import List


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B"):
        self.model = SentenceTransformer(model_name_or_path, device="cuda:1")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True, prompt_name="document", batch_size=1).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)



entity_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/entity_vector"
entity_client = chromadb.PersistentClient(path=entity_path)
entity_collection = entity_client.get_collection(name="all_entity")



def retrieve_from_graph(collection, query_embeddings, n_results):

    # 尝试检索结果
    query_embeddings = query_embeddings
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        include=["embeddings","documents", "distances","metadatas"]
    )

    return results

if __name__ == "__main__":
    entity = "党的十九届五中全会"
    embedding_model = SentenceTransformerEmbeddings()
    query_embedding = embedding_model.embed_query(entity)
    results = retrieve_from_graph(entity_collection, query_embedding, 150)
    entities = results['documents'][0]
    distances = results['distances'][0]
    print(f"Entities similar to '{entity}':")
    for i, entity in enumerate(entities):
        print(f"{i + 1}. {entity}")
        print("Distances:")
        print(f"{i + 1}. {distances[i]}")
        print("---" * 50)
