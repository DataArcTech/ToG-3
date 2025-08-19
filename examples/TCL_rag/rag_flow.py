from typing import List
import json

from rag_factory.registry import EmbeddingRegistry, LLMRegistry, VectorStoreRegistry, RetrieverRegistry, RerankerRegistry
from rag_factory.data_model import Document



class TCL_RAG:
    def __init__(
        self,
        *,
        llm_config=None,
        embedding_config=None,
        vector_store_config=None,
        bm25_retriever_config=None,
        retriever_config=None,
        reranker_config=None,
    ):
        self.llm = LLMRegistry.create(**llm_config)
        self.embedding = EmbeddingRegistry.create(**embedding_config)
        self.vector_store = VectorStoreRegistry.create(**vector_store_config, embedding=self.embedding)

        documents = self._load_data(bm25_retriever_config["data_path"])
        self.bm25_retriever = RetrieverRegistry.create(
            "bm25",
            documents=documents,
            preprocess_func=self.chinese_preprocessing_func,
            k=bm25_retriever_config["k"]
        )
        self.retriever = RetrieverRegistry.create(**retriever_config, vectorstore=self.vector_store)
        self.multi_path_retriever = RetrieverRegistry.create("multipath", retrievers=[self.bm25_retriever, self.retriever])
        self.reranker = RerankerRegistry.create(**reranker_config)

    def invoke(self, query: str, k: int = None):
        return self.multi_path_retriever.invoke(query, top_k=k)

    def rerank(self, query: str, documents: List[Document], k: int = None, batch_size: int = 8):
        return self.reranker.rerank(query, documents, k, batch_size)

    def _load_data(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            docs = []
            for item in data:
                content = item.get("full_content", "")
                metadata = {"title": item.get("original_filename", "")}
                docs.append(Document(content=content, metadata=metadata))
        return docs

    def chinese_preprocessing_func(self, text: str) -> str:
        import jieba
        return " ".join(jieba.cut(text))


    def answer(self, query: str, documents: List[Document]):

        template = (
            "你是一位工业领域的专家。根据以下检索到的材料回答用户问题。"
            "如果回答所需信息未在材料中出现，请说明无法找到相关信息。\n\n"
            "{context}\n\n"
            "用户问题：{question}\n"
            "答复："
        )
        context = "\n".join([doc.content for doc in documents])
        prompt = template.format(question=query, context=context)
        messages = [
            {"role": "system", "content": "你是一位工业领域的专家。"},
            {"role": "user", "content": prompt}
        ]
        return self.llm.chat(messages)




