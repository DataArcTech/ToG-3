# test_faiss_vectorstore.py
import numpy as np
from rag_factory.Store.VectorStore.VectorStore_Faiss import FaissVectorStore, Document
from rag_factory.Embed import EmbeddingRegistry
from rag_factory.Store import VectorStoreRegistry
from rag_factory.Retrieval import RetrieverRegistry

if __name__ == "__main__":
    # 初始化
    embeddings = EmbeddingRegistry.create(name="huggingface", model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B")
    # vs = VectorStoreRegistry.load(name="faiss", folder_path="/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/faiss_test_store", embedding=embeddings)
    # print(vs)
    # vs = VectorStoreRegistry.create(name="faiss", embedding=embeddings)
    vs = FaissVectorStore.load_local(folder_path="/data/FinAi_Mapping_Knowledge/chenmingzhen/test_faiss_store", embeddings=embeddings, index_name="index")
    # # # 添加一些文本
    # texts = ["苹果是一种水果", "香蕉是黄色的", "猫是一种动物", "狗喜欢跑步"]
    # # ids = vs.add_texts(texts, metadatas=[{"type": "test"} for _ in texts])

    # documents = [Document(content=text, metadata={"type": "test"}) for text in texts]
    # vectorstore = vs.from_documents(documents, embedding=embeddings)
    # vectorstore.save_local(folder_path="/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/faiss_test_store")
    # print(f"添加的文档ID: {ids}")

    # # 相似性搜索
    # results = vs.similarity_search("苹果", k=2)
    # print("\n=== 相似性搜索结果 ===")
    # for doc in results:
    #     print(f"内容: {doc.content}, 元数据: {doc.metadata}")

    # # 带分数的搜索
    # results_with_score = vs.similarity_search_with_score("苹果", k=2)
    # print("\n=== 带分数的搜索结果 ===")
    # for doc, score in results_with_score:
    #     print(f"内容: {doc.content}, 分数: {score}")

    # # 最大边际相关性搜索
    # mmr_results = vs.max_marginal_relevance_search("苹果", k=2, fetch_k=3)
    # print("\n=== MMR搜索结果 ===")
    # for doc in mmr_results:
    #     print(f"内容: {doc.content}")

    # # 保存到本地
    # save_path = "./faiss_test_store"
    # vs.save_local(save_path)
    # print(f"\n索引已保存到: {save_path}")

    # 从本地加载
    # loaded_vs = FaissVectorStore.load_local(save_path, embeddings)
    # load_results = loaded_vs.similarity_search("苹果", k=2)
    # print("\n=== 从本地加载后的搜索结果 ===")
    # for doc in load_results:
    #     print(f"内容: {doc.content}")

    retriever = vs.as_retriever(search_kwargs={"k": 2})
    # retriever = RetrieverRegistry.create(name="vectorstore", vectorstore=vs)
    results = retriever.invoke("文件名称：GB$T 2828.1-2012 计数抽样检验程序 第1部分：按接收质量限(AQL)检索的逐批检验抽样计划.pdf\n4 不合格的表示 4. 1 总则\n不合格的程度以不合格品百分数(见 3.1.8和 3.1.9)或每百单位产品不合格数(见 3.1.10和3.1.11)表示。表7、表8和表10 是基于假定不合格的出现是随机且统计独立的。如果已知产品的某个不合格可能由某一条件引起，此条件还可能引起其他一些不合格，则应仅考虑该产品是否为合格品，而不管该产品有多少个不合格。")
    print(results)
