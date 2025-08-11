from rag_flow import TCL_RAG
import yaml

# 加载配置文件
with open('/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/TCL_rag/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

llm_config = config['llm']
embedding_config = config['embedding']
reranker_config = config['reranker']
bm25_retriever_config = config['bm25']
retriever_config = config['retriever']
vector_store_config = config['store']




if __name__ == "__main__":

    rag = TCL_RAG(llm_config=llm_config, 
                embedding_config=embedding_config, 
                reranker_config=reranker_config, 
                retriever_config=retriever_config, 
                vector_store_config=vector_store_config,
                bm25_retriever_config=bm25_retriever_config)

    result = rag.invoke("毛细管设计规范按照什么标准",k=20)

    answer = rag.answer("毛细管设计规范按照什么标准",result)


    print(answer)