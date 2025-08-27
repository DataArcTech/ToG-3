import sys
import os
import json

# 添加 RAG-Factory 目录到 Python 路径
rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)


from rag_factory.documents.GraphExtractor import GraphExtractor
from rag_factory.documents.Prompt import KG_TRIPLES_PROMPT
from rag_factory.documents.schema import Document
from rag_factory.llms import OpenAILLM
from rag_factory.Store.GraphStore.graphrag_neo4j import Neo4jGraphStore
from rag_factory.Store.GraphStore.GraphNode import EntityNode
from rag_factory.Embed import HuggingFaceEmbeddings
from rag_factory.documents.pydantic_schema import GraphTriples


llm = OpenAILLM(
    model_name="gpt-5-mini",
    api_key="sk-xxxxxx",
    base_url="https://api.gptsapi.net/v1",
)

extractor = GraphExtractor(
    llm=llm,
    extract_prompt=KG_TRIPLES_PROMPT,
    response_format=GraphTriples,
)

storage = Neo4jGraphStore(
        url="bolt://localhost:7680",
        username="neo4j",
        password="12345678",
        database="neo4j",
        embedding=HuggingFaceEmbeddings(
            model_name="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
            model_kwargs={'device': 'cuda:0'}
        )
    )

async def graph_construction(documents: list[Document]):
    """
    图构建函数：调用extractor抽取实体和关系，并存储到图数据库
    """
    # 调用extractor进行实体和关系抽取
    result = await extractor.acall(documents)
    
    # 打印抽取结果
    print(f"成功处理 {len(result)} 个文档")
    
    for i, doc in enumerate(result):
        entities = doc.metadata.get("entities", [])
        relations = doc.metadata.get("relations", [])
        
        print(f"\n文档 {i+1}:")
        print(f"  抽取到 {len(entities)} 个实体")
        print(f"  抽取到 {len(relations)} 个关系")
        
        # 显示抽取的实体
        if entities:
            print("  实体列表:")
            for entity in entities:
                print(f"    - {entity.name} ({entity.label})")
        
        # 显示抽取的关系
        if relations:
            print("  关系列表:")
            for relation in relations:
                print(f"    - {relation.head_id} --{relation.label}--> {relation.tail_id}")
        
        # 存储到图数据库
        for entity in entities:
            await storage.upsert_entity(entity)
        for relation in relations:
            await storage.upsert_relation(relation)
        await storage.upsert_document(doc)

    await storage.merge_node()
    await storage.vectorize_existing_nodes()
    
    print("\n图构建完成！")


if __name__ == "__main__":
    import asyncio

    # with open("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/simple_graphrag/2026国考公务员行测-资料部分.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # documents = []
    # for i, item in enumerate(data):
    #     if i == 0:
    #         continue
    #     content = item["content"]
    #     # metadata = item["metadata"]
    #     metadata = {"file_name": item["file_name"]}
    #     metadata["chunk_id"] = f"chunk_{hash(content)}"
    #     documents.append(Document(content=content, metadata=metadata))


    # asyncio.run(graph_construction(documents))

    async def run_operations():
        await storage.merge_node()
        await storage.vectorize_existing_nodes()
    
    asyncio.run(run_operations())

    print("图构建完成！")