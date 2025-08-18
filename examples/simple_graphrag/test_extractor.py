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
from rag_factory.documents.parse_fn import parse_extraction_result
from rag_factory.Store.GraphStore.GraphNode import EntityNode
from rag_factory.Embed import HuggingFaceEmbeddings




# llm = OpenAILLM(
#     model_name="gpt-5-mini",
#     api_key="sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2",
#     base_url="https://api.gptsapi.net/v1",
# )

# extractor = GraphExtractor(
#     llm=llm,
#     extract_prompt=KG_TRIPLES_PROMPT,
#     parse_fn=parse_extraction_result,
# )


# input_text = "基本概念 容斥原理指把包含于某内容中的所有对象的数目先计算出来,然后再把计数时重复计算的数目排斥出去,使得计算的结果既无遗漏又无重复。"


# # result = extractor.acall([Document(content=input_text)])
# chunk_id = f"chunk_{hash(input_text)}"
# result = extractor(documents=[Document(content=input_text, metadata={"chunk_id": chunk_id})])

# print(result)




import asyncio

async def test_upsert_entity():
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

    # entity = EntityNode(
    #     label="实体",
    #     name="实体",
    #     metadatas={
    #         "entity_description": ["实体描述", "实体描述2"],
    #         "source_chunk_id": ["111", "222"],
    #     }
    # )

    # await storage.upsert_entity(entity)
    # await storage.merge_node()
    # await storage.vectorize_existing_nodes()
    results = await storage.search("容斥原理", k=3, search_type="query")
    # for r in results:
    #     print("实体:", r["node"]["name"], "score:", r["score"])
    #     for rel in r["relations"]:
    #         print("  --", rel["relation_type"], "->", rel["neighbor"]["name"])
    for r in results:
        chunk_preview = r['chunk']['content'][:50].replace("\n", " ")  # 只显示前50字符
        print(f"Chunk: {chunk_preview}..., 相似度: {r['score']:.4f}")
        for entity in r['entities']:
            print(f"  实体: {entity['node']['name']}")
            for rel in entity['relations']:
                neighbor_name = rel['neighbor'].get('name', rel['neighbor'].get('content', '未知'))
                print(f"    -- 关系: {rel['relation_type']} -> {neighbor_name}")

# 运行异步函数
asyncio.run(test_upsert_entity())


# import asyncio

# async def main():
#     doc = result[0] if isinstance(result, list) else result
#     for entity in doc.metadata.get("entities", []):
#         await storage.upsert_entity(entity)

#     for relation in doc.metadata.get("relations", []):
#         await storage.upsert_relation(relation)

#     await storage.upsert_document(doc)

# asyncio.run(main())