
import os
import sys

rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)
import asyncio
from rag_factory.llms.openai_llm import OpenAILLM
from rag_factory.documents.event_GraphExtractor import HyperRAGGraphExtractor
from rag_factory.documents.schema import Document
from rag_factory.documents.prompt_test import TCL_PROMPT, KnowledgeStructure, TCL_CLEAN_PROMPT
from pathlib import Path
import json
from typing import List


# ------------------------------ 配置参数 ------------------------------
INPUT_FILE = Path("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/TCL_Graph/dataarc_parse/data/parsed_data_by_dataarc.json")
OUTPUT_DIR = Path("/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/TCL_Graph/new_TCL")
FINAL_OUTPUT_FILE = OUTPUT_DIR / "extracted_documents.json"
BATCH_SIZE = 10


# ------------------------------ 工具函数 ------------------------------
def serialize_documents(docs: List[Document]) -> List[dict]:
    return [
        {
            "content": doc.content,
            "metadata": doc.metadata
        } for doc in docs
    ]


def save_to_file(file_path: Path, data: List[dict]):
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ------------------------------ 主处理逻辑 ------------------------------
async def main():
    print(f"[信息] 正在读取文档数据: {INPUT_FILE}")
    docs = []
    try:
        with INPUT_FILE.open("r", encoding="utf-8") as f:
            document_data = json.load(f)
            for item in document_data:
                content = item.get('chunk', '')
                chunk_id = "chunk_" + str(hash(content))
                source = item.get('file_name', '')
                docs.append(Document(content=content, metadata={"source": source,"chunk_id": chunk_id}))
    except Exception as e:
        print(f"[错误] 无法读取输入文件: {e}")
        return

    print(f"[信息] 成功读取 {len(docs)} 个文档")

    # 初始化 LLM 和抽取器
    llm = OpenAILLM(
        model_name="gpt-4.1-mini",
        api_key="sk-xxxxx",
        base_url="https://api.gptsapi.net/v1"
    )

    extractor = HyperRAGGraphExtractor(
        llm=llm,
        extract_prompt=TCL_PROMPT,
        response_format=KnowledgeStructure,
        enable_cleaning=True,
        clean_prompt=TCL_CLEAN_PROMPT
    )

    all_result_docs = []
    total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, i in enumerate(range(0, len(docs), BATCH_SIZE), start=1):
        batch_docs = docs[i:i + BATCH_SIZE]
        print(f"\n[批次 {batch_num}/{total_batches}] 开始处理 {len(batch_docs)} 个文档...")

        try:
            result_docs = await extractor.acall(batch_docs)
        except Exception as e:
            print(f"[错误] 批次 {batch_num} 处理失败: {e}")
            continue

        all_result_docs.extend(result_docs)
        print(f"[批次 {batch_num}] 完成，已累计处理 {len(all_result_docs)} 个文档")

        if batch_num % 1 == 0 or batch_num == total_batches:
            temp_file = OUTPUT_DIR / f"temp_extracted_batch_{batch_num}.json"
            save_to_file(temp_file, serialize_documents(all_result_docs))
            print(f"[保存] 批次 {batch_num} 临时结果已保存: {temp_file}")

    # 保存最终结果
    print("\n[信息] 所有批次处理完成，开始保存最终结果...")
    save_to_file(FINAL_OUTPUT_FILE, serialize_documents(all_result_docs))
    print(f"[完成] 所有文档保存成功，共处理 {len(all_result_docs)} 个文档")
    print(f"[路径] {FINAL_OUTPUT_FILE}")

    # 统计信息
    total_events = sum(len(doc.metadata.get("events", [])) for doc in all_result_docs)
    total_mentions = sum(len(doc.metadata.get("mentions", [])) for doc in all_result_docs)
    total_entity_relations = sum(len(doc.metadata.get("entity_relations", [])) for doc in all_result_docs)
    total_event_relations = sum(len(doc.metadata.get("event_relations", [])) for doc in all_result_docs)

    print("\n📊 处理统计：")
    print(f"- 总文档数: {len(all_result_docs)}")
    print(f"- 总事件数: {total_events}")
    print(f"- 总实体提及数: {total_mentions}")
    print(f"- 总实体关系数: {total_entity_relations}")
    print(f"- 总事件关系数: {total_event_relations}")


# ------------------------------ 启动入口 ------------------------------
if __name__ == "__main__":
    asyncio.run(main())