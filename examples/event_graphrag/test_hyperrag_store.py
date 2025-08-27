#!/usr/bin/env python3
"""
HyperRAGNeo4jStore 核心功能
调用LLM并将数据存入Neo4j数据库
"""

import asyncio
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any
import sys

rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, rag_factory_path)

from rag_factory.store.graph.event_graphrag import HyperRAGNeo4jStore
from rag_factory.data_model.document import Document
from rag_factory.graph.extractor.event import HyperRAGGraphExtractor
from rag_factory.prompts.prompt import HYPERRAG_EXTRACTION_PROMPT
from rag_factory.data_model.schema import KnowledgeStructure
from rag_factory.embeddings.huggingface import HuggingFaceEmbeddings
from rag_factory.llm.openai import OpenAILLM
from config_example import NEO4J_CONFIG, EMBEDDING_CONFIG, DOCUMENT_CONFIG, LOG_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


def load_documents_from_file(file_path: str, max_documents: int = None) -> List[Document]:
    """
    从文件加载文档数据
    
    Args:
        file_path: 文档文件路径
        max_documents: 最大处理文档数量，None表示处理所有文档
        
    Returns:
        文档列表
    """
    if not os.path.exists(file_path):
        logger.error(f"文档文件不存在: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 限制处理文档数量
        if max_documents:
            data = data[:max_documents]
        
        documents = []
        
        # 处理文档数据
        for i, item in enumerate(data):
            content = item["content"]
            metadata = {"source": "2026国考公务员行测-资料部分"}
            documents.append(Document(content=content, metadata=metadata))
        
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
        
    except Exception as e:
        logger.error(f"加载文档失败: {e}")
        return []


async def extract_graph_structure(documents: List[Document], llm: OpenAILLM) -> List[Document]:
    """
    Extract 阶段：使用 HyperRAGGraphExtractor 提取图结构
    
    Args:
        documents: 原始文档列表
        llm: 语言模型实例
        
    Returns:
        包含图结构的文档列表
    """
    logger.info("开始 Extract 阶段：提取图结构...")
    
    try:
        # 创建 HyperRAGGraphExtractor
        extractor = HyperRAGGraphExtractor(
            llm=llm,
            extract_prompt=HYPERRAG_EXTRACTION_PROMPT,
            response_format=KnowledgeStructure,
            max_concurrent=5
        )
        
        # 提取图结构
        logger.info(f"使用 HyperRAGGraphExtractor 处理 {len(documents)} 个文档...")
        processed_documents = await extractor.acall(documents, show_progress=True)
        
        # 统计提取结果
        total_events = 0
        total_mentions = 0
        total_event_relations = 0
        total_entity_relations = 0
        
        for i, doc in enumerate(processed_documents):
            events = doc.metadata.get("events", [])
            mentions = doc.metadata.get("mentions", [])
            event_relations = doc.metadata.get("event_relations", [])
            entity_relations = doc.metadata.get("entity_relations", [])
            
            total_events += len(events)
            total_mentions += len(mentions)
            total_event_relations += len(event_relations)
            total_entity_relations += len(entity_relations)
            
            logger.debug(f"文档 {i+1}: {len(events)} 个事件, {len(mentions)} 个提及, "
                        f"{len(event_relations)} 个事件关系, {len(entity_relations)} 个实体关系")
        
        logger.info(f"Extract 阶段完成！总计提取:")
        logger.info(f"  - 事件: {total_events} 个")
        logger.info(f"  - 提及: {total_mentions} 个")
        logger.info(f"  - 事件关系: {total_event_relations} 个")
        logger.info(f"  - 实体关系: {total_entity_relations} 个")
        
        return processed_documents
        
    except Exception as e:
        logger.error(f"Extract 阶段失败: {e}")
        import traceback
        traceback.print_exc()
        return documents  # 返回原始文档


async def store_documents_to_neo4j(documents: List[Document]) -> bool:
    """
    将文档存储到Neo4j数据库
    
    Args:
        documents: 要存储的文档列表
        
    Returns:
        存储是否成功
    """
    logger.info("开始存储文档到Neo4j数据库")
    
    # 初始化嵌入模型
    logger.info("初始化HuggingFace嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_CONFIG["model_path"], 
        model_kwargs={'device': EMBEDDING_CONFIG["device"]}
    )
    
    # 初始化存储
    store = HyperRAGNeo4jStore(
        url=NEO4J_CONFIG["url"],
        username=NEO4J_CONFIG["username"], 
        password=NEO4J_CONFIG["password"],
        database=NEO4J_CONFIG["database"],
        embedding=embedding_model
    )
    
    try:
        # 健康检查
        logger.info("检查数据库连接状态...")
        health = await store.health_check()
        logger.info(f"数据库状态: {health['status']}")
        
        if health['status'] != 'healthy':
            logger.error("数据库连接异常，无法继续")
            return False
        
        # 过滤已存在的文档
        logger.info("过滤重复文档...")
        filtered_docs = await store.filter_existing_chunks(documents)
        logger.info(f"需要存储的文档数量: {len(filtered_docs)}")
        
        if not filtered_docs:
            logger.info("所有文档都已存在，无需重复存储")
            return True
        
        # 存储图结构
        logger.info("存储文档到图数据库...")
        success = await store.store_hyperrag_graph(filtered_docs)
        
        if success:
            logger.info("文档存储成功")
            
            # 获取统计信息
            logger.info("获取存储统计信息...")
            stats = await store.get_graph_statistics()
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            
            return True
        else:
            logger.error("文档存储失败")
            return False
        
    except Exception as e:
        logger.error(f"存储过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await store.close()


async def main():
    """主函数"""
    logger.info("HyperRAGNeo4jStore 文档处理工具")
    logger.info("=" * 50)
    
    # 初始化 LLM
    logger.info("初始化 OpenAI LLM...")
    try:
        from config_example import LLM_CONFIG   
        llm = OpenAILLM(
            model_name=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"]
        )
        logger.info(f"LLM 初始化成功: {LLM_CONFIG['model_name']}")
    except ImportError:
        logger.warning("未找到 config.py 文件，使用默认配置")
        llm = OpenAILLM(
            model_name=LLM_CONFIG["model_name"],
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"]
        )
    except Exception as e:
        logger.error(f"LLM 初始化失败: {e}")
        return
    
    # 阶段 1: Load - 加载文档
    logger.info("阶段 1: Load - 加载文档")
    logger.info(f"从文件加载文档: {DOCUMENT_CONFIG['file_path']}")
    documents = load_documents_from_file(
        DOCUMENT_CONFIG['file_path'], 
        DOCUMENT_CONFIG['max_documents']
    )
    
    if not documents:
        logger.error("没有加载到任何文档，程序退出")
        return
    
    # 阶段 2: Extract - 提取图结构
    logger.info("\n阶段 2: Extract - 提取图结构")
    processed_documents = await extract_graph_structure(documents, llm)
    
    if not processed_documents:
        logger.error("图结构提取失败，程序退出")
        return
    
    # 阶段 3: Store - 存储到数据库
    logger.info("\n阶段 3: Store - 存储到数据库")
    success = await store_documents_to_neo4j(processed_documents)
    
    if success:
        logger.info("文档处理完成！")
    else:
        logger.error("文档处理失败！")


if __name__ == "__main__":
    asyncio.run(main())
