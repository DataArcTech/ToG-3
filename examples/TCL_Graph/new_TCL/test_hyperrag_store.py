"""
HyperRAGNeo4jStore 核心功能
调用LLM并将数据存入Neo4j数据库
"""

import asyncio
import json
import os
import logging
from typing import List
import sys

rag_factory_path = os.path.join(os.path.dirname(__file__), "..", "..","..")
sys.path.insert(0, rag_factory_path)
from rag_factory.Store.GraphStore.event_graphrag_neo4j import HyperRAGNeo4jStore
from rag_factory.documents.schema import Document

from rag_factory.Embed import HuggingFaceEmbeddings
from config_example import NEO4J_CONFIG, EMBEDDING_CONFIG, LOG_CONFIG, DOCUMENT_CONFIG

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format=LOG_CONFIG["format"]
)
logger = logging.getLogger(__name__)


async def store_documents_to_neo4j(store: HyperRAGNeo4jStore, documents: List[Document]) -> bool:
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
    # Load - 加载文档
    logger.info("阶段 1: Load - 加载文档")
    logger.info(f"从文件加载文档: {DOCUMENT_CONFIG['file_path']}")

    with open(DOCUMENT_CONFIG["file_path"], "r") as f:
        documents = json.load(f)
        documents = [Document(content=doc["content"], metadata=doc["metadata"]) for doc in documents]

    
    if not documents:
        logger.error("没有加载到任何文档，程序退出")
        return
    
    
    #  Store - 存储到数据库
    logger.info("\n阶段 3: Store - 存储到数据库")
    success = await store_documents_to_neo4j(store, documents)
    
    if success:
        logger.info("文档处理完成！")
    else:
        logger.error("文档处理失败！")


if __name__ == "__main__":
    asyncio.run(main())
