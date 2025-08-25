#!/usr/bin/env python3
"""
配置文件示例
包含数据库连接、模型路径、LLM配置等参数
"""

# Neo4j数据库配置
NEO4J_CONFIG = {
    "url": "bolt://localhost:7681",
    "username": "neo4j",
    "password": "12345678",
    "database": "neo4j"
}

# 嵌入模型配置
EMBEDDING_CONFIG = {
    "model_path": "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B",
    "device": "cuda:0"  # 可选: "cpu", "cuda:0", "cuda:1" 等
}

# LLM配置
LLM_CONFIG = {
    "model_name": "gpt-5-mini",  # 或其他支持的模型
    "api_key": "xxxx",  # 请替换为实际的 API key
    "base_url": "https://api.gptsapi.net/v1",  # 或其他 API 端点
    # "max_tokens": 4000,
    # "temperature": 0.1
}

# 文档配置
DOCUMENT_CONFIG = {
    "file_path": "/data/FinAi_Mapping_Knowledge/chenmingzhen/RAG-Factory/examples/simple_graphrag/2026国考公务员行测-资料部分.json",  # 文档文件路径
    "max_documents": None  # 最大处理文档数量，None表示处理所有文档
}

# 图提取配置
EXTRACTION_CONFIG = {
    "max_concurrent": 50,  # 最大并发数
    "show_progress": True,  # 是否显示进度条
    "batch_size": 10  # 批处理大小
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",  # 日志级别: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}


