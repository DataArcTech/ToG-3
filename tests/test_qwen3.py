# test_qwen3_reranker.py
from rag_factory.data_model import Document
from rag_factory.registry import RerankerRegistry
from rag_factory.retrieval.reranker.qwen3 import Qwen3Reranker


if __name__ == "__main__":
    # 测试数据 - 更有针对性的文档
    documents = [
        Document(content="人工智能是计算机科学的一个重要分支，专注于创建能够模拟人类智能的机器。", metadata={"id": "1", "topic": "AI"}),
        Document(content="机器学习是实现人工智能的一种重要方法，通过数据训练模型来进行预测。", metadata={"id": "2", "topic": "ML"}),
        Document(content="深度学习使用多层神经网络来学习数据的复杂模式和特征。", metadata={"id": "3", "topic": "DL"}),
        Document(content="今天天气很好，阳光明媚，适合外出散步和运动。", metadata={"id": "4", "topic": "weather"}),
        Document(content="Python是一种广泛使用的编程语言，特别适合数据科学和机器学习。", metadata={"id": "5", "topic": "programming"}),
    ]
    
    # 测试查询
    query = "什么是机器学习和深度学习？"
    
    print("=== 测试文档列表 ===")
    for doc in documents:
        print(f"ID: {doc.metadata['id']}, 主题: {doc.metadata['topic']}")
        print(f"内容: {doc.content}\n")
    
    print(f"查询: {query}\n")
    
    # 测试Qwen3Reranker
    print("=== 测试Qwen3Reranker ===")
    try:
        model_path = "xxx"  # 替换为实际的模型路径
        
        # 创建Qwen3Reranker实例
        qwen3_reranker = Qwen3Reranker(
            model_name_or_path=model_path,
            max_length=4096,
            device_id="cuda:0"  
        )
        
        print("Qwen3Reranker初始化成功")
        
        # 测试重排序功能
        print("\n--- 测试重排序（返回所有文档）---")
        reranked_docs = qwen3_reranker.rerank(query, documents)
        
        print("重排序结果:")
        for i, doc in enumerate(reranked_docs):
            print(f"{i+1}. ID: {doc.metadata['id']}, 主题: {doc.metadata['topic']}")
            print(f"   内容: {doc.content}")
        
        # 测试k参数
        print("\n--- 测试k=3参数 ---")
        top3_docs = qwen3_reranker.rerank(query, documents, k=3)
        
        print("前3个最相关文档:")
        for i, doc in enumerate(top3_docs):
            print(f"{i+1}. ID: {doc.metadata['id']}, 主题: {doc.metadata['topic']}")
            print(f"   内容: {doc.content}")
        
        # 测试批处理大小
        print("\n--- 测试batch_size=2 ---")
        batch_result = qwen3_reranker.rerank(query, documents, k=2, batch_size=2)
        
        print("批处理结果（前2个）:")
        for i, doc in enumerate(batch_result):
            print(f"{i+1}. ID: {doc.metadata['id']}, 主题: {doc.metadata['topic']}")
            print(f"   内容: {doc.content}")
            
    except Exception as e:
        print(f"Qwen3Reranker测试失败: {e}")
    
    # 测试注册表功能
    print("\n=== 测试RerankerRegistry中的qwen3 ===")
    if RerankerRegistry.is_registered("qwen3"):
        print("qwen3已在注册表中注册")
    else:
        print("qwen3未在注册表中注册")
    
    print("\n测试结束!")