# 图RAG示例

## 配置示例
```yaml
# examples/graphrag/config.yaml
dataset:
  dataset_name: test_samples
  chunk_size: 1024
  chunk_overlap: 20

llm:
  type: OpenAICompatible
  base_url: "http://your-llm-server/v1"
  model: "your-model"

embedding:
  type: OpenAICompatibleEmbedding
  model: "text-embedding-3-large"
  dimension: 1024

storage:
  type: graph_store
  url: "bolt://localhost:7687"
  username: "neo4j"
  password: "your-password"

rag:
  solution: "graph_rag"
  max_paths_per_chunk: 2
  max_cluster_size: 5
```

## 运行脚本
```bash
# examples/graphrag/run.sh
python main.py --config examples/graphrag/config.yaml
```

## 典型输出
```json
{
  "question": "实体A和实体B的关系是什么?",
  "answer": "实体A是实体B的母公司",
  "evidence": [
    "从文本块1提取的三元组: (实体A, 控股, 实体B)",
    "从知识图谱检索的路径: 实体A->控股->实体B"
  ]
}
```

## 常见问题
1. **图数据库连接失败**
   - 检查Neo4j服务是否运行
   - 验证配置中的用户名密码

2. **知识抽取效果不佳**
   - 调整max_paths_per_chunk参数
   - 优化知识抽取提示词

3. **查询响应慢**
   - 减少similarity_top_k值
   - 添加图数据库索引