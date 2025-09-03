# 行测多GPU测试脚本使用文档

## 概述
这是一套用于行测题目多GPU并行测试的脚本集合，包含四种不同的测试模式：图检索、RAG检索、LLM直接回答和综合对比测试。

## 核心组件

### `rag_flow_example.py` - RAG流程核心类
这是整个测试系统的核心组件，提供了 `TCL_RAG` 类，实现了行测题目的多种回答方式：

#### 主要功能
- **多路径检索**: 结合BM25和向量检索的混合检索策略
- **智能重排序**: 使用reranker对检索结果进行质量排序
- **多种回答模式**: 支持图检索、RAG检索和LLM直接回答
- **答案质量评估**: 使用AI评判器对比不同方法的答案质量
- **知识点匹配**: 评估不同方法的知识点命中率

#### 核心方法
- `invoke(query, k)`: 执行多路径检索
- `rerank(query, documents, k)`: 对检索结果进行重排序
- `graph_answer(query, documents)`: 基于图检索生成答案
- `rag_answer(query, documents)`: 基于RAG检索生成答案
- `llm_answer(query)`: LLM直接生成答案
- `judge_answer(...)`: 评估和对比不同答案的质量
- `rewrite_query(query)`: 智能重写查询以优化检索效果
- `mark_knowledge(query)`: 标记题目所需知识点
- `match_knowledge(...)`: 匹配知识点命中率

#### 技术特点
- 支持中文预处理（使用jieba分词）
- 可配置的LLM、Embedding、VectorStore等组件
- 灵活的配置管理，支持多GPU部署
- 使用思维链（Chain-of-Thought）推理方式

## 脚本说明

### 1. `test_example_graph_multi_gpu.py` - 图检索测试
- **功能**: 测试基于知识图谱的行测题目检索和回答
- **特点**: 结合RAG和图检索，使用PPR算法进行实体扩展
- **输出**: 图检索结果、种子节点、检索时间、正确率统计等

### 2. `test_example_rag_multi_gpu.py` - RAG检索测试  
- **功能**: 测试传统RAG检索和回答
- **特点**: 使用向量检索+重排序+生成的方式
- **输出**: RAG检索结果、相关文档、回答时间、正确率统计等

### 3. `test_example_llm_multi_gpu.py` - LLM直接回答测试
- **功能**: 测试LLM直接回答行测题目
- **特点**: 不进行检索，直接基于题目生成答案
- **输出**: LLM回答结果、回答时间、正确率统计等

### 4. `test_example_match_multi_gpu.py` - 综合对比测试
- **功能**: 对比三种方法的性能和质量
- **特点**: 使用知识点匹配评估答案质量，统计命中率和正确率
- **输出**: 综合对比结果、知识点匹配率、性能统计等

### 5. `test_example_rewrite.py` - 查询重写工具
- **功能**: 将原始题目重写为适合检索的查询
- **特点**: 提取题目核心知识点，优化检索效果
- **输出**: 重写后的查询和知识点列表

## 使用方法

### 基本配置
```python
# 在main()函数中修改以下参数
gpu_ids = [0, 2, 3, 4, 5, 6, 7]  # 要使用的GPU设备ID
test_rounds = 3                    # 每个GPU的测试轮次
query_id_list = [i for i in range(1, 28)]  # 要测试的题目编号
```

### 运行步骤
1. **准备配置文件**: 确保对应的YAML配置文件存在
2. **准备测试数据**: 确保行测题目JSON文件路径正确
3. **设置输出路径**: 修改结果保存路径
4. **执行脚本**: `python script_name.py`

### 执行顺序建议
1. 先运行 `test_example_rewrite.py` 重写查询
2. 再运行三个独立测试脚本生成基础结果
3. 最后运行对比测试脚本进行综合分析

## 输出结果
- 每个GPU生成独立的结果文件
- 文件名格式: `{base_name}_gpu{gpu_id}_round{round_num}.json`
- 包含详细的测试结果、性能指标、正确率统计和知识点匹配率

## 注意事项
- 确保所有GPU设备可用且有足够显存
- 配置文件中的模型路径和设备设置需要正确
- 测试数据格式需要符合预期结构
- 建议先用少量数据测试脚本运行正常性
- 需要先运行查询重写脚本生成优化后的查询

## 文件结构
```
examples/TCL_rag/
├── rag_flow_example.py                    # RAG流程核心类
├── prompt.py                              # 提示词模板文件
├── test_example_graph_multi_gpu.py        # 图检索测试脚本
├── test_example_rag_multi_gpu.py          # RAG检索测试脚本
├── test_example_llm_multi_gpu.py          # LLM直接回答测试脚本
├── test_example_match_multi_gpu.py        # 综合对比测试脚本
├── test_example_rewrite.py                # 查询重写工具脚本
├── config_graph.yaml                      # 图检索配置文件
├── config_rag.yaml                        # RAG配置文件
├── config_llm.yaml                        # LLM配置文件
└── analysis_README.md                     # 本使用文档
```

## 依赖要求
- Python 3.7+
- PyTorch (支持CUDA)
- 相关RAG和知识图谱库
- 多GPU环境支持
- jieba (中文分词)
- Neo4j (知识图谱数据库)
