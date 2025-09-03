# 申论多GPU测试脚本使用文档

## 概述
这是一套用于申论题目多GPU并行测试的脚本集合，包含四种不同的测试模式：图检索、RAG检索、LLM直接回答和综合对比测试。

## 核心组件

### `rag_flow_example_commentary.py` - RAG流程核心类
这是整个测试系统的核心组件，提供了 `TCL_RAG` 类，实现了申论题目的多种回答方式：

#### 主要功能
- **多路径检索**: 结合BM25和向量检索的混合检索策略
- **智能重排序**: 使用reranker对检索结果进行质量排序
- **多种回答模式**: 支持图检索、RAG检索和LLM直接回答
- **答案质量评估**: 使用AI评判器对比不同方法的答案质量

#### 核心方法
- `invoke(query, k)`: 执行多路径检索
- `rerank(query, documents, k)`: 对检索结果进行重排序
- `graph_answer(query, materials, documents)`: 基于图检索生成答案
- `rag_answer(query, materials, documents)`: 基于RAG检索生成答案
- `llm_answer(query, materials)`: LLM直接生成答案
- `judge_answer(...)`: 评估和对比不同答案的质量
- `rewrite_query(query)`: 智能重写查询以优化检索效果

#### 技术特点
- 支持中文预处理（使用jieba分词）
- 可配置的LLM、Embedding、VectorStore等组件
- 灵活的配置管理，支持多GPU部署

### `prompt.py` - 提示词模板库
包含所有申论相关的提示词模板，为不同任务提供标准化的提示词：

#### 主要提示词模板
- **COMMENTARY_GRAPH_PROMPT**: 图检索模式下的申论答案生成提示词
- **COMMENTARY_RAG_PROMPT**: RAG检索模式下的申论答案生成提示词  
- **COMMENTARY_LLM_PROMPT**: LLM直接回答模式下的申论答案生成提示词
- **COMMENTARY_REWRITE_PROMPT**: 查询重写提示词，用于分析题目类型
- **COMMENTARY_RERANK_PROMPT**: 检索结果重排序提示词
- **COMMENTARY_JUDGE_PROMPT**: 答案质量评估提示词

#### 提示词特点
- 针对申论题目特点设计，包含政治正确性和政策敏感性要求
- 支持多种申论题型（概括归纳、综合分析、解决问题等）
- 提供标准化的评分维度和评估标准

## 脚本说明

### 1. `commentary_graph_multi_gpu.py` - 图检索测试
- **功能**: 测试基于知识图谱的申论题目检索和回答
- **特点**: 结合RAG和图检索，使用PPR算法进行实体扩展
- **输出**: 图检索结果、种子节点、检索时间等

### 2. `commentary_rag_multi_gpu.py` - RAG检索测试  
- **功能**: 测试传统RAG检索和回答
- **特点**: 使用向量检索+重排序+生成的方式
- **输出**: RAG检索结果、相关文档、回答时间等

### 3. `commentary_llm_multi_gpu.py` - LLM直接回答测试
- **功能**: 测试LLM直接回答申论题目
- **特点**: 不进行检索，直接基于题目和材料生成答案
- **输出**: LLM回答结果、回答时间等

### 4. `commentary_match_multi_gpu.py` - 综合对比测试
- **功能**: 对比三种方法的性能和质量
- **特点**: 使用AI评判器评估答案质量，统计得分和获胜次数
- **输出**: 综合对比结果、质量评估、性能统计等

### 5. `commentary_query_rewrite.py` - 查询重写工具
- **功能**: 对申论题目进行智能重写，优化检索效果
- **特点**: 使用LLM分析题目类型，生成更精确的查询语句
- **输出**: 重写后的查询语句，保存为JSON格式文件

## 使用方法

### 基本配置
```python
# 在main()函数中修改以下参数
gpu_ids = [0, 1, 2, 3, 4, 6, 7]  # 要使用的GPU设备ID
test_rounds = 1                    # 每个GPU的测试轮次
query_id_count = 17                # 要测试前多少道题目
```

### 运行步骤
1. **准备配置文件**: 确保对应的YAML配置文件存在
2. **准备测试数据**: 确保申论真题JSON文件路径正确
3. **设置输出路径**: 修改结果保存路径
4. **执行脚本**: `python script_name.py`

### 执行顺序建议
1. 先运行三个独立测试脚本生成基础结果
2. 再运行对比测试脚本进行综合分析

## 查询重写功能

### 功能说明
`commentary_query_rewrite.py` 提供了智能查询重写功能，能够：
- 分析申论题目的类型（概括归纳、综合分析、解决问题等）
- 生成更精确的查询语句，提高检索效果
- 批量处理多个题目，生成重写后的查询数据集

### 使用方法
```python
# 运行查询重写脚本
python commentary_query_rewrite.py
```

### 输出结果
- 生成包含原始题目和重写后查询的JSON文件
- 文件格式：`{question_content, rewritten_query, materials, answer}`
- 可用于后续的检索和回答任务中

### 配置说明
在 `config_commentary.yaml` 中的 `rewrite` 部分配置：
- `path`: 重写结果保存路径
- 相关LLM配置用于查询重写任务

## 输出结果
- 每个GPU生成独立的结果文件
- 文件名格式: `{base_name}_gpu{gpu_id}_round{round_num}.json`
- 包含详细的测试结果、性能指标和质量评估

## 注意事项
- 确保所有GPU设备可用且有足够显存
- 配置文件中的模型路径和设备设置需要正确
- 测试数据格式需要符合预期结构
- 建议先用少量数据测试脚本运行正常性

## 提示词模板使用

### 模板选择
根据不同的任务需求选择合适的提示词模板：
- **答案生成**: 使用 `COMMENTARY_*_PROMPT` 系列模板
- **查询重写**: 使用 `COMMENTARY_REWRITE_PROMPT`
- **结果重排序**: 使用 `COMMENTARY_RERANK_PROMPT`
- **质量评估**: 使用 `COMMENTARY_JUDGE_PROMPT`

### 自定义提示词
可以在 `prompt.py` 中修改或添加新的提示词模板：
- 保持政治正确性和政策敏感性
- 针对具体申论题型进行优化
- 确保输出格式的一致性

## 文件结构
```
examples/TCL_rag/
├── rag_flow_example_commentary.py     # RAG流程核心类
├── commentary_graph_multi_gpu.py      # 图检索测试脚本
├── commentary_rag_multi_gpu.py        # RAG检索测试脚本
├── commentary_llm_multi_gpu.py        # LLM直接回答测试脚本
├── commentary_match_multi_gpu.py      # 综合对比测试脚本
├── commentary_query_rewrite.py        # 查询重写工具脚本
├── prompt.py                          # 提示词模板库
├── config_commentary.yaml             # 图检索配置文件
├── config_commentary_llm.yaml         # LLM/RAG配置文件
└── commentary_README.md               # 本使用文档
```

## 依赖要求
- Python 3.7+
- PyTorch (支持CUDA)
- 相关RAG和知识图谱库
- 多GPU环境支持
- jieba (中文分词)
