# # without chunk info
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Any
# from langchain_core.embeddings import Embeddings
# import hdbscan
# from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
# import pandas as pd
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import normalize
# import warnings
# from Qwen3_Reranker import Qwen3Reranker
# warnings.filterwarnings('ignore')

# # 封装为 LangChain 所需的 Embeddings 接口
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", device: str = "cuda:1"):
#         self.model = SentenceTransformer(model_name_or_path, device=device)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode(text, convert_to_numpy=True, prompt = "Instruct: Given a natural language question, retrieve the most relevant text passages from a collection of documents to support answer generation.\nQuery:").tolist()

#     def __call__(self, text: str) -> List[float]:
#         return self.embed_query(text)

# class EntityClusteringPipeline:
#     def __init__(self, embedding_model_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_4B", 
#                  reranker_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_4B",
#                  device: str = "cuda:1"):
#         """
#         初始化实体聚类管道
        
#         Args:
#             embedding_model_path: 嵌入模型路径
#             device: 设备（如：cuda:1）
#         """
#         self.embeddings = SentenceTransformerEmbeddings(embedding_model_path, device)
#         self.reranker_path = reranker_path
#         self.reranker = Qwen3Reranker(
#             model_name_or_path=self.reranker_path,
#             max_length=2048,
#             instruction="Given the user query, retrieval the relevant passages",
#             device_id = "cuda:7" 
            
#         )
#         self.entities = []
#         self.entity_vectors = None
#         self.clusters = None
#         self.cluster_labels = None
        
#     def load_entities_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
#         """
#         从JSONL文件加载实体数据
        
#         Args:
#             file_path: JSONL文件路径
            
#         Returns:
#             实体列表
#         """
#         entities = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     entity = json.loads(line.strip())
#                     entities.append(entity)
#                 except json.JSONDecodeError as e:
#                     print(f"解析行时出错: {e}")
#                     continue
        
#         print(f"成功加载 {len(entities)} 个实体")
#         return entities
    
#     def extract_entity_names(self, entities: List[Dict[str, Any]]) -> List[str]:
#         """
#         提取实体名称
        
#         Args:
#             entities: 实体列表
            
#         Returns:
#             实体名称列表
#         """
#         entity_names = []
#         for entity in entities:
#             if 'name' in entity:
#                 entity_names.append(entity['name'])
        
#         print(f"提取到 {len(entity_names)} 个实体名称")
#         return entity_names
    


#     def vectorize_entities(self, entity_names: List[str]) -> np.ndarray:
#         """
#         对实体名称进行向量化
        
#         Args:
#             entity_names: 实体名称列表
            
#         Returns:
#             向量化结果
#         """
#         print("开始向量化实体名称...")
        
#         batch_size = 100
#         all_vectors = []
        
#         for i in range(0, len(entity_names), batch_size):
#             batch = entity_names[i:i+batch_size]
#             vectors = self.embeddings.embed_documents(batch)  # 使用嵌入模型进行向量化
#             all_vectors.extend(vectors)
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"已处理 {i + len(batch)} / {len(entity_names)} 个实体")
        
#         vectors_array = np.array(all_vectors)
#         print(f"向量化完成，形状: {vectors_array.shape}")
#         return vectors_array
    
#     def perform_clustering(self, vectors: np.ndarray, min_cluster_size: int = 2, 
#                           min_samples: int = 1, cluster_selection_epsilon: float = 0.1,
#                           use_cosine: bool = True) -> np.ndarray:
#         """
#         使用HDBSCAN进行聚类
        
#         Args:
#             vectors: 向量矩阵
#             min_cluster_size: 最小簇大小
#             min_samples: 最小样本数
#             cluster_selection_epsilon: 聚类选择epsilon
#             use_cosine: 是否使用余弦距离
            
#         Returns:
#             聚类标签
#         """
#         print("开始HDBSCAN聚类...")
        
#         if use_cosine:
#             # 方法1：使用余弦距离矩阵（推荐方法）
#             print("计算余弦距离矩阵...")
            
#             # 先对向量进行L2归一化，确保数值稳定性
#             normalized_vectors = normalize(vectors, norm='l2')
            
#             # 计算余弦距离矩阵
#             distance_matrix = cosine_distances(normalized_vectors)
            
#             # 使用预计算的距离矩阵
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 cluster_selection_epsilon=cluster_selection_epsilon,
#                 metric='precomputed'
#             )
#             cluster_labels = clusterer.fit_predict(distance_matrix)
#             print("使用余弦距离矩阵聚类成功")
#         else:
#             # 方法2：使用欧几里得距离（备选方案）
#             print("使用欧几里得距离进行聚类...")
            
#             # 先对向量进行L2归一化，使欧几里得距离近似余弦距离
#             normalized_vectors = normalize(vectors, norm='l2')
            
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 cluster_selection_epsilon=cluster_selection_epsilon,
#                 metric='euclidean'
#             )
#             cluster_labels = clusterer.fit_predict(normalized_vectors)
#             print("使用L2归一化向量+欧几里得距离聚类成功")
        
#         # 统计聚类结果
#         n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
#         n_noise = list(cluster_labels).count(-1)
        
#         print(f"聚类完成:")
#         print(f"  - 簇数量: {n_clusters}")
#         print(f"  - 噪声点数量: {n_noise}")
#         print(f"  - 噪声点比例: {n_noise/len(cluster_labels):.2%}")
        
#         return cluster_labels
    
#     def analyze_clusters(self, entity_names: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
#         """
#         分析聚类结果
        
#         Args:
#             entity_names: 实体名称列表
#             cluster_labels: 聚类标签
            
#         Returns:
#             聚类结果字典
#         """
#         clusters = defaultdict(list)
        
#         for name, label in zip(entity_names, cluster_labels):
#             clusters[label].append(name)
        
#         return dict(clusters)
    
#     def find_cluster_representatives(self, clusters: Dict[int, List[str]], 
#                                    entity_names: List[str], vectors: np.ndarray) -> Dict[int, str]:
#         """
#         为每个簇找到代表性实体（最接近簇中心的实体）
        
#         Args:
#             clusters: 聚类结果
#             entity_names: 实体名称列表
#             vectors: 向量矩阵
            
#         Returns:
#             每个簇的代表实体
#         """
#         representatives = {}
#         name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if cluster_id == -1:  # 跳过噪声点
#                 continue
                
#             if len(cluster_entities) == 1:
#                 representatives[cluster_id] = cluster_entities[0]
#                 continue
            
#             # 获取簇中所有实体的向量
#             cluster_indices = [name_to_idx[name] for name in cluster_entities]
#             cluster_vectors = vectors[cluster_indices]
            
#             # 计算簇中心
#             cluster_center = np.mean(cluster_vectors, axis=0)
            
#             # 找到最接近中心的实体
#             similarities = cosine_similarity([cluster_center], cluster_vectors)[0]
#             best_idx = np.argmax(similarities)
#             representatives[cluster_id] = cluster_entities[best_idx]
        
#         return representatives
    
#     def create_alias_mapping(self, clusters: Dict[int, List[str]], 
#                            representatives: Dict[int, str]) -> Dict[str, str]:
#         """
#         创建别称映射
        
#         Args:
#             clusters: 聚类结果
#             representatives: 簇代表实体
            
#         Returns:
#             别称映射字典
#         """
#         alias_mapping = {}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if cluster_id == -1:  # 噪声点保持原样
#                 for entity in cluster_entities:
#                     alias_mapping[entity] = entity
#                 continue
            
#             representative = representatives[cluster_id]
#             for entity in cluster_entities:
#                 alias_mapping[entity] = representative
        
#         return alias_mapping
    
#     def visualize_clusters(self, vectors: np.ndarray, cluster_labels: np.ndarray, 
#                           entity_names: List[str], method: str = 'tsne', 
#                           sample_size: int = 1000, figsize: tuple = (15, 10)):
#         """
#         可视化聚类结果(推荐使用TSNE降维)
        
#         Args:
#             vectors: 向量矩阵
#             cluster_labels: 聚类标签
#             entity_names: 实体名称列表
#             method: 降维方法 ('pca' 或 'tsne')
#             sample_size: 采样大小（处理大数据集）
#             figsize: 图形大小
#         """
#         # 设置中文字体
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
        
#         # 如果数据量太大，进行采样
#         if len(vectors) > sample_size:
#             np.random.seed(42)  # 确保可重现性
#             indices = np.random.choice(len(vectors), sample_size, replace=False)
#             vectors_sample = vectors[indices]
#             labels_sample = cluster_labels[indices]
#             names_sample = [entity_names[i] for i in indices]
#         else:
#             vectors_sample = vectors
#             labels_sample = cluster_labels
#             names_sample = entity_names
        
#         # 对向量进行归一化
#         vectors_normalized = normalize(vectors_sample, norm='l2')
        
#         # 降维
#         print(f"正在进行{method.upper()}降维...")
#         if method == 'pca':
#             reducer = PCA(n_components=2, random_state=42)
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         elif method == 'tsne':
#             reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors_normalized)-1), metric='cosine')
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         else:
#             raise ValueError("method must be 'pca' or 'tsne'")
        
#         # 创建子图
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
#         # 左图：所有点的散点图
#         unique_labels = sorted(set(labels_sample))
#         n_clusters = len([l for l in unique_labels if l != -1])
        
#         # 使用更好的颜色方案
#         if n_clusters <= 10:
#             colors = plt.cm.tab10(np.linspace(0, 1, 10))
#         else:
#             colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
#         # 为噪声点设置特殊颜色
#         color_map = {}
#         color_idx = 0
#         for label in unique_labels:
#             if label == -1:
#                 color_map[label] = 'gray'
#             else:
#                 color_map[label] = colors[color_idx % len(colors)]
#                 color_idx += 1
        
#         # 绘制所有点
#         for label in unique_labels:
#             mask = labels_sample == label
#             if label == -1:  # 噪声点
#                 ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                            c='gray', marker='x', alpha=0.5, s=20, label='Noise')
#             else:
#                 ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                            c=[color_map[label]], marker='o', alpha=0.7, s=30, 
#                            label=f'Cluster {label}')
        
#         ax1.set_title(f'All Clusters ({method.upper()})', fontsize=14)
#         ax1.set_xlabel('Component 1', fontsize=12)
#         ax1.set_ylabel('Component 2', fontsize=12)
#         ax1.grid(True, alpha=0.3)
        
#         # 添加图例（只显示前15个）
#         handles, labels = ax1.get_legend_handles_labels()
#         if len(handles) > 15:
#             ax1.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left')
#         else:
#             ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         # 右图：显示最大的几个簇并标注代表性实体
#         cluster_sizes = [(label, sum(labels_sample == label)) for label in unique_labels if label != -1]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         # 只显示前10个最大的簇
#         top_clusters = [label for label, size in cluster_sizes[:10]]
        
#         for label in top_clusters:
#             mask = labels_sample == label
#             cluster_points = vectors_2d[mask]
#             cluster_names = [names_sample[i] for i in range(len(names_sample)) if labels_sample[i] == label]
            
#             # 绘制簇
#             ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                        c=[color_map[label]], marker='o', alpha=0.7, s=40, 
#                        label=f'Cluster {label} ({len(cluster_names)})')
            
#             # 添加簇中心和代表实体标注
#             center_x, center_y = np.mean(cluster_points, axis=0)
            
#             # 找到最接近中心的点
#             distances = np.sum((cluster_points - [center_x, center_y])**2, axis=1)
#             closest_idx = np.argmin(distances)
            
#             # 标注代表实体
#             if len(cluster_names) > 1:  # 只对多元素簇标注
#                 representative = cluster_names[closest_idx]
#                 ax2.annotate(representative, 
#                            (cluster_points[closest_idx, 0], cluster_points[closest_idx, 1]),
#                            xytext=(5, 5), textcoords='offset points', 
#                            fontsize=8, alpha=0.8,
#                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
#         ax2.set_title(f'Top 10 Largest Clusters with Labels ({method.upper()})', fontsize=14)
#         ax2.set_xlabel('Component 1', fontsize=12)
#         ax2.set_ylabel('Component 2', fontsize=12)
#         ax2.grid(True, alpha=0.3)
#         ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         plt.tight_layout()
#         plt.savefig(f'entity_clustering_visualization_{method}_without_chunk_info.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # 打印一些可视化统计信息
#         print(f"\n可视化统计信息:")
#         print(f"  - 可视化样本数: {len(vectors_sample)}")
#         print(f"  - 降维方法: {method.upper()}")
#         print(f"  - 显示的簇数: {len(top_clusters)}")
        
#         if method == 'pca':
#             print(f"  - 主成分解释的方差比: {reducer.explained_variance_ratio_}")
#             print(f"  - 累计方差解释比: {np.sum(reducer.explained_variance_ratio_):.3f}")
    
#     def save_results(self, alias_mapping: Dict[str, str], clusters: Dict[int, List[str]], 
#                     representatives: Dict[int, str], output_path: str = "clustering_results.json"):
#         """
#         保存聚类结果
        
#         Args:
#             alias_mapping: 别称映射
#             clusters: 聚类结果
#             representatives: 簇代表实体
#             output_path: 输出文件路径
#         """
#         results = {
#             "alias_mapping": alias_mapping,
#             "clusters": {str(k): v for k, v in clusters.items()},
#             "representatives": {str(k): v for k, v in representatives.items()},
#             "statistics": {
#                 "total_entities": len(alias_mapping),
#                 "unique_representatives": len(set(alias_mapping.values())),
#                 "compression_ratio": len(set(alias_mapping.values())) / len(alias_mapping)
#             }
#         }
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
        
#         print(f"结果已保存到: {output_path}")
    
#     def run_pipeline(self, file_path: str, min_cluster_size: int = 2, 
#                     min_samples: int = 1, cluster_selection_epsilon: float = 0.1,
#                     use_cosine: bool = True, visualize: bool = True, 
#                     output_path: str = "clustering_results.json"):
#         """
#         运行完整的聚类管道
        
#         Args:
#             file_path: 输入文件路径
#             min_cluster_size: 最小簇大小
#             min_samples: 最小样本数
#             cluster_selection_epsilon: 聚类选择epsilon
#             use_cosine: 是否使用余弦距离
#             visualize: 是否可视化结果
#             output_path: 输出文件路径
#         """
#         # 1. 加载数据
#         entities = self.load_entities_from_jsonl(file_path)
#         entity_names = self.extract_entity_names(entities)
        
#         # 2. 向量化
#         vectors = self.vectorize_entities(entity_names)
        
#         # 3. 聚类
#         cluster_labels = self.perform_clustering(
#             vectors, min_cluster_size, min_samples, cluster_selection_epsilon, use_cosine
#         )
        
#         # 4. 分析结果
#         clusters = self.analyze_clusters(entity_names, cluster_labels)
#         representatives = self.find_cluster_representatives(clusters, entity_names, vectors)
#         alias_mapping = self.create_alias_mapping(clusters, representatives)
        
#         # 5. 可视化
#         if visualize:
#             # 使用两种降维方法
#             self.visualize_clusters(vectors, cluster_labels, entity_names, method='tsne')
#             # self.visualize_clusters(vectors, cluster_labels, entity_names, method='pca')
        
#         # 6. 保存结果
#         self.save_results(alias_mapping, clusters, representatives, output_path)
        
#         # 7. 打印统计信息
#         self.print_statistics(clusters, alias_mapping)
        
#         return alias_mapping, clusters, representatives
    
#     def print_statistics(self, clusters: Dict[int, List[str]], alias_mapping: Dict[str, str]):
#         """
#         打印统计信息
#         """
#         print("\n=== 聚类统计信息 ===")
#         print(f"总实体数: {len(alias_mapping)}")
#         print(f"簇数量: {len([k for k in clusters.keys() if k != -1])}")
#         print(f"噪声点数: {len(clusters.get(-1, []))}")
#         print(f"唯一代表实体数: {len(set(alias_mapping.values()))}")
#         print(f"压缩率: {len(set(alias_mapping.values())) / len(alias_mapping):.2%}")
        
#         # 显示簇大小分布
#         cluster_sizes = [len(v) for k, v in clusters.items() if k != -1]
#         if cluster_sizes:
#             print(f"簇大小统计:")
#             print(f"  - 平均大小: {np.mean(cluster_sizes):.2f}")
#             print(f"  - 中位数大小: {np.median(cluster_sizes):.2f}")
#             print(f"  - 最大簇大小: {max(cluster_sizes)}")
#             print(f"  - 最小簇大小: {min(cluster_sizes)}")
        
#         # 显示一些示例簇
#         print("\n=== 示例聚类结果 ===")
#         cluster_sizes = [(k, len(v)) for k, v in clusters.items() if k != -1]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         for i, (cluster_id, size) in enumerate(cluster_sizes[:5]):
#             entities = clusters[cluster_id]
#             print(f"簇 {cluster_id} (大小: {size}):")
#             print(f"  实体: {entities}")
#             print()

# # 使用示例
# if __name__ == "__main__":
#     # 创建聚类管道
#     pipeline = EntityClusteringPipeline()
    
#     # 运行聚类
#     file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA.jsonl"
    
#     alias_mapping, clusters, representatives = pipeline.run_pipeline(
#         file_path=file_path,
#         min_cluster_size=2,
#         min_samples=1,
#         cluster_selection_epsilon=0.08,
#         use_cosine=True,
#         visualize=True,
#         output_path="entity_clustering_results_without_chunk_info.json"
#     )




















# with chunk info
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Any, Optional, Tuple
# from langchain_core.embeddings import Embeddings
# import hdbscan
# from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
# import pandas as pd
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import normalize
# import warnings
# from Qwen3_Reranker import Qwen3Reranker
# import chromadb
# warnings.filterwarnings('ignore')

# # 封装为 LangChain 所需的 Embeddings 接口
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", device: str = "cuda:1"):
#         self.model = SentenceTransformer(model_name_or_path, device=device)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode(text, convert_to_numpy=True, prompt = "Instruct: Given a natural language question, retrieve the most relevant text passages from a collection of documents to support answer generation.\nQuery:").tolist()

#     def __call__(self, text: str) -> List[float]:
#         return self.embed_query(text)

# class EntityClusteringPipeline:
#     def __init__(self, embedding_model_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", 
#                  reranker_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B",
#                  vector_db_path: str = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/vector_data",
#                  collection_name: str = "edu_chunks",
#                  device: str = "cuda:1"):
#         """
#         初始化实体聚类管道
        
#         Args:
#             embedding_model_path: 嵌入模型路径
#             reranker_path: 重排序模型路径
#             vector_db_path: 向量数据库路径
#             collection_name: 向量数据库集合名称
#             device: 设备（如：cuda:1）
#         """
#         self.embeddings = SentenceTransformerEmbeddings(embedding_model_path, device)
#         self.reranker_path = reranker_path
#         self.reranker = Qwen3Reranker(
#             model_name_or_path=self.reranker_path,
#             max_length=2048,
#             instruction="Given the user query, retrieve the relevant passages",
#             device_id="cuda:7" 
#         )
        
#         # 初始化向量数据库
#         self.vector_db_path = vector_db_path
#         self.collection_name = collection_name
#         self.client = None
#         self.collection = None
#         self._init_vector_db()
        
#         self.entities = []
#         self.entity_vectors = None
#         self.clusters = None
#         self.cluster_labels = None
#         self.entity_enriched_texts = []  # 存储增强后的实体文本
        
#     def _init_vector_db(self):
#         """初始化向量数据库连接"""
#         try:
#             self.client = chromadb.PersistentClient(path=self.vector_db_path)
#             self.collection = self.client.get_collection(name=self.collection_name)
#             print(f"成功连接向量数据库: {self.collection_name}")
#         except Exception as e:
#             print(f"连接向量数据库失败: {e}")
#             self.client = None
#             self.collection = None
    
#     def load_entities_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
#         """
#         从JSONL文件加载实体数据
        
#         Args:
#             file_path: JSONL文件路径
            
#         Returns:
#             实体列表
#         """
#         entities = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     entity = json.loads(line.strip())
#                     entities.append(entity)
#                 except json.JSONDecodeError as e:
#                     print(f"解析行时出错: {e}")
#                     continue
        
#         print(f"成功加载 {len(entities)} 个实体")
#         return entities
    
#     def extract_entity_data(self, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Dict]]]:
#         """
#         提取实体名称和位置信息
        
#         Args:
#             entities: 实体列表
            
#         Returns:
#             实体名称列表和位置信息列表
#         """
#         entity_names = []
#         entity_positions = []
        
#         for entity in entities:
#             if 'name' in entity:
#                 entity_names.append(entity['name'])
#                 # 获取位置信息，如果不存在则设为空列表
#                 positions = entity.get('positions', [])
#                 entity_positions.append(positions)
        
#         print(f"提取到 {len(entity_names)} 个实体名称")
#         return entity_names, entity_positions
    
#     def _rerank_documents(self, query: str, documents: List[str], batch_size: int = 25) -> List[Tuple[str, float]]:
#         """
#         使用重排序模型对文档进行重排序
        
#         Args:
#             query: 查询文本
#             documents: 文档列表
#             batch_size: 批处理大小
            
#         Returns:
#             排序后的文档列表，包含文档内容和得分
#         """
#         if not documents:
#             return []
        
#         scored_docs = []
#         num_docs = len(documents)
        
#         for start_idx in range(0, num_docs, batch_size):
#             end_idx = min(start_idx + batch_size, num_docs)
#             batch_docs = documents[start_idx:end_idx]
            
#             # 创建查询-文档对
#             pairs = [(query, doc) for doc in batch_docs]
            
#             try:
#                 # 计算重排序得分
#                 scores = self.reranker.compute_scores(pairs)
#                 scored_docs.extend(zip(batch_docs, scores))
#             except Exception as e:
#                 print(f"重排序处理批次 {start_idx}-{end_idx} 时出错: {e}")
#                 # 如果重排序失败，给予默认得分
#                 scored_docs.extend(zip(batch_docs, [0.0] * len(batch_docs)))
        
#         # 按得分降序排列
#         scored_docs.sort(key=lambda x: x[1], reverse=True)
#         return scored_docs
    
#     def _get_entity_definition_chunks(self, entity_name: str, positions: List[Dict], 
#                                     top_k: int = 3, score_threshold: float = 0.0) -> List[str]:
#         """
#         根据实体位置信息获取定义相关的chunk
        
#         Args:
#             entity_name: 实体名称
#             positions: 位置信息列表
#             top_k: 返回前k个最相关的chunk
#             score_threshold: 得分阈值，低于此值的chunk将被过滤
            
#         Returns:
#             定义相关的chunk列表
#         """
#         if not positions or self.collection is None:
#             return []
        
#         # 构建查询
#         query = f"实体名{entity_name}的定义是什么？"
        
#         # 从positions中提取chunk ID
#         chunk_ids = []
#         for position in positions:
#             if all(key in position for key in ['art_id', 'par_id', 'sen_id']):
#                 chunk_id = str(position['art_id']) + '-' + str(position['par_id']) + '-' + str(position['sen_id'])
#                 chunk_ids.append(chunk_id)
        
#         if not chunk_ids:
#             return []
        
#         try:
#             # 从向量数据库获取chunk内容
#             result = self.collection.get(ids=chunk_ids)
#             documents = result.get('documents', [])
            
#             if not documents:
#                 return []
            
#             # 使用重排序模型找出最相关的定义chunk
#             scored_docs = self._rerank_documents(query, documents)
            
#             # 过滤低分chunk并返回top_k
#             filtered_docs = [doc for doc, score in scored_docs if score >= score_threshold]
#             return filtered_docs[:top_k]
            
#         except Exception as e:
#             print(f"获取实体 {entity_name} 的定义chunk时出错: {e}")
#             return []
    
#     def preprocess_entity_names(self, entity_names: List[str], 
#                               entity_positions: List[List[Dict]], 
#                               max_definition_length: int = 1024) -> List[str]:
#         """
#         预处理实体名称，添加定义信息以增强向量化效果
        
#         Args:
#             entity_names: 实体名称列表
#             entity_positions: 实体位置信息列表
#             max_definition_length: 定义文本的最大长度
            
#         Returns:
#             预处理后的实体文本列表（实体名称 + 定义信息）
#         """
#         print("开始预处理实体名称，添加定义信息...")
        
#         enriched_texts = []
        
#         for i, entity_name in enumerate(entity_names):
#             # 获取对应的位置信息
#             positions = entity_positions[i] if i < len(entity_positions) else []
            
#             # 获取定义相关的chunk
#             definition_chunks = self._get_entity_definition_chunks(entity_name, positions)
            
#             # 构建增强文本
#             if definition_chunks:
#                 # 合并定义chunks并限制长度
#                 definition_text = " ".join(definition_chunks)
#                 if len(definition_text) > max_definition_length:
#                     definition_text = definition_text[:max_definition_length] + "..."
                
#                 # 结合实体名称和定义
#                 enriched_text = f"{entity_name} 定义: {definition_text}"
#             else:
#                 # 如果没有找到定义chunk，只使用实体名称
#                 enriched_text = entity_name
            
#             enriched_texts.append(enriched_text)
            
#             # 进度显示
#             if (i + 1) % 100 == 0:
#                 print(f"已处理 {i + 1}/{len(entity_names)} 个实体")
        
#         print(f"预处理完成，共处理 {len(enriched_texts)} 个实体")
        
#         # 保存增强后的文本以供后续分析
#         self.entity_enriched_texts = enriched_texts
        
#         return enriched_texts
    
#     def vectorize_entities(self, entity_texts: List[str]) -> np.ndarray:
#         """
#         对实体文本进行向量化
        
#         Args:
#             entity_texts: 实体文本列表（可能包含定义信息）
            
#         Returns:
#             向量化结果
#         """
#         print("开始向量化实体文本...")
        
#         batch_size = 100
#         all_vectors = []
        
#         for i in range(0, len(entity_texts), batch_size):
#             batch = entity_texts[i:i+batch_size]
#             vectors = self.embeddings.embed_documents(batch)
#             all_vectors.extend(vectors)
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"已处理 {i + len(batch)} / {len(entity_texts)} 个实体")
        
#         vectors_array = np.array(all_vectors)
#         print(f"向量化完成，形状: {vectors_array.shape}")
#         return vectors_array
    
#     def perform_clustering(self, vectors: np.ndarray, min_cluster_size: int = 2, 
#                           min_samples: int = 1, cluster_selection_epsilon: float = 0.1,
#                           use_cosine: bool = True) -> np.ndarray:
#         """
#         使用HDBSCAN进行聚类
        
#         Args:
#             vectors: 向量矩阵
#             min_cluster_size: 最小簇大小
#             min_samples: 最小样本数
#             cluster_selection_epsilon: 聚类选择epsilon
#             use_cosine: 是否使用余弦距离
            
#         Returns:
#             聚类标签
#         """
#         print("开始HDBSCAN聚类...")
        
#         if use_cosine:
#             print("计算余弦距离矩阵...")
            
#             # 先对向量进行L2归一化，确保数值稳定性
#             normalized_vectors = normalize(vectors, norm='l2')
            
#             # 计算余弦距离矩阵
#             distance_matrix = cosine_distances(normalized_vectors)
            
#             # 使用预计算的距离矩阵
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 cluster_selection_epsilon=cluster_selection_epsilon,
#                 metric='precomputed'
#             )
#             cluster_labels = clusterer.fit_predict(distance_matrix)
#             print("使用余弦距离矩阵聚类成功")
#         else:
#             print("使用欧几里得距离进行聚类...")
            
#             # 先对向量进行L2归一化，使欧几里得距离近似余弦距离
#             normalized_vectors = normalize(vectors, norm='l2')
            
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 cluster_selection_epsilon=cluster_selection_epsilon,
#                 metric='euclidean'
#             )
#             cluster_labels = clusterer.fit_predict(normalized_vectors)
#             print("使用L2归一化向量+欧几里得距离聚类成功")
        
#         # 统计聚类结果
#         n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
#         n_noise = list(cluster_labels).count(-1)
        
#         print(f"聚类完成:")
#         print(f"  - 簇数量: {n_clusters}")
#         print(f"  - 噪声点数量: {n_noise}")
#         print(f"  - 噪声点比例: {n_noise/len(cluster_labels):.2%}")
        
#         return cluster_labels
    
#     def analyze_clusters(self, entity_names: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
#         """
#         分析聚类结果
        
#         Args:
#             entity_names: 实体名称列表
#             cluster_labels: 聚类标签
            
#         Returns:
#             聚类结果字典
#         """
#         clusters = defaultdict(list)
        
#         for name, label in zip(entity_names, cluster_labels):
#             clusters[label].append(name)
        
#         return dict(clusters)
    
#     def find_cluster_representatives(self, clusters: Dict[int, List[str]], 
#                                    entity_names: List[str], vectors: np.ndarray) -> Dict[int, str]:
#         """
#         为每个簇找到代表性实体（最接近簇中心的实体）
        
#         Args:
#             clusters: 聚类结果
#             entity_names: 实体名称列表
#             vectors: 向量矩阵
            
#         Returns:
#             每个簇的代表实体
#         """
#         representatives = {}
#         name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if cluster_id == -1:  # 跳过噪声点
#                 continue
                
#             if len(cluster_entities) == 1:
#                 representatives[cluster_id] = cluster_entities[0]
#                 continue
            
#             # 获取簇中所有实体的向量
#             cluster_indices = [name_to_idx[name] for name in cluster_entities]
#             cluster_vectors = vectors[cluster_indices]
            
#             # 计算簇中心
#             cluster_center = np.mean(cluster_vectors, axis=0)
            
#             # 找到最接近中心的实体
#             similarities = cosine_similarity([cluster_center], cluster_vectors)[0]
#             best_idx = np.argmax(similarities)
#             representatives[cluster_id] = cluster_entities[best_idx]
        
#         return representatives
    
#     def create_alias_mapping(self, clusters: Dict[int, List[str]], 
#                            representatives: Dict[int, str]) -> Dict[str, str]:
#         """
#         创建别称映射
        
#         Args:
#             clusters: 聚类结果
#             representatives: 簇代表实体
            
#         Returns:
#             别称映射字典
#         """
#         alias_mapping = {}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if cluster_id == -1:  # 噪声点保持原样
#                 for entity in cluster_entities:
#                     alias_mapping[entity] = entity
#                 continue
            
#             representative = representatives[cluster_id]
#             for entity in cluster_entities:
#                 alias_mapping[entity] = representative
        
#         return alias_mapping
    
#     def visualize_clusters(self, vectors: np.ndarray, cluster_labels: np.ndarray, 
#                           entity_names: List[str], method: str = 'tsne', 
#                           sample_size: int = 1000, figsize: tuple = (15, 10)):
#         """
#         可视化聚类结果
        
#         Args:
#             vectors: 向量矩阵
#             cluster_labels: 聚类标签
#             entity_names: 实体名称列表
#             method: 降维方法 ('pca' 或 'tsne')
#             sample_size: 采样大小（处理大数据集）
#             figsize: 图形大小
#         """
#         # 设置中文字体
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
        
#         # 如果数据量太大，进行采样
#         if len(vectors) > sample_size:
#             np.random.seed(42)
#             indices = np.random.choice(len(vectors), sample_size, replace=False)
#             vectors_sample = vectors[indices]
#             labels_sample = cluster_labels[indices]
#             names_sample = [entity_names[i] for i in indices]
#         else:
#             vectors_sample = vectors
#             labels_sample = cluster_labels
#             names_sample = entity_names
        
#         # 对向量进行归一化
#         vectors_normalized = normalize(vectors_sample, norm='l2')
        
#         # 降维
#         print(f"正在进行{method.upper()}降维...")
#         if method == 'pca':
#             reducer = PCA(n_components=2, random_state=42)
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         elif method == 'tsne':
#             reducer = TSNE(n_components=2, random_state=42, 
#                          perplexity=min(30, len(vectors_normalized)-1), 
#                          metric='cosine')
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         else:
#             raise ValueError("method must be 'pca' or 'tsne'")
        
#         # 创建子图
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
#         # 左图：所有点的散点图
#         unique_labels = sorted(set(labels_sample))
#         n_clusters = len([l for l in unique_labels if l != -1])
        
#         # 使用更好的颜色方案
#         if n_clusters <= 10:
#             colors = plt.cm.tab10(np.linspace(0, 1, 10))
#         else:
#             colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
#         # 为噪声点设置特殊颜色
#         color_map = {}
#         color_idx = 0
#         for label in unique_labels:
#             if label == -1:
#                 color_map[label] = 'gray'
#             else:
#                 color_map[label] = colors[color_idx % len(colors)]
#                 color_idx += 1
        
#         # 绘制所有点
#         for label in unique_labels:
#             mask = labels_sample == label
#             if label == -1:  # 噪声点
#                 ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                            c='gray', marker='x', alpha=0.5, s=20, label='Noise')
#             else:
#                 ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                            c=[color_map[label]], marker='o', alpha=0.7, s=30, 
#                            label=f'Cluster {label}')
        
#         ax1.set_title(f'All Clusters ({method.upper()})', fontsize=14)
#         ax1.set_xlabel('Component 1', fontsize=12)
#         ax1.set_ylabel('Component 2', fontsize=12)
#         ax1.grid(True, alpha=0.3)
        
#         # 添加图例（只显示前15个）
#         handles, labels = ax1.get_legend_handles_labels()
#         if len(handles) > 15:
#             ax1.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left')
#         else:
#             ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         # 右图：显示最大的几个簇并标注代表性实体
#         cluster_sizes = [(label, sum(labels_sample == label)) for label in unique_labels if label != -1]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         # 只显示前10个最大的簇
#         top_clusters = [label for label, size in cluster_sizes[:10]]
        
#         for label in top_clusters:
#             mask = labels_sample == label
#             cluster_points = vectors_2d[mask]
#             cluster_names = [names_sample[i] for i in range(len(names_sample)) if labels_sample[i] == label]
            
#             # 绘制簇
#             ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                        c=[color_map[label]], marker='o', alpha=0.7, s=40, 
#                        label=f'Cluster {label} ({len(cluster_names)})')
            
#             # 添加簇中心和代表实体标注
#             center_x, center_y = np.mean(cluster_points, axis=0)
            
#             # 找到最接近中心的点
#             distances = np.sum((cluster_points - [center_x, center_y])**2, axis=1)
#             closest_idx = np.argmin(distances)
            
#             # 标注代表实体
#             if len(cluster_names) > 1:  # 只对多元素簇标注
#                 representative = cluster_names[closest_idx]
#                 ax2.annotate(representative, 
#                            (cluster_points[closest_idx, 0], cluster_points[closest_idx, 1]),
#                            xytext=(5, 5), textcoords='offset points', 
#                            fontsize=8, alpha=0.8,
#                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
#         ax2.set_title(f'Top 10 Largest Clusters with Labels ({method.upper()})', fontsize=14)
#         ax2.set_xlabel('Component 1', fontsize=12)
#         ax2.set_ylabel('Component 2', fontsize=12)
#         ax2.grid(True, alpha=0.3)
#         ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         plt.tight_layout()
#         plt.savefig(f'entity_clustering_visualization_{method}.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # 打印可视化统计信息
#         print(f"\n可视化统计信息:")
#         print(f"  - 可视化样本数: {len(vectors_sample)}")
#         print(f"  - 降维方法: {method.upper()}")
#         print(f"  - 显示的簇数: {len(top_clusters)}")
        
#         if method == 'pca':
#             print(f"  - 主成分解释的方差比: {reducer.explained_variance_ratio_}")
#             print(f"  - 累计方差解释比: {np.sum(reducer.explained_variance_ratio_):.3f}")
    
#     def save_results(self, alias_mapping: Dict[str, str], clusters: Dict[int, List[str]], 
#                     representatives: Dict[int, str], output_path: str = "clustering_results.json"):
#         """
#         保存聚类结果
        
#         Args:
#             alias_mapping: 别称映射
#             clusters: 聚类结果
#             representatives: 簇代表实体
#             output_path: 输出文件路径
#         """
#         results = {
#             "alias_mapping": alias_mapping,
#             "clusters": {str(k): v for k, v in clusters.items()},
#             "representatives": {str(k): v for k, v in representatives.items()},
#             "statistics": {
#                 "total_entities": len(alias_mapping),
#                 "unique_representatives": len(set(alias_mapping.values())),
#                 "compression_ratio": len(set(alias_mapping.values())) / len(alias_mapping)
#             }
#         }
        
#         # 如果有增强文本，也保存一份样本
#         if self.entity_enriched_texts:
#             results["sample_enriched_texts"] = self.entity_enriched_texts[:10]  # 保存前10个样本
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
        
#         print(f"结果已保存到: {output_path}")
    
#     def run_pipeline(self, file_path: str, min_cluster_size: int = 2, 
#                     min_samples: int = 1, cluster_selection_epsilon: float = 0.1,
#                     use_cosine: bool = True, visualize: bool = True, 
#                     output_path: str = "clustering_results.json"):
#         """
#         运行完整的聚类管道
        
#         Args:
#             file_path: 输入文件路径
#             min_cluster_size: 最小簇大小
#             min_samples: 最小样本数
#             cluster_selection_epsilon: 聚类选择epsilon
#             use_cosine: 是否使用余弦距离
#             visualize: 是否可视化结果
#             output_path: 输出文件路径
#         """
#         # 1. 加载数据
#         entities = self.load_entities_from_jsonl(file_path)
#         entity_names, entity_positions = self.extract_entity_data(entities)
        
#         # 2. 预处理实体名称，添加定义信息
#         enriched_entity_texts = self.preprocess_entity_names(entity_names, entity_positions)
        
#         # 3. 向量化增强后的实体文本
#         vectors = self.vectorize_entities(enriched_entity_texts)
        
#         # 4. 聚类
#         cluster_labels = self.perform_clustering(
#             vectors, min_cluster_size, min_samples, cluster_selection_epsilon, use_cosine
#         )
        
#         # 5. 分析结果（仍使用原始实体名称进行分析）
#         clusters = self.analyze_clusters(entity_names, cluster_labels)
#         representatives = self.find_cluster_representatives(clusters, entity_names, vectors)
#         alias_mapping = self.create_alias_mapping(clusters, representatives)
        
#         # 6. 可视化
#         if visualize:
#             self.visualize_clusters(vectors, cluster_labels, entity_names, method='tsne')
        
#         # 7. 保存结果
#         self.save_results(alias_mapping, clusters, representatives, output_path)
        
#         # 8. 打印统计信息
#         self.print_statistics(clusters, alias_mapping)
        
#         return alias_mapping, clusters, representatives
    
#     def print_statistics(self, clusters: Dict[int, List[str]], alias_mapping: Dict[str, str]):
#         """
#         打印统计信息
#         """
#         print("\n=== 聚类统计信息 ===")
#         print(f"总实体数: {len(alias_mapping)}")
#         print(f"簇数量: {len([k for k in clusters.keys() if k != -1])}")
#         print(f"噪声点数: {len(clusters.get(-1, []))}")
#         print(f"唯一代表实体数: {len(set(alias_mapping.values()))}")
#         print(f"压缩率: {len(set(alias_mapping.values())) / len(alias_mapping):.2%}")
        
#         # 显示簇大小分布
#         cluster_sizes = [len(v) for k, v in clusters.items() if k != -1]
#         if cluster_sizes:
#             print(f"簇大小统计:")
#             print(f"  - 平均大小: {np.mean(cluster_sizes):.2f}")
#             print(f"  - 中位数大小: {np.median(cluster_sizes):.2f}")
#             print(f"  - 最大簇大小: {max(cluster_sizes)}")
#             print(f"  - 最小簇大小: {min(cluster_sizes)}")
        
#         # 显示一些示例簇
#         print("\n=== 示例聚类结果 ===")
#         cluster_sizes = [(k, len(v)) for k, v in clusters.items() if k != -1]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         for i, (cluster_id, size) in enumerate(cluster_sizes[:5]):
#             entities = clusters[cluster_id]
#             print(f"簇 {cluster_id} (大小: {size}):")
#             print(f"  实体: {entities}")
#             print()
    
#     def analyze_enriched_entities(self, entity_names: List[str], 
#                                 entity_positions: List[List[Dict]], 
#                                 output_path: str = "entity_analysis.json"):
#         """
#         分析增强实体的效果
        
#         Args:
#             entity_names: 原始实体名称列表
#             entity_positions: 实体位置信息列表
#             output_path: 分析结果输出路径
#         """
#         print("分析实体增强效果...")
        
#         analysis_results = {
#             "total_entities": len(entity_names),
#             "entities_with_positions": 0,
#             "entities_with_definitions": 0,
#             "average_definition_length": 0,
#             "sample_enriched_entities": []
#         }
        
#         total_definition_length = 0
#         entities_with_definitions = 0
        
#         for i, entity_name in enumerate(entity_names[:100]):  # 分析前100个实体作为样本
#             positions = entity_positions[i] if i < len(entity_positions) else []
            
#             if positions:
#                 analysis_results["entities_with_positions"] += 1
                
#                 # 获取定义chunks
#                 definition_chunks = self._get_entity_definition_chunks(entity_name, positions)
                
#                 if definition_chunks:
#                     entities_with_definitions += 1
#                     definition_text = " ".join(definition_chunks)
#                     total_definition_length += len(definition_text)
                    
#                     # 添加样本
#                     if len(analysis_results["sample_enriched_entities"]) < 10:
#                         analysis_results["sample_enriched_entities"].append({
#                             "entity_name": entity_name,
#                             "definition_preview": definition_text[:200] + "..." if len(definition_text) > 200 else definition_text,
#                             "definition_length": len(definition_text),
#                             "num_chunks": len(definition_chunks)
#                         })
        
#         # 更新统计信息
#         analysis_results["entities_with_definitions"] = entities_with_definitions
#         if entities_with_definitions > 0:
#             analysis_results["average_definition_length"] = total_definition_length / entities_with_definitions
        
#         # 保存分析结果
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
#         # 打印分析结果
#         print(f"\n=== 实体增强分析结果 ===")
#         print(f"总实体数（样本）: {analysis_results['total_entities']}")
#         print(f"有位置信息的实体数: {analysis_results['entities_with_positions']}")
#         print(f"有定义信息的实体数: {analysis_results['entities_with_definitions']}")
#         print(f"平均定义长度: {analysis_results['average_definition_length']:.1f} 字符")
#         print(f"定义覆盖率: {analysis_results['entities_with_definitions']/analysis_results['total_entities']:.2%}")
        
#         return analysis_results
    
#     def compare_clustering_results(self, entity_names: List[str], 
#                                  entity_positions: List[List[Dict]],
#                                  clustering_params: Dict = None):
#         """
#         比较使用和不使用定义信息的聚类效果
        
#         Args:
#             entity_names: 实体名称列表
#             entity_positions: 实体位置信息列表
#             clustering_params: 聚类参数
#         """
#         if clustering_params is None:
#             clustering_params = {
#                 "min_cluster_size": 2,
#                 "min_samples": 1,
#                 "cluster_selection_epsilon": 0.1,
#                 "use_cosine": True
#             }
        
#         print("比较聚类效果：使用 vs 不使用定义信息")
        
#         # 1. 不使用定义信息的聚类
#         print("\n--- 基准聚类（仅实体名称）---")
#         vectors_baseline = self.vectorize_entities(entity_names)
#         labels_baseline = self.perform_clustering(vectors_baseline, **clustering_params)
#         clusters_baseline = self.analyze_clusters(entity_names, labels_baseline)
        
#         # 2. 使用定义信息的聚类
#         print("\n--- 增强聚类（实体名称 + 定义）---")
#         enriched_texts = self.preprocess_entity_names(entity_names, entity_positions)
#         vectors_enriched = self.vectorize_entities(enriched_texts)
#         labels_enriched = self.perform_clustering(vectors_enriched, **clustering_params)
#         clusters_enriched = self.analyze_clusters(entity_names, labels_enriched)
        
#         # 3. 比较结果
#         comparison_results = {
#             "baseline": {
#                 "num_clusters": len([k for k in clusters_baseline.keys() if k != -1]),
#                 "num_noise": len(clusters_baseline.get(-1, [])),
#                 "noise_ratio": len(clusters_baseline.get(-1, [])) / len(entity_names),
#                 "largest_cluster_size": max([len(v) for k, v in clusters_baseline.items() if k != -1], default=0)
#             },
#             "enriched": {
#                 "num_clusters": len([k for k in clusters_enriched.keys() if k != -1]),
#                 "num_noise": len(clusters_enriched.get(-1, [])),
#                 "noise_ratio": len(clusters_enriched.get(-1, [])) / len(entity_names),
#                 "largest_cluster_size": max([len(v) for k, v in clusters_enriched.items() if k != -1], default=0)
#             }
#         }
        
#         print(f"\n=== 聚类效果比较 ===")
#         print(f"基准方法（仅实体名称）:")
#         print(f"  - 簇数量: {comparison_results['baseline']['num_clusters']}")
#         print(f"  - 噪声点数: {comparison_results['baseline']['num_noise']}")
#         print(f"  - 噪声比例: {comparison_results['baseline']['noise_ratio']:.2%}")
#         print(f"  - 最大簇大小: {comparison_results['baseline']['largest_cluster_size']}")
        
#         print(f"\n增强方法（实体名称 + 定义）:")
#         print(f"  - 簇数量: {comparison_results['enriched']['num_clusters']}")
#         print(f"  - 噪声点数: {comparison_results['enriched']['num_noise']}")
#         print(f"  - 噪声比例: {comparison_results['enriched']['noise_ratio']:.2%}")
#         print(f"  - 最大簇大小: {comparison_results['enriched']['largest_cluster_size']}")
        
#         # 保存比较结果
#         with open("clustering_comparison.json", 'w', encoding='utf-8') as f:
#             json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
#         return comparison_results, clusters_baseline, clusters_enriched

# # 使用示例
# if __name__ == "__main__":
#     # 创建聚类管道
#     pipeline = EntityClusteringPipeline(
#         embedding_model_path="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_4B",
#         reranker_path="/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_4B",
#         vector_db_path="/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/vector_data",
#         collection_name="edu_chunks",
#         device="cuda:1"
#     )
    
#     # 运行聚类
#     file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA.jsonl"
    
#     # 选项1：运行完整管道（推荐）
#     alias_mapping, clusters, representatives = pipeline.run_pipeline(
#         file_path=file_path,
#         min_cluster_size=2,
#         min_samples=1,
#         cluster_selection_epsilon=0.1,
#         use_cosine=True,
#         visualize=True,
#         output_path="entity_clustering_results.json"
#     )

















# # with chunk info avg embedding

# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Any, Optional, Tuple
# from langchain_core.embeddings import Embeddings
# import hdbscan
# from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
# import pandas as pd
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import normalize
# import warnings
# from Qwen3_Reranker import Qwen3Reranker
# import chromadb
# warnings.filterwarnings('ignore')

# # 封装为 LangChain 所需的 Embeddings 接口
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", device: str = "cuda:1"):
#         self.model = SentenceTransformer(model_name_or_path, device=device)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode(text, convert_to_numpy=True, prompt = "Instruct: Given a natural language question, retrieve the most relevant text passages from a collection of documents to support answer generation.\nQuery:").tolist()

#     def __call__(self, text: str) -> List[float]:
#         return self.embed_query(text)

# class EntityClusteringPipeline:
#     def __init__(self, embedding_model_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", 
#                  reranker_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_0.6B",
#                  vector_db_path: str = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/vector_data",
#                  collection_name: str = "edu_chunks",
#                  device: str = "cuda:1"):
#         """
#         初始化实体聚类管道
        
#         Args:
#             embedding_model_path: 嵌入模型路径
#             reranker_path: 重排序模型路径
#             vector_db_path: 向量数据库路径
#             collection_name: 向量数据库集合名称
#             device: 设备（如：cuda:1）
#         """
#         self.embeddings = SentenceTransformerEmbeddings(embedding_model_path, device)
#         self.reranker_path = reranker_path
#         self.reranker = Qwen3Reranker(
#             model_name_or_path=self.reranker_path,
#             max_length=2048,
#             instruction="Given the user query, retrieve the relevant passages",
#             device_id="cuda:7" 
#         )
        
#         # 初始化向量数据库
#         self.vector_db_path = vector_db_path
#         self.collection_name = collection_name
#         self.client = None
#         self.collection = None
#         self._init_vector_db()
        
#         self.entities = []
#         self.entity_vectors = None
#         self.clusters = None
#         self.cluster_labels = None
#         self.entity_definition_chunks = []  # 存储每个实体的定义chunks
        
#     def _init_vector_db(self):
#         """初始化向量数据库连接"""
#         try:
#             self.client = chromadb.PersistentClient(path=self.vector_db_path)
#             self.collection = self.client.get_collection(name=self.collection_name)
#             print(f"成功连接向量数据库: {self.collection_name}")
#         except Exception as e:
#             print(f"连接向量数据库失败: {e}")
#             self.client = None
#             self.collection = None
    
#     def load_entities_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
#         """
#         从JSONL文件加载实体数据
        
#         Args:
#             file_path: JSONL文件路径
            
#         Returns:
#             实体列表
#         """
#         entities = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     entity = json.loads(line.strip())
#                     entities.append(entity)
#                 except json.JSONDecodeError as e:
#                     print(f"解析行时出错: {e}")
#                     continue
        
#         print(f"成功加载 {len(entities)} 个实体")
#         return entities
    
#     def extract_entity_data(self, entities: List[Dict[str, Any]]) -> Tuple[List[str], List[List[Dict]]]:
#         """
#         提取实体名称和位置信息
        
#         Args:
#             entities: 实体列表
            
#         Returns:
#             实体名称列表和位置信息列表
#         """
#         entity_names = []
#         entity_positions = []
        
#         for entity in entities:
#             if 'name' in entity:
#                 entity_names.append(entity['name'])
#                 # 获取位置信息，如果不存在则设为空列表
#                 positions = entity.get('positions', [])
#                 entity_positions.append(positions)
        
#         print(f"提取到 {len(entity_names)} 个实体名称")
#         return entity_names, entity_positions
    
#     def _rerank_documents(self, query: str, documents: List[str], batch_size: int = 25) -> List[Tuple[str, float]]:
#         """
#         使用重排序模型对文档进行重排序
        
#         Args:
#             query: 查询文本
#             documents: 文档列表
#             batch_size: 批处理大小
            
#         Returns:
#             排序后的文档列表，包含文档内容和得分
#         """
#         if not documents:
#             return []
        
#         scored_docs = []
#         num_docs = len(documents)
        
#         for start_idx in range(0, num_docs, batch_size):
#             end_idx = min(start_idx + batch_size, num_docs)
#             batch_docs = documents[start_idx:end_idx]
            
#             # 创建查询-文档对
#             pairs = [(query, doc) for doc in batch_docs]
            
#             try:
#                 # 计算重排序得分
#                 scores = self.reranker.compute_scores(pairs)
#                 scored_docs.extend(zip(batch_docs, scores))
#             except Exception as e:
#                 print(f"重排序处理批次 {start_idx}-{end_idx} 时出错: {e}")
#                 # 如果重排序失败，给予默认得分
#                 scored_docs.extend(zip(batch_docs, [0.0] * len(batch_docs)))
        
#         # 按得分降序排列
#         scored_docs.sort(key=lambda x: x[1], reverse=True)
#         return scored_docs
    
#     def _get_entity_definition_chunks(self, entity_name: str, positions: List[Dict], 
#                                     top_k: int = 3, score_threshold: float = 0.0) -> List[str]:
#         """
#         根据实体位置信息获取定义相关的chunk
        
#         Args:
#             entity_name: 实体名称
#             positions: 位置信息列表
#             top_k: 返回前k个最相关的chunk
#             score_threshold: 得分阈值，低于此值的chunk将被过滤
            
#         Returns:
#             定义相关的chunk列表
#         """
#         if not positions or self.collection is None:
#             return []
        
#         # 构建查询
#         query = f"实体名{entity_name}的定义是什么？"
        
#         # 从positions中提取chunk ID
#         chunk_ids = []
#         for position in positions:
#             if all(key in position for key in ['art_id', 'par_id', 'sen_id']):
#                 chunk_id = str(position['art_id']) + '-' + str(position['par_id']) + '-' + str(position['sen_id'])
#                 chunk_ids.append(chunk_id)
        
#         if not chunk_ids:
#             return []
        
#         try:
#             # 从向量数据库获取chunk内容
#             result = self.collection.get(ids=chunk_ids)
#             documents = result.get('documents', [])
            
#             if not documents:
#                 return []
            
#             # 使用重排序模型找出最相关的定义chunk
#             scored_docs = self._rerank_documents(query, documents)
            
#             # 过滤低分chunk并返回top_k
#             filtered_docs = [doc for doc, score in scored_docs if score >= score_threshold]
#             return filtered_docs[:top_k]
            
#         except Exception as e:
#             print(f"获取实体 {entity_name} 的定义chunk时出错: {e}")
#             return []
    
#     def prepare_entity_definition_pairs(self, entity_names: List[str], 
#                                       entity_positions: List[List[Dict]]) -> Tuple[List[str], List[List[str]]]:
#         """
#         为每个实体准备定义chunks，用于分别向量化
        
#         Args:
#             entity_names: 实体名称列表
#             entity_positions: 实体位置信息列表
            
#         Returns:
#             实体名称列表和对应的定义chunks列表
#         """
#         print("开始准备实体和定义chunks...")
        
#         entity_definition_chunks = []
        
#         for i, entity_name in enumerate(entity_names):
#             # 获取对应的位置信息
#             positions = entity_positions[i] if i < len(entity_positions) else []
            
#             # 获取定义相关的chunk
#             definition_chunks = self._get_entity_definition_chunks(entity_name, positions)
#             entity_definition_chunks.append(definition_chunks)
            
#             # 进度显示
#             if (i + 1) % 100 == 0:
#                 print(f"已处理 {i + 1}/{len(entity_names)} 个实体")
        
#         print(f"准备完成，共处理 {len(entity_definition_chunks)} 个实体")
        
#         # 保存定义chunks以供后续分析
#         self.entity_definition_chunks = entity_definition_chunks
        
#         return entity_names, entity_definition_chunks
    
#     def vectorize_entities_with_definitions(self, entity_names: List[str], 
#                                           entity_definition_chunks: List[List[str]]) -> np.ndarray:
#         """
#         分别向量化实体名称和定义chunks，然后取平均值
        
#         Args:
#             entity_names: 实体名称列表
#             entity_definition_chunks: 每个实体对应的定义chunks列表
            
#         Returns:
#             平均向量化结果
#         """
#         print("开始分别向量化实体名称和定义chunks...")
        
#         # 1. 向量化实体名称
#         print("正在向量化实体名称...")
#         entity_vectors = []
#         batch_size = 100
        
#         for i in range(0, len(entity_names), batch_size):
#             batch = entity_names[i:i+batch_size]
#             vectors = self.embeddings.embed_documents(batch)
#             entity_vectors.extend(vectors)
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"已处理实体名称 {i + len(batch)} / {len(entity_names)}")
        
#         entity_vectors = np.array(entity_vectors)
#         print(f"实体名称向量化完成，形状: {entity_vectors.shape}")
        
#         # 2. 向量化定义chunks并计算平均值
#         print("正在向量化定义chunks...")
#         definition_vectors = []
        
#         for i, chunks in enumerate(entity_definition_chunks):
#             if chunks:
#                 # 如果有定义chunks，向量化所有chunks并取平均
#                 chunk_vectors = self.embeddings.embed_documents(chunks)
#                 chunk_vectors = np.array(chunk_vectors)
#                 # 计算定义chunks的平均向量
#                 avg_definition_vector = np.mean(chunk_vectors, axis=0)
#                 definition_vectors.append(avg_definition_vector)
#             else:
#                 # 如果没有定义chunks，使用零向量
#                 definition_vectors.append(np.zeros(entity_vectors.shape[1]))
            
#             # 进度显示
#             if (i + 1) % 100 == 0:
#                 print(f"已处理定义chunks {i + 1} / {len(entity_definition_chunks)}")
        
#         definition_vectors = np.array(definition_vectors)
#         print(f"定义chunks向量化完成，形状: {definition_vectors.shape}")
        
#         # 3. 计算实体向量和定义向量的平均值
#         print("正在计算实体向量和定义向量的平均值...")
        
#         # 统计有定义的实体数量
#         entities_with_definitions = sum(1 for chunks in entity_definition_chunks if chunks)
#         entities_without_definitions = len(entity_definition_chunks) - entities_with_definitions
        
#         print(f"有定义的实体数量: {entities_with_definitions}")
#         print(f"无定义的实体数量: {entities_without_definitions}")
        
#         # 对于有定义的实体，计算实体向量和定义向量的平均值
#         # 对于无定义的实体，只使用实体向量
#         combined_vectors = []
        
#         for i in range(len(entity_names)):
#             entity_vector = entity_vectors[i]
#             definition_vector = definition_vectors[i]
            
#             # 检查是否有有效的定义向量（非零向量）
#             if np.any(definition_vector != 0):
#                 # 有定义：计算实体向量和定义向量的平均值
#                 combined_vector = (entity_vector + definition_vector) / 2
#             else:
#                 # 无定义：只使用实体向量
#                 combined_vector = entity_vector
            
#             combined_vectors.append(combined_vector)
        
#         combined_vectors = np.array(combined_vectors)
#         print(f"向量合并完成，最终形状: {combined_vectors.shape}")
        
#         return combined_vectors
    
#     def perform_clustering(self, vectors: np.ndarray, min_cluster_size: int = 2, 
#                           min_samples: int = 1, cluster_selection_epsilon: float = 0.1,
#                           use_cosine: bool = True) -> np.ndarray:
#         """
#         使用HDBSCAN进行聚类
        
#         Args:
#             vectors: 向量矩阵
#             min_cluster_size: 最小簇大小
#             min_samples: 最小样本数
#             cluster_selection_epsilon: 聚类选择epsilon
#             use_cosine: 是否使用余弦距离
            
#         Returns:
#             聚类标签
#         """
#         print("开始HDBSCAN聚类...")
        
#         if use_cosine:
#             print("计算余弦距离矩阵...")
            
#             # 先对向量进行L2归一化，确保数值稳定性
#             normalized_vectors = normalize(vectors, norm='l2')
            
#             # 计算余弦距离矩阵
#             distance_matrix = cosine_distances(normalized_vectors)
            
#             # 使用预计算的距离矩阵
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 cluster_selection_epsilon=cluster_selection_epsilon,
#                 metric='precomputed'
#             )
#             cluster_labels = clusterer.fit_predict(distance_matrix)
#             print("使用余弦距离矩阵聚类成功")
#         else:
#             print("使用欧几里得距离进行聚类...")
            
#             # 先对向量进行L2归一化，使欧几里得距离近似余弦距离
#             normalized_vectors = normalize(vectors, norm='l2')
            
#             clusterer = hdbscan.HDBSCAN(
#                 min_cluster_size=min_cluster_size,
#                 min_samples=min_samples,
#                 cluster_selection_epsilon=cluster_selection_epsilon,
#                 metric='euclidean'
#             )
#             cluster_labels = clusterer.fit_predict(normalized_vectors)
#             print("使用L2归一化向量+欧几里得距离聚类成功")
        
#         # 统计聚类结果
#         n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
#         n_noise = list(cluster_labels).count(-1)
        
#         print(f"聚类完成:")
#         print(f"  - 簇数量: {n_clusters}")
#         print(f"  - 噪声点数量: {n_noise}")
#         print(f"  - 噪声点比例: {n_noise/len(cluster_labels):.2%}")
        
#         return cluster_labels
    
#     def analyze_clusters(self, entity_names: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
#         """
#         分析聚类结果
        
#         Args:
#             entity_names: 实体名称列表
#             cluster_labels: 聚类标签
            
#         Returns:
#             聚类结果字典
#         """
#         clusters = defaultdict(list)
        
#         for name, label in zip(entity_names, cluster_labels):
#             clusters[label].append(name)
        
#         return dict(clusters)
    
#     def find_cluster_representatives(self, clusters: Dict[int, List[str]], 
#                                    entity_names: List[str], vectors: np.ndarray) -> Dict[int, str]:
#         """
#         为每个簇找到代表性实体（最接近簇中心的实体）
        
#         Args:
#             clusters: 聚类结果
#             entity_names: 实体名称列表
#             vectors: 向量矩阵
            
#         Returns:
#             每个簇的代表实体
#         """
#         representatives = {}
#         name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if cluster_id == -1:  # 跳过噪声点
#                 continue
                
#             if len(cluster_entities) == 1:
#                 representatives[cluster_id] = cluster_entities[0]
#                 continue
            
#             # 获取簇中所有实体的向量
#             cluster_indices = [name_to_idx[name] for name in cluster_entities]
#             cluster_vectors = vectors[cluster_indices]
            
#             # 计算簇中心
#             cluster_center = np.mean(cluster_vectors, axis=0)
            
#             # 找到最接近中心的实体
#             similarities = cosine_similarity([cluster_center], cluster_vectors)[0]
#             best_idx = np.argmax(similarities)
#             representatives[cluster_id] = cluster_entities[best_idx]
        
#         return representatives
    
#     def create_alias_mapping(self, clusters: Dict[int, List[str]], 
#                            representatives: Dict[int, str]) -> Dict[str, str]:
#         """
#         创建别称映射
        
#         Args:
#             clusters: 聚类结果
#             representatives: 簇代表实体
            
#         Returns:
#             别称映射字典
#         """
#         alias_mapping = {}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if cluster_id == -1:  # 噪声点保持原样
#                 for entity in cluster_entities:
#                     alias_mapping[entity] = entity
#                 continue
            
#             representative = representatives[cluster_id]
#             for entity in cluster_entities:
#                 alias_mapping[entity] = representative
        
#         return alias_mapping
    
#     def visualize_clusters(self, vectors: np.ndarray, cluster_labels: np.ndarray, 
#                           entity_names: List[str], method: str = 'tsne', 
#                           sample_size: int = 1000, figsize: tuple = (15, 10)):
#         """
#         可视化聚类结果
        
#         Args:
#             vectors: 向量矩阵
#             cluster_labels: 聚类标签
#             entity_names: 实体名称列表
#             method: 降维方法 ('pca' 或 'tsne')
#             sample_size: 采样大小（处理大数据集）
#             figsize: 图形大小
#         """
#         # 设置中文字体
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
        
#         # 如果数据量太大，进行采样
#         if len(vectors) > sample_size:
#             np.random.seed(42)
#             indices = np.random.choice(len(vectors), sample_size, replace=False)
#             vectors_sample = vectors[indices]
#             labels_sample = cluster_labels[indices]
#             names_sample = [entity_names[i] for i in indices]
#         else:
#             vectors_sample = vectors
#             labels_sample = cluster_labels
#             names_sample = entity_names
        
#         # 对向量进行归一化
#         vectors_normalized = normalize(vectors_sample, norm='l2')
        
#         # 降维
#         print(f"正在进行{method.upper()}降维...")
#         if method == 'pca':
#             reducer = PCA(n_components=2, random_state=42)
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         elif method == 'tsne':
#             reducer = TSNE(n_components=2, random_state=42, 
#                          perplexity=min(30, len(vectors_normalized)-1), 
#                          metric='cosine')
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         else:
#             raise ValueError("method must be 'pca' or 'tsne'")
        
#         # 创建子图
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
#         # 左图：所有点的散点图
#         unique_labels = sorted(set(labels_sample))
#         n_clusters = len([l for l in unique_labels if l != -1])
        
#         # 使用更好的颜色方案
#         if n_clusters <= 10:
#             colors = plt.cm.tab10(np.linspace(0, 1, 10))
#         else:
#             colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
#         # 为噪声点设置特殊颜色
#         color_map = {}
#         color_idx = 0
#         for label in unique_labels:
#             if label == -1:
#                 color_map[label] = 'gray'
#             else:
#                 color_map[label] = colors[color_idx % len(colors)]
#                 color_idx += 1
        
#         # 绘制所有点
#         for label in unique_labels:
#             mask = labels_sample == label
#             if label == -1:  # 噪声点
#                 ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                            c='gray', marker='x', alpha=0.5, s=20, label='Noise')
#             else:
#                 ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                            c=[color_map[label]], marker='o', alpha=0.7, s=30, 
#                            label=f'Cluster {label}')
        
#         ax1.set_title(f'All Clusters ({method.upper()})', fontsize=14)
#         ax1.set_xlabel('Component 1', fontsize=12)
#         ax1.set_ylabel('Component 2', fontsize=12)
#         ax1.grid(True, alpha=0.3)
        
#         # 添加图例（只显示前15个）
#         handles, labels = ax1.get_legend_handles_labels()
#         if len(handles) > 15:
#             ax1.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left')
#         else:
#             ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         # 右图：显示最大的几个簇并标注代表性实体
#         cluster_sizes = [(label, sum(labels_sample == label)) for label in unique_labels if label != -1]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         # 只显示前10个最大的簇
#         top_clusters = [label for label, size in cluster_sizes[:10]]
        
#         for label in top_clusters:
#             mask = labels_sample == label
#             cluster_points = vectors_2d[mask]
#             cluster_names = [names_sample[i] for i in range(len(names_sample)) if labels_sample[i] == label]
            
#             # 绘制簇
#             ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                        c=[color_map[label]], marker='o', alpha=0.7, s=40, 
#                        label=f'Cluster {label} ({len(cluster_names)})')
            
#             # 添加簇中心和代表实体标注
#             center_x, center_y = np.mean(cluster_points, axis=0)
            
#             # 找到最接近中心的点
#             distances = np.sum((cluster_points - [center_x, center_y])**2, axis=1)
#             closest_idx = np.argmin(distances)
            
#             # 标注代表实体
#             if len(cluster_names) > 1:  # 只对多元素簇标注
#                 representative = cluster_names[closest_idx]
#                 ax2.annotate(representative, 
#                            (cluster_points[closest_idx, 0], cluster_points[closest_idx, 1]),
#                            xytext=(5, 5), textcoords='offset points', 
#                            fontsize=8, alpha=0.8,
#                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
#         ax2.set_title(f'Top 10 Largest Clusters with Labels ({method.upper()})', fontsize=14)
#         ax2.set_xlabel('Component 1', fontsize=12)
#         ax2.set_ylabel('Component 2', fontsize=12)
#         ax2.grid(True, alpha=0.3)
#         ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         plt.tight_layout()
#         plt.savefig(f'entity_clustering_visualization_{method}.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         # 打印可视化统计信息
#         print(f"\n可视化统计信息:")
#         print(f"  - 可视化样本数: {len(vectors_sample)}")
#         print(f"  - 降维方法: {method.upper()}")
#         print(f"  - 显示的簇数: {len(top_clusters)}")
        
#         if method == 'pca':
#             print(f"  - 主成分解释的方差比: {reducer.explained_variance_ratio_}")
#             print(f"  - 累计方差解释比: {np.sum(reducer.explained_variance_ratio_):.3f}")
    
#     def save_results(self, alias_mapping: Dict[str, str], clusters: Dict[int, List[str]], 
#                     representatives: Dict[int, str], output_path: str = "clustering_results.json"):
#         """
#         保存聚类结果
        
#         Args:
#             alias_mapping: 别称映射
#             clusters: 聚类结果
#             representatives: 簇代表实体
#             output_path: 输出文件路径
#         """
#         results = {
#             "alias_mapping": alias_mapping,
#             "clusters": {str(k): v for k, v in clusters.items()},
#             "representatives": {str(k): v for k, v in representatives.items()},
#             "statistics": {
#                 "total_entities": len(alias_mapping),
#                 "unique_representatives": len(set(alias_mapping.values())),
#                 "compression_ratio": len(set(alias_mapping.values())) / len(alias_mapping),
#                 "entities_with_definitions": sum(1 for chunks in self.entity_definition_chunks if chunks),
#                 "entities_without_definitions": sum(1 for chunks in self.entity_definition_chunks if not chunks)
#             }
#         }
        
#         # 如果有定义chunks，也保存一份样本
#         if self.entity_definition_chunks:
#             sample_info = []
#             for i, chunks in enumerate(self.entity_definition_chunks[:10]):  # 前10个样本
#                 sample_info.append({
#                     "entity_index": i,
#                     "num_definition_chunks": len(chunks),
#                     "definition_chunks_preview": chunks[:2] if chunks else []  # 只显示前2个chunk
#                 })
#             results["sample_definition_chunks"] = sample_info
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
        
#         print(f"结果已保存到: {output_path}")
    
#     def run_pipeline(self, file_path: str, min_cluster_size: int = 2, 
#                     min_samples: int = 1, cluster_selection_epsilon: float = 0.1,
#                     use_cosine: bool = True, visualize: bool = True, 
#                     output_path: str = "clustering_results.json"):
#         """
#         运行完整的聚类管道
        
#         Args:
#             file_path: 输入文件路径
#             min_cluster_size: 最小簇大小
#             min_samples: 最小样本数
#             cluster_selection_epsilon: 聚类选择epsilon
#             use_cosine: 是否使用余弦距离
#             visualize: 是否可视化结果
#             output_path: 输出文件路径
#         """
#         # 1. 加载数据
#         entities = self.load_entities_from_jsonl(file_path)
#         entity_names, entity_positions = self.extract_entity_data(entities)
        
#         # 2. 准备实体和定义chunks
#         entity_names, entity_definition_chunks = self.prepare_entity_definition_pairs(entity_names, entity_positions)
        
#         # 3. 分别向量化实体名称和定义chunks，然后取平均值
#         vectors = self.vectorize_entities_with_definitions(entity_names, entity_definition_chunks)
        
#         # 4. 聚类
#         cluster_labels = self.perform_clustering(
#             vectors, min_cluster_size, min_samples, cluster_selection_epsilon, use_cosine
#         )
        
#         # 5. 分析结果
#         clusters = self.analyze_clusters(entity_names, cluster_labels)
#         representatives = self.find_cluster_representatives(clusters, entity_names, vectors)
#         alias_mapping = self.create_alias_mapping(clusters, representatives)
        
#         # 6. 可视化
#         if visualize:
#             self.visualize_clusters(vectors, cluster_labels, entity_names, method='tsne')
        
#         # 7. 保存结果
#         self.save_results(alias_mapping, clusters, representatives, output_path)
        
#         # 8. 打印统计信息
#         self.print_statistics(clusters, alias_mapping)
        
#         return alias_mapping, clusters, representatives
    
#     def print_statistics(self, clusters: Dict[int, List[str]], alias_mapping: Dict[str, str]):
#         """
#         打印统计信息
#         """
#         print("\n=== 聚类统计信息 ===")
#         print(f"总实体数: {len(alias_mapping)}")
#         print(f"簇数量: {len([k for k in clusters.keys() if k != -1])}")
#         print(f"噪声点数: {len(clusters.get(-1, []))}")
#         print(f"唯一代表实体数: {len(set(alias_mapping.values()))}")
#         print(f"压缩率: {len(set(alias_mapping.values())) / len(alias_mapping):.2%}")
        
#         # 显示定义覆盖率
#         entities_with_definitions = sum(1 for chunks in self.entity_definition_chunks if chunks)
#         entities_without_definitions = len(self.entity_definition_chunks) - entities_with_definitions
#         print(f"有定义的实体数: {entities_with_definitions}")
#         print(f"无定义的实体数: {entities_without_definitions}")
#         print(f"定义覆盖率: {entities_with_definitions / len(self.entity_definition_chunks):.2%}")
        
#         # 显示簇大小分布
#         cluster_sizes = [len(v) for k, v in clusters.items() if k != -1]
#         if cluster_sizes:
#             print(f"簇大小统计:")
#             print(f"  - 平均大小: {np.mean(cluster_sizes):.2f}")
#             print(f"  - 中位数大小: {np.median(cluster_sizes):.2f}")
#             print(f"  - 最大簇大小: {max(cluster_sizes)}")
#             print(f"  - 最小簇大小: {min(cluster_sizes)}")
        
#         # 显示一些示例簇
#         print("\n=== 示例聚类结果 ===")
#         cluster_sizes = [(k, len(v)) for k, v in clusters.items() if k != -1]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         for i, (cluster_id, size) in enumerate(cluster_sizes[:5]):
#             entities = clusters[cluster_id]
#             print(f"簇 {cluster_id} (大小: {size}):")
#             print(f"  实体: {entities}")
#             print()
    
#     def analyze_definition_coverage(self, entity_names: List[str], 
#                                   entity_positions: List[List[Dict]], 
#                                   output_path: str = "definition_coverage_analysis.json"):
#         """
#         分析定义覆盖率和质量
        
#         Args:
#             entity_names: 原始实体名称列表
#             entity_positions: 实体位置信息列表
#             output_path: 分析结果输出路径
#         """
#         print("分析定义覆盖率和质量...")
        
#         analysis_results = {
#             "total_entities": len(entity_names),
#             "entities_with_positions": 0,
#             "entities_with_definitions": 0,
#             "average_chunks_per_entity": 0,
#             "average_chunk_length": 0,
#             "sample_entities": []
#         }
        
#         total_chunks = 0
#         total_chunk_length = 0
#         chunk_count = 0
#         entities_with_definitions = 0
        
#         for i, entity_name in enumerate(entity_names[:100]):  # 分析前100个实体作为样本
#             positions = entity_positions[i] if i < len(entity_positions) else []
            
#             if positions:
#                 analysis_results["entities_with_positions"] += 1
                
#                 # 获取定义chunks
#                 definition_chunks = self._get_entity_definition_chunks(entity_name, positions)
                
#                 if definition_chunks:
#                     entities_with_definitions += 1
#                     total_chunks += len(definition_chunks)
                    
#                     for chunk in definition_chunks:
#                         total_chunk_length += len(chunk)
#                         chunk_count += 1
                    
#                     # 添加样本
#                     if len(analysis_results["sample_entities"]) < 10:
#                         analysis_results["sample_entities"].append({
#                             "entity_name": entity_name,
#                             "num_chunks": len(definition_chunks),
#                             "chunks_preview": [chunk[:100] + "..." if len(chunk) > 100 else chunk 
#                                              for chunk in definition_chunks[:2]]  # 只显示前2个chunk的预览
#                         })
        
#         # 更新统计信息
#         analysis_results["entities_with_definitions"] = entities_with_definitions
#         if entities_with_definitions > 0:
#             analysis_results["average_chunks_per_entity"] = total_chunks / entities_with_definitions
#         if chunk_count > 0:
#             analysis_results["average_chunk_length"] = total_chunk_length / chunk_count
        
#         # 保存分析结果
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
#         # 打印分析结果
#         print(f"\n=== 定义覆盖率分析结果 ===")
#         print(f"总实体数（样本）: {analysis_results['total_entities']}")
#         print(f"有位置信息的实体数: {analysis_results['entities_with_positions']}")
#         print(f"有定义信息的实体数: {analysis_results['entities_with_definitions']}")
#         print(f"平均每个实体的chunks数: {analysis_results['average_chunks_per_entity']:.2f}")
#         print(f"平均chunk长度: {analysis_results['average_chunk_length']:.1f} 字符")
#         print(f"定义覆盖率: {analysis_results['entities_with_definitions']/analysis_results['total_entities']:.2%}")
        
#         return analysis_results
    
#     def compare_vectorization_methods(self, entity_names: List[str], 
#                                     entity_positions: List[List[Dict]],
#                                     clustering_params: Dict = None):
#         """
#         比较不同向量化方法的聚类效果
        
#         Args:
#             entity_names: 实体名称列表
#             entity_positions: 实体位置信息列表
#             clustering_params: 聚类参数
#         """
#         if clustering_params is None:
#             clustering_params = {
#                 "min_cluster_size": 2,
#                 "min_samples": 1,
#                 "cluster_selection_epsilon": 0.1,
#                 "use_cosine": True
#             }
        
#         print("比较不同向量化方法的聚类效果...")
        
#         # 1. 仅使用实体名称
#         print("\n--- 方法1: 仅实体名称 ---")
#         vectors_entity_only = self.vectorize_entities_simple(entity_names)
#         labels_entity_only = self.perform_clustering(vectors_entity_only, **clustering_params)
#         clusters_entity_only = self.analyze_clusters(entity_names, labels_entity_only)
        
#         # 2. 拼接实体名称和定义
#         print("\n--- 方法2: 实体名称 + 定义拼接 ---")
#         enriched_texts = self.create_enriched_texts(entity_names, entity_positions)
#         vectors_concatenated = self.vectorize_entities_simple(enriched_texts)
#         labels_concatenated = self.perform_clustering(vectors_concatenated, **clustering_params)
#         clusters_concatenated = self.analyze_clusters(entity_names, labels_concatenated)
        
#         # 3. 实体名称和定义分别向量化后平均
#         print("\n--- 方法3: 实体向量与定义向量平均 ---")
#         entity_names_prepared, entity_definition_chunks = self.prepare_entity_definition_pairs(entity_names, entity_positions)
#         vectors_averaged = self.vectorize_entities_with_definitions(entity_names_prepared, entity_definition_chunks)
#         labels_averaged = self.perform_clustering(vectors_averaged, **clustering_params)
#         clusters_averaged = self.analyze_clusters(entity_names, labels_averaged)
        
#         # 4. 比较结果
#         comparison_results = {
#             "entity_only": {
#                 "num_clusters": len([k for k in clusters_entity_only.keys() if k != -1]),
#                 "num_noise": len(clusters_entity_only.get(-1, [])),
#                 "noise_ratio": len(clusters_entity_only.get(-1, [])) / len(entity_names),
#                 "largest_cluster_size": max([len(v) for k, v in clusters_entity_only.items() if k != -1], default=0)
#             },
#             "concatenated": {
#                 "num_clusters": len([k for k in clusters_concatenated.keys() if k != -1]),
#                 "num_noise": len(clusters_concatenated.get(-1, [])),
#                 "noise_ratio": len(clusters_concatenated.get(-1, [])) / len(entity_names),
#                 "largest_cluster_size": max([len(v) for k, v in clusters_concatenated.items() if k != -1], default=0)
#             },
#             "averaged": {
#                 "num_clusters": len([k for k in clusters_averaged.keys() if k != -1]),
#                 "num_noise": len(clusters_averaged.get(-1, [])),
#                 "noise_ratio": len(clusters_averaged.get(-1, [])) / len(entity_names),
#                 "largest_cluster_size": max([len(v) for k, v in clusters_averaged.items() if k != -1], default=0)
#             }
#         }
        
#         print(f"\n=== 向量化方法比较结果 ===")
#         for method, results in comparison_results.items():
#             print(f"{method}:")
#             print(f"  - 簇数量: {results['num_clusters']}")
#             print(f"  - 噪声点数: {results['num_noise']}")
#             print(f"  - 噪声比例: {results['noise_ratio']:.2%}")
#             print(f"  - 最大簇大小: {results['largest_cluster_size']}")
#             print()
        
#         # 保存比较结果
#         with open("vectorization_methods_comparison.json", 'w', encoding='utf-8') as f:
#             json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
#         return comparison_results, clusters_entity_only, clusters_concatenated, clusters_averaged
    
#     def vectorize_entities_simple(self, texts: List[str]) -> np.ndarray:
#         """
#         简单的实体向量化（用于比较）
        
#         Args:
#             texts: 文本列表
            
#         Returns:
#             向量化结果
#         """
#         print("开始简单向量化...")
        
#         batch_size = 100
#         all_vectors = []
        
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i+batch_size]
#             vectors = self.embeddings.embed_documents(batch)
#             all_vectors.extend(vectors)
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"已处理 {i + len(batch)} / {len(texts)} 个文本")
        
#         vectors_array = np.array(all_vectors)
#         print(f"简单向量化完成，形状: {vectors_array.shape}")
#         return vectors_array
    
#     def create_enriched_texts(self, entity_names: List[str], 
#                             entity_positions: List[List[Dict]], 
#                             max_definition_length: int = 1024) -> List[str]:
#         """
#         创建拼接的增强文本（用于比较）
        
#         Args:
#             entity_names: 实体名称列表
#             entity_positions: 实体位置信息列表
#             max_definition_length: 定义文本的最大长度
            
#         Returns:
#             增强文本列表
#         """
#         print("创建拼接的增强文本...")
        
#         enriched_texts = []
        
#         for i, entity_name in enumerate(entity_names):
#             # 获取对应的位置信息
#             positions = entity_positions[i] if i < len(entity_positions) else []
            
#             # 获取定义相关的chunk
#             definition_chunks = self._get_entity_definition_chunks(entity_name, positions)
            
#             # 构建增强文本
#             if definition_chunks:
#                 # 合并定义chunks并限制长度
#                 definition_text = " ".join(definition_chunks)
#                 if len(definition_text) > max_definition_length:
#                     definition_text = definition_text[:max_definition_length] + "..."
                
#                 # 结合实体名称和定义
#                 enriched_text = f"{entity_name} 定义: {definition_text}"
#             else:
#                 # 如果没有找到定义chunk，只使用实体名称
#                 enriched_text = entity_name
            
#             enriched_texts.append(enriched_text)
            
#             # 进度显示
#             if (i + 1) % 100 == 0:
#                 print(f"已处理 {i + 1}/{len(entity_names)} 个实体")
        
#         print(f"增强文本创建完成，共处理 {len(enriched_texts)} 个实体")
#         return enriched_texts

# # 使用示例
# if __name__ == "__main__":
#     # 创建聚类管道
#     pipeline = EntityClusteringPipeline(
#         embedding_model_path="/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_4B",
#         reranker_path="/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_4B",
#         vector_db_path="/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/vector_DB/vector_data",
#         collection_name="edu_chunks",
#         device="cuda:1"
#     )
    
#     # 运行聚类
#     file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA.jsonl"
    
#     # 选项1：运行完整管道（使用新的向量平均方法）
#     alias_mapping, clusters, representatives = pipeline.run_pipeline(
#         file_path=file_path,
#         min_cluster_size=2,
#         min_samples=1,
#         cluster_selection_epsilon=0.1,
#         use_cosine=True,
#         visualize=True,
#         output_path="entity_clustering_results_averaged.json"
#     )
    
#     # 选项2：比较不同向量化方法
#     # entities = pipeline.load_entities_from_jsonl(file_path)
#     # entity_names, entity_positions = pipeline.extract_entity_data(entities)
#     # comparison_results, clusters_entity_only, clusters_concatenated, clusters_averaged = pipeline.compare_vectorization_methods(
#     #     entity_names, entity_positions
#     # )
    
#     # 选项3：分析定义覆盖率
#     # entities = pipeline.load_entities_from_jsonl(file_path)
#     # entity_names, entity_positions = pipeline.extract_entity_data(entities)
#     # coverage_analysis = pipeline.analyze_definition_coverage(entity_names, entity_positions)





















































# # NOTE cluster with GMM 
# # with GMM clustering
# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict, Any, Optional
# from langchain_core.embeddings import Embeddings
# from sklearn.mixture import GaussianMixture
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.preprocessing import normalize
# from sklearn.metrics import silhouette_score, calinski_harabasz_score
# import warnings
# from Qwen3_Reranker import Qwen3Reranker
# warnings.filterwarnings('ignore')

# # 封装为 LangChain 所需的 Embeddings 接口
# class SentenceTransformerEmbeddings(Embeddings):
#     def __init__(self, model_name_or_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_0.6B", device: str = "cuda:1"):
#         self.model = SentenceTransformer(model_name_or_path, device=device)

#     def embed_documents(self, texts: List[str]) -> List[List[float]]:
#         return self.model.encode(texts, convert_to_numpy=True).tolist()

#     def embed_query(self, text: str) -> List[float]:
#         return self.model.encode(text, convert_to_numpy=True, prompt = "Instruct: Given a natural language question, retrieve the most relevant text passages from a collection of documents to support answer generation.\nQuery:").tolist()

#     def __call__(self, text: str) -> List[float]:
#         return self.embed_query(text)

# class EntityClusteringPipeline:
#     def __init__(self, embedding_model_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_embedding_4B", 
#                  reranker_path: str = "/finance_ML/dataarc_syn_database/model/Qwen/qwen_reranker_4B",
#                  device: str = "cuda:1"):
#         """
#         初始化实体聚类管道
        
#         Args:
#             embedding_model_path: 嵌入模型路径
#             reranker_path: 重排序模型路径
#             device: 设备（如：cuda:1）
#         """
#         self.embeddings = SentenceTransformerEmbeddings(embedding_model_path, device)
#         self.reranker_path = reranker_path
#         self.reranker = Qwen3Reranker(
#             model_name_or_path=self.reranker_path,
#             max_length=2048,
#             instruction="Given the user query, retrieval the relevant passages",
#             device_id = "cuda:7" 
#         )
#         self.entities = []
#         self.entity_vectors = None
#         self.clusters = None
#         self.cluster_labels = None
#         self.gmm_model = None
        
#     def load_entities_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
#         """
#         从JSONL文件加载实体数据
        
#         Args:
#             file_path: JSONL文件路径
            
#         Returns:
#             实体列表
#         """
#         entities = []
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 try:
#                     entity = json.loads(line.strip())
#                     entities.append(entity)
#                 except json.JSONDecodeError as e:
#                     print(f"解析行时出错: {e}")
#                     continue
        
#         print(f"成功加载 {len(entities)} 个实体")
#         return entities
    
#     def extract_entity_names(self, entities: List[Dict[str, Any]]) -> List[str]:
#         """
#         提取实体名称
        
#         Args:
#             entities: 实体列表
            
#         Returns:
#             实体名称列表
#         """
#         entity_names = []
#         for entity in entities:
#             if 'name' in entity:
#                 entity_names.append(entity['name'])
        
#         print(f"提取到 {len(entity_names)} 个实体名称")
#         return entity_names

#     def vectorize_entities(self, entity_names: List[str]) -> np.ndarray:
#         """
#         对实体名称进行向量化
        
#         Args:
#             entity_names: 实体名称列表
            
#         Returns:
#             向量化结果
#         """
#         print("开始向量化实体名称...")
        
#         batch_size = 100
#         all_vectors = []
        
#         for i in range(0, len(entity_names), batch_size):
#             batch = entity_names[i:i+batch_size]
#             vectors = self.embeddings.embed_documents(batch)
#             all_vectors.extend(vectors)
            
#             if (i // batch_size + 1) % 10 == 0:
#                 print(f"已处理 {i + len(batch)} / {len(entity_names)} 个实体")
        
#         vectors_array = np.array(all_vectors)
#         print(f"向量化完成，形状: {vectors_array.shape}")
#         return vectors_array

#     def find_optimal_n_components(self, vectors: np.ndarray, max_components: int = 50, 
#                                  min_components: int = 2) -> int:
#         """
#         使用多种指标找到最优的聚类数量
        
#         Args:
#             vectors: 向量矩阵
#             max_components: 最大聚类数
#             min_components: 最小聚类数
            
#         Returns:
#             最优聚类数
#         """
#         print("正在寻找最优聚类数量...")
        
#         # 对向量进行归一化
#         normalized_vectors = normalize(vectors, norm='l2')
        
#         # 限制最大聚类数（避免过拟合）
#         max_components = min(max_components, len(vectors) // 5)
        
#         scores = {'n_components': [], 'aic': [], 'bic': [], 'silhouette': [], 'calinski_harabasz': []}
        
#         for n in range(min_components, max_components + 1):
#             try:
#                 # 拟合GMM模型
#                 gmm = GaussianMixture(
#                     n_components=n, 
#                     covariance_type='full', 
#                     random_state=42,
#                     max_iter=100,
#                     tol=1e-3
#                 )
#                 gmm.fit(normalized_vectors)
                
#                 # 计算各种指标
#                 labels = gmm.predict(normalized_vectors)
#                 aic = gmm.aic(normalized_vectors)
#                 bic = gmm.bic(normalized_vectors)
                
#                 # 计算轮廓系数（需要至少2个簇）
#                 if len(np.unique(labels)) > 1:
#                     silhouette = silhouette_score(normalized_vectors, labels)
#                     calinski_harabasz = calinski_harabasz_score(normalized_vectors, labels)
#                 else:
#                     silhouette = -1
#                     calinski_harabasz = 0
                
#                 scores['n_components'].append(n)
#                 scores['aic'].append(aic)
#                 scores['bic'].append(bic)
#                 scores['silhouette'].append(silhouette)
#                 scores['calinski_harabasz'].append(calinski_harabasz)
                
#                 if n % 10 == 0:
#                     print(f"已测试 {n} 个聚类数")
                    
#             except Exception as e:
#                 print(f"聚类数 {n} 时出错: {e}")
#                 continue
        
#         # 找到最优聚类数
#         scores_df = pd.DataFrame(scores)
        
#         # 使用BIC最小值作为主要指标（BIC在模型复杂度和拟合度之间平衡）
#         best_n_bic = scores_df.loc[scores_df['bic'].idxmin(), 'n_components']
        
#         # 使用轮廓系数最大值作为参考
#         valid_silhouette = scores_df[scores_df['silhouette'] > -1]
#         if not valid_silhouette.empty:
#             best_n_silhouette = valid_silhouette.loc[valid_silhouette['silhouette'].idxmax(), 'n_components']
#         else:
#             best_n_silhouette = best_n_bic
        
#         # 综合考虑（优先考虑BIC）
#         optimal_n = int(best_n_bic)
        
#         print(f"最优聚类数选择结果:")
#         print(f"  - 基于BIC的最优聚类数: {best_n_bic}")
#         print(f"  - 基于轮廓系数的最优聚类数: {best_n_silhouette}")
#         print(f"  - 最终选择的聚类数: {optimal_n}")
        
#         # 绘制评估曲线
#         self.plot_evaluation_curves(scores_df, optimal_n)
        
#         return optimal_n

#     def plot_evaluation_curves(self, scores_df: pd.DataFrame, optimal_n: int):
#         """
#         绘制模型评估曲线
        
#         Args:
#             scores_df: 评估分数数据框
#             optimal_n: 最优聚类数
#         """
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
#         # AIC曲线
#         axes[0, 0].plot(scores_df['n_components'], scores_df['aic'], 'b-', marker='o')
#         axes[0, 0].axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal n={optimal_n}')
#         axes[0, 0].set_xlabel('Number of Components')
#         axes[0, 0].set_ylabel('AIC')
#         axes[0, 0].set_title('AIC Score')
#         axes[0, 0].legend()
#         axes[0, 0].grid(True, alpha=0.3)
        
#         # BIC曲线
#         axes[0, 1].plot(scores_df['n_components'], scores_df['bic'], 'g-', marker='o')
#         axes[0, 1].axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal n={optimal_n}')
#         axes[0, 1].set_xlabel('Number of Components')
#         axes[0, 1].set_ylabel('BIC')
#         axes[0, 1].set_title('BIC Score')
#         axes[0, 1].legend()
#         axes[0, 1].grid(True, alpha=0.3)
        
#         # 轮廓系数曲线
#         valid_silhouette = scores_df[scores_df['silhouette'] > -1]
#         if not valid_silhouette.empty:
#             axes[1, 0].plot(valid_silhouette['n_components'], valid_silhouette['silhouette'], 'r-', marker='o')
#             axes[1, 0].axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal n={optimal_n}')
#         axes[1, 0].set_xlabel('Number of Components')
#         axes[1, 0].set_ylabel('Silhouette Score')
#         axes[1, 0].set_title('Silhouette Score')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True, alpha=0.3)
        
#         # Calinski-Harabasz指数曲线
#         valid_ch = scores_df[scores_df['calinski_harabasz'] > 0]
#         if not valid_ch.empty:
#             axes[1, 1].plot(valid_ch['n_components'], valid_ch['calinski_harabasz'], 'purple', marker='o')
#             axes[1, 1].axvline(x=optimal_n, color='r', linestyle='--', label=f'Optimal n={optimal_n}')
#         axes[1, 1].set_xlabel('Number of Components')
#         axes[1, 1].set_ylabel('Calinski-Harabasz Index')
#         axes[1, 1].set_title('Calinski-Harabasz Index')
#         axes[1, 1].legend()
#         axes[1, 1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('gmm_evaluation_curves.png', dpi=300, bbox_inches='tight')
#         plt.show()

#     def perform_clustering(self, vectors: np.ndarray, n_components: Optional[int] = None, 
#                           covariance_type: str = 'full', max_iter: int = 100,
#                           auto_select: bool = True) -> np.ndarray:
#         """
#         使用GMM进行聚类
        
#         Args:
#             vectors: 向量矩阵
#             n_components: 聚类数量（如果为None且auto_select为True，则自动选择）
#             covariance_type: 协方差类型 ('full', 'tied', 'diag', 'spherical')
#             max_iter: 最大迭代次数
#             auto_select: 是否自动选择最优聚类数
            
#         Returns:
#             聚类标签
#         """
#         print("开始GMM聚类...")
        
#         # 对向量进行L2归一化
#         normalized_vectors = normalize(vectors, norm='l2')
        
#         # 自动选择最优聚类数
#         if n_components is None and auto_select:
#             n_components = self.find_optimal_n_components(vectors)
#         elif n_components is None:
#             # 如果没有指定聚类数且不自动选择，使用启发式方法
#             n_components = min(50, max(2, int(np.sqrt(len(vectors)))))
#             print(f"使用启发式方法选择聚类数: {n_components}")
        
#         # 拟合GMM模型
#         self.gmm_model = GaussianMixture(
#             n_components=n_components,
#             covariance_type=covariance_type,
#             random_state=42,
#             max_iter=max_iter,
#             tol=1e-3,
#             reg_covar=1e-6  # 正则化项，防止协方差矩阵奇异
#         )
        
#         print(f"正在拟合GMM模型 (n_components={n_components}, covariance_type={covariance_type})...")
#         self.gmm_model.fit(normalized_vectors)
        
#         # 预测聚类标签
#         cluster_labels = self.gmm_model.predict(normalized_vectors)
        
#         # 获取聚类概率
#         cluster_probs = self.gmm_model.predict_proba(normalized_vectors)
        
#         # 计算每个点的聚类置信度（最大概率）
#         cluster_confidence = np.max(cluster_probs, axis=1)
        
#         # 统计聚类结果
#         n_clusters = len(set(cluster_labels))
        
#         print(f"GMM聚类完成:")
#         print(f"  - 聚类数量: {n_clusters}")
#         print(f"  - 模型收敛: {self.gmm_model.converged_}")
#         print(f"  - 迭代次数: {self.gmm_model.n_iter_}")
#         print(f"  - 平均聚类置信度: {np.mean(cluster_confidence):.3f}")
#         print(f"  - 低置信度点数量 (<0.5): {np.sum(cluster_confidence < 0.5)}")
        
#         # 计算并打印模型质量指标
#         if len(np.unique(cluster_labels)) > 1:
#             silhouette = silhouette_score(normalized_vectors, cluster_labels)
#             calinski_harabasz = calinski_harabasz_score(normalized_vectors, cluster_labels)
#             aic = self.gmm_model.aic(normalized_vectors)
#             bic = self.gmm_model.bic(normalized_vectors)
            
#             print(f"  - 轮廓系数: {silhouette:.3f}")
#             print(f"  - Calinski-Harabasz指数: {calinski_harabasz:.3f}")
#             print(f"  - AIC: {aic:.3f}")
#             print(f"  - BIC: {bic:.3f}")
        
#         return cluster_labels

#     def analyze_clusters(self, entity_names: List[str], cluster_labels: np.ndarray) -> Dict[int, List[str]]:
#         """
#         分析聚类结果
        
#         Args:
#             entity_names: 实体名称列表
#             cluster_labels: 聚类标签
            
#         Returns:
#             聚类结果字典
#         """
#         clusters = defaultdict(list)
        
#         for name, label in zip(entity_names, cluster_labels):
#             clusters[label].append(name)
        
#         return dict(clusters)

#     def find_cluster_representatives(self, clusters: Dict[int, List[str]], 
#                                    entity_names: List[str], vectors: np.ndarray) -> Dict[int, str]:
#         """
#         为每个簇找到代表性实体
        
#         Args:
#             clusters: 聚类结果
#             entity_names: 实体名称列表
#             vectors: 向量矩阵
            
#         Returns:
#             每个簇的代表实体
#         """
#         representatives = {}
#         name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
#         for cluster_id, cluster_entities in clusters.items():
#             if len(cluster_entities) == 1:
#                 representatives[cluster_id] = cluster_entities[0]
#                 continue
            
#             # 获取簇中所有实体的向量
#             cluster_indices = [name_to_idx[name] for name in cluster_entities]
#             cluster_vectors = vectors[cluster_indices]
            
#             if self.gmm_model is not None:
#                 # 使用GMM模型的簇中心
#                 cluster_center = self.gmm_model.means_[cluster_id]
#             else:
#                 # 计算簇中心
#                 cluster_center = np.mean(cluster_vectors, axis=0)
            
#             # 找到最接近中心的实体
#             similarities = cosine_similarity([cluster_center], cluster_vectors)[0]
#             best_idx = np.argmax(similarities)
#             representatives[cluster_id] = cluster_entities[best_idx]
        
#         return representatives

#     def create_alias_mapping(self, clusters: Dict[int, List[str]], 
#                            representatives: Dict[int, str]) -> Dict[str, str]:
#         """
#         创建别称映射
        
#         Args:
#             clusters: 聚类结果
#             representatives: 簇代表实体
            
#         Returns:
#             别称映射字典
#         """
#         alias_mapping = {}
        
#         for cluster_id, cluster_entities in clusters.items():
#             representative = representatives[cluster_id]
#             for entity in cluster_entities:
#                 alias_mapping[entity] = representative
        
#         return alias_mapping

#     def visualize_clusters(self, vectors: np.ndarray, cluster_labels: np.ndarray, 
#                           entity_names: List[str], method: str = 'tsne', 
#                           sample_size: int = 1000, figsize: tuple = (15, 10)):
#         """
#         可视化聚类结果
        
#         Args:
#             vectors: 向量矩阵
#             cluster_labels: 聚类标签
#             entity_names: 实体名称列表
#             method: 降维方法 ('pca' 或 'tsne')
#             sample_size: 采样大小
#             figsize: 图形大小
#         """
#         # 设置中文字体
#         plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
#         plt.rcParams['axes.unicode_minus'] = False
        
#         # 如果数据量太大，进行采样
#         if len(vectors) > sample_size:
#             np.random.seed(42)
#             indices = np.random.choice(len(vectors), sample_size, replace=False)
#             vectors_sample = vectors[indices]
#             labels_sample = cluster_labels[indices]
#             names_sample = [entity_names[i] for i in indices]
#         else:
#             vectors_sample = vectors
#             labels_sample = cluster_labels
#             names_sample = entity_names
        
#         # 对向量进行归一化
#         vectors_normalized = normalize(vectors_sample, norm='l2')
        
#         # 降维
#         print(f"正在进行{method.upper()}降维...")
#         if method == 'pca':
#             reducer = PCA(n_components=2, random_state=42)
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         elif method == 'tsne':
#             reducer = TSNE(n_components=2, random_state=42, 
#                           perplexity=min(30, len(vectors_normalized)-1), 
#                           metric='cosine')
#             vectors_2d = reducer.fit_transform(vectors_normalized)
#         else:
#             raise ValueError("method must be 'pca' or 'tsne'")
        
#         # 创建子图
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
#         # 左图：所有点的散点图
#         unique_labels = sorted(set(labels_sample))
#         n_clusters = len(unique_labels)
        
#         # 使用更好的颜色方案
#         if n_clusters <= 10:
#             colors = plt.cm.tab10(np.linspace(0, 1, 10))
#         else:
#             colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
#         color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
        
#         # 绘制所有点
#         for label in unique_labels:
#             mask = labels_sample == label
#             ax1.scatter(vectors_2d[mask, 0], vectors_2d[mask, 1], 
#                        c=[color_map[label]], marker='o', alpha=0.7, s=30, 
#                        label=f'Cluster {label}')
        
#         # 如果有GMM模型，绘制置信椭圆
#         if self.gmm_model is not None and method == 'pca':
#             self.plot_confidence_ellipses(ax1, vectors_2d, labels_sample, reducer)
        
#         ax1.set_title(f'All Clusters ({method.upper()}) - GMM', fontsize=14)
#         ax1.set_xlabel('Component 1', fontsize=12)
#         ax1.set_ylabel('Component 2', fontsize=12)
#         ax1.grid(True, alpha=0.3)
        
#         # 添加图例
#         handles, labels = ax1.get_legend_handles_labels()
#         if len(handles) > 15:
#             ax1.legend(handles[:15], labels[:15], bbox_to_anchor=(1.05, 1), loc='upper left')
#         else:
#             ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         # 右图：显示最大的几个簇
#         cluster_sizes = [(label, sum(labels_sample == label)) for label in unique_labels]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         top_clusters = [label for label, size in cluster_sizes[:10]]
        
#         for label in top_clusters:
#             mask = labels_sample == label
#             cluster_points = vectors_2d[mask]
#             cluster_names = [names_sample[i] for i in range(len(names_sample)) if labels_sample[i] == label]
            
#             ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], 
#                        c=[color_map[label]], marker='o', alpha=0.7, s=40, 
#                        label=f'Cluster {label} ({len(cluster_names)})')
            
#             # 标注代表实体
#             if len(cluster_names) > 1:
#                 center_x, center_y = np.mean(cluster_points, axis=0)
#                 distances = np.sum((cluster_points - [center_x, center_y])**2, axis=1)
#                 closest_idx = np.argmin(distances)
#                 representative = cluster_names[closest_idx]
                
#                 ax2.annotate(representative, 
#                            (cluster_points[closest_idx, 0], cluster_points[closest_idx, 1]),
#                            xytext=(5, 5), textcoords='offset points', 
#                            fontsize=8, alpha=0.8,
#                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
#         ax2.set_title(f'Top 10 Largest Clusters with Labels ({method.upper()}) - GMM', fontsize=14)
#         ax2.set_xlabel('Component 1', fontsize=12)
#         ax2.set_ylabel('Component 2', fontsize=12)
#         ax2.grid(True, alpha=0.3)
#         ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
#         plt.tight_layout()
#         plt.savefig(f'entity_clustering_visualization_{method}_gmm.png', dpi=300, bbox_inches='tight')
#         plt.show()
        
#         print(f"\n可视化统计信息:")
#         print(f"  - 可视化样本数: {len(vectors_sample)}")
#         print(f"  - 降维方法: {method.upper()}")
#         print(f"  - 聚类数量: {n_clusters}")

#     def plot_confidence_ellipses(self, ax, vectors_2d, labels, reducer):
#         """
#         绘制GMM的置信椭圆（仅适用于PCA降维）
#         """
#         if not hasattr(self.gmm_model, 'means_'):
#             return
            
#         try:
#             # 将高维的GMM参数投影到2D空间
#             means_2d = reducer.transform(self.gmm_model.means_)
            
#             for i, (mean, covar) in enumerate(zip(means_2d, self.gmm_model.covariances_)):
#                 if i in labels:  # 只绘制存在的簇
#                     # 计算2D协方差矩阵（简化版本）
#                     covar_2d = np.eye(2) * 0.5  # 使用简化的协方差矩阵
                    
#                     # 绘制95%置信椭圆
#                     eigenvals, eigenvecs = np.linalg.eigh(covar_2d)
#                     angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
#                     width, height = 2 * np.sqrt(eigenvals * 5.991)  # 95% confidence
                    
#                     from matplotlib.patches import Ellipse
#                     ellipse = Ellipse(mean, width, height, angle=angle, 
#                                     alpha=0.2, facecolor='gray', edgecolor='black', linestyle='--')
#                     ax.add_patch(ellipse)
#         except Exception as e:
#             print(f"绘制置信椭圆时出错: {e}")

#     def save_results(self, alias_mapping: Dict[str, str], clusters: Dict[int, List[str]], 
#                     representatives: Dict[int, str], output_path: str = "clustering_results_gmm.json"):
#         """
#         保存聚类结果
#         """
#         results = {
#             "alias_mapping": alias_mapping,
#             "clusters": {str(k): v for k, v in clusters.items()},
#             "representatives": {str(k): v for k, v in representatives.items()},
#             "model_info": {
#                 "algorithm": "GMM",
#                 "n_components": self.gmm_model.n_components if self.gmm_model else None,
#                 "covariance_type": self.gmm_model.covariance_type if self.gmm_model else None,
#                 "converged": self.gmm_model.converged_ if self.gmm_model else None,
#                 "n_iter": self.gmm_model.n_iter_ if self.gmm_model else None
#             },
#             "statistics": {
#                 "total_entities": len(alias_mapping),
#                 "unique_representatives": len(set(alias_mapping.values())),
#                 "compression_ratio": len(set(alias_mapping.values())) / len(alias_mapping)
#             }
#         }
        
#         with open(output_path, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
        
#         print(f"结果已保存到: {output_path}")

#     def run_pipeline(self, file_path: str, n_components: Optional[int] = None,
#                     covariance_type: str = 'full', max_iter: int = 100,
#                     auto_select: bool = True, visualize: bool = True, 
#                     output_path: str = "clustering_results_gmm.json"):
#         """
#         运行完整的聚类管道
        
#         Args:
#             file_path: 输入文件路径
#             n_components: 聚类数量（None表示自动选择）
#             covariance_type: 协方差类型
#             max_iter: 最大迭代次数
#             auto_select: 是否自动选择最优聚类数
#             visualize: 是否可视化结果
#             output_path: 输出文件路径
#         """
#         # 1. 加载数据
#         entities = self.load_entities_from_jsonl(file_path)
#         entity_names = self.extract_entity_names(entities)
        
#         # 2. 向量化
#         vectors = self.vectorize_entities(entity_names)
        
#         # 3. 聚类
#         cluster_labels = self.perform_clustering(
#             vectors, n_components, covariance_type, max_iter, auto_select
#         )
        
#         # 4. 分析结果
#         clusters = self.analyze_clusters(entity_names, cluster_labels)
#         representatives = self.find_cluster_representatives(clusters, entity_names, vectors)
#         alias_mapping = self.create_alias_mapping(clusters, representatives)
        
#         # 5. 可视化
#         if visualize:
#             self.visualize_clusters(vectors, cluster_labels, entity_names, method='tsne')
#             # self.visualize_clusters(vectors, cluster_labels, entity_names, method='pca')
        
#         # 6. 保存结果
#         self.save_results(alias_mapping, clusters, representatives, output_path)
        
#         # 7. 打印统计信息
#         self.print_statistics(clusters, alias_mapping)
        
#         return alias_mapping, clusters, representatives
    
#     def print_statistics(self, clusters: Dict[int, List[str]], alias_mapping: Dict[str, str]):
#         """
#         打印统计信息
#         """
#         print("\n=== GMM聚类统计信息 ===")
#         print(f"总实体数: {len(alias_mapping)}")
#         print(f"簇数量: {len(clusters)}")
#         print(f"唯一代表实体数: {len(set(alias_mapping.values()))}")
#         print(f"压缩率: {len(set(alias_mapping.values())) / len(alias_mapping):.2%}")
        
#         # 显示簇大小分布
#         cluster_sizes = [len(v) for v in clusters.values()]
#         if cluster_sizes:
#             print(f"簇大小统计:")
#             print(f"  - 平均大小: {np.mean(cluster_sizes):.2f}")
#             print(f"  - 中位数大小: {np.median(cluster_sizes):.2f}")
#             print(f"  - 最大簇大小: {max(cluster_sizes)}")
#             print(f"  - 最小簇大小: {min(cluster_sizes)}")
        
#         # 显示一些示例簇
#         print("\n=== 示例GMM聚类结果 ===")
#         cluster_sizes = [(k, len(v)) for k, v in clusters.items()]
#         cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
#         for i, (cluster_id, size) in enumerate(cluster_sizes[:5]):
#             entities = clusters[cluster_id]
#             print(f"簇 {cluster_id} (大小: {size}):")
#             print(f"  实体: {entities}")
#             print()

#     def get_cluster_probabilities(self, entity_names: List[str], vectors: np.ndarray) -> Dict[str, np.ndarray]:
#         """
#         获取每个实体属于各个簇的概率
        
#         Args:
#             entity_names: 实体名称列表
#             vectors: 向量矩阵
            
#         Returns:
#             实体名称到概率分布的映射
#         """
#         if self.gmm_model is None:
#             raise ValueError("GMM模型未训练，请先运行聚类")
        
#         normalized_vectors = normalize(vectors, norm='l2')
#         probs = self.gmm_model.predict_proba(normalized_vectors)
        
#         prob_dict = {}
#         for name, prob in zip(entity_names, probs):
#             prob_dict[name] = prob
        
#         return prob_dict

#     def analyze_uncertain_entities(self, entity_names: List[str], vectors: np.ndarray, 
#                                   threshold: float = 0.6) -> List[Dict[str, Any]]:
#         """
#         分析聚类不确定的实体
        
#         Args:
#             entity_names: 实体名称列表
#             vectors: 向量矩阵
#             threshold: 置信度阈值
            
#         Returns:
#             不确定实体的详细信息
#         """
#         if self.gmm_model is None:
#             raise ValueError("GMM模型未训练，请先运行聚类")
        
#         normalized_vectors = normalize(vectors, norm='l2')
#         probs = self.gmm_model.predict_proba(normalized_vectors)
#         max_probs = np.max(probs, axis=1)
        
#         uncertain_entities = []
#         for i, (name, prob, max_prob) in enumerate(zip(entity_names, probs, max_probs)):
#             if max_prob < threshold:
#                 # 找到前两个最高概率的簇
#                 top_clusters = np.argsort(prob)[-2:][::-1]
#                 uncertain_entities.append({
#                     'name': name,
#                     'max_probability': max_prob,
#                     'top_cluster': top_clusters[0],
#                     'top_cluster_prob': prob[top_clusters[0]],
#                     'second_cluster': top_clusters[1],
#                     'second_cluster_prob': prob[top_clusters[1]],
#                     'uncertainty': 1 - max_prob
#                 })
        
#         # 按不确定性排序
#         uncertain_entities.sort(key=lambda x: x['uncertainty'], reverse=True)
        
#         print(f"\n=== 聚类不确定实体分析 ===")
#         print(f"低置信度实体数量 (<{threshold}): {len(uncertain_entities)}")
        
#         if uncertain_entities:
#             print("\n前10个最不确定的实体:")
#             for i, entity in enumerate(uncertain_entities[:10]):
#                 print(f"{i+1}. {entity['name']}")
#                 print(f"   最高概率: {entity['max_probability']:.3f} (簇 {entity['top_cluster']})")
#                 print(f"   第二高概率: {entity['second_cluster_prob']:.3f} (簇 {entity['second_cluster']})")
#                 print(f"   不确定性: {entity['uncertainty']:.3f}")
#                 print()
        
#         return uncertain_entities

# # 使用示例
# if __name__ == "__main__":
#     # 创建GMM聚类管道
#     pipeline = EntityClusteringPipeline()
    
#     # 运行聚类
#     file_path = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA.jsonl"
    
#     # 方法1: 自动选择最优聚类数
#     alias_mapping, clusters, representatives = pipeline.run_pipeline(
#         file_path=file_path,
#         n_components=None,  # 自动选择
#         covariance_type='full',
#         max_iter=100,
#         auto_select=True,
#         visualize=True,
#         output_path="entity_clustering_results_gmm_auto.json"
#     )
    
#     # 方法2: 手动指定聚类数
#     # alias_mapping, clusters, representatives = pipeline.run_pipeline(
#     #     file_path=file_path,
#     #     n_components=20,  # 手动指定
#     #     covariance_type='full',
#     #     max_iter=100,
#     #     auto_select=False,
#     #     visualize=True,
#     #     output_path="entity_clustering_results_gmm_manual.json"
#     # )
    
#     # 分析不确定的实体
#     entities = pipeline.load_entities_from_jsonl(file_path)
#     entity_names = pipeline.extract_entity_names(entities)
#     vectors = pipeline.vectorize_entities(entity_names)
    
#     uncertain_entities = pipeline.analyze_uncertain_entities(entity_names, vectors, threshold=0.6)
    
#     print(f"\n=== 最终结果摘要 ===")
#     print(f"原始实体数: {len(entity_names)}")
#     print(f"聚类后代表实体数: {len(set(alias_mapping.values()))}")
#     print(f"压缩率: {len(set(alias_mapping.values())) / len(entity_names):.2%}")
#     print(f"不确定实体数: {len(uncertain_entities)}")
    
#     # 保存不确定实体信息
#     if uncertain_entities:
#         with open("uncertain_entities_gmm.json", 'w', encoding='utf-8') as f:
#             json.dump(uncertain_entities, f, ensure_ascii=False, indent=2)
#         print(f"不确定实体信息已保存到: uncertain_entities_gmm.json")