# import json
# import numpy as np
# import pandas as pd
# import networkx as nx
# from collections import defaultdict, Counter
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import argparse

# class EntityCommunityDetector:
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.edges = []
#         self.entities = set()
#         self.entity_to_id = {}
#         self.id_to_entity = {}
#         self.graph = None
#         self.communities = None
        
#     def load_data(self):
#         """加载实体连接数据"""
#         print("正在加载数据...")
#         with open(self.file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 data = json.loads(line.strip())
#                 self.edges.append(data)
#                 self.entities.add(data['node1'])
#                 self.entities.add(data['node2'])
        
#         # 创建实体到ID的映射
#         self.entities = list(self.entities)
#         self.entity_to_id = {entity: idx for idx, entity in enumerate(self.entities)}
#         self.id_to_entity = {idx: entity for idx, entity in enumerate(self.entities)}
        
#         print(f"加载完成：{len(self.entities)} 个实体，{len(self.edges)} 条边")
        
#     def build_graph(self):
#         """构建图结构"""
#         print("正在构建图结构...")
#         # 构建NetworkX图
#         self.graph = nx.Graph()
        
#         # 添加节点
#         for entity in self.entities:
#             self.graph.add_node(entity)
        
#         # 添加边，使用normalize_weight作为权重
#         for edge in self.edges:
#             node1, node2 = edge['node1'], edge['node2']
#             # weight = edge['normlize_weight']
#             weight = edge['weight']
            
#             # 如果边已存在，取最大权重
#             if self.graph.has_edge(node1, node2):
#                 existing_weight = self.graph[node1][node2]['weight']
#                 weight = max(weight, existing_weight)
            
#             self.graph.add_edge(node1, node2, weight=weight)
        
#         print(f"图构建完成：{self.graph.number_of_nodes()} 个节点，{self.graph.number_of_edges()} 条边")
        
#     def louvain_detection(self):
#         """使用Louvain算法进行社区检测"""
#         print("正在执行Louvain社区检测...")
        
#         try:
#             import community as community_louvain
#             partition = community_louvain.best_partition(self.graph, weight='weight')
            
#             # 转换为聚类标签
#             community_labels = np.zeros(len(self.entities))
#             for entity, community_id in partition.items():
#                 entity_idx = self.entity_to_id[entity]
#                 community_labels[entity_idx] = community_id
                
#             self.communities = community_labels.astype(int)
            
#             # 计算模块度
#             modularity = community_louvain.modularity(partition, self.graph, weight='weight')
#             print(f"Louvain检测完成，发现 {len(set(community_labels))} 个社区，模块度: {modularity:.3f}")
            
#             return community_labels.astype(int), modularity
            
#         except ImportError:
#             print("错误：未安装python-louvain包")
#             print("请使用以下命令安装：pip install python-louvain")
#             return None, None
    
#     def leiden_detection(self, resolution=1.0):
#         """使用Leiden算法进行社区检测"""
#         print("正在执行Leiden社区检测...")
        
#         try:
#             import leidenalg
#             import igraph as ig
            
#             # 将NetworkX图转换为igraph
#             print("正在转换图格式...")
            
#             # 创建igraph图
#             g = ig.Graph()
            
#             # 添加节点
#             node_list = list(self.graph.nodes())
#             g.add_vertices(len(node_list))
            
#             # 创建节点名称到索引的映射
#             node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            
#             # 添加边和权重
#             edges = []
#             weights = []
#             for edge in self.graph.edges(data=True):
#                 node1, node2, data = edge
#                 idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
#                 edges.append((idx1, idx2))
#                 weights.append(data.get('weight', 1.0))
            
#             g.add_edges(edges)
#             g.es['weight'] = weights
            
#             # 执行Leiden算法
#             print(f"正在执行Leiden算法，分辨率参数: {resolution}")
            
#             # 使用RBConfigurationVertexPartition支持分辨率参数
#             if resolution != 1.0:
#                 partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
#                                                    weights=weights, resolution_parameter=resolution)
#             else:
#                 # 对于默认分辨率，使用ModularityVertexPartition
#                 partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, 
#                                                    weights=weights)
            
#             # 转换为聚类标签
#             community_labels = np.zeros(len(self.entities))
#             for idx, community_id in enumerate(partition.membership):
#                 entity = node_list[idx]
#                 entity_idx = self.entity_to_id[entity]
#                 community_labels[entity_idx] = community_id
                
#             self.communities = community_labels.astype(int)
            
#             # 计算模块度
#             modularity = partition.modularity
#             print(f"Leiden检测完成，发现 {len(set(community_labels))} 个社区，模块度: {modularity:.3f}")
            
#             return community_labels.astype(int), modularity
            
#         except ImportError:
#             print("错误：未安装所需的包")
#             print("请使用以下命令安装：")
#             print("pip install leidenalg")
#             print("pip install python-igraph")
#             return None, None
#         except Exception as e:
#             print(f"Leiden算法执行失败: {e}")
#             return None, None
    
#     def label_propagation_detection(self):
#         """使用标签传播算法进行社区检测"""
#         print("正在执行标签传播社区检测...")
        
#         try:
#             # 使用NetworkX内置的标签传播算法
#             communities = list(nx.community.label_propagation_communities(self.graph))
            
#             # 转换为聚类标签
#             community_labels = np.zeros(len(self.entities))
#             for community_id, community in enumerate(communities):
#                 for entity in community:
#                     entity_idx = self.entity_to_id[entity]
#                     community_labels[entity_idx] = community_id
            
#             self.communities = community_labels.astype(int)
            
#             # 计算模块度
#             modularity = nx.community.modularity(self.graph, communities, weight='weight')
#             print(f"标签传播检测完成，发现 {len(communities)} 个社区，模块度: {modularity:.3f}")
            
#             return community_labels.astype(int), modularity
            
#         except Exception as e:
#             print(f"标签传播算法执行失败: {e}")
#             return None, None
    
#     def greedy_modularity_detection(self):
#         """使用贪心模块度优化算法进行社区检测"""
#         print("正在执行贪心模块度优化社区检测...")
        
#         try:
#             # 使用NetworkX内置的贪心模块度优化算法
#             communities = list(nx.community.greedy_modularity_communities(self.graph, weight='weight'))
            
#             # 转换为聚类标签
#             community_labels = np.zeros(len(self.entities))
#             for community_id, community in enumerate(communities):
#                 for entity in community:
#                     entity_idx = self.entity_to_id[entity]
#                     community_labels[entity_idx] = community_id
            
#             self.communities = community_labels.astype(int)
            
#             # 计算模块度
#             modularity = nx.community.modularity(self.graph, communities, weight='weight')
#             print(f"贪心模块度优化检测完成，发现 {len(communities)} 个社区，模块度: {modularity:.3f}")
            
#             return community_labels.astype(int), modularity
            
#         except Exception as e:
#             print(f"贪心模块度优化算法执行失败: {e}")
#             return None, None
    
#     def compare_methods(self):
#         """比较不同的社区检测方法"""
#         print("\n比较不同的社区检测方法:")
#         print("=" * 60)
        
#         methods = [
#             ("Louvain", self.louvain_detection),
#             ("Leiden", self.leiden_detection),
#             # ("Label Propagation", self.label_propagation_detection),
#             # ("Greedy Modularity", self.greedy_modularity_detection)
#         ]
        
#         results = {}
        
#         for method_name, method_func in methods:
#             print(f"\n正在测试 {method_name} 方法...")
#             try:
#                 if method_name == "Leiden":
#                     # 测试不同的分辨率参数
#                     best_modularity = -1
#                     best_communities = None
#                     best_resolution = 1.0
                    
#                     # 测试分辨率参数，包括默认值
#                     resolutions = [0.5, 1.0, 1.5, 2.0]
#                     for resolution in resolutions:
#                         try:
#                             communities, modularity = method_func(resolution)
#                             if modularity is not None and modularity > best_modularity:
#                                 best_modularity = modularity
#                                 best_communities = communities
#                                 best_resolution = resolution
#                             print(f"  分辨率 {resolution}: 模块度 {modularity:.3f}, 社区数 {len(set(communities))}")
#                         except Exception as e:
#                             print(f"  分辨率 {resolution}: 执行失败 - {e}")
                    
#                     results[method_name] = {
#                         'communities': best_communities,
#                         'modularity': best_modularity,
#                         'num_communities': len(set(best_communities)) if best_communities is not None else 0,
#                         'resolution': best_resolution
#                     }
#                     if best_communities is not None:
#                         print(f"最佳分辨率: {best_resolution}, 模块度: {best_modularity:.3f}")
#                     else:
#                         print("所有分辨率参数都失败了")
#                 else:
#                     communities, modularity = method_func()
#                     results[method_name] = {
#                         'communities': communities,
#                         'modularity': modularity,
#                         'num_communities': len(set(communities)) if communities is not None else 0
#                     }
                    
#             except Exception as e:
#                 print(f"{method_name} 方法执行失败: {e}")
#                 results[method_name] = {
#                     'communities': None,
#                     'modularity': None,
#                     'num_communities': 0
#                 }
        
#         # 显示比较结果
#         print("\n比较结果:")
#         print("-" * 60)
#         print(f"{'方法':<15} {'社区数':<10} {'模块度':<10} {'状态'}")
#         print("-" * 60)
        
#         for method_name, result in results.items():
#             status = "成功" if result['communities'] is not None else "失败"
#             modularity_str = f"{result['modularity']:.3f}" if result['modularity'] is not None else "N/A"
#             print(f"{method_name:<15} {result['num_communities']:<10} {modularity_str:<10} {status}")
        
#         # 选择最佳方法
#         best_method = max(results.items(), 
#                          key=lambda x: x[1]['modularity'] if x[1]['modularity'] is not None else -1)
        
#         if best_method[1]['communities'] is not None:
#             print(f"\n最佳方法: {best_method[0]} (模块度: {best_method[1]['modularity']:.3f})")
#             self.communities = best_method[1]['communities']
#             return best_method[0], best_method[1]
#         else:
#             print("\n所有方法都失败了")
#             return None, None
    
#     def analyze_communities(self):
#         """分析社区检测结果"""
#         if self.communities is None:
#             print("请先执行社区检测")
#             return
        
#         print("\n社区检测结果分析:")
#         print("-" * 50)
        
#         # 统计每个社区的大小
#         community_sizes = Counter(self.communities)
#         print(f"总共发现 {len(community_sizes)} 个社区")
        
#         # 按社区大小排序
#         sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
        
#         for i, (community_id, size) in enumerate(sorted_communities[:10]):  # 只显示前10个最大的社区
#             print(f"社区 {community_id} (排名 {i+1}): {size} 个实体")
            
#             # 显示该社区中的实体（最多显示10个）
#             community_entities = [self.entities[j] for j in range(len(self.entities)) 
#                                 if self.communities[j] == community_id]
            
#             if len(community_entities) <= 10:
#                 print(f"  实体: {', '.join(community_entities)}")
#             else:
#                 print(f"  实体: {', '.join(community_entities[:10])}... (共{len(community_entities)}个)")
#             print()
        
#         if len(sorted_communities) > 10:
#             print(f"... 还有 {len(sorted_communities) - 10} 个社区")
        
#         # 统计社区大小分布
#         sizes = list(community_sizes.values())
#         print(f"\n社区大小统计:")
#         print(f"  平均大小: {np.mean(sizes):.2f}")
#         print(f"  中位数大小: {np.median(sizes):.2f}")
#         print(f"  最大社区: {max(sizes)} 个实体")
#         print(f"  最小社区: {min(sizes)} 个实体")
#         print(f"  标准差: {np.std(sizes):.2f}")
    
#     def save_results(self, output_file, method_name=None):
#         """保存社区检测结果"""
#         if self.communities is None:
#             print("请先执行社区检测")
#             return
        
#         print(f"正在保存结果到 {output_file}")
        
#         # 创建结果数据
#         results = []
#         for i, entity in enumerate(self.entities):
#             # 获取原始实体类型信息
#             entity_type = "未知"
#             for edge in self.edges:
#                 if edge['node1'] == entity:
#                     entity_type = edge['node1_type']
#                     break
#                 elif edge['node2'] == entity:
#                     entity_type = edge['node2_type']
#                     break
            
#             results.append({
#                 'entity': entity,
#                 'entity_type': entity_type,
#                 'community_id': int(self.communities[i]),
#                 'method': method_name if method_name else "unknown"
#             })
        
#         # 保存为JSON Lines格式
#         with open(output_file, 'w', encoding='utf-8') as f:
#             for result in results:
#                 f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
#         print(f"结果已保存到 {output_file}")
        
#         # 保存社区统计信息
#         stats_file = output_file.replace('.jsonl', '_stats.json')
#         community_stats = {}
#         for i, entity in enumerate(self.entities):
#             community_id = int(self.communities[i])
#             if community_id not in community_stats:
#                 community_stats[community_id] = {
#                     'entities': [],
#                     'size': 0,
#                     'entity_types': Counter()
#                 }
#             community_stats[community_id]['entities'].append(entity)
#             community_stats[community_id]['size'] += 1
            
#             # 统计实体类型
#             entity_type = "未知"
#             for edge in self.edges:
#                 if edge['node1'] == entity:
#                     entity_type = edge['node1_type']
#                     break
#                 elif edge['node2'] == entity:
#                     entity_type = edge['node2_type']
#                     break
#             community_stats[community_id]['entity_types'][entity_type] += 1
        
#         # 转换Counter为普通字典以便JSON序列化
#         for community_id in community_stats:
#             community_stats[community_id]['entity_types'] = dict(community_stats[community_id]['entity_types'])
        
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             json.dump(community_stats, f, ensure_ascii=False, indent=2)
        
#         print(f"社区统计信息已保存到 {stats_file}")
    
#     def visualize_communities(self, output_dir=None):
#         """可视化社区检测结果"""
#         if self.communities is None:
#             print("请先执行社区检测")
#             return
        
#         if output_dir:
#             Path(output_dir).mkdir(exist_ok=True)
        
#         # 1. 社区大小分布
#         plt.figure(figsize=(12, 8))
        
#         plt.subplot(2, 2, 1)
#         community_sizes = Counter(self.communities)
#         plt.bar(range(len(community_sizes)), sorted(community_sizes.values(), reverse=True))
#         plt.xlabel('社区排名')
#         plt.ylabel('实体数量')
#         plt.title('社区大小分布 (按大小排序)')
        
#         # 2. 社区大小直方图
#         plt.subplot(2, 2, 2)
#         sizes = list(community_sizes.values())
#         plt.hist(sizes, bins=min(20, len(set(sizes))), alpha=0.7, edgecolor='black')
#         plt.xlabel('社区大小')
#         plt.ylabel('频次')
#         plt.title('社区大小直方图')
        
#         # 3. 权重分布
#         plt.subplot(2, 2, 3)
#         weights = [edge['normlize_weight'] for edge in self.edges]
#         plt.hist(weights, bins=50, alpha=0.7, edgecolor='black')
#         plt.xlabel('标准化权重')
#         plt.ylabel('频次')
#         plt.title('边权重分布')
        
#         # 4. 社区数量统计
#         plt.subplot(2, 2, 4)
#         sizes_stats = {
#             '小社区 (1-5)': sum(1 for s in sizes if 1 <= s <= 5),
#             '中社区 (6-20)': sum(1 for s in sizes if 6 <= s <= 20),
#             '大社区 (21-50)': sum(1 for s in sizes if 21 <= s <= 50),
#             '超大社区 (>50)': sum(1 for s in sizes if s > 50)
#         }
        
#         plt.pie(sizes_stats.values(), labels=sizes_stats.keys(), autopct='%1.1f%%')
#         plt.title('社区规模分布')
        
#         plt.tight_layout()
        
#         if output_dir:
#             plt.savefig(f"{output_dir}/community_analysis.png", dpi=300, bbox_inches='tight')
#         plt.show()

# def main():
#     parser = argparse.ArgumentParser(description='实体社区检测分析')
#     parser.add_argument('--input', '-i', required=True, help='输入文件路径')
#     parser.add_argument('--output', '-o', default='community_entities.jsonl', help='输出文件路径')
#     parser.add_argument('--method', '-m', default='compare', 
#                        choices=['louvain', 'leiden', 'label_propagation', 'greedy_modularity', 'compare'], 
#                        help='社区检测方法')
#     parser.add_argument('--resolution', '-r', type=float, default=1.0, 
#                        help='Leiden算法的分辨率参数')
#     parser.add_argument('--visualize', '-v', action='store_true', help='生成可视化图表')
#     parser.add_argument('--output_dir', '-d', help='可视化图表输出目录')
    
#     args = parser.parse_args()
    
#     # 创建社区检测器
#     detector = EntityCommunityDetector(args.input)
    
#     # 执行社区检测流程
#     detector.load_data()
#     detector.build_graph()
    
#     # 选择社区检测方法
#     method_name = args.method
#     if args.method == 'louvain':
#         detector.louvain_detection()
#     elif args.method == 'leiden':
#         detector.leiden_detection(args.resolution)
#     elif args.method == 'label_propagation':
#         detector.label_propagation_detection()
#     elif args.method == 'greedy_modularity':
#         detector.greedy_modularity_detection()
#     elif args.method == 'compare':
#         method_name, _ = detector.compare_methods()
    
#     # 分析结果
#     detector.analyze_communities()
    
#     # 保存结果
#     detector.save_results(args.output, method_name)
    
#     # 可视化（可选）
#     if args.visualize:
#         detector.visualize_communities(args.output_dir)

# if __name__ == "__main__":
#     # 如果直接运行，使用默认参数
#     import sys
#     if len(sys.argv) == 1:
#         # 默认设置
#         input_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_edge_gpt4.1_mini_new_type_with_QA.jsonl"
#         output_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/undirected_community_entities.jsonl"
        
#         print("使用默认参数运行...")
#         detector = EntityCommunityDetector(input_file)
#         detector.load_data()
#         detector.build_graph()
        
#         # 比较所有方法并选择最佳
#         method_name, _ = detector.compare_methods()
        
#         detector.analyze_communities()
#         detector.save_results(output_file, method_name)
        
#         print("社区检测完成！")
#     else:
#         main()









import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import matplotlib.font_manager as fm


font_path = '/data/FinAi_Mapping_Knowledge/chenmingzhen/.fonts/simsun.ttc'

font = fm.FontProperties(fname=font_path, size=16)

times = "/data/FinAi_Mapping_Knowledge/chenmingzhen/.fonts/Times_New_Roman.ttf"
custom_font = fm.FontProperties(fname=times, size=16)

class EntityCommunityDetector:
    def __init__(self, file_path, directed=True):
        self.file_path = file_path
        self.directed = directed
        self.edges = []
        self.entities = set()
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.graph = None
        self.communities = None
        
    def load_data(self):
        """加载实体连接数据"""
        print("正在加载数据...")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.edges.append(data)
                self.entities.add(data['node1'])
                self.entities.add(data['node2'])
        
        # 创建实体到ID的映射
        self.entities = list(self.entities)
        self.entity_to_id = {entity: idx for idx, entity in enumerate(self.entities)}
        self.id_to_entity = {idx: entity for idx, entity in enumerate(self.entities)}
        
        print(f"加载完成：{len(self.entities)} 个实体，{len(self.edges)} 条边")
        
    def build_graph(self):
        """构建图结构"""
        print(f"正在构建{'有向' if self.directed else '无向'}图结构...")
        
        # 构建NetworkX图
        if self.directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        
        # 添加节点
        for entity in self.entities:
            self.graph.add_node(entity)
        
        # 添加边，使用normalize_weight作为权重
        for edge in self.edges:
            node1, node2 = edge['node1'], edge['node2']
            weight = edge['normlize_weight']
            
            # 如果边已存在，取最大权重
            if self.graph.has_edge(node1, node2):
                existing_weight = self.graph[node1][node2]['weight']
                weight = max(weight, existing_weight)
            
            self.graph.add_edge(node1, node2, weight=weight)
        
        print(f"图构建完成：{self.graph.number_of_nodes()} 个节点，{self.graph.number_of_edges()} 条边")
        
    def leiden_detection(self, resolution=1.0):
        """使用Leiden算法进行社区检测"""
        print("正在执行Leiden社区检测...")
        
        try:
            import leidenalg
            import igraph as ig
            
            # 将NetworkX图转换为igraph
            print("正在转换图格式...")
            
            # 创建igraph图
            g = ig.Graph(directed=self.directed)
            
            # 添加节点
            node_list = list(self.graph.nodes())
            g.add_vertices(len(node_list))
            
            # 创建节点名称到索引的映射
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            
            # 添加边和权重
            edges = []
            weights = []
            
            if self.directed:
                # 有向图处理
                for edge in self.graph.edges(data=True):
                    node1, node2, data = edge
                    idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                    edges.append((idx1, idx2))
                    weights.append(data.get('weight', 1.0))
            else:
                # 无向图处理
                for edge in self.graph.edges(data=True):
                    node1, node2, data = edge
                    idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                    edges.append((idx1, idx2))
                    weights.append(data.get('weight', 1.0))
            
            g.add_edges(edges)
            g.es['weight'] = weights
            
            # 执行Leiden算法
            print(f"正在执行Leiden算法，分辨率参数: {resolution}")
            
            # 根据图类型选择合适的分区类型
            if self.directed:
                # 有向图使用CPMVertexPartition
                if resolution != 1.0:
                    partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, 
                                                       weights=weights, resolution_parameter=resolution)
                else:
                    partition = leidenalg.find_partition(g, leidenalg.CPMVertexPartition, 
                                                       weights=weights)
            else:
                # 无向图使用RBConfigurationVertexPartition或ModularityVertexPartition
                if resolution != 1.0:
                    partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
                                                       weights=weights, resolution_parameter=resolution)
                else:
                    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, 
                                                       weights=weights)
            
            # 转换为聚类标签
            community_labels = np.zeros(len(self.entities))
            for idx, community_id in enumerate(partition.membership):
                entity = node_list[idx]
                entity_idx = self.entity_to_id[entity]
                community_labels[entity_idx] = community_id
                
            self.communities = community_labels.astype(int)
            
            # 计算模块度
            modularity = partition.modularity
            print(f"Leiden检测完成，发现 {len(set(community_labels))} 个社区，模块度: {modularity:.3f}")
            
            return community_labels.astype(int), modularity
            
        except ImportError:
            print("错误：未安装所需的包")
            print("请使用以下命令安装：")
            print("pip install leidenalg")
            print("pip install python-igraph")
            return None, None
        except Exception as e:
            print(f"Leiden算法执行失败: {e}")
            return None, None
    
    def optimize_resolution(self, resolution_range=(0.01, 3.0), num_points=20):
        """优化分辨率参数"""
        print(f"\n正在优化分辨率参数，范围: {resolution_range}, 测试点数: {num_points}")
        print("=" * 60)
        
        resolutions = np.linspace(resolution_range[0], resolution_range[1], num_points)
        best_modularity = -1
        best_communities = None
        best_resolution = 1.0
        
        results = []
        
        for resolution in resolutions:
            try:
                communities, modularity = self.leiden_detection(resolution)
                if modularity is not None and communities is not None:
                    num_communities = len(set(communities))
                    results.append({
                        'resolution': resolution,
                        'modularity': modularity,
                        'num_communities': num_communities,
                        'communities': communities
                    })
                    
                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_communities = communities
                        best_resolution = resolution
                    
                    print(f"分辨率 {resolution:.3f}: 模块度 {modularity:.3f}, 社区数 {num_communities}")
                else:
                    print(f"分辨率 {resolution:.3f}: 执行失败")
                    
            except Exception as e:
                print(f"分辨率 {resolution:.3f}: 执行失败 - {e}")
        
        if best_communities is not None:
            print(f"\n最佳分辨率: {best_resolution:.3f}, 模块度: {best_modularity:.3f}")
            self.communities = best_communities
            
            # 绘制分辨率优化结果
            self.plot_resolution_optimization(results)
            
            return best_resolution, best_modularity
        else:
            print("\n所有分辨率参数都失败了")
            return None, None
    
    def plot_resolution_optimization(self, results):
        """绘制分辨率优化结果"""
        if not results:
            return
            
        resolutions = [r['resolution'] for r in results]
        modularities = [r['modularity'] for r in results]
        num_communities = [r['num_communities'] for r in results]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(resolutions, modularities, 'b-o', markersize=4)
        plt.xlabel('分辨率参数',fontproperties=font)
        plt.ylabel('模块度',fontproperties=font)
        plt.title('分辨率 vs 模块度', fontproperties=font)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(resolutions, num_communities, 'r-o', markersize=4)
        plt.xlabel('分辨率参数', fontproperties=font)
        plt.ylabel('社区数量', fontproperties=font)
        plt.title('分辨率 vs 社区数量', fontproperties=font)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_communities(self):
        """分析社区检测结果"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        print("\n社区检测结果分析:")
        print("-" * 50)
        
        # 统计每个社区的大小
        community_sizes = Counter(self.communities)
        print(f"总共发现 {len(community_sizes)} 个社区")
        
        # 按社区大小排序
        sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
        
        for i, (community_id, size) in enumerate(sorted_communities[:10]):  # 只显示前10个最大的社区
            print(f"社区 {community_id} (排名 {i+1}): {size} 个实体")
            
            # 显示该社区中的实体（最多显示10个）
            community_entities = [self.entities[j] for j in range(len(self.entities)) 
                                if self.communities[j] == community_id]
            
            if len(community_entities) <= 10:
                print(f"  实体: {', '.join(community_entities)}")
            else:
                print(f"  实体: {', '.join(community_entities[:10])}... (共{len(community_entities)}个)")
            print()
        
        if len(sorted_communities) > 10:
            print(f"... 还有 {len(sorted_communities) - 10} 个社区")
        
        # 统计社区大小分布
        sizes = list(community_sizes.values())
        print(f"\n社区大小统计:")
        print(f"  平均大小: {np.mean(sizes):.2f}")
        print(f"  中位数大小: {np.median(sizes):.2f}")
        print(f"  最大社区: {max(sizes)} 个实体")
        print(f"  最小社区: {min(sizes)} 个实体")
        print(f"  标准差: {np.std(sizes):.2f}")
        
        # 如果是有向图，分析入度和出度
        if self.directed:
            print(f"\n有向图分析:")
            in_degrees = dict(self.graph.in_degree())
            out_degrees = dict(self.graph.out_degree())
            
            print(f"  平均入度: {np.mean(list(in_degrees.values())):.2f}")
            print(f"  平均出度: {np.mean(list(out_degrees.values())):.2f}")
            
            # 找出入度和出度最高的节点
            max_in_node = max(in_degrees, key=in_degrees.get)
            max_out_node = max(out_degrees, key=out_degrees.get)
            print(f"  最大入度节点: {max_in_node} (入度: {in_degrees[max_in_node]})")
            print(f"  最大出度节点: {max_out_node} (出度: {out_degrees[max_out_node]})")
    
    def save_results(self, output_file, method_name="Leiden", resolution=None):
        """保存社区检测结果"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        print(f"正在保存结果到 {output_file}")
        
        # 创建结果数据
        results = []
        for i, entity in enumerate(self.entities):
            # 获取原始实体类型信息
            entity_type = "未知"
            for edge in self.edges:
                if edge['node1'] == entity:
                    entity_type = edge['node1_type']
                    break
                elif edge['node2'] == entity:
                    entity_type = edge['node2_type']
                    break
            
            # 计算节点的度信息
            if self.directed:
                in_degree = self.graph.in_degree(entity)
                out_degree = self.graph.out_degree(entity)
                degree_info = {'in_degree': in_degree, 'out_degree': out_degree}
            else:
                degree = self.graph.degree(entity)
                degree_info = {'degree': degree}
            
            result = {
                'entity': entity,
                'entity_type': entity_type,
                'community_id': int(self.communities[i]),
                'method': method_name,
                'graph_type': 'directed' if self.directed else 'undirected'
            }
            result.update(degree_info)
            
            if resolution is not None:
                result['resolution'] = resolution
                
            results.append(result)
        
        # 保存为JSON Lines格式
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"结果已保存到 {output_file}")
        
        # 保存社区统计信息
        stats_file = output_file.replace('.jsonl', '_stats.json')
        community_stats = {}
        for i, entity in enumerate(self.entities):
            community_id = int(self.communities[i])
            if community_id not in community_stats:
                community_stats[community_id] = {
                    'entities': [],
                    'size': 0,
                    'entity_types': Counter()
                }
            community_stats[community_id]['entities'].append(entity)
            community_stats[community_id]['size'] += 1
            
            # 统计实体类型
            entity_type = "未知"
            for edge in self.edges:
                if edge['node1'] == entity:
                    entity_type = edge['node1_type']
                    break
                elif edge['node2'] == entity:
                    entity_type = edge['node2_type']
                    break
            community_stats[community_id]['entity_types'][entity_type] += 1
        
        # 转换Counter为普通字典以便JSON序列化
        for community_id in community_stats:
            community_stats[community_id]['entity_types'] = dict(community_stats[community_id]['entity_types'])
        
        # 添加图类型信息
        meta_info = {
            'graph_type': 'directed' if self.directed else 'undirected',
            'method': method_name,
            'total_entities': len(self.entities),
            'total_edges': len(self.edges),
            'num_communities': len(community_stats)
        }
        
        if resolution is not None:
            meta_info['resolution'] = resolution
        
        stats_data = {
            'meta': meta_info,
            'communities': community_stats
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        print(f"社区统计信息已保存到 {stats_file}")
    
    def visualize_communities(self, output_dir=None):
        """可视化社区检测结果"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)
        
        # 创建更大的图形
        plt.figure(figsize=(15, 10))
        
        # 1. 社区大小分布
        plt.subplot(2, 3, 1)
        community_sizes = Counter(self.communities)
        plt.bar(range(len(community_sizes)), sorted(community_sizes.values(), reverse=True))
        plt.xlabel('社区排名', fontproperties=font)
        plt.ylabel('实体数量', fontproperties=font)
        plt.title('社区大小分布 (按大小排序)', fontproperties=font)
        
        # 2. 社区大小直方图
        plt.subplot(2, 3, 2)
        sizes = list(community_sizes.values())
        plt.hist(sizes, bins=min(20, len(set(sizes))), alpha=0.7, edgecolor='black')
        plt.xlabel('社区大小', fontproperties=font)
        plt.ylabel('频次', fontproperties=font)
        plt.title('社区大小直方图', fontproperties=font)
        
        # 3. 权重分布
        plt.subplot(2, 3, 3)
        weights = [edge['normlize_weight'] for edge in self.edges]
        plt.hist(weights, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('标准化权重', fontproperties=font)
        plt.ylabel('频次', fontproperties=font)
        plt.title('边权重分布', fontproperties=font)
        
        # 4. 社区数量统计
        plt.subplot(2, 3, 4)
        sizes_stats = {
            '小社区 (1-5)': sum(1 for s in sizes if 1 <= s <= 5),
            '中社区 (6-20)': sum(1 for s in sizes if 6 <= s <= 20),
            '大社区 (21-50)': sum(1 for s in sizes if 21 <= s <= 50),
            '超大社区 (>50)': sum(1 for s in sizes if s > 50)
        }
        
        plt.pie(sizes_stats.values(), labels=sizes_stats.keys(), autopct='%1.1f%%')
        plt.title('社区规模分布', fontproperties=font)
        
        # 5. 度分布
        plt.subplot(2, 3, 5)
        if self.directed:
            in_degrees = [self.graph.in_degree(node) for node in self.graph.nodes()]
            out_degrees = [self.graph.out_degree(node) for node in self.graph.nodes()]
            plt.hist(in_degrees, bins=30, alpha=0.7, label='入度', edgecolor='black')
            plt.hist(out_degrees, bins=30, alpha=0.7, label='出度', edgecolor='black')
            plt.xlabel('度', fontproperties=font)
            plt.ylabel('频次', fontproperties=font)
            plt.title('度分布 (有向图)', fontproperties=font)
            plt.legend()
        else:
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            plt.hist(degrees, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('度', fontproperties=font)
            plt.ylabel('频次', fontproperties=font)
            plt.title('度分布 (无向图)', fontproperties=font)
        
        # 6. 图的基本信息
        plt.subplot(2, 3, 6)
        info_text = f"图类型: {'有向图' if self.directed else '无向图'}\n"
        info_text += f"节点数: {self.graph.number_of_nodes()}\n"
        info_text += f"边数: {self.graph.number_of_edges()}\n"
        info_text += f"社区数: {len(set(self.communities))}\n"
        
        if self.directed:
            # 计算强连通分量
            scc = list(nx.strongly_connected_components(self.graph))
            info_text += f"强连通分量数: {len(scc)}\n"
            info_text += f"最大强连通分量: {max(len(c) for c in scc)}\n"
        else:
            # 计算连通分量
            cc = list(nx.connected_components(self.graph))
            info_text += f"连通分量数: {len(cc)}\n"
            info_text += f"最大连通分量: {max(len(c) for c in cc)}\n"
        
        plt.text(0.1, 0.5, info_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='center')
        plt.axis('off')
        plt.title('图的基本信息', fontproperties=font)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/community_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='实体社区检测分析 (Leiden算法)')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', default='community_entities.jsonl', help='输出文件路径')
    parser.add_argument('--resolution', '-r', type=float, default=0.1, 
                       help='Leiden算法的分辨率参数')
    parser.add_argument('--directed', '-d', action='store_true', 
                       help='使用有向图 (默认为无向图)')
    parser.add_argument('--optimize', action='store_true', 
                       help='优化分辨率参数')
    parser.add_argument('--resolution_range', nargs=2, type=float, default=[0.01, 3.0],
                       help='分辨率参数优化范围')
    parser.add_argument('--num_points', type=int, default=20,
                       help='分辨率参数优化测试点数')
    parser.add_argument('--visualize', '-v', action='store_true', help='生成可视化图表')
    parser.add_argument('--output_dir', help='可视化图表输出目录')
    
    args = parser.parse_args()
    
    # 创建社区检测器
    detector = EntityCommunityDetector(args.input, directed=args.directed)
    
    # 执行社区检测流程
    detector.load_data()
    detector.build_graph()
    
    # 选择执行方式
    if args.optimize:
        # 优化分辨率参数
        best_resolution, best_modularity = detector.optimize_resolution(
            resolution_range=tuple(args.resolution_range),
            num_points=args.num_points
        )
        if best_resolution is not None:
            detector.save_results(args.output, "Leiden", best_resolution)
    else:
        # 使用指定的分辨率参数
        communities, modularity = detector.leiden_detection(args.resolution)
        if communities is not None:
            detector.save_results(args.output, "Leiden", args.resolution)
    
    # 分析结果
    detector.analyze_communities()
    
    # 可视化（可选）
    if args.visualize:
        detector.visualize_communities(args.output_dir)

if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    if len(sys.argv) == 1:
        # 默认设置
        input_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/graph_with_description/new_edge_gpt4.1_mini_with_desciption.jsonl"
        output_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/graph_with_description/directed_community_entities.jsonl"
        
        print("使用默认参数运行...")
        detector = EntityCommunityDetector(input_file, directed=True)  # 默认使用有向图
        detector.load_data()
        detector.build_graph()
        
        # 优化分辨率参数
        best_resolution, best_modularity = detector.optimize_resolution()
        
        detector.analyze_communities()
        if best_resolution is not None:
            detector.save_results(output_file, "Leiden", best_resolution)
        
        print("社区检测完成！")
    else:
        main()