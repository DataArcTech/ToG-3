import json
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

import matplotlib.font_manager as fm


font_path = '/data/FinAi_Mapping_Knowledge/chenmingzhen/.fonts/simsun.ttc'

font = fm.FontProperties(fname=font_path, size=16)

times = "/data/FinAi_Mapping_Knowledge/chenmingzhen/.fonts/Times_New_Roman.ttf"
custom_font = fm.FontProperties(fname=times, size=16)


class EntityCommunityDetector:
    def __init__(self, file_path):
        self.file_path = file_path
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
        print("正在构建图结构...")
        # 构建NetworkX图
        self.graph = nx.Graph()
        
        # 添加节点
        for entity in self.entities:
            self.graph.add_node(entity)
        
        # 添加边，使用normalize_weight作为权重
        for edge in self.edges:
            node1, node2 = edge['node1'], edge['node2']
            # weight = edge['normlize_weight']
            weight = edge['weight']
            
            # 如果边已存在，取最大权重
            if self.graph.has_edge(node1, node2):
                existing_weight = self.graph[node1][node2]['weight']
                weight = max(weight, existing_weight)
            
            self.graph.add_edge(node1, node2, weight=weight)
        
        print(f"图构建完成：{self.graph.number_of_nodes()} 个节点，{self.graph.number_of_edges()} 条边")
        
    def louvain_detection(self):
        """使用Louvain算法进行社区检测"""
        print("正在执行Louvain社区检测...")
        
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph, weight='weight')
            
            # 转换为聚类标签
            community_labels = np.zeros(len(self.entities))
            for entity, community_id in partition.items():
                entity_idx = self.entity_to_id[entity]
                community_labels[entity_idx] = community_id
                
            self.communities = community_labels.astype(int)
            
            # 计算模块度
            modularity = community_louvain.modularity(partition, self.graph, weight='weight')
            print(f"Louvain检测完成，发现 {len(set(community_labels))} 个社区，模块度: {modularity:.3f}")
            
            return community_labels.astype(int), modularity
            
        except ImportError:
            print("错误：未安装python-louvain包")
            print("请使用以下命令安装：pip install python-louvain")
            return None, None
    
    def leiden_detection(self, resolution=1.0):
        """使用Leiden算法进行社区检测"""
        print("正在执行Leiden社区检测...")
        
        try:
            import leidenalg
            import igraph as ig
            
            # 将NetworkX图转换为igraph
            print("正在转换图格式...")
            
            # 创建igraph图
            g = ig.Graph()
            
            # 添加节点
            node_list = list(self.graph.nodes())
            g.add_vertices(len(node_list))
            
            # 创建节点名称到索引的映射
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            
            # 添加边和权重
            edges = []
            weights = []
            for edge in self.graph.edges(data=True):
                node1, node2, data = edge
                idx1, idx2 = node_to_idx[node1], node_to_idx[node2]
                edges.append((idx1, idx2))
                weights.append(data.get('weight', 1.0))
            
            g.add_edges(edges)
            g.es['weight'] = weights
            
            # 执行Leiden算法
            print(f"正在执行Leiden算法，分辨率参数: {resolution}")
            
            # 使用RBConfigurationVertexPartition支持分辨率参数
            if resolution != 1.0:
                partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, 
                                                   weights=weights, resolution_parameter=resolution)
            else:
                # 对于默认分辨率，使用ModularityVertexPartition
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
    
    def label_propagation_detection(self):
        """使用标签传播算法进行社区检测"""
        print("正在执行标签传播社区检测...")
        
        try:
            # 使用NetworkX内置的标签传播算法
            communities = list(nx.community.label_propagation_communities(self.graph))
            
            # 转换为聚类标签
            community_labels = np.zeros(len(self.entities))
            for community_id, community in enumerate(communities):
                for entity in community:
                    entity_idx = self.entity_to_id[entity]
                    community_labels[entity_idx] = community_id
            
            self.communities = community_labels.astype(int)
            
            # 计算模块度
            modularity = nx.community.modularity(self.graph, communities, weight='weight')
            print(f"标签传播检测完成，发现 {len(communities)} 个社区，模块度: {modularity:.3f}")
            
            return community_labels.astype(int), modularity
            
        except Exception as e:
            print(f"标签传播算法执行失败: {e}")
            return None, None
    
    def greedy_modularity_detection(self):
        """使用贪心模块度优化算法进行社区检测"""
        print("正在执行贪心模块度优化社区检测...")
        
        try:
            # 使用NetworkX内置的贪心模块度优化算法
            communities = list(nx.community.greedy_modularity_communities(self.graph, weight='weight'))
            
            # 转换为聚类标签
            community_labels = np.zeros(len(self.entities))
            for community_id, community in enumerate(communities):
                for entity in community:
                    entity_idx = self.entity_to_id[entity]
                    community_labels[entity_idx] = community_id
            
            self.communities = community_labels.astype(int)
            
            # 计算模块度
            modularity = nx.community.modularity(self.graph, communities, weight='weight')
            print(f"贪心模块度优化检测完成，发现 {len(communities)} 个社区，模块度: {modularity:.3f}")
            
            return community_labels.astype(int), modularity
            
        except Exception as e:
            print(f"贪心模块度优化算法执行失败: {e}")
            return None, None
    
    def compare_methods(self):
        """比较不同的社区检测方法"""
        print("\n比较不同的社区检测方法:")
        print("=" * 60)
        
        methods = [
            ("Louvain", self.louvain_detection),
            ("Leiden", self.leiden_detection),
            # ("Label Propagation", self.label_propagation_detection),
            # ("Greedy Modularity", self.greedy_modularity_detection)
        ]
        
        results = {}
        
        for method_name, method_func in methods:
            print(f"\n正在测试 {method_name} 方法...")
            try:
                if method_name == "Leiden":
                    # 测试不同的分辨率参数
                    best_modularity = -1
                    best_communities = None
                    best_resolution = 1.0
                    
                    # 测试分辨率参数，包括默认值
                    resolutions = [0.5, 1.0, 1.5, 2.0]
                    for resolution in resolutions:
                        try:
                            communities, modularity = method_func(resolution)
                            if modularity is not None and modularity > best_modularity:
                                best_modularity = modularity
                                best_communities = communities
                                best_resolution = resolution
                            print(f"  分辨率 {resolution}: 模块度 {modularity:.3f}, 社区数 {len(set(communities))}")
                        except Exception as e:
                            print(f"  分辨率 {resolution}: 执行失败 - {e}")
                    
                    results[method_name] = {
                        'communities': best_communities,
                        'modularity': best_modularity,
                        'num_communities': len(set(best_communities)) if best_communities is not None else 0,
                        'resolution': best_resolution
                    }
                    if best_communities is not None:
                        print(f"最佳分辨率: {best_resolution}, 模块度: {best_modularity:.3f}")
                    else:
                        print("所有分辨率参数都失败了")
                else:
                    communities, modularity = method_func()
                    results[method_name] = {
                        'communities': communities,
                        'modularity': modularity,
                        'num_communities': len(set(communities)) if communities is not None else 0
                    }
                    
            except Exception as e:
                print(f"{method_name} 方法执行失败: {e}")
                results[method_name] = {
                    'communities': None,
                    'modularity': None,
                    'num_communities': 0
                }
        
        # 显示比较结果
        print("\n比较结果:")
        print("-" * 60)
        print(f"{'方法':<15} {'社区数':<10} {'模块度':<10} {'状态'}")
        print("-" * 60)
        
        for method_name, result in results.items():
            status = "成功" if result['communities'] is not None else "失败"
            modularity_str = f"{result['modularity']:.3f}" if result['modularity'] is not None else "N/A"
            print(f"{method_name:<15} {result['num_communities']:<10} {modularity_str:<10} {status}")
        
        # 选择最佳方法
        best_method = max(results.items(), 
                         key=lambda x: x[1]['modularity'] if x[1]['modularity'] is not None else -1)
        
        if best_method[1]['communities'] is not None:
            print(f"\n最佳方法: {best_method[0]} (模块度: {best_method[1]['modularity']:.3f})")
            self.communities = best_method[1]['communities']
            return best_method[0], best_method[1]
        else:
            print("\n所有方法都失败了")
            return None, None
    
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
    
    def save_results(self, output_file, method_name=None):
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
            
            results.append({
                'entity': entity,
                'entity_type': entity_type,
                'community_id': int(self.communities[i]),
                'method': method_name if method_name else "unknown"
            })
        
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
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(community_stats, f, ensure_ascii=False, indent=2)
        
        print(f"社区统计信息已保存到 {stats_file}")
    
    def visualize_network_graph(self, output_dir=None, max_nodes=100, figsize=(200, 150)):
        """可视化网络图"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        print("正在生成网络图可视化...")
        
        # 如果节点太多，选择子图
        if len(self.graph.nodes()) > max_nodes:
            print(f"节点数量过多({len(self.graph.nodes())})，随机选择{max_nodes}个节点进行可视化")
            # 优先选择度数较高的节点
            node_degrees = dict(self.graph.degree())
            selected_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
        
        # 创建社区颜色映射
        community_colors = {}
        unique_communities = set()
        for node in subgraph.nodes():
            if node in self.entity_to_id:
                community_id = self.communities[self.entity_to_id[node]]
                unique_communities.add(community_id)
        
        # 生成颜色
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
        for i, community_id in enumerate(unique_communities):
            community_colors[community_id] = colors[i]
        
        # 设置节点颜色
        node_colors = []
        for node in subgraph.nodes():
            if node in self.entity_to_id:
                community_id = self.communities[self.entity_to_id[node]]
                node_colors.append(community_colors[community_id])
            else:
                node_colors.append('#808080')  # 灰色表示未知
        
        # 计算布局
        print("正在计算图布局...")
        try:
            # 使用spring layout，根据权重调整
            pos = nx.spring_layout(subgraph, k=1, iterations=50, weight='weight')
        except:
            # 如果spring layout失败，使用随机布局
            pos = nx.random_layout(subgraph)
        
        # 创建图形
        plt.figure(figsize=figsize)
        
        # 绘制边
        edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
        edge_widths = [min(w * 2, 5) for w in edge_weights]  # 限制边的宽度
        
        nx.draw_networkx_edges(subgraph, pos, 
                             width=edge_widths, 
                             alpha=0.6, 
                             edge_color='gray')
        
        # 绘制节点
        node_sizes = [subgraph.degree(node) * 100 + 100 for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, 
                             node_color=node_colors, 
                             node_size=node_sizes,
                             alpha=0.8)
        
        # 绘制标签（只对度数较高的节点）
        high_degree_nodes = [node for node in subgraph.nodes() if subgraph.degree(node) > 2]
        if len(high_degree_nodes) <= 20:  # 只有少于20个高度数节点时才显示标签
            labels = {node: node[:10] + '...' if len(node) > 10 else node for node in high_degree_nodes}
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)
        
        plt.title(f"实体关系网络图 (社区检测结果)\n显示 {len(subgraph.nodes())} 个节点，{len(unique_communities)} 个社区", 
                 fontsize=16, fontweight='bold', fontproperties=font)
        plt.axis('off')
        
        # 添加图例
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=community_colors[cid], 
                                    markersize=10, label=f'社区 {cid}')
                         for cid in sorted(unique_communities)]
        
        if len(legend_elements) <= 20:  # 只有少于20个社区时才显示图例
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/network_graph.pdf", format='pdf', dpi=600, bbox_inches='tight')
        plt.show()
    
    def visualize_community_subgraphs(self, output_dir=None, top_n=6, figsize=(20, 15)):
        """可视化最大的几个社区的子图"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        print(f"正在生成前{top_n}个社区的子图可视化...")
        
        # 统计社区大小
        community_sizes = Counter(self.communities)
        top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, (community_id, size) in enumerate(top_communities):
            if i >= len(axes):
                break
            
            # 获取社区内的节点
            community_nodes = [self.entities[j] for j in range(len(self.entities)) 
                             if self.communities[j] == community_id]
            
            # 创建子图
            subgraph = self.graph.subgraph(community_nodes)
            
            # 计算布局
            if len(community_nodes) > 1:
                try:
                    pos = nx.spring_layout(subgraph, k=1, iterations=50)
                except:
                    pos = nx.random_layout(subgraph)
            else:
                pos = {community_nodes[0]: (0, 0)}
            
            # 绘制子图
            ax = axes[i]
            
            # 绘制边
            if subgraph.edges():
                edge_weights = [subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()]
                edge_widths = [min(w * 2, 5) for w in edge_weights]
                nx.draw_networkx_edges(subgraph, pos, ax=ax, width=edge_widths, alpha=0.6, edge_color='gray')
            
            # 绘制节点
            node_sizes = [subgraph.degree(node) * 50 + 50 for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos, ax=ax, node_color='lightblue', 
                                 node_size=node_sizes, alpha=0.8)
            
            # 绘制标签
            if len(community_nodes) <= 10:
                labels = {node: node[:8] + '...' if len(node) > 8 else node for node in community_nodes}
                nx.draw_networkx_labels(subgraph, pos, labels, ax=ax, font_size=8)
            
            ax.set_title(f"社区 {community_id} ({size} 个节点)", fontweight='bold', fontproperties=font)
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(len(top_communities), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("主要社区子图可视化", fontsize=16, fontweight='bold', fontproperties=font)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/community_subgraphs.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_communities(self, output_dir=None):
        """可视化社区检测结果"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        if output_dir:
            Path(output_dir).mkdir(exist_ok=True)
        
        print("正在生成统计图表...")
        
        # 1. 社区大小分布
        plt.figure(figsize=(20, 15))
        
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
        weights = [edge['weight'] for edge in self.edges]
        plt.hist(weights, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('权重', fontproperties=font)
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
        degrees = [d for n, d in self.graph.degree()]
        plt.hist(degrees, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('节点度数', fontproperties=font)
        plt.ylabel('频次', fontproperties=font)
        plt.title('节点度分布', fontproperties=font)
        plt.yscale('log')
        
        # 6. 社区内外连接比例
        plt.subplot(2, 3, 6)
        intra_edges = 0  # 社区内部边数
        inter_edges = 0  # 社区间边数
        
        for edge in self.graph.edges():
            node1, node2 = edge
            if node1 in self.entity_to_id and node2 in self.entity_to_id:
                comm1 = self.communities[self.entity_to_id[node1]]
                comm2 = self.communities[self.entity_to_id[node2]]
                if comm1 == comm2:
                    intra_edges += 1
                else:
                    inter_edges += 1
        
        plt.bar(['社区内部', '社区间'], [intra_edges, inter_edges])
        plt.ylabel('边数', fontproperties=font)
        plt.title('社区内外连接分布', fontproperties=font)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/community_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 生成网络图可视化
        self.visualize_network_graph(output_dir)
        
        # 生成社区子图可视化
        self.visualize_community_subgraphs(output_dir)
    
    def visualize_interactive_network(self, output_dir=None, max_nodes=200):
        """生成交互式网络图（使用plotly）"""
        if self.communities is None:
            print("请先执行社区检测")
            return
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.offline import plot
        except ImportError:
            print("未安装plotly，跳过交互式可视化")
            return
        
        print("正在生成交互式网络图...")
        
        # 如果节点太多，选择子图
        if len(self.graph.nodes()) > max_nodes:
            print(f"节点数量过多({len(self.graph.nodes())})，随机选择{max_nodes}个节点进行可视化")
            node_degrees = dict(self.graph.degree())
            selected_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph(selected_nodes)
        else:
            subgraph = self.graph
        
        # 计算布局
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # 准备节点数据
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # 节点信息
            degree = subgraph.degree(node)
            if node in self.entity_to_id:
                community_id = self.communities[self.entity_to_id[node]]
                node_text.append(f"实体: {node}<br>社区: {community_id}<br>度数: {degree}")
                node_color.append(community_id)
            else:
                node_text.append(f"实体: {node}<br>度数: {degree}")
                node_color.append(-1)
            
            node_size.append(max(degree * 5, 10))
        
        # 准备边数据
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = subgraph[edge[0]][edge[1]].get('weight', 1)
            edge_info.append(f"权重: {weight}")
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=0.5, color='gray'),
                                hoverinfo='none',
                                mode='lines'))
        
        # 添加节点
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers',
                                hoverinfo='text',
                                text=node_text,
                                marker=dict(size=node_size,
                                          color=node_color,
                                          colorscale='Set3',
                                          line=dict(width=2))))
        
        fig.update_layout(title=f"交互式实体关系网络图<br><sub>显示 {len(subgraph.nodes())} 个节点</sub>",
                         titlefont_size=16,
                         showlegend=False,
                         hovermode='closest',
                         margin=dict(b=20,l=5,r=5,t=40),
                         annotations=[ dict(
                             text="拖拽节点进行交互，悬停查看详情",
                             showarrow=False,
                             xref="paper", yref="paper",
                             x=0.005, y=-0.002,
                             xanchor='left', yanchor='bottom',
                             font=dict(color='gray', size=12)
                         )],
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        
        if output_dir:
            plot(fig, filename=f"{output_dir}/interactive_network.html", auto_open=False)
            print(f"交互式网络图已保存到 {output_dir}/interactive_network.html")
        else:
            fig.show()
    
    def generate_graph_statistics(self, output_dir=None):
        """生成详细的图统计信息"""
        if self.graph is None:
            print("请先构建图")
            return
        
        print("正在生成图统计信息...")
        
        stats = {
            'basic_stats': {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_connected(self.graph),
                'num_components': nx.number_connected_components(self.graph)
            },
            'degree_stats': {
                'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
                'max_degree': max(dict(self.graph.degree()).values()),
                'min_degree': min(dict(self.graph.degree()).values())
            },
            'clustering': {
                'avg_clustering': nx.average_clustering(self.graph),
                'transitivity': nx.transitivity(self.graph)
            }
        }
        
        if self.communities is not None:
            community_sizes = Counter(self.communities)
            stats['community_stats'] = {
                'num_communities': len(community_sizes),
                'largest_community': max(community_sizes.values()),
                'smallest_community': min(community_sizes.values()),
                'avg_community_size': np.mean(list(community_sizes.values())),
                'modularity': self._calculate_modularity()
            }
        
        # 打印统计信息
        print("\n图统计信息:")
        print("=" * 50)
        print(f"节点数: {stats['basic_stats']['num_nodes']}")
        print(f"边数: {stats['basic_stats']['num_edges']}")
        print(f"密度: {stats['basic_stats']['density']:.4f}")
        print(f"连通性: {'连通' if stats['basic_stats']['is_connected'] else '非连通'}")
        print(f"连通分量数: {stats['basic_stats']['num_components']}")
        print(f"平均度数: {stats['degree_stats']['avg_degree']:.2f}")
        print(f"最大度数: {stats['degree_stats']['max_degree']}")
        print(f"平均聚类系数: {stats['clustering']['avg_clustering']:.4f}")
        print(f"传递性: {stats['clustering']['transitivity']:.4f}")
        
        if 'community_stats' in stats:
            print(f"社区数: {stats['community_stats']['num_communities']}")
            print(f"最大社区大小: {stats['community_stats']['largest_community']}")
            print(f"平均社区大小: {stats['community_stats']['avg_community_size']:.2f}")
            print(f"模块度: {stats['community_stats']['modularity']:.4f}")
        
        # 保存统计信息
        if output_dir:
            with open(f"{output_dir}/graph_statistics.json", 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
            print(f"统计信息已保存到 {output_dir}/graph_statistics.json")
        
        return stats
    
    def _calculate_modularity(self):
        """计算模块度"""
        if self.communities is None:
            return 0
        
        # 创建社区分区
        communities = {}
        for i, community_id in enumerate(self.communities):
            if community_id not in communities:
                communities[community_id] = []
            communities[community_id].append(self.entities[i])
        
        # 转换为networkx社区格式
        community_list = [set(nodes) for nodes in communities.values()]
        
        try:
            return nx.community.modularity(self.graph, community_list, weight='weight')
        except:
            return 0

def main():
    parser = argparse.ArgumentParser(description='实体社区检测分析')
    parser.add_argument('--input', '-i', required=True, help='输入文件路径')
    parser.add_argument('--output', '-o', default='community_entities.jsonl', help='输出文件路径')
    parser.add_argument('--method', '-m', default='compare', 
                       choices=['louvain', 'leiden', 'label_propagation', 'greedy_modularity', 'compare'], 
                       help='社区检测方法')
    parser.add_argument('--resolution', '-r', type=float, default=1.0, 
                       help='Leiden算法的分辨率参数')
    parser.add_argument('--visualize', '-v', action='store_true', help='生成可视化图表')
    parser.add_argument('--output_dir', '-d', help='可视化图表输出目录')
    parser.add_argument('--interactive', action='store_true', help='生成交互式网络图')
    parser.add_argument('--max_nodes', type=int, default=100, help='网络图最大节点数')
    
    args = parser.parse_args()
    
    # 创建社区检测器
    detector = EntityCommunityDetector(args.input)
    
    # 执行社区检测流程
    detector.load_data()
    detector.build_graph()
    
    # 选择社区检测方法
    method_name = args.method
    if args.method == 'louvain':
        detector.louvain_detection()
    elif args.method == 'leiden':
        detector.leiden_detection(args.resolution)
    elif args.method == 'label_propagation':
        detector.label_propagation_detection()
    elif args.method == 'greedy_modularity':
        detector.greedy_modularity_detection()
    elif args.method == 'compare':
        method_name, _ = detector.compare_methods()
    
    # 分析结果
    detector.analyze_communities()
    
    # 生成图统计信息
    detector.generate_graph_statistics(args.output_dir)
    
    # 保存结果
    detector.save_results(args.output, method_name)
    
    # 可视化（可选）
    if args.visualize:
        detector.visualize_communities(args.output_dir)
    
    # 交互式可视化（可选）
    if args.interactive:
        detector.visualize_interactive_network(args.output_dir, args.max_nodes)

if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    if len(sys.argv) == 1:
        # 默认设置
        input_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_edge_gpt4.1_mini_new_type_with_QA.jsonl"
        output_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/undirected_community_entities_1.jsonl"
        output_dir = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/visualizations"
        
        print("使用默认参数运行...")
        detector = EntityCommunityDetector(input_file)
        detector.load_data()
        detector.build_graph()
        
        # 比较所有方法并选择最佳
        method_name, _ = detector.compare_methods()
        
        detector.analyze_communities()
        detector.generate_graph_statistics(output_dir)
        detector.save_results(output_file, method_name)
        
        # 生成所有可视化
        detector.visualize_communities(output_dir)
        detector.visualize_interactive_network(output_dir)
        
        print("社区检测完成！")
    else:
        main()