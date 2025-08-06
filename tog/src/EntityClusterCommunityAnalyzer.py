import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EntityClusteringAnalyzer:
    def __init__(self, clustering_file, community_file):
        """
        初始化分析器
        
        Args:
            clustering_file: 语义聚类结果文件路径
            community_file: 社区划分结果文件路径
        """
        self.clustering_file = clustering_file
        self.community_file = community_file
        self.clustering_data = None
        self.community_data = None
        self.entity_to_cluster = {}
        self.entity_to_community = {}
        self.analysis_results = {}
        
    def load_data(self):
        """加载数据文件"""
        try:
            # 加载语义聚类结果
            with open(self.clustering_file, 'r', encoding='utf-8') as f:
                self.clustering_data = json.load(f)
            
            # 加载社区划分结果
            with open(self.community_file, 'r', encoding='utf-8') as f:
                self.community_data = json.load(f)
            
            print(f"成功加载数据:")
            print(f"- 语义聚类簇数: {len(self.clustering_data['clusters'])}")
            print(f"- 社区数: {len(self.community_data)}")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
        
        return True
    
    def preprocess_data(self):
        """预处理数据，建立实体到聚类和社区的映射"""
        # 建立实体到聚类的映射
        for cluster_id, entities in self.clustering_data['clusters'].items():
            for entity in entities:
                self.entity_to_cluster[entity] = cluster_id
        
        # 建立实体到社区的映射
        for community_id, community_info in self.community_data.items():
            entities = community_info['entities']
            for entity in entities:
                self.entity_to_community[entity] = community_id
        
        print(f"预处理完成:")
        print(f"- 聚类中的实体数: {len(self.entity_to_cluster)}")
        print(f"- 社区中的实体数: {len(self.entity_to_community)}")
    
    def analyze_consistency(self):
        """分析聚类与社区划分的一致性"""
        consistency_results = {}
        
        # 分析每个聚类的一致性
        for cluster_id, entities in self.clustering_data['clusters'].items():
            cluster_communities = []
            missing_entities = []
            
            for entity in entities:
                if entity in self.entity_to_community:
                    cluster_communities.append(self.entity_to_community[entity])
                else:
                    missing_entities.append(entity)
            
            # 计算一致性指标
            if cluster_communities:
                community_counter = Counter(cluster_communities)
                most_common_community = community_counter.most_common(1)[0]
                consistency_ratio = most_common_community[1] / len(cluster_communities)
                coverage_ratio = len(cluster_communities) / len(entities)
                
                consistency_results[cluster_id] = {
                    'entities': entities,
                    'total_entities': len(entities),
                    'found_in_community': len(cluster_communities),
                    'missing_entities': missing_entities,
                    'communities': dict(community_counter),
                    'most_common_community': most_common_community[0],
                    'consistency_ratio': consistency_ratio,
                    'coverage_ratio': coverage_ratio,
                    'is_consistent': len(community_counter) == 1
                }
            else:
                consistency_results[cluster_id] = {
                    'entities': entities,
                    'total_entities': len(entities),
                    'found_in_community': 0,
                    'missing_entities': missing_entities,
                    'communities': {},
                    'most_common_community': None,
                    'consistency_ratio': 0,
                    'coverage_ratio': 0,
                    'is_consistent': False
                }
        
        self.analysis_results['consistency'] = consistency_results
        return consistency_results
    
    def analyze_cross_distribution(self):
        """分析跨分布情况"""
        # 分析社区中的聚类分布
        community_cluster_distribution = defaultdict(lambda: defaultdict(int))
        
        for entity, community_id in self.entity_to_community.items():
            if entity in self.entity_to_cluster:
                cluster_id = self.entity_to_cluster[entity]
                community_cluster_distribution[community_id][cluster_id] += 1
        
        # 分析聚类中的社区分布
        cluster_community_distribution = defaultdict(lambda: defaultdict(int))
        
        for entity, cluster_id in self.entity_to_cluster.items():
            if entity in self.entity_to_community:
                community_id = self.entity_to_community[entity]
                cluster_community_distribution[cluster_id][community_id] += 1
        
        self.analysis_results['cross_distribution'] = {
            'community_cluster': dict(community_cluster_distribution),
            'cluster_community': dict(cluster_community_distribution)
        }
    
    def calculate_metrics(self):
        """计算各种评估指标"""
        consistency_results = self.analysis_results['consistency']
        
        # 基本统计
        total_clusters = len(consistency_results)
        consistent_clusters = sum(1 for r in consistency_results.values() if r['is_consistent'])
        
        # 计算平均一致性比率
        avg_consistency_ratio = np.mean([r['consistency_ratio'] for r in consistency_results.values()])
        
        # 计算覆盖率
        total_entities_in_clusters = sum(r['total_entities'] for r in consistency_results.values())
        total_entities_found = sum(r['found_in_community'] for r in consistency_results.values())
        coverage_ratio = total_entities_found / total_entities_in_clusters if total_entities_in_clusters > 0 else 0
        
        metrics = {
            'total_clusters': total_clusters,
            'consistent_clusters': consistent_clusters,
            'consistency_percentage': (consistent_clusters / total_clusters) * 100,
            'avg_consistency_ratio': avg_consistency_ratio,
            'coverage_ratio': coverage_ratio,
            'total_entities_in_clusters': total_entities_in_clusters,
            'total_entities_found': total_entities_found
        }
        
        self.analysis_results['metrics'] = metrics
        return metrics
    
    def visualize_consistency(self):
        """可视化一致性分析结果"""
        consistency_results = self.analysis_results['consistency']
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('实体语义聚类与社区划分一致性分析', fontsize=16, fontweight='bold')
        
        # 1. 一致性比率分布
        ratios = [r['consistency_ratio'] for r in consistency_results.values()]
        axes[0, 0].hist(ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel('一致性比率')
        axes[0, 0].set_ylabel('聚类数量')
        axes[0, 0].set_title('聚类一致性比率分布')
        axes[0, 0].axvline(np.mean(ratios), color='red', linestyle='--', 
                          label=f'平均值: {np.mean(ratios):.3f}')
        axes[0, 0].legend()
        
        # 2. 聚类大小分布
        cluster_sizes = [r['total_entities'] for r in consistency_results.values()]
        axes[0, 1].hist(cluster_sizes, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_xlabel('聚类大小')
        axes[0, 1].set_ylabel('聚类数量')
        axes[0, 1].set_title('聚类大小分布')
        
        # 3. 一致性状态饼图
        consistent_count = sum(1 for r in consistency_results.values() if r['is_consistent'])
        inconsistent_count = len(consistency_results) - consistent_count
        
        labels = ['完全一致', '不一致']
        sizes = [consistent_count, inconsistent_count]
        colors = ['#90EE90', '#FFB6C1']
        
        axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('聚类一致性状态分布')
        
        # 4. 覆盖率分析
        found_entities = [r['found_in_community'] for r in consistency_results.values()]
        total_entities = [r['total_entities'] for r in consistency_results.values()]
        missing_entities = [total - found for total, found in zip(total_entities, found_entities)]
        
        x = range(len(consistency_results))
        width = 0.35
        
        axes[1, 1].bar([i - width/2 for i in x], found_entities, width, 
                      label='在社区中找到', color='lightblue', alpha=0.7)
        axes[1, 1].bar([i + width/2 for i in x], missing_entities, width, 
                      label='未在社区中找到', color='lightcoral', alpha=0.7)
        
        axes[1, 1].set_xlabel('聚类ID')
        axes[1, 1].set_ylabel('实体数量')
        axes[1, 1].set_title('实体覆盖情况')
        axes[1, 1].legend()
        
        # 只显示前10个聚类的标签
        if len(consistency_results) > 10:
            step = len(consistency_results) // 10
            tick_positions = list(range(0, len(consistency_results), step))
            tick_labels = [list(consistency_results.keys())[i] for i in tick_positions]
            axes[1, 1].set_xticks(tick_positions)
            axes[1, 1].set_xticklabels(tick_labels, rotation=45)
        else:
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(list(consistency_results.keys()), rotation=45)
        
        plt.tight_layout()
        plt.savefig('entity_clustering_consistency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_cross_distribution(self):
        """可视化跨分布情况"""
        cross_dist = self.analysis_results['cross_distribution']
        
        # 创建聚类-社区分布热力图
        cluster_community_data = cross_dist['cluster_community']
        
        # 准备数据
        clusters = list(cluster_community_data.keys())
        communities = list(set(comm for cluster_comms in cluster_community_data.values() 
                             for comm in cluster_comms.keys()))
        
        # 限制显示的数量以避免图表过于复杂
        max_display = 20
        if len(clusters) > max_display:
            clusters = clusters[:max_display]
        if len(communities) > max_display:
            communities = communities[:max_display]
        
        # 创建矩阵
        matrix = np.zeros((len(clusters), len(communities)))
        
        for i, cluster in enumerate(clusters):
            for j, community in enumerate(communities):
                if cluster in cluster_community_data and community in cluster_community_data[cluster]:
                    matrix[i, j] = cluster_community_data[cluster][community]
        
        # 绘制热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix, 
                   xticklabels=communities, 
                   yticklabels=clusters,
                   annot=True, 
                   fmt='.0f',
                   cmap='YlOrRd',
                   cbar_kws={'label': '实体数量'})
        
        plt.title('聚类-社区分布热力图')
        plt.xlabel('社区ID')
        plt.ylabel('聚类ID')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('cluster_community_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """生成详细的分析报告"""
        consistency_results = self.analysis_results['consistency']
        metrics = self.analysis_results['metrics']
        
        report = []
        report.append("=" * 60)
        report.append("实体语义聚类与社区划分一致性分析报告")
        report.append("=" * 60)
        report.append("")
        
        # 总体概览
        report.append("## 总体概览")
        report.append(f"- 总聚类数: {metrics['total_clusters']}")
        report.append(f"- 完全一致的聚类数: {metrics['consistent_clusters']}")
        report.append(f"- 一致性百分比: {metrics['consistency_percentage']:.2f}%")
        report.append(f"- 平均一致性比率: {metrics['avg_consistency_ratio']:.3f}")
        report.append(f"- 实体覆盖率: {metrics['coverage_ratio']:.3f}")
        report.append(f"- 聚类中的总实体数: {metrics['total_entities_in_clusters']}")
        report.append(f"- 在社区中找到的实体数: {metrics['total_entities_found']}")
        report.append("")
        
        # 详细分析
        report.append("## 详细分析结果")
        report.append("")
        
        # 按一致性比率排序
        sorted_clusters = sorted(consistency_results.items(), 
                               key=lambda x: x[1]['consistency_ratio'], 
                               reverse=True)
        
        for cluster_id, result in sorted_clusters:
            report.append(f"### 聚类 {cluster_id}")
            report.append(f"- 实体数量: {result['total_entities']}")
            report.append(f"- 在社区中找到: {result['found_in_community']}")
            report.append(f"- 一致性比率: {result['consistency_ratio']:.3f}")
            report.append(f"- 是否完全一致: {'是' if result['is_consistent'] else '否'}")
            
            if result['most_common_community']:
                report.append(f"- 主要社区: {result['most_common_community']}")
            
            if result['communities']:
                report.append("- 社区分布:")
                for comm_id, count in result['communities'].items():
                    report.append(f"  * 社区 {comm_id}: {count} 个实体")
            
            if result['missing_entities']:
                report.append(f"- 缺失实体数: {len(result['missing_entities'])}")
                if len(result['missing_entities']) <= 5:
                    report.append(f"- 缺失实体: {', '.join(result['missing_entities'])}")
            
            report.append("")
        
        # 问题分析
        report.append("## 问题分析")
        report.append("")
        
        # 识别问题聚类
        problem_clusters = [(cid, result) for cid, result in consistency_results.items() 
                          if not result['is_consistent']]
        
        if problem_clusters:
            report.append("### 不一致的聚类")
            for cluster_id, result in problem_clusters:
                report.append(f"- 聚类 {cluster_id}: 跨 {len(result['communities'])} 个社区")
        
        # 识别覆盖率低的聚类
        low_coverage_clusters = [(cid, result) for cid, result in consistency_results.items() 
                               if result['coverage_ratio'] < 0.5]
        
        if low_coverage_clusters:
            report.append("")
            report.append("### 覆盖率低的聚类 (< 50%)")
            for cluster_id, result in low_coverage_clusters:
                coverage = result['found_in_community'] / result['total_entities']
                report.append(f"- 聚类 {cluster_id}: 覆盖率 {coverage:.2f}")
        
        # 建议
        report.append("")
        report.append("## 建议")
        report.append("")
        
        if metrics['consistency_percentage'] < 50:
            report.append("- 聚类与社区划分一致性较低，建议重新审视聚类策略")
        
        if metrics['coverage_ratio'] < 0.8:
            report.append("- 实体覆盖率较低，建议检查数据预处理过程")
        
        if len(problem_clusters) > 0:
            report.append("- 存在跨社区的聚类，建议分析这些聚类的语义合理性")
        
        # 保存报告
        with open('entity_clustering_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("详细报告已保存到 'entity_clustering_analysis_report.txt'")
        return report
    
    def save_results_to_excel(self):
        """将分析结果保存到Excel文件"""
        consistency_results = self.analysis_results['consistency']
        
        # 准备数据
        data = []
        for cluster_id, result in consistency_results.items():
            data.append({
                '聚类ID': cluster_id,
                '实体数量': result['total_entities'],
                '在社区中找到': result['found_in_community'],
                '缺失实体数': len(result['missing_entities']),
                '一致性比率': result['consistency_ratio'],
                '是否完全一致': '是' if result['is_consistent'] else '否',
                '主要社区': result['most_common_community'] if result['most_common_community'] else 'N/A',
                '社区数量': len(result['communities']),
                '社区分布': str(result['communities'])
            })
        
        df = pd.DataFrame(data)
        df.to_excel('entity_clustering_analysis_results.xlsx', index=False)
        print("分析结果已保存到 'entity_clustering_analysis_results.xlsx'")
    
    def run_full_analysis(self):
        """运行完整的分析流程"""
        print("开始实体聚类与社区划分一致性分析...")
        print("=" * 50)
        
        # 1. 加载数据
        if not self.load_data():
            return
        
        # 2. 预处理数据
        self.preprocess_data()
        
        # 3. 分析一致性
        print("\n正在分析一致性...")
        self.analyze_consistency()
        
        # 4. 分析跨分布
        print("正在分析跨分布情况...")
        self.analyze_cross_distribution()
        
        # 5. 计算指标
        print("正在计算评估指标...")
        metrics = self.calculate_metrics()
        
        # 6. 生成可视化
        print("正在生成可视化图表...")
        self.visualize_consistency()
        self.visualize_cross_distribution()
        
        # 7. 生成报告
        print("正在生成详细报告...")
        self.generate_detailed_report()
        
        # 8. 保存Excel结果
        print("正在保存Excel结果...")
        self.save_results_to_excel()
        
        print("\n分析完成！")
        print("=" * 50)
        print("生成的文件:")
        print("- entity_clustering_consistency_analysis.png (一致性分析图)")
        print("- cluster_community_heatmap.png (分布热力图)")
        print("- entity_clustering_analysis_report.txt (详细报告)")
        print("- entity_clustering_analysis_results.xlsx (Excel结果)")


# 使用示例
if __name__ == "__main__":
    # 文件路径
    # clustering_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/entity_clustering_results.json"
    clustering_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/entity_clustering_results_averaged.json"
    community_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/undirected_community_entities_stats.json"
    
    # 创建分析器实例
    analyzer = EntityClusteringAnalyzer(clustering_file, community_file)
    
    # 运行完整分析
    analyzer.run_full_analysis()
    
    # 可以单独访问分析结果
    print("\n快速统计:")
    metrics = analyzer.analysis_results['metrics']
    print(f"一致性百分比: {metrics['consistency_percentage']:.2f}%")
    print(f"平均一致性比率: {metrics['avg_consistency_ratio']:.3f}")
    print(f"实体覆盖率: {metrics['coverage_ratio']:.3f}")