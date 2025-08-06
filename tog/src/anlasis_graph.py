import json
import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

font_path = '/data/FinAi_Mapping_Knowledge/chenmingzhen/.fonts/simsun.ttc'

font = fm.FontProperties(fname=font_path, size=16)

times = "/data/FinAi_Mapping_Knowledge/chenmingzhen/.fonts/Times_New_Roman.ttf"
custom_font = fm.FontProperties(fname=times, size=16)


class ClusterPositionAnalyzer:
    def __init__(self, cluster_stats_file, entity_table_file):
        self.cluster_stats_file = cluster_stats_file
        self.entity_table_file = entity_table_file
        self.cluster_stats = {}
        self.entity_info = {}
        self.enriched_clusters = {}
        
    def load_data(self):
        """加载聚类统计和实体表数据"""
        print("正在加载数据...")
        
        # 加载聚类统计结果
        with open(self.cluster_stats_file, 'r', encoding='utf-8') as f:
            self.cluster_stats = json.load(f)
            # data = json.load(f)
            # self.cluster_stats = data["communities"]

        print(f"已加载 {len(self.cluster_stats)} 个聚类")
        
        # 加载实体表
        with open(self.entity_table_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.entity_info[data['name']] = data
        print(f"已加载 {len(self.entity_info)} 个实体信息")
        
    def enrich_clusters_with_positions(self):
        """为聚类结果添加位置信息"""
        print("正在为聚类结果添加位置信息...")
        
        self.enriched_clusters = {}
        
        for cluster_id, entities in self.cluster_stats.items():
            enriched_entities = []
            
            for entity_name in entities:
                entity_data = {
                    'name': entity_name,
                    'id': None,
                    'type': [],
                    'positions': [],
                    'found_in_table': False
                }
                
                # 查找实体信息
                if entity_name in self.entity_info:
                    entity_info = self.entity_info[entity_name]
                    entity_data.update({
                        'id': entity_info.get('id'),
                        'type': entity_info.get('type', []),
                        'positions': entity_info.get('positions', []),
                        'found_in_table': True
                    })
                
                enriched_entities.append(entity_data)
            
            self.enriched_clusters[cluster_id] = enriched_entities
            
        print("位置信息添加完成")
        
    def analyze_cluster_positions(self):
        """分析每个聚类的位置信息统计"""
        print("正在分析聚类位置信息...")
        
        cluster_analysis = {}
        
        for cluster_id, entities in self.enriched_clusters.items():
            analysis = {
                'cluster_id': cluster_id,
                'total_entities': len(entities),
                'entities_with_positions': 0,
                'entities_without_positions': 0,
                'total_positions': 0,
                'unique_articles': set(),
                'unique_paragraphs': set(),
                'unique_sentences': set(),
                'article_distribution': Counter(),
                'paragraph_distribution': Counter(),
                'sentence_distribution': Counter(),
                'type_distribution': Counter(),
                'entities_detail': []
            }
            
            for entity in entities:
                entity_detail = {
                    'name': entity['name'],
                    'id': entity['id'],
                    'type': entity['type'],
                    'position_count': len(entity['positions']),
                    'found_in_table': entity['found_in_table']
                }
                
                if entity['positions']:
                    analysis['entities_with_positions'] += 1
                    analysis['total_positions'] += len(entity['positions'])
                    
                    for pos in entity['positions']:
                        art_id = pos.get('art_id')
                        par_id = pos.get('par_id')
                        sen_id = pos.get('sen_id')
                        
                        if art_id is not None:
                            analysis['unique_articles'].add(art_id)
                            analysis['article_distribution'][art_id] += 1
                        
                        if par_id is not None:
                            analysis['unique_paragraphs'].add(f"{art_id}_{par_id}")
                            analysis['paragraph_distribution'][f"{art_id}_{par_id}"] += 1
                        
                        if sen_id is not None:
                            analysis['unique_sentences'].add(f"{art_id}_{par_id}_{sen_id}")
                            analysis['sentence_distribution'][f"{art_id}_{par_id}_{sen_id}"] += 1
                else:
                    analysis['entities_without_positions'] += 1
                
                # 统计实体类型
                for entity_type in entity['type']:
                    analysis['type_distribution'][entity_type] += 1
                
                analysis['entities_detail'].append(entity_detail)
            
            # 转换set为数量
            analysis['unique_articles_count'] = len(analysis['unique_articles'])
            analysis['unique_paragraphs_count'] = len(analysis['unique_paragraphs'])
            analysis['unique_sentences_count'] = len(analysis['unique_sentences'])
            
            # 删除set对象（不能序列化）
            del analysis['unique_articles']
            del analysis['unique_paragraphs']
            del analysis['unique_sentences']
            
            cluster_analysis[cluster_id] = analysis
            
        return cluster_analysis
    
    def generate_summary_report(self, cluster_analysis):
        """生成汇总报告"""
        print("正在生成汇总报告...")
        
        summary = {
            'total_clusters': len(cluster_analysis),
            'total_entities': sum(analysis['total_entities'] for analysis in cluster_analysis.values()),
            'total_entities_with_positions': sum(analysis['entities_with_positions'] for analysis in cluster_analysis.values()),
            'total_positions': sum(analysis['total_positions'] for analysis in cluster_analysis.values()),
            'coverage_rate': 0,
            'cluster_size_distribution': Counter(),
            'position_coverage_by_cluster': {},
            'top_articles': Counter(),
            'top_types': Counter(),
            'clusters_by_size': []
        }
        
        # 计算覆盖率
        if summary['total_entities'] > 0:
            summary['coverage_rate'] = summary['total_entities_with_positions'] / summary['total_entities']
        
        # 统计各种分布
        for cluster_id, analysis in cluster_analysis.items():
            # 聚类大小分布
            summary['cluster_size_distribution'][analysis['total_entities']] += 1
            
            # 位置覆盖率
            if analysis['total_entities'] > 0:
                coverage = analysis['entities_with_positions'] / analysis['total_entities']
                summary['position_coverage_by_cluster'][cluster_id] = coverage
            
            # 文章分布
            for art_id, count in analysis['article_distribution'].items():
                summary['top_articles'][art_id] += count
            
            # 类型分布
            for entity_type, count in analysis['type_distribution'].items():
                summary['top_types'][entity_type] += count
            
            # 按大小排序的聚类
            summary['clusters_by_size'].append({
                'cluster_id': cluster_id,
                'size': analysis['total_entities'],
                'position_coverage': coverage if analysis['total_entities'] > 0 else 0,
                'unique_articles': analysis['unique_articles_count']
            })
        
        # 按聚类大小排序
        summary['clusters_by_size'].sort(key=lambda x: x['size'], reverse=True)
        
        return summary
    
    def save_results(self, output_dir="./cluster_analysis_results"):
        """保存分析结果"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. 保存富化后的聚类数据
        enriched_file = f"{output_dir}/enriched_clusters.json"
        with open(enriched_file, 'w', encoding='utf-8') as f:
            json.dump(self.enriched_clusters, f, ensure_ascii=False, indent=2)
        print(f"富化聚类数据已保存到: {enriched_file}")
        
        # 2. 分析聚类位置信息
        cluster_analysis = self.analyze_cluster_positions()
        analysis_file = f"{output_dir}/cluster_position_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(cluster_analysis, f, ensure_ascii=False, indent=2)
        print(f"聚类位置分析已保存到: {analysis_file}")
        
        # 3. 生成汇总报告
        summary = self.generate_summary_report(cluster_analysis)
        summary_file = f"{output_dir}/summary_report.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"汇总报告已保存到: {summary_file}")
        
        # 4. 生成详细报告
        self.generate_detailed_report(cluster_analysis, summary, f"{output_dir}/detailed_report.txt")
        
        return cluster_analysis, summary
    
    def generate_detailed_report(self, cluster_analysis, summary, output_file):
        """生成详细的文本报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("聚类位置信息分析详细报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 汇总信息
            f.write("汇总信息:\n")
            f.write(f"总聚类数: {summary['total_clusters']}\n")
            f.write(f"总实体数: {summary['total_entities']}\n")
            f.write(f"有位置信息的实体数: {summary['total_entities_with_positions']}\n")
            f.write(f"位置覆盖率: {summary['coverage_rate']:.2%}\n")
            f.write(f"总位置数: {summary['total_positions']}\n\n")
            
            # Top文章
            f.write("出现频率最高的文章:\n")
            for art_id, count in summary['top_articles'].most_common(10):
                f.write(f"  文章 {art_id}: {count} 次\n")
            f.write("\n")
            
            # Top类型
            f.write("最常见的实体类型:\n")
            for entity_type, count in summary['top_types'].most_common(10):
                f.write(f"  {entity_type}: {count} 个\n")
            f.write("\n")
            
            # 按大小排序的聚类详情
            f.write("聚类详情 (按大小排序):\n")
            f.write("-" * 50 + "\n")
            
            for cluster_info in summary['clusters_by_size'][:20]:  # 只显示前20个
                cluster_id = cluster_info['cluster_id']
                analysis = cluster_analysis[cluster_id]
                
                f.write(f"\n聚类 {cluster_id}:\n")
                f.write(f"  实体数: {analysis['total_entities']}\n")
                f.write(f"  有位置信息: {analysis['entities_with_positions']}\n")
                f.write(f"  位置覆盖率: {cluster_info['position_coverage']:.2%}\n")
                f.write(f"  涉及文章数: {analysis['unique_articles_count']}\n")
                f.write(f"  涉及段落数: {analysis['unique_paragraphs_count']}\n")
                f.write(f"  涉及句子数: {analysis['unique_sentences_count']}\n")
                
                # 显示部分实体
                f.write("  实体列表:\n")
                for entity in analysis['entities_detail'][:10]:  # 只显示前10个
                    status = "✓" if entity['found_in_table'] else "✗"
                    f.write(f"    {status} {entity['name']} (位置数: {entity['position_count']})\n")
                
                if len(analysis['entities_detail']) > 10:
                    f.write(f"    ... 还有 {len(analysis['entities_detail']) - 10} 个实体\n")
        
        print(f"详细报告已保存到: {output_file}")
    
    def visualize_results(self, cluster_analysis, summary, output_dir="./cluster_analysis_results"):
        """生成可视化图表"""
        plt.style.use('default')
        
        # 1. 聚类大小分布
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 聚类大小分布
        cluster_sizes = [info['size'] for info in summary['clusters_by_size']]
        axes[0, 0].hist(cluster_sizes, bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('聚类大小', fontproperties=font)
        axes[0, 0].set_ylabel('聚类数量', fontproperties=font)
        axes[0, 0].set_title('聚类大小分布', fontproperties=font)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 位置覆盖率分布
        coverage_rates = [info['position_coverage'] for info in summary['clusters_by_size']]
        axes[0, 1].hist(coverage_rates, bins=20, alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel('位置覆盖率', fontproperties=font)
        axes[0, 1].set_ylabel('聚类数量', fontproperties=font)
        axes[0, 1].set_title('位置覆盖率分布', fontproperties=font)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top文章分布
        top_articles = summary['top_articles'].most_common(10)
        if top_articles:
            articles, counts = zip(*top_articles)
            axes[1, 0].bar(range(len(articles)), counts, color='orange', alpha=0.7)
            axes[1, 0].set_xlabel('文章ID', fontproperties=font)
            axes[1, 0].set_ylabel('实体出现次数', fontproperties=font)
            axes[1, 0].set_title('Top 10 文章中的实体分布', fontproperties=font)
            axes[1, 0].set_xticks(range(len(articles)))
            axes[1, 0].set_xticklabels(articles, rotation=45)
        
        # 实体类型分布
        top_types = summary['top_types'].most_common(10)
        if top_types:
            types, counts = zip(*top_types)
            axes[1, 1].pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('实体类型分布', fontproperties=font)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_analysis_visualization.png", dpi=300, bbox_inches='tight')
        print(f"可视化图表已保存到: {output_dir}/cluster_analysis_visualization.png")
        plt.show()
        
        # 2. 聚类详细分析图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 聚类大小 vs 位置覆盖率散点图
        sizes = [info['size'] for info in summary['clusters_by_size']]
        coverages = [info['position_coverage'] for info in summary['clusters_by_size']]
        
        scatter = ax.scatter(sizes, coverages, alpha=0.6, s=60, c=range(len(sizes)), cmap='viridis')
        ax.set_xlabel('聚类大小', fontproperties=font)
        ax.set_ylabel('位置覆盖率', fontproperties=font)
        ax.set_title('聚类大小 vs 位置覆盖率', fontproperties=font)
        ax.grid(True, alpha=0.3)
        
        # 添加颜色条
        # plt.colorbar(scatter, ax=ax, label='聚类索引', fontproperties=font)
        cbar = plt.colorbar(scatter, ax=ax)                 # 创建 Colorbar
        cbar.set_label('聚类索引', fontproperties=font)       # 设置标签及字体
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/cluster_size_vs_coverage.png", dpi=300, bbox_inches='tight')
        print(f"散点图已保存到: {output_dir}/cluster_size_vs_coverage.png")
        plt.show()
    # def visualize_results(self, cluster_analysis, summary, output_dir: str = "./cluster_analysis_results"):
    #     """Generate visualization charts (English labels)."""
    #     import os
    #     import matplotlib.pyplot as plt

    #     # Make sure the output directory exists
    #     os.makedirs(output_dir, exist_ok=True)

    #     plt.style.use("default")

    #     # 1. Overview charts
    #     fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    #     # Cluster‑size distribution
    #     cluster_sizes = [info["size"] for info in summary["clusters_by_size"]]
    #     axes[0, 0].hist(cluster_sizes, bins=20, alpha=0.7, color="skyblue")
    #     axes[0, 0].set_xlabel("Cluster Size")
    #     axes[0, 0].set_ylabel("Number of Clusters")
    #     axes[0, 0].set_title("Cluster Size Distribution")
    #     axes[0, 0].grid(True, alpha=0.3)

    #     # Position‑coverage distribution
    #     coverage_rates = [info["position_coverage"] for info in summary["clusters_by_size"]]
    #     axes[0, 1].hist(coverage_rates, bins=20, alpha=0.7, color="lightgreen")
    #     axes[0, 1].set_xlabel("Position Coverage")
    #     axes[0, 1].set_ylabel("Number of Clusters")
    #     axes[0, 1].set_title("Position Coverage Distribution")
    #     axes[0, 1].grid(True, alpha=0.3)

    #     # Top‑article entity counts
    #     top_articles = summary["top_articles"].most_common(10)
    #     if top_articles:
    #         articles, counts = zip(*top_articles)
    #         axes[1, 0].bar(range(len(articles)), counts, color="orange", alpha=0.7)
    #         axes[1, 0].set_xlabel("Article ID")
    #         axes[1, 0].set_ylabel("Entity Occurrences")
    #         axes[1, 0].set_title("Entity Distribution in Top‑10 Articles")
    #         axes[1, 0].set_xticks(range(len(articles)))
    #         axes[1, 0].set_xticklabels(articles, rotation=45)

    #     # Entity‑type distribution
    #     top_types = summary["top_types"].most_common(10)
    #     if top_types:
    #         types_, counts = zip(*top_types)
    #         axes[1, 1].pie(counts, labels=types_, autopct="%1.1f%%", startangle=90)
    #         axes[1, 1].set_title("Entity Type Distribution")

    #     plt.tight_layout()
    #     fig1_path = os.path.join(output_dir, "cluster_analysis_visualization.png")
    #     # plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    #     plt.savefig(fig1_path, dpi=300)
    #     print(f"Visualization saved to: {fig1_path}")
    #     plt.show()

    #     # 2. Scatter plot: cluster size vs. position coverage
    #     fig, ax = plt.subplots(figsize=(12, 8))

    #     scatter = ax.scatter(
    #         cluster_sizes,
    #         coverage_rates,
    #         alpha=0.6,
    #         s=60,
    #         c=range(len(cluster_sizes)),
    #         cmap="viridis",
    #     )
    #     ax.set_xlabel("Cluster Size")
    #     ax.set_ylabel("Position Coverage")
    #     ax.set_title("Cluster Size vs. Position Coverage")
    #     ax.grid(True, alpha=0.3)

    #     plt.colorbar(scatter, ax=ax, label="Cluster Index")

    #     plt.tight_layout()
    #     fig2_path = os.path.join(output_dir, "cluster_size_vs_coverage.png")
    #     plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    #     print(f"Scatter plot saved to: {fig2_path}")
    #     plt.show()

    
    def run_analysis(self, output_dir="./cluster_analysis_results"):
        """运行完整分析流程"""
        print("开始聚类位置信息分析...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 富化聚类数据
        self.enrich_clusters_with_positions()
        
        # 3. 保存结果
        cluster_analysis, summary = self.save_results(output_dir)
        
        # 4. 生成可视化
        self.visualize_results(cluster_analysis, summary, output_dir)
        
        print("\n分析完成！")
        print(f"结果已保存到: {output_dir}")
        print(f"总体位置覆盖率: {summary['coverage_rate']:.2%}")
        print(f"最大聚类大小: {max(info['size'] for info in summary['clusters_by_size'])}")
        print(f"平均聚类大小: {sum(info['size'] for info in summary['clusters_by_size']) / len(summary['clusters_by_size']):.1f}")
        
        return cluster_analysis, summary

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='聚类位置信息分析')
    parser.add_argument('--cluster_stats', '-c', required=True, help='聚类统计文件路径')
    parser.add_argument('--entity_table', '-e', required=True, help='实体表文件路径')
    parser.add_argument('--output_dir', '-o', default='/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/cluster_analysis_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = ClusterPositionAnalyzer(args.cluster_stats, args.entity_table)
    analyzer.run_analysis(args.output_dir)

if __name__ == "__main__":
    # 如果直接运行，使用默认参数
    import sys
    if len(sys.argv) == 1:
        # 默认文件路径
        cluster_stats_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/directed_community_entities_stats.json"
        entity_table_file = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA_with_titles.jsonl"
        
        print("使用默认参数运行...")
        analyzer = ClusterPositionAnalyzer(cluster_stats_file, entity_table_file)
        analyzer.run_analysis()
    else:
        main()