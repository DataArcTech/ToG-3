"""
教育文档实体抽取处理系统
基于词典匹配的方法对chunk进行实体识别
"""

import json
import os
import re
import ahocorasick
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityMatcher:
    """实体匹配器"""
    
    def __init__(self, entities_file_path: str):
        """
        初始化实体匹配器
        
        Args:
            entities_file_path: 实体文件路径
        """
        self.entities_file_path = entities_file_path
        self.entities_dict = {}  # {entity_id: {"name": entity_name, "type": entity_types}}
        self.automaton = ahocorasick.Automaton()
        self.max_entity_length = 0
        
        # 加载实体数据
        self._load_entities()
        self._build_automaton()
    
    def _load_entities(self):
        """从JSONL文件加载实体数据"""
        logger.info(f"正在加载实体文件: {self.entities_file_path}")
        
        try:
            with open(self.entities_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entity_record = json.loads(line)
                        entity_id = entity_record.get('id')
                        entity_name = entity_record.get('name', '').strip()
                        entity_types = entity_record.get('type', [])
                        
                        if entity_id is not None and entity_name:
                            self.entities_dict[entity_id] = {
                                'name': entity_name,
                                'type': entity_types
                            }
                            self.max_entity_length = max(self.max_entity_length, len(entity_name))
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"第 {line_num} 行JSON解析失败: {e}")
                        continue
            
            logger.info(f"成功加载 {len(self.entities_dict)} 个有效实体")
            
        except Exception as e:
            logger.error(f"加载实体文件失败: {e}")
            raise
    
    def _build_automaton(self):
        """构建AC自动机用于高效匹配"""
        logger.info("正在构建AC自动机...")
        
        for entity_id, entity_info in self.entities_dict.items():
            entity_name = entity_info['name']
            # 支持大小写不敏感匹配
            self.automaton.add_word(entity_name.lower(), {
                'entity_id': entity_id,
                'entity_name': entity_name,
                'entity_type': entity_info['type']
            })
        
        self.automaton.make_automaton()
        logger.info(f"AC自动机构建完成，包含 {len(self.entities_dict)} 个实体")
    
    def _is_valid_boundary(self, text: str, start: int, end: int) -> bool:
        """
        检查匹配的实体是否在有效的词边界上
        
        Args:
            text: 原始文本
            start: 匹配开始位置
            end: 匹配结束位置
            
        Returns:
            bool: 是否为有效边界
        """
        # 检查前边界
        if start > 0:
            prev_char = text[start - 1]
            curr_char = text[start]
            # 如果前一个字符是字母数字，当前字符也是字母数字，则可能是词的一部分
            if prev_char.isalnum() and curr_char.isalnum():
                return False
        
        # 检查后边界
        if end < len(text):
            prev_char = text[end - 1]
            next_char = text[end]
            # 如果当前字符是字母数字，下一个字符也是字母数字，则可能是词的一部分
            if prev_char.isalnum() and next_char.isalnum():
                return False
        
        return True
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中抽取实体
        
        Args:
            text: 待抽取的文本
            
        Returns:
            List[Dict]: 抽取到的实体列表
        """
        if not text.strip():
            return []
        
        results = []
        text_lower = text.lower()
        matched_positions = set()  # 用于避免重叠匹配
        
        # 使用AC自动机进行匹配
        for end_idx, entity_info in self.automaton.iter(text_lower):
            entity_name = entity_info['entity_name']
            start_idx = end_idx - len(entity_name) + 1
            
            # 检查是否与已匹配的位置重叠
            if any(pos in matched_positions for pos in range(start_idx, end_idx + 1)):
                continue
            
            # 检查词边界
            if not self._is_valid_boundary(text, start_idx, end_idx + 1):
                continue
            
            # 获取原始文本中的匹配内容
            matched_text = text[start_idx:end_idx + 1]
            
            # 记录匹配位置
            for pos in range(start_idx, end_idx + 1):
                matched_positions.add(pos)
            
            results.append({
                'entity_id': entity_info['entity_id'],
                'entity_name': entity_name,
                'entity_type': entity_info['entity_type'],
                'matched_text': matched_text,
                'start_position': start_idx,
                'end_position': end_idx + 1
            })
        
        # 按起始位置排序
        results.sort(key=lambda x: x['start_position'])
        
        return results
    
    def get_entity_statistics(self) -> Dict[str, Any]:
        """获取实体统计信息"""
        # 统计实体类型分布
        type_distribution = defaultdict(int)
        for entity_info in self.entities_dict.values():
            for entity_type in entity_info['type']:
                type_distribution[entity_type] += 1
        
        return {
            'total_entities': len(self.entities_dict),
            'max_entity_length': self.max_entity_length,
            'entity_length_distribution': self._get_length_distribution(),
            'entity_type_distribution': dict(type_distribution)
        }
    
    def _get_length_distribution(self) -> Dict[str, int]:
        """获取实体长度分布"""
        length_dist = defaultdict(int)
        for entity_info in self.entities_dict.values():
            entity_name = entity_info['name']
            length_dist[len(entity_name)] += 1
        return dict(length_dist)


class ChunkProcessor:
    """Chunk处理器"""
    
    def __init__(self, entity_matcher: EntityMatcher, output_directory: str = None):
        """
        初始化处理器
        
        Args:
            entity_matcher: 实体匹配器实例
            output_directory: 输出目录路径
        """
        self.entity_matcher = entity_matcher
        self.output_directory = output_directory
    
    def process_chunk_file(self, file_path: str, input_directory: str) -> Tuple[int, int]:
        """
        处理单个chunk文件
        
        Args:
            file_path: chunk文件路径
            input_directory: 输入目录的根路径
            
        Returns:
            Tuple[int, int]: (处理的chunk数量, 发现实体的chunk数量)
        """
        logger.info(f"正在处理文件: {file_path}")
        
        try:
            processed_chunks = []
            chunks_with_entities = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        
                        # 获取需要处理的内容 - 现在是chunk字段
                        chunk_content = chunk_data.get('chunk', '').strip()
                        
                        # 进行实体抽取
                        entities = self.entity_matcher.extract_entities(chunk_content)
                        
                        # 添加实体信息到chunk数据
                        chunk_data['entities'] = entities
                        
                        if entities:
                            chunks_with_entities += 1
                        
                        processed_chunks.append(chunk_data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"文件 {file_path} 第 {line_num} 行JSON解析失败: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"处理文件 {file_path} 第 {line_num} 行时出错: {e}")
                        continue
            
            # 计算输出文件路径
            output_file_path = self._get_output_file_path(file_path, input_directory)
            
            # 写入处理后的数据到输出目录
            self._write_processed_chunks(output_file_path, processed_chunks)
            
            logger.info(f"文件处理完成: {len(processed_chunks)} 个chunk, {chunks_with_entities} 个包含实体")
            logger.info(f"输出文件: {output_file_path}")
            return len(processed_chunks), chunks_with_entities
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
            return 0, 0
    
    def _get_output_file_path(self, input_file_path: str, input_directory: str) -> str:
        """
        计算输出文件路径，保持原有的目录结构
        
        Args:
            input_file_path: 输入文件的完整路径
            input_directory: 输入目录的根路径
            
        Returns:
            str: 输出文件路径
        """
        # 获取相对于输入目录的相对路径
        relative_path = os.path.relpath(input_file_path, input_directory)
        
        # 构建输出文件路径
        output_file_path = os.path.join(self.output_directory, relative_path)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file_path)
        os.makedirs(output_dir, exist_ok=True)
        
        return output_file_path
    
    def _write_processed_chunks(self, file_path: str, processed_chunks: List[Dict]):
        """写入处理后的chunk数据"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for chunk in processed_chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"写入文件 {file_path} 失败: {e}")
            raise
    
    def process_single_file(self, file_path: str, output_file_path: str = None) -> Dict[str, Any]:
        """
        处理单个文件
        
        Args:
            file_path: 输入文件路径
            output_file_path: 输出文件路径，如果为None则使用默认输出目录
            
        Returns:
            Dict: 处理统计信息
        """
        logger.info(f"开始处理文件: {file_path}")
        
        try:
            processed_chunks = []
            chunks_with_entities = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        chunk_data = json.loads(line)
                        
                        # 获取需要处理的内容 - 现在是chunk字段
                        chunk_content = chunk_data.get('chunk', '').strip()
                        
                        # 进行实体抽取
                        entities = self.entity_matcher.extract_entities(chunk_content)
                        
                        # 添加实体信息到chunk数据
                        chunk_data['entities'] = entities
                        
                        if entities:
                            chunks_with_entities += 1
                        
                        processed_chunks.append(chunk_data)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"文件 {file_path} 第 {line_num} 行JSON解析失败: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"处理文件 {file_path} 第 {line_num} 行时出错: {e}")
                        continue
            
            # 确定输出文件路径
            if output_file_path is None:
                if self.output_directory:
                    filename = os.path.basename(file_path)
                    name, ext = os.path.splitext(filename)
                    output_file_path = os.path.join(self.output_directory, f"{name}_with_entities{ext}")
                else:
                    name, ext = os.path.splitext(file_path)
                    output_file_path = f"{name}_with_entities{ext}"
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # 写入处理后的数据
            self._write_processed_chunks(output_file_path, processed_chunks)
            
            # 返回统计信息
            statistics = {
                'input_file': file_path,
                'output_file': output_file_path,
                'total_chunks': len(processed_chunks),
                'chunks_with_entities': chunks_with_entities,
                'entity_coverage': chunks_with_entities / len(processed_chunks) if processed_chunks else 0
            }
            
            logger.info("文件处理完成统计:")
            logger.info(f"  输入文件: {file_path}")
            logger.info(f"  输出文件: {output_file_path}")
            logger.info(f"  总chunk数: {len(processed_chunks)}")
            logger.info(f"  包含实体的chunk数: {chunks_with_entities}")
            logger.info(f"  实体覆盖率: {statistics['entity_coverage']:.2%}")
            
            return statistics
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
            raise
    
    def process_all_chunks(self, chunk_directory: str) -> Dict[str, Any]:
        """
        处理所有chunk文件
        
        Args:
            chunk_directory: chunk文件目录
            
        Returns:
            Dict: 处理统计信息
        """
        logger.info(f"开始处理目录: {chunk_directory}")
        
        if self.output_directory:
            logger.info(f"输出目录: {self.output_directory}")
            # 确保输出根目录存在
            os.makedirs(self.output_directory, exist_ok=True)
        
        # 获取所有.jsonl文件
        chunk_files = []
        for root, dirs, files in os.walk(chunk_directory):
            for file in files:
                if file.endswith('.jsonl'):
                    chunk_files.append(os.path.join(root, file))
        
        logger.info(f"找到 {len(chunk_files)} 个chunk文件")
        
        # 统计信息
        total_chunks = 0
        total_chunks_with_entities = 0
        processed_files = 0
        failed_files = 0
        
        # 处理每个文件
        for file_path in tqdm(chunk_files, desc="处理chunk文件"):
            try:
                chunks_count, entities_count = self.process_chunk_file(file_path, chunk_directory)
                total_chunks += chunks_count
                total_chunks_with_entities += entities_count
                processed_files += 1
            except Exception as e:
                logger.error(f"处理文件 {file_path} 时发生错误: {e}")
                failed_files += 1
        
        # 返回统计信息
        statistics = {
            'processed_files': processed_files,
            'failed_files': failed_files,
            'total_files': len(chunk_files),
            'total_chunks': total_chunks,
            'chunks_with_entities': total_chunks_with_entities,
            'entity_coverage': total_chunks_with_entities / total_chunks if total_chunks > 0 else 0,
            'input_directory': chunk_directory,
            'output_directory': self.output_directory
        }
        
        logger.info("处理完成统计:")
        logger.info(f"  输入目录: {chunk_directory}")
        logger.info(f"  输出目录: {self.output_directory}")
        logger.info(f"  处理文件数: {processed_files}/{len(chunk_files)}")
        logger.info(f"  失败文件数: {failed_files}")
        logger.info(f"  总chunk数: {total_chunks}")
        logger.info(f"  包含实体的chunk数: {total_chunks_with_entities}")
        logger.info(f"  实体覆盖率: {statistics['entity_coverage']:.2%}")
        
        return statistics


def main():
    """主函数"""
    # 配置路径
    # ENTITIES_FILE = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt_4.1-mini.jsonl"
    ENTITIES_FILE = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/graph/data/new_graph_gpt4.1_mini_new_type_with_QA.jsonl"
    CHUNK_FILE = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/datasets/200_que.jsonl"
    OUTPUT_DIRECTORY = "/data/FinAi_Mapping_Knowledge/chenmingzhen/EDU_POC/parse_datasets/chunked_data_with_entities"
    
    try:
        # 初始化实体匹配器
        logger.info("=== 教育文档实体抽取系统启动 ===")
        entity_matcher = EntityMatcher(ENTITIES_FILE)
        
        # 打印实体统计信息
        stats = entity_matcher.get_entity_statistics()
        logger.info(f"实体统计信息:")
        logger.info(f"  总实体数: {stats['total_entities']}")
        logger.info(f"  最大实体长度: {stats['max_entity_length']}")
        logger.info(f"  实体类型分布: {stats['entity_type_distribution']}")
        
        # 初始化chunk处理器，指定输出目录
        processor = ChunkProcessor(entity_matcher, OUTPUT_DIRECTORY)
        
        # 处理单个文件
        processing_stats = processor.process_single_file(CHUNK_FILE)
        
        # 输出最终统计
        logger.info("=== 处理完成 ===")
        logger.info(f"最终统计信息: {processing_stats}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise


if __name__ == "__main__":
    main()