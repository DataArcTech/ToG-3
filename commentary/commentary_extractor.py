import json
import re
import os
import traceback
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class CommentaryQuestion:
    question_id: str
    materials: List[str]
    question_content: str
    answer: str
    answer_explanation: str

class CommentaryExtractor:
    """
    申论真题内容提取器
    从JSON文件中提取题目、材料、参考答案等内容并结构化
    """
    
    def __init__(self, input_files: List[str], output_file: str):
        self.input_files = input_files
        self.output_file = output_file
        self.materials_pattern = re.compile(r'(?:材料|【材料】?)\s*([0-9一二三四五六七八九十百千万]+)(?=\s|$|】)')

        
    def extract_materials(self, content_list: List[Dict]) -> List[Dict]:
        """提取材料列表"""
        materials = []
        for item in content_list:
            content = item.get('content', '').strip()
            if not '材料' in content:
                continue
            if '材料' in content and not content.startswith('材料') and not content.startswith('【材料'):
                continue
            print("materials content: ", content)    
            match = self.materials_pattern.search(content)
            if match:
                print("materials match: ", match)
                material_num = match.group(1)
                material_content = content.strip()
                materials.append({
                    'material_id': material_num,
                    'content': material_content
                })
        return materials
    
    def split_questions_by_number(self, content: str) -> List[Dict]:
        """按照（一）（二）（三）（四）拆分题目"""
        questions = []
        
        # 使用正则表达式匹配题目编号和内容
        # 匹配格式：（一）题目内容（分数）要求：...
        question_patterns = [
            r'[（(]([一二三四])[）)]([\s\S]*?)(?=[（(][二三四][）)]|$)',
            r'([一二三四])、([\s\S]*?)(?=([一二三四])、|$)',
            r'【问题([一二三四])】([\s\S]*?)(?=【问题([一二三四])】|$)'
        ]
        
        for pattern in question_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            if matches:
                for match in matches:
                    # print("match: ", match)
                    question_num = match.group(1)
                    question_content = match.group(2).strip()
                    print("question_num: ", question_num)
                    print("question_content: ", question_content)
            
                    questions.append({
                        'question_id': question_num,
                        'question_content': question_content,
                        'answer': '',
                        'answer_explanation': ''
                    })
                
        
        return questions

    def extract_questions(self, content_list: List[Dict]) -> List[Dict]:
        """提取题目信息"""
        questions = []
        
        for item in content_list:
            content = item.get('content', '')
            
            if '三、作答要求' in content or '一、注意事项' in content or '二、给定材料' in content or '参考答案' in content:
                continue
                
            # 检查是否包含题目内容
            question_numbers = ['（一）', '（二）', '（三）', '（四）', '(一)', '(二)', '(三)', '(四)', '一、', '二、', '三、', '四、', '问题一', '问题二', '问题三']
            if any(question_number in content for question_number in question_numbers):
                # 使用拆分方法提取题目
                questions.extend(self.split_questions_by_number(content))
            
        return questions
    
    def extract_answers(self, content_list: List[Dict]) -> List[Dict]:
        """提取参考答案"""
        answers = []
        
        for item in content_list:
            content = item.get('content', '')
            
            # 查找参考答案部分
            if '参考答案' in content:
                # print("content: ", content)
                # 提取各个答案
                answer_patterns = [
                    (r'一、参考答案\s*([\s\S]*?)(?=二、参考答案|$)', '一'),
                    (r'二、参考答案\s*([\s\S]*?)(?=三、参考答案|$)', '二'),
                    (r'三、参考答案\s*([\s\S]*?)(?=四、参考答案|$)', '三'),
                    (r'四、参考答案\s*([\s\S]*?)(?=四、参考答案|$)', '四'),
                    (r'【试题一】参考答案\s*([\s\S]*?)(?=【试题二】参考答案|$)', '一'),
                    (r'【试题二】参考答案\s*([\s\S]*?)(?=【试题三】参考答案|$)', '二'),
                    (r'【试题三】参考答案\s*([\s\S]*?)(?=【试题四】参考答案|$)', '三'),
                    (r'【问题一参考答案】\s*([\s\S]*?)(?=【问题二参考答案】|$)', '一'),
                    (r'【问题二参考答案】\s*([\s\S]*?)(?=【问题三参考答案】|$)', '二'),
                    (r'【问题三参考答案】\s*([\s\S]*?)(?=【问题四参考答案】|$)', '三')
                ]
                
                for pattern, answer_id in answer_patterns:
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        # print("match: ", match)
                        # print("pattern: ", pattern)
                        # print("answer_id: ", answer_id)
                        answer_content = match.group(1).strip()
                        answers.append({
                            'answer_id': answer_id,
                            'content': answer_content
                        })
                
        return answers
    
    def merge_questions(self, questions: List[Dict], materials: List[Dict], answers: List[Dict]) -> List[Dict]:
        """合并题目和答案"""
        # 创建答案映射
        answer_map = {answer['answer_id']: answer['content'] for answer in answers}
        # print("answer_map: ", answer_map)
        
        # 合并题目和答案
        for question in questions:
            question_id = question['question_id']
            question['materials'] = materials
            if question_id in answer_map:
                question['answer'] = answer_map[question_id]
                
        return questions
    
    def process_file(self, file_path: str) -> List[CommentaryQuestion]:
        """处理单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_list = json.load(f)
            
            materials = self.extract_materials(content_list)
            # print("materials: ", materials)
            question_contents = self.extract_questions(content_list)
            # print("questions:", question_contents)
            answers = self.extract_answers(content_list)
            # print("answers: ", answers)
            questions = self.merge_questions(question_contents, materials, answers)
            
            return questions
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            print("完整堆栈信息:")
            traceback.print_exc()  # 直接打印错误堆栈
            return None
    
    def process_files(self) -> List[Dict[str, Any]]:
        """处理目录下的所有JSON文件"""
        results = []
        file_count = 0
        for file_path in self.input_files:
            result = self.process_file(file_path)
            print(f"成功处理 {len(result)} 个题目")
            if result:
                results.extend(result)
                file_count += 1
        print(f"成功处理 {file_count} 个文件")
        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """保存结果到JSON文件"""
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
    
    def extract_and_save(self):
        """提取内容并保存结果"""
        print(f"开始处理目录: {self.input_files}")
        results = self.process_files()
        # print("results: ", results)
        
        if results:
            print(f"成功处理 {len(results)} 个题目")
            self.save_results(results, self.output_file)
            return results
        else:
            print("没有找到可处理的文件")
            return []

def main():
    input_files = ["commentary/2020年深圳市公考《申论》题（一卷）及参考答案.json", 
    "commentary/2020年深圳市公考《申论》题（二卷）及参考答案.json",
    "commentary/2023年深圳市公务员考试《申论》（一卷）真题及答案.json",
    "commentary/2023年深圳市公务员考试《申论》（二卷）真题及答案.json",
    "commentary/2024年深圳市公务员考试《申论》（二卷）题及参考答案.json"]
    output_file = "commentary/申论真题2020-2024.json"
    extractor = CommentaryExtractor(input_files, output_file)

    try:
        extractor.extract_and_save()
    except Exception as e:
        print(f"提取过程中出现错误: {e}")
        print("完整堆栈信息:")
        traceback.print_exc()  # 直接打印错误堆栈

if __name__ == "__main__":
    main()