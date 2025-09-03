import json
import re
import traceback
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DataAnalysisQuestion:
    """资料分析题目数据结构"""
    question_number: str  # 题号，如 "86"
    question_text: str    # 题目内容
    materials: str        # 材料内容
    answer: str           # 正确答案
    explanation: str   # 答案解析


@dataclass
class DataAnalysisAnswer:
    """资料分析答案解析数据结构"""
    question_number: str  # 题号，如 "86"
    explanation: str      # 解析内容
    answer: str           # 正确答案


class DataAnalysisExtractor:
    """资料分析题目和答案解析提取器"""
    
    def __init__(self, questions_file: str, answers_file: str):
        """
        初始化提取器
        
        Args:
            questions_file: 题目文件路径（img_content.json）
            answers_file: 答案解析文件路径（解析_layout.json）
        """
        self.questions_file = questions_file
        self.answers_file = answers_file
        self.questions_data = None
        self.answers_data = None
        self.question_numbers = set()
        self.materials_start_patterns = [
            re.compile(r"根据以下资料"),
            re.compile(r"[（(]([一二三四])[）)]"),
            re.compile(r"([一二三四])\s*、\s*"),
            re.compile(r"[（(]材料([0-9]*)[）)]")
        ]
        self.materials_end_pattern = re.compile(r"^\d+、\s*.+$")
        
    def load_data(self) -> None:
        """加载两个JSON文件的数据"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            print(f"成功加载题目文件: {self.questions_file}")
            
            with open(self.answers_file, 'r', encoding='utf-8') as f:
                self.answers_data = json.load(f)
            print(f"成功加载答案解析文件: {self.answers_file}")
        except Exception as e:
            print(f"加载文件失败: {e}")
            raise
    
    def find_data_analysis_section(self) -> Tuple[int, int]:
        """
        找到资料分析部分的起始和结束位置
        
        Returns:
            Tuple[int, int]: (起始索引, 结束索引)
        """
        start_idx = -1
        end_idx = -1
        for i, item in enumerate(self.questions_data):
            if "资料分析" in item.get("content", ""):
                print("资料分析开始：", item.get("content"))
                start_idx = i
                break
        
        end_idx = len(self.questions_data)
        return start_idx, end_idx
    
    def extract_questions(self, start_idx: int, end_idx: int) -> List[DataAnalysisQuestion]:
        """
        提取资料分析题目
        
        Args:
            start_idx: 资料分析部分起始索引
            end_idx: 资料分析部分结束索引
            
        Returns:
            List[DataAnalysisQuestion]: 题目列表
        """
        questions = []
        materials = []
        materials_start = False
        
        current_question = None
        current_options = []
        
        for i in range(start_idx+1, end_idx):
            item = self.questions_data[i]
            # print("item: ", item)
            item_text = item.get("content", "")
            if isinstance(item_text, dict) and item_text.__contains__("content"):
                item_text = str(item_text["content"])
            text = item_text.strip()
            
            if not text:
                continue

            # if self.materials_start_pattern.search(text):
            #     print("资料开始：", text)
            #     materials_start = True
            #     materials = []
            #     continue

            # # 如果遇到题目编号，停止收集材料
            # if self.materials_end_pattern.search(text):
            #     print("资料结束：", text)
            #     print("materials: ", materials)
            #     materials_start = False
                
            # # 收集材料内容
            # if materials_start:
            #     materials.append(text)
            
            for pattern in self.materials_start_patterns:
                materials_match = pattern.search(text)
                if materials_match:
                    break
            if materials_match:
                print("资料开始：", text)
                materials_start = True
                materials = []
                materials.append(text)
                continue
                
            # 检查是否是题目编号
            question_patterns = [
                # r'(\d+)\.\s+([\s\S]+)',
                r'(\d+)\s*、\s*([\s\S]+)'
            ]
            print("text: ", text)
            for pattern in question_patterns:
                question_match = re.search(pattern, text)
                if question_match:
                    # print("question_match: ", question_match)
                    break
            if question_match:
                if materials_start:
                    print("资料结束：", text)
                    materials_start = False
                # 保存前一个题目
                if current_question:
                    current_question.question_text = current_question.question_text + "\n" + "\n".join(current_options)
                    questions.append(current_question)
                
                # 开始新题目
                question_number = question_match.group(1)
                self.question_numbers.add(question_number)
                question_text = question_match.group(2)
                current_question = DataAnalysisQuestion(
                    question_number=question_number,
                    question_text=question_text,
                    materials=materials,
                    answer="",
                    explanation=""
                )
                current_options = []
            elif materials_start:
                materials.append(text)
            # 检查是否是选项
            elif text.startswith(('A.', 'B.', 'C.', 'D.')) and current_question:
                current_options.append(text)
        
        # 添加最后一个题目
        if current_question:
            current_question.question_text = current_question.question_text + "\n" + "\n".join(current_options)
            questions.append(current_question)
        
        return questions
    
    def extract_answers(self) -> List[DataAnalysisAnswer]:
        """
        提取资料分析题目的答案解析
        
        Returns:
            List[DataAnalysisAnswer]: 答案解析列表
        """
        answers = []
        question_number = None
        print("self.question_numbers: ", self.question_numbers)
        
        for i in range(len(self.answers_data)):
            item = self.answers_data[i]
            
            text = item.get("content", "").strip().lstrip('\n')
            # 匹配解析标题，如 "【解析 86】"
            question_patterns = [
                r'【解析\s*(\d+)】\.\s+([\s\S]+)',
                r'【解析\s*([一二三四])】\.\s+([\s\S]+)',
                # r'(\d+)\.\s+([\s\S]+)',
                r'(\d+)\s*、\s*([\s\S]+)'
            ]
            for pattern in question_patterns:
                match = re.search(pattern, text)
                if match:
                    question_number = match.group(1)
                    break
            print("answer_match: ", match)
            if match and match.group(1) in self.question_numbers:
                question_number = match.group(1)
                print("question_number: ", question_number)
                    
                # 查找对应的解析内容
                # explanation = self._find_explanation_content(i)
                explanation = match.group(2).strip()
                    
                if explanation:
                    print("解析: ", question_number)
                    print("explanation: ", explanation)
                    answer = self._find_answer_content(explanation)
                    print("answer: ", answer)
                    answers.append(DataAnalysisAnswer(
                        question_number=question_number,
                        explanation=explanation,
                        answer=answer
                    ))
        
        return answers
    
    def _find_explanation_content(self, start_index: int) -> str:
        """
        查找解析标题对应的解析内容
        
        Args:
            header_item: 解析标题项
            
        Returns:
            str: 解析内容
        """
        explanation_parts = []
        print("start_index: ", start_index)
        
        # 从标题后开始收集解析内容，直到遇到下一个标题或页面结束
        for i in range(start_index + 1, len(self.answers_data)):
            item = self.answers_data[i]
            print("item: ", item)
            # 如果遇到新的解析标题，停止收集
            
            text = item.get("content", "")
            match = re.search(r'【解析\s*(\d+)】', text)
            if match:
                print("end_item: ", item.get("content"))
                break
                
            # 收集解析内容
            if item.get("content", "").strip():
                explanation_parts.append(item.get("content", "").strip())
        
        return "\n".join(explanation_parts)
    
    def _find_answer_content(self, explanation: str) -> str:
        """
        查找解析内容对应的正确答案
        """
        answer_patterns = [
            r'故正确答案为\s*([A-D])',
            r'正确答案[:：]\s*([A-D])',
            r'选择\s*([A-D])\s*选项'
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, explanation)
            if match:
                return match.group(1)
        return ""

    def match_questions_and_answers(self, questions: List[DataAnalysisQuestion], 
                                  answers: List[DataAnalysisAnswer]) -> List[DataAnalysisQuestion]:
        """
        将题目和答案解析进行匹配
        
        Args:
            questions: 题目列表
            answers: 答案解析列表
            
        Returns:
            List[DataAnalysisQuestion]: 包含答案解析的完整题目列表
        """
        # 创建答案解析的查找字典
        answers_dict = {answer.question_number: answer for answer in answers}
        
        # 为每个题目添加答案解析
        for question in questions:
            if question.question_number in answers_dict:
                question.answer = answers_dict[question.question_number].answer
                question.explanation = answers_dict[question.question_number].explanation   
            else:
                question.answer = "未找到答案解析"
        
        return questions
    
    def extract_all(self) -> List[DataAnalysisQuestion]:
        """
        提取所有资料分析题目和答案解析
        
        Returns:
            List[DataAnalysisQuestion]: 完整的资料分析题目列表
        """
        if not self.questions_data or not self.answers_data:
            self.load_data()
        
        # 找到资料分析部分
        start_idx, end_idx = self.find_data_analysis_section()
        
        if start_idx == -1:
            print("未找到资料分析部分")
            return []
        
        print(f"找到资料分析部分: 索引 {start_idx} 到 {end_idx}")
        
        # 提取题目
        questions = self.extract_questions(start_idx, end_idx)
        print(f"提取到 {len(questions)} 道资料分析题目\n\n")
        
        # 提取答案解析
        answers = self.extract_answers()
        print(f"提取到 {len(answers)} 个答案解析\n\n")
        
        # 匹配题目和答案解析
        complete_questions = self.match_questions_and_answers(questions, answers)
        
        return complete_questions
    
    def save_to_json(self, questions: List[DataAnalysisQuestion], output_file: str) -> None:
        """
        将提取的题目保存为JSON文件
        
        Args:
            questions: 题目列表
            output_file: 输出文件路径
        """
        # 转换为字典格式以便JSON序列化
        questions_dict = []
        for q in questions:
            questions_dict.append({
                "question_number": q.question_number,
                "question_text": str(q.materials) + "\n" + q.question_text,
                "materials": "",
                "answer": q.answer,
                "explanation": q.explanation
            })
        
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(questions_dict, f, ensure_ascii=False, indent=2)
        
        print(f"题目已保存到: {output_file}")
    
    def print_summary(self, questions: List[DataAnalysisQuestion]) -> None:
        """
        打印提取结果的摘要
        
        Args:
            questions: 题目列表
        """
        print(f"\n=== 资料分析题目提取摘要 ===")
        print(f"总共提取到 {len(questions)} 道题目")
        
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. 第{q.question_number}题")
            print(f"   题目: {q.question_text[:50]}...")
            print(f"   是否有解析: {'是' if q.answer != '未找到答案解析' else '否'}")


def main():
    """主函数示例"""
    # 文件路径
    questions_file = "analysis/2013深圳市公务员录用考试《行测》真题.json"
    answers_file = "analysis/2013深圳市公务员录用考试《行测》真题答案及解析.json"

    example_paths = ["analysis/资料分析题目_例题.json",
                     "analysis/资料分析题目2024.json",
                     "analysis/资料分析题目2020.json",
                     "analysis/资料分析题目2019.json",
                     "analysis/资料分析题目2018.json",
                     "analysis/资料分析题目2017.json",
                     "analysis/资料分析题目2016.json",
                     "analysis/资料分析题目2015.json",
                     "analysis/资料分析题目2014.json",
                     "analysis/资料分析题目2013.json",
                     "analysis/资料分析题目2012.json",
                     "analysis/资料分析题目2011.json",
                     "analysis/资料分析题目2010.json",
                     "analysis/资料分析题目2009.json",
                     "analysis/资料分析题目2008.json"]
    example_data = []
    for example_path in example_paths:
        with open(example_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        for example in examples:
            example_data.append(DataAnalysisQuestion(
                question_number=example['question_number'], 
                question_text=example['question_text'], 
                materials=example['materials'], 
                answer=example['answer'], 
                explanation=example['explanation']))
    print("example_data_num: ", len(example_data))
    
    # 创建提取器
    # extractor = DataAnalysisExtractor(questions_file, answers_file)
    
    # try:
    #     # 提取所有资料分析题目
    #     questions = extractor.extract_all()
        
    #     if questions:
    #         # 打印摘要
    #         # extractor.print_summary(questions)
            
    #         # 保存到JSON文件
    #         extractor.save_to_json(questions, "analysis/资料分析题目2013.json")
            
    #         # 打印第一道题目的详细信息作为示例
    #         # if questions:
    #         #     first_q = questions[0]
    #         #     print(f"\n=== 第一道题目详细信息 ===")
    #         #     print(f"题号: {first_q.question_number}")
    #         #     print(f"题目: {first_q.question_text}")
    #         #     print(f"选项: {first_q.options}")
    #         #     print(f"材料: {first_q.materials[:100]}...")
    #         #     print(f"解析: {first_q.answer[:200]}...")
        
    # except Exception as e:
    #     print(f"提取过程中出现错误: {e}")
    #     print("完整堆栈信息:")
    #     traceback.print_exc()  # 直接打印错误堆栈
    #     # 或者获取堆栈字符串
    #     error_msg = traceback.format_exc()
    #     print("格式化后的错误信息:\n", error_msg)


if __name__ == "__main__":
    main()
