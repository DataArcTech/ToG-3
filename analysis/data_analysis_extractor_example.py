import json
import re
import traceback
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DataAnalysisQuestion:
    """资料分析题目数据结构"""
    question_number: str = ""  # 题号，如 "86"
    question_text: str = ""   # 题目内容
    title_lv1: str = ""
    title_lv2: str = ""
    knowledge1: str = ""
    knowledge2: str = ""
    materials: str = ""        # 材料内容
    answer: str = ""           # 正确答案
    explanation: str = ""   # 答案解析


@dataclass
class DataAnalysisAnswer:
    """资料分析答案解析数据结构"""
    question_number: str = "" # 题号，如 "86"
    explanation: str = ""      # 解析内容
    answer: str = ""           # 正确答案


class DataAnalysisExtractor:
    """资料分析题目和答案解析提取器"""
    
    def __init__(self, example_file: str):
        """
        初始化提取器
        """
        self.example_file = example_file
        self.questions_data = None
        self.delete_pattern = r"打开华图在线.*?扫码查看本题解析"
        self.materials_end_pattern = re.compile(r"^\d+\.\s+.+$")
        
    def load_data(self) -> None:
        """加载JSON文件的数据"""
        try:
            with open(self.example_file, 'r', encoding='utf-8') as f:
                self.questions_data = json.load(f)
            print(f"成功加载题目文件: {self.example_file}")
        except Exception as e:
            print(f"加载文件失败: {e}")
            raise
    
    def extract_questions(self, questions_data: List[Dict]) -> List[DataAnalysisQuestion]:
        """
        提取资料分析题目
    
            
        Returns:
            List[DataAnalysisQuestion]: 题目列表
        """
        questions = []
        separators = ["题型判断", "华图思路", "题目讲解"]
        split_pattern = "|".join(map(re.escape, sorted(separators, key=len, reverse=True)))
        real_pattern1 = r"\( *\d{4} *[\u4e00-\u9fa5]+ *\)"
        real_pattern2 = r"\（ *\d{4} *[\u4e00-\u9fa5]+ *\）"
        
        current_question = None
        
        for item in questions_data:
            item_text = item.get("content", "")
            if isinstance(item_text, dict) and item_text.__contains__("content"):
                item_text = str(item_text["content"])
            text = item_text.strip()
            
            if not text or "题目讲解" not in text:
                continue

            matches = re.findall(real_pattern1, text)
            if not matches:
                matches = re.findall(real_pattern2, text)

            if not matches:
                continue

            cleaned_text = re.sub(self.delete_pattern, "", text, flags=re.DOTALL)


            split_text = re.split(split_pattern, cleaned_text)
            # print("split_text: ", split_text)
            title_lv1=item.get("metadata", "").get("title_lv1", "") 
            title_lv2=item.get("metadata", "").get("title_lv2", "")
            title_match = re.findall(r"第[一二三四五六七八九十]节\s*([\u4e00-\u9fa5]+)", title_lv2)
            if title_match:
                title_match = title_match[0]
            else:
                title_match = ""
            # 收集材料内容
            current_question = DataAnalysisQuestion(
                question_text=split_text[0],
                title_lv1=title_lv1,
                title_lv2=title_lv2,
                knowledge1=self._find_knowledge(split_text[1]),
                knowledge2=title_match,
                answer=self._find_answer_content(split_text[-1]),
                explanation=split_text[-1].replace('|', '').strip()
            )

            questions.append(current_question)
        
        return questions

    def _find_knowledge(self, text: str) -> str:
        """
        查找知识点
        """
        knowledge_match = re.findall(r".*?本题考查\s*([^\|]+)\。", text)
        if knowledge_match:
            return knowledge_match[0]
        return ""

    def _find_answer_content(self, explanation: str) -> str:
        """
        查找解析内容对应的正确答案
        """
        match = re.search(r'选择\s*([A-D])\s*选项', explanation)
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
        if not self.questions_data:
            self.load_data()
        
        # 提取题目
        questions = self.extract_questions(self.questions_data)
        print(f"提取到 {len(questions)} 道资料分析题目\n\n")
        
        return questions
    
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
                "title_lv1":q.title_lv1,
                "title_lv2":q.title_lv2,
                "knowledge1": q.knowledge1,
                "knowledge2": q.knowledge2,
                "question_text": q.question_text,
                "materials": q.materials,
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
    example_file = "analysis/2026国考公务员行测-资料部分(例题精讲).json"
    
    # 创建提取器
    extractor = DataAnalysisExtractor(example_file)
    
    try:
        # 提取所有资料分析题目
        questions = extractor.extract_all()
        
        if questions:
            # 打印摘要
            # extractor.print_summary(questions)
            
            # 保存到JSON文件
            extractor.save_to_json(questions, "analysis/资料分析题目_例题.json")
            
            # 打印第一道题目的详细信息作为示例
            # if questions:
            #     first_q = questions[0]
            #     print(f"\n=== 第一道题目详细信息 ===")
            #     print(f"题号: {first_q.question_number}")
            #     print(f"题目: {first_q.question_text}")
            #     print(f"选项: {first_q.options}")
            #     print(f"材料: {first_q.materials[:100]}...")
            #     print(f"解析: {first_q.answer[:200]}...")
        
    except Exception as e:
        print(f"提取过程中出现错误: {e}")
        print("完整堆栈信息:")
        traceback.print_exc()  # 直接打印错误堆栈
        # 或者获取堆栈字符串
        error_msg = traceback.format_exc()
        print("格式化后的错误信息:\n", error_msg)


if __name__ == "__main__":
    main()
