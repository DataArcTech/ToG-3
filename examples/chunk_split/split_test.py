import sys
import os

# 添加 RAG-Factory 目录到 Python 路径
rag_factory_path = os.path.join(os.path.dirname(__file__), "..","..")
sys.path.insert(0, rag_factory_path)

import re
import json
import numpy as np

from rag_factory.parser.Spliter import RecursiveCharacterTextSplitter, SemanticChunker, TokenTextSplitter, MarkdownHeaderTextSplitter
from rag_factory.Embed.Embedding_Huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from typing import List

class LayoutSpliter:
    """
        适用于dots ocr 解析的layout文件
    """
    def __init__(self, 
                begin_pattern = r'^([(（]材料|\d+[)、.]|（材料|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+、|\d+. |阅读下列材料|统计表:)'
                ):
        """
        begin_pattern: 单独分割的开头格式
        """
        self.begin_pattern = begin_pattern

        
    @staticmethod
    def _lv1_heading(paragraph):
        return bool(re.match(r'^第[一二三四五六七八九十零百千万]+[章]', paragraph.strip()))  
    
    @staticmethod
    def _lv2_heading(paragraph):
        return bool(re.match(r'^第[一二三四五六七八九十零百千万]+[节]', paragraph.strip()))
    
    def split_text_with_title(self, data, begin_pattern = None ) -> List:
        """
            对有固定一级标题，二级标题格式的数据进行分块
            输入：layout json 数据
            输出：metadata格式：
            {
                "title_lv1":,
                "title_lv2":,
                "table":,
                "figure":,
                "text":
            }
        """
        pattern = begin_pattern or self.begin_pattern
        title_lv1 = ''
        title_lv2 = ''
        temp_text = ''
        table = []
        figure = []
        meta_data = []
        for index, row in enumerate(tqdm(data)):
            category = row.get('category','')
            text = row.get('text','')
            if index == len(data):
                meta_data.append({
                        "title_lv1":title_lv1,
                        "title_lv2":title_lv2,
                        "table":table,
                        "figure":figure,
                        "text":temp_text
                    })
            if category in ['Page-footer', 'Page-header'] :
                continue
            if category in ["Section-header", "Title"]:
                if len(temp_text) >= 10:
                    meta_data.append({
                        "title_lv1":title_lv1,
                        "title_lv2":title_lv2,
                        "table":table,
                        "figure":figure,
                        "text":temp_text
                    })
                    temp_text = ''
                    table = []
                    figure = []

                text = re.sub(r'^[#*]+\s*', '', text, flags=re.MULTILINE)
                if self._lv1_heading(text):
                    title_lv1 = text
                    title_lv2 = ''
                elif self._lv2_heading(text):
                    title_lv2 = text
                else:
                    temp_text +=" "+text
            elif category == 'Table':
                table.append(text)
                temp_text += text
            elif category == 'Picture':
                if text != '':
                    figure.append(f"page_{row['page_no']}_{row['index']}")
                    temp_text += f"<figure> {text} </figure>"
            else:
                text = re.sub(r'^[#*]+\s*', '', text)
                if re.match(pattern, text.strip()) and len(temp_text)>5: 
                    meta_data.append({
                        "title_lv1":title_lv1,
                        "title_lv2":title_lv2,
                        "table":table,
                        "figure":figure,
                        "text":temp_text
                    })
                    temp_text = text
                    table = []
                    figure = []
                else:
                    temp_text += " "+text
        return meta_data
    
    def split_text(self, data) -> List[str]:
        """
            将每个标题间的内容分成一个块
            输入：layout json 数据
            输出：分块文本
        """
        temp_text = ''
        temp = []
        for row in data:
            category = row.get('category','')
            text = str(row.get("text","")).lstrip('#').strip()
            if category == 'Page-header' or category == 'Page-footer':
                continue
            
            if category == 'Picture':
                p_dir = f'page_{row.get("page_no","")}_{row.get("index","")}'
                text = f"<figure> ({p_dir}) {text} </figure>"
            if category in ["Section-header", 'Title']:
                temp.append(temp_text)
                temp_text = text
                continue
            for t in text.split("\n\n"):
                if re.match(self.begin_pattern, t.lstrip("*#").strip()) :
                    temp.append(temp_text)
                    temp_text = t
                else:
                    temp_text += " "+ t
        
        temp.append(temp_text)
        temp = [x for x in temp if x != ""]
        return temp


if __name__ == "__main__":

    md_file = '/home/yangcehao/RAG-Factory/examples/dot_ocr_chunk_split/test_file/2026国考公务员行测-资料部分_markdown.md'
    layout_file = '/home/yangcehao/RAG-Factory/examples/dot_ocr_chunk_split/test_file/2026国考公务员行测-资料部分_img_content.json'


    with open(layout_file, 'r', encoding='utf-8') as f:
        layout_data = json.load(f)


    with open(md_file, 'r', encoding='utf-8') as f:
        md_texts = f.read()


    layout_spliter = LayoutSpliter()
    text_spliter = MarkdownHeaderTextSplitter()

    layout_result = layout_spliter.split_text_with_title(layout_data, begin_pattern = r'^\s*(?:.?)?(例|【例|（例|【解析|典型真题)\s*')

    md_result = text_spliter.split_text(md_texts)

    with open('/home/yangcehao/RAG-Factory/examples/dot_ocr_chunk_split/test1.json', 'w', encoding='utf-8') as f:
        json.dump(layout_result, f, ensure_ascii=False, indent=4)

    with open('/home/yangcehao/RAG-Factory/examples/dot_ocr_chunk_split/test2.json', 'w', encoding='utf-8') as f:
        json.dump(md_result, f, ensure_ascii=False, indent=4)
