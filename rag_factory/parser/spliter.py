import re
import os
import json

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter
)
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import List, Dict, Optional, Literal

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name_or_path: str):
        self.model = SentenceTransformer(model_name_or_path, device="cuda")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True, prompt_name="Document", batch_size=1).tolist()

    def __call__(self, text: str) -> List[float]:
        return self.embed_query(text)
    

class CustomerSpliter:
    def __init__(self, 
                begin_pattern = r'^([(（]材料|\d+[)、.]|（材料|[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\d+、|\d+. |阅读下列材料|统计表:)'
                ):
        self.begin_pattern = begin_pattern

        
    @staticmethod
    def _lv1_heading(paragraph):
        return bool(re.match(r'^第[一二三四五六七八九十零百千万]+[章]', paragraph.strip()))  
    
    @staticmethod
    def _lv2_heading(paragraph):
        return bool(re.match(r'^第[一二三四五六七八九十零百千万]+[节]', paragraph.strip()))
    
    def split_text_with_title(self, data) -> List:
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
                if re.match(r'^\s*(?:.?)?(例|【例|（例|【解析|典型真题)\s*', text.strip()) and len(temp_text)>5: 
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


class TextSpliter:
    def __init__(
        self,
        splitter_type: Literal["recursive", "semantic"] = "recursive",
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        min_chunk_size: int = 512,
        separators: Optional[List[str]] = None,
        embeddings_model: Optional[str] = None
    ):
        """
        初始化Markdown处理器
        
        参数:
            splitter_type: 分割器类型("recursive", "semantic")
            chunk_size: 每个文本块的最大长度(仅适用于recursive)
            chunk_overlap: 块之间的重叠长度(仅适用于recursive)
            min_chunk_size: 每个块的最小长度(仅适用于semantic)
            separators: 文本分割符列表(仅适用于recursive)
            embeddings_model: 语义分块使用的嵌入模型(仅适用于semantic)
        """
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.encoding = 'utf-8'
        
        self.text_splitter = self._initialize_splitter(
            splitter_type,
            chunk_size,
            chunk_overlap,
            min_chunk_size,
            separators,
            embeddings_model
        )
    
    def _initialize_splitter(
        self,
        splitter_type: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int,
        separators: Optional[List[str]],
        embeddings_model: Optional[str]
    ):
        """根据类型初始化不同的文本分割器"""
        default_separators = ["\n\n", "\n", "。", "！", "？"]
        
        if splitter_type == "semantic":
            embeddings = SentenceTransformerEmbeddings(model_name_or_path=embeddings_model)
            return SemanticChunker(embeddings=embeddings, min_chunk_size=min_chunk_size)
        else: 
            return RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators or default_separators
            )


    def split_text(
        self,
        text,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        
        result = []
        for chunk in chunks:
            chunk_data = {"text": chunk}
            if metadata:
                chunk_data.update(metadata)
            result.append(chunk_data)
        
        return result
    


if __name__ == "__main__":
    md_file = '/home/yangcehao/edu_project/dots_ocr_result/parsed/2026国考公务员行测-资料部分_markdown.md'
    layout_file = '/home/yangcehao/edu_project/dots_ocr_result/parsed/2026国考公务员行测-资料部分_layout.json'
    with open(layout_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    spliter = CustomerSpliter()
    metadata = spliter.split_text(data)
    
    with open('/home/yangcehao/RAG-Factory/rag_factory/parser/text.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # def add_newlines_before_options(text):
    #     """
    #     在 A. B. C. D. 前添加换行符
    #     """
    #     # 匹配 A. B. C. D. （可能前面有空格）
    #     pattern = r'(?<!\n)([ABCD]\.)'
    #     result = re.sub(pattern, r'\n\1', text)
    #     return result

    # file = '/home/yangcehao/edu_project/dots_ocr_result/new_chunked/v2/2026国考公务员行测-资料部分.json'
    # with open(file, 'r', encoding='utf-8') as f:
    #     data = json.load(f)
    
    # for row in data:
    #     text = row['content']
    #     text = add_newlines_before_options(text)
    #     row['content'] = text
    #     row['metadata']['text'] = text

    # with open('/home/yangcehao/RAG-Factory/rag_factory/parser/text1.json', 'w', encoding='utf-8') as f:
    #      json.dump(data, f, ensure_ascii=False, indent=4)
    