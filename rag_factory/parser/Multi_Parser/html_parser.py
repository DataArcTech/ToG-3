import requests
import re
import os
from readability import Document
from markdownify import markdownify as md
import hashlib


class HtmlParser:
    def _url_to_markdown(self,url: str) -> str:
        try:
            resp = requests.get(url, timeout=10)
        except Exception as e:
            print(e)
        resp.encoding = resp.apparent_encoding
        return self._parser_html(resp.text)


    def _file_to_markdown(self,file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self._parser_html(html_content)



    def _parser_html(self, html_content):
        doc = Document(html_content)
        content_html = doc.summary()
        title = doc.title()

        markdown_text = md(
            content_html,
            # strip=['span'],      
            convert=['img', 'table', 'div'],  
            heading_style="ATX"
        )

        return f"# {title}\n\n{markdown_text}"
    
    def __call__(self, fnm, output_dir='./output'):
        if re.match(r"^https?://", fnm.strip()):  
            file_name = hashlib.md5(fnm.encode("utf-8")).hexdigest()[:10]
            md = self._url_to_markdown(fnm)
        elif os.path.exists(fnm): 
            md = self._file_to_markdown(fnm)
            file_name = os.path.basename(fnm).rsplit(".", 1)[0]
        else:
            raise ValueError(f"Unknown input: {fnm} (no URL or html file)")
        
        output_path = os.path.join(output_dir, file_name+"_markdown.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        return md

if __name__ == "__main__":
    parser = HtmlParser()
    parser('https://en.wikipedia.org/wiki/Artificial_intelligence', output_dir='/home/yangcehao/doc_analysis/MultiTypeParser/test')