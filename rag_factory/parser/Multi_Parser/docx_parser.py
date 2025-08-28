from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
import re
import pandas as pd
from collections import Counter
from io import BytesIO
import os
import base64
import pypandoc
import tempfile
from pdf_parser import DotsOCRParser
from docx.table import Table
from docx.text.paragraph import Paragraph

class DocxParser:

    def _convert_docx_to_pdf(self, fnm):
        pypandoc.download_pandoc()
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            pdf_path = tmp_pdf.name

        try:
            if isinstance(fnm, str):

                pypandoc.convert_file(fnm, 'pdf', outputfile=pdf_path)
            else:
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_docx:
                    tmp_docx.write(fnm)
                    tmp_docx_path = tmp_docx.name
                pypandoc.convert_file(tmp_docx_path, 'pdf', outputfile=pdf_path)
                os.remove(tmp_docx_path)  

            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
        finally:
            os.remove(pdf_path) 

        return pdf_bytes    

    
    def _parse_docx(self, fnm, image_dir):
        self.doc = Document(fnm) if isinstance(fnm, str) else Document(BytesIO(fnm))
        os.makedirs(image_dir, exist_ok=True)
        md_output = []
        NAMESPACE = {
        "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    }

        def iter_block_items(doc):
            parent_elm = doc.element.body
            for child in parent_elm:
                if child.tag.endswith('tbl'):
                    yield Table(child, doc)
                elif child.tag.endswith('p'):
                    yield Paragraph(child, doc)

        img_num = 1
        for item in iter_block_items(self.doc):
            if isinstance(item, Table):
                table_html = "<table>\n"
                for row in item.rows:
                    table_html += "  <tr>\n"
                    for cell in row.cells:
                        table_html += f"    <td>{cell.text.strip()}</td>\n"
                    table_html += "  </tr>\n"
                table_html += "</table>\n"
                md_output.append(table_html)

            else:  # Paragraph
                if item.text.strip():
                    md_output.append(item.text.strip())
                
                for run in item.runs:
                    inline_shapes = run._element.findall(".//a:blip", namespaces=NAMESPACE)
                    for blip in inline_shapes:
                        rId = blip.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
                        if rId:
                            image_part = self.doc.part.related_parts[rId]
                            image_data = image_part.blob

                            img_ext = image_part.content_type.split("/")[-1]  # 例如 "png", "jpeg"
                            img_name = f"img_{img_num}.{img_ext}"
                            img_path = os.path.join(image_dir, img_name)
                            with open(img_path, "wb") as f:
                                f.write(image_data)
                            img_num +=1
                            md_output.append(f"![image]({img_path})")

                    if 'lastRenderedPageBreak' in run._element.xml or '<w:br w:type="page"/>' in run._element.xml:
                        md_output.append("\n---\n")

        return "\n\n".join(md_output)
    
    def __call__(self, fnm, output_dir='./output', use_ocr=False):
        if isinstance(fnm, str):
            file_name = os.path.basename(fnm).rsplit(".docx", 1)[0]
        elif isinstance(fnm, bytes):
            prefix_bytes = fnm[:10]
            prefix_hex = prefix_bytes.hex()
            # prefix_b64 = base64.urlsafe_b64encode(prefix_bytes).decode('utf-8')
            file_name= f"{prefix_hex}"
        else:
            file_name="input_docx"
        output_path = os.path.join(output_dir, file_name+"_markdown.md")
        if use_ocr:
            pdf_parser = DotsOCRParser(
                output_dir=output_dir, use_hf=False
            )
            pdf_byte = self._convert_docx_to_pdf(fnm)
            results = pdf_parser.parse_pdf(pdf_byte, 
                                           file_name, 
                                           prompt_mode="prompt_layout_all_en", 
                                           save_dir=os.path.join(output_dir, file_name)
                                           )
            pdf_parser.process_output(filename=file_name, results=results, output_dir=output_dir, save_layout=False)
            
        else:
            image_dir = os.path.join(output_dir, file_name+"/image")
            md_output = self._parse_docx(fnm, image_dir=image_dir )
            with open(output_path, 'w', encoding='utf-8') as output_f:
                output_f.write(md_output)

    
if __name__ == "__main__":
    parser = DocxParser()
    s = parser('/home/yangcehao/doc_analysis/MultiTypeParser/Untitled.docx', 
               output_dir='/home/yangcehao/doc_analysis/MultiTypeParser/test', use_ocr=True)

