from docx_parser import DocxParser
from excel_parser import ExcelParser
from pdf_parser import DotsOCRParser
from ppt_parser import PptParser
from html_parser import HtmlParser
import argparse
import os
import glob
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import logging

def main():
    arg = argparse.ArgumentParser(
        description="Multi type parser",
    )

    arg.add_argument(
        "--input_path", type=str,
        help="Input file or dir path"
    )
    arg.add_argument(
        "--output_path", type=str,
        help="Output dir path"
    )
    arg.add_argument(
        "--parse_type", type=str, default="all",
        help="Choose one or multi parse type from [pdf, docx, excel, ppt, html, image] or all. (use ',' to connect each type) "
    )
    args = arg.parse_args()

    
    parse_type = args.parse_type.split(",")

    if "all" in parse_type:
        available_type = ["pdf", "docx", "excel", "ppt", "html", "image"]
    else:
        available_type = list(set(parse_type) & set(["pdf", "docx", "excel", "ppt", "html", "image"]))

    pdf_suffixes = [".pdf"]
    docx_suffiex = [".docx"]
    excel_suffiex = [".csv", ".xlsx"]
    ppt_suffiex = [".pptx"]
    html_suffiex = [".html"]
    img_suffiex = [".jpg", ".jpeg", ".png"]


    if os.path.isdir(args.input_path):
        docs_paths = sorted(Path(args.input_path).glob('*'))
        pdf_files = [file for file in docs_paths if file.suffix in pdf_suffixes]
        docx_files = [file for file in docs_paths if file.suffix in docx_suffiex]
        excel_files = [file for file in docs_paths if file.suffix in excel_suffiex]
        ppt_files = [file for file in docs_paths if file.suffix in ppt_suffiex]
        html_files = [file for file in docs_paths if file.suffix in html_suffiex]
        img_files = [file for file in docs_paths if file.suffix in img_suffiex]
    
        if "pdf" in available_type:
            print("Parsing pdf files")
            parser = DotsOCRParser(output_dir=args.output_path, use_hf=True)
            for f in tqdm(pdf_files):
                results = parser.parse_file(
                        str(f), 
                        prompt_mode="prompt_layout_all_en",
                        )
        if "docx" in available_type:
            print("Parsing docx files")
            parser = DocxParser()
            for f in tqdm(docx_files):
                result = parser(str(f), output_dir=args.output_path)
        if "excel" in available_type:
            print("Parsing excel files")
            parser = ExcelParser()
            for f in tqdm(excel_files):
                result = parser(str(f), output_dir=args.output_path)
        if "ppt" in available_type:
            print("Parsing ppt files")
            parser = PptParser()
            for f in tqdm(ppt_files):
                result = parser(str(f), output_dir=args.output_path)
        if "html" in available_type:
            print("Parsing html files")
            parser = HtmlParser()
            for f in tqdm(html_files):
                result = parser(str(f), output_dir=args.output_path)
        if "image" in available_type:
            print("Parsing image files")
            parser = DotsOCRParser(output_dir=args.output_path, use_hf=True)
            for f in tqdm(img_files):
                results = parser.parse_file(
                        str(f), 
                        prompt_mode="prompt_layout_all_en",
                        )

    elif os.path.isfile(args.input_path):

        filename, file_ext = os.path.splitext(os.path.basename(args.input_path))
        if file_ext in pdf_suffixes or file_ext in img_suffiex:
            parser = DotsOCRParser(output_dir=args.output_path, use_hf=True)
            result = parser.parse_file(
                        str(args.input_path), 
                        prompt_mode="prompt_layout_all_en",
                        )
        elif file_ext in docx_suffiex:
            parser = DocxParser()       
            result = parser(args.input_path, output_dir=args.output_path)
        elif file_ext in excel_suffiex:
            parser = ExcelParser()
            result = parser(args.input_path, output_dir=args.output_path)
        elif file_ext in ppt_suffiex:
            parser = PptParser()
            result = parser(args.input_path, output_dir=args.output_path)
        elif file_ext in html_suffiex:
            parser = HtmlParser()
            result = parser(args.input_path, output_dir=args.output_path)

    else:
        try:
            if urlparse(args.input_path):
                parser = HtmlParser()
                result = parser(args.input_path, output_dir=args.output_path)
        except:
            print(f"'{args.input_path}' no exist or can not parse")

if __name__ == "__main__":
    main()
 