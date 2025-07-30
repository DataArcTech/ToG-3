from __future__ import annotations
import base64
import subprocess
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
import json
import logging
from .Base import Parser


class DoclingParser(Parser):
    """
    Docling document parsing utility class.

    Specialized in parsing Office documents and HTML files, converting the content
    into structured data and generating markdown and JSON output.
    """

    # Define Docling-specific formats
    HTML_FORMATS = {".html", ".htm", ".xhtml"}

    def __init__(self) -> None:
        """Initialize DoclingParser"""
        super().__init__()

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse PDF document using Docling

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object for easier handling
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

            name_without_suff = pdf_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = pdf_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=pdf_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_pdf: {str(e)}")
            raise

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse document using Docling based on file extension

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        # Convert to Path object
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")

        # Get file extension
        ext = file_path.suffix.lower()

        # Choose appropriate parser based on file type
        if ext == ".pdf":
            return self.parse_pdf(file_path, output_dir, method, lang, **kwargs)
        elif ext in self.OFFICE_FORMATS:
            return self.parse_office_doc(file_path, output_dir, lang, **kwargs)
        elif ext in self.HTML_FORMATS:
            return self.parse_html(file_path, output_dir, lang, **kwargs)
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Docling only supports PDF files, Office formats ({', '.join(self.OFFICE_FORMATS)}) "
                f"and HTML formats ({', '.join(self.HTML_FORMATS)})"
            )

    def _run_docling_command(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        file_stem: str,
        **kwargs,
    ) -> None:
        """
        Run docling command line tool

        Args:
            input_path: Path to input file or directory
            output_dir: Output directory path
            file_stem: File stem for creating subdirectory
            **kwargs: Additional parameters for docling command
        """
        # Create subdirectory structure similar to MinerU
        file_output_dir = Path(output_dir) / file_stem / "docling"
        file_output_dir.mkdir(parents=True, exist_ok=True)

        cmd_json = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "json",
            str(input_path),
        ]
        cmd_md = [
            "docling",
            "--output",
            str(file_output_dir),
            "--to",
            "md",
            str(input_path),
        ]

        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            docling_subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                docling_subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result_json = subprocess.run(cmd_json, **docling_subprocess_kwargs)
            result_md = subprocess.run(cmd_md, **docling_subprocess_kwargs)
            logging.info("Docling command executed successfully")
            if result_json.stdout:
                logging.debug(f"JSON cmd output: {result_json.stdout}")
            if result_md.stdout:
                logging.debug(f"Markdown cmd output: {result_md.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error running docling command: {e}")
            if e.stderr:
                logging.error(f"Error details: {e.stderr}")
            raise
        except FileNotFoundError:
            raise RuntimeError(
                "docling command not found. Please ensure Docling is properly installed."
            )

    def _read_output_files(
        self,
        output_dir: Path,
        file_stem: str,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Read the output files generated by docling and convert to MinerU format

        Args:
            output_dir: Output directory
            file_stem: File name without extension

        Returns:
            Tuple containing (content list JSON, Markdown text)
        """
        # Use subdirectory structure similar to MinerU
        file_subdir = output_dir / file_stem / "docling"
        md_file = file_subdir / f"{file_stem}.md"
        json_file = file_subdir / f"{file_stem}.json"

        # Read markdown content
        md_content = ""
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    md_content = f.read()
            except Exception as e:
                logging.warning(f"Could not read markdown file {md_file}: {e}")

        # Read JSON content and convert format
        content_list = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    docling_content = json.load(f)
                    # Convert docling format to minerU format
                    content_list = self.read_from_block_recursive(
                        docling_content["body"],
                        "body",
                        file_subdir,
                        0,
                        "0",
                        docling_content,
                    )
            except Exception as e:
                logging.warning(f"Could not read or convert JSON file {json_file}: {e}")
        return content_list, md_content

    def read_from_block_recursive(
        self,
        block,
        type: str,
        output_dir: Path,
        cnt: int,
        num: str,
        docling_content: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        content_list = []
        if not block.get("children"):
            cnt += 1
            content_list.append(self.read_from_block(block, type, output_dir, cnt, num))
        else:
            if type not in ["groups", "body"]:
                cnt += 1
                content_list.append(
                    self.read_from_block(block, type, output_dir, cnt, num)
                )
            members = block["children"]
            for member in members:
                cnt += 1
                member_tag = member["$ref"]
                member_type = member_tag.split("/")[1]
                member_num = member_tag.split("/")[2]
                member_block = docling_content[member_type][int(member_num)]
                content_list.extend(
                    self.read_from_block_recursive(
                        member_block,
                        member_type,
                        output_dir,
                        cnt,
                        member_num,
                        docling_content,
                    )
                )
        return content_list

    def read_from_block(
        self, block, type: str, output_dir: Path, cnt: int, num: str
    ) -> Dict[str, Any]:
        if type == "texts":
            if block["label"] == "formula":
                return {
                    "type": "equation",
                    "img_path": "",
                    "text": block["orig"],
                    "text_format": "unkown",
                    "page_idx": cnt // 10,
                }
            else:
                return {
                    "type": "text",
                    "text": block["orig"],
                    "page_idx": cnt // 10,
                }
        elif type == "pictures":
            try:
                base64_uri = block["image"]["uri"]
                base64_str = base64_uri.split(",")[1]
                # Create images directory within the docling subdirectory
                image_dir = output_dir / "images"
                image_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                image_path = image_dir / f"image_{num}.png"
                with open(image_path, "wb") as f:
                    f.write(base64.b64decode(base64_str))
                return {
                    "type": "image",
                    "img_path": str(image_path.resolve()),  # Convert to absolute path
                    "image_caption": block.get("caption", ""),
                    "image_footnote": block.get("footnote", ""),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process image {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Image processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }
        else:
            try:
                return {
                    "type": "table",
                    "img_path": "",
                    "table_caption": block.get("caption", ""),
                    "table_footnote": block.get("footnote", ""),
                    "table_body": block.get("data", []),
                    "page_idx": cnt // 10,
                }
            except Exception as e:
                logging.warning(f"Failed to process table {num}: {e}")
                return {
                    "type": "text",
                    "text": f"[Table processing failed: {block.get('caption', '')}]",
                    "page_idx": cnt // 10,
                }

    def parse_office_doc(
        self,
        doc_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse office document directly using Docling

        Supported formats: .doc, .docx, .ppt, .pptx, .xls, .xlsx

        Args:
            doc_path: Path to the document file
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Document file does not exist: {doc_path}")

            if doc_path.suffix.lower() not in self.OFFICE_FORMATS:
                raise ValueError(f"Unsupported office format: {doc_path.suffix}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=doc_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_office_doc: {str(e)}")
            raise

    def parse_html(
        self,
        html_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse HTML document using Docling

        Supported formats: .html, .htm, .xhtml

        Args:
            html_path: Path to the HTML file
            output_dir: Output directory path
            lang: Document language for optimization
            **kwargs: Additional parameters for docling command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        try:
            # Convert to Path object
            html_path = Path(html_path)
            if not html_path.exists():
                raise FileNotFoundError(f"HTML file does not exist: {html_path}")

            if html_path.suffix.lower() not in self.HTML_FORMATS:
                raise ValueError(f"Unsupported HTML format: {html_path.suffix}")

            name_without_suff = html_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = html_path.parent / "docling_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Run docling command
            self._run_docling_command(
                input_path=html_path,
                output_dir=base_output_dir,
                file_stem=name_without_suff,
                **kwargs,
            )

            # Read the generated output files
            content_list, _ = self._read_output_files(
                base_output_dir, name_without_suff
            )
            return content_list

        except Exception as e:
            logging.error(f"Error in parse_html: {str(e)}")
            raise

    def check_installation(self) -> bool:
        """
        Check if Docling is properly installed

        Returns:
            bool: True if installation is valid, False otherwise
        """
        try:
            # Prepare subprocess parameters to hide console window on Windows
            import platform

            subprocess_kwargs = {
                "capture_output": True,
                "text": True,
                "check": True,
                "encoding": "utf-8",
                "errors": "ignore",
            }

            # Hide console window on Windows
            if platform.system() == "Windows":
                subprocess_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

            result = subprocess.run(["docling", "--version"], **subprocess_kwargs)
            logging.debug(f"Docling version: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.debug(
                "Docling is not properly installed. "
                "Please ensure it is installed correctly."
            )
            return False


