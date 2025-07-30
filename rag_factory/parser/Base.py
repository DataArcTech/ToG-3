# type: ignore
"""
Generic Document Parser Utility

This module provides functionality for parsing PDF and image documents using MinerU 2.0 library,
and converts the parsing results into markdown and JSON formats

Note: MinerU 2.0 no longer includes LibreOffice document conversion module.
For Office documents (.doc, .docx, .ppt, .pptx), please convert them to PDF format first.
"""

from __future__ import annotations


import json
import argparse
import base64
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import (
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Any,
    TypeVar,
)

T = TypeVar("T")


class Parser:
    """
    Base class for document parsing utilities.

    Defines common functionality and constants for parsing different document types.
    """

    # Define common file formats
    OFFICE_FORMATS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
    IMAGE_FORMATS = {".png", ".jpeg", ".jpg", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    TEXT_FORMATS = {".txt", ".md"}

    # Class-level logger
    logger = logging.getLogger(__name__)

    def __init__(self) -> None:
        """Initialize the base parser."""
        pass

    @staticmethod
    def convert_office_to_pdf(
        doc_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        Convert Office document (.doc, .docx, .ppt, .pptx, .xls, .xlsx) to PDF.
        Requires LibreOffice to be installed.

        Args:
            doc_path: Path to the Office document file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            # Convert to Path object for easier handling
            doc_path = Path(doc_path)
            if not doc_path.exists():
                raise FileNotFoundError(f"Office document does not exist: {doc_path}")

            name_without_suff = doc_path.stem

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = doc_path.parent / "libreoffice_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)

            # Create temporary directory for PDF conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Convert to PDF using LibreOffice
                logging.info(f"Converting {doc_path.name} to PDF using LibreOffice...")

                # Prepare subprocess parameters to hide console window on Windows
                import platform

                # Try LibreOffice commands in order of preference
                commands_to_try = ["libreoffice", "soffice"]

                conversion_successful = False
                for cmd in commands_to_try:
                    try:
                        convert_cmd = [
                            cmd,
                            "--headless",
                            "--convert-to",
                            "pdf",
                            "--outdir",
                            str(temp_path),
                            str(doc_path),
                        ]

                        # Prepare conversion subprocess parameters
                        convert_subprocess_kwargs = {
                            "capture_output": True,
                            "text": True,
                            "timeout": 60,  # 60 second timeout
                            "encoding": "utf-8",
                            "errors": "ignore",
                        }

                        # Hide console window on Windows
                        if platform.system() == "Windows":
                            convert_subprocess_kwargs["creationflags"] = (
                                subprocess.CREATE_NO_WINDOW
                            )

                        result = subprocess.run(
                            convert_cmd, **convert_subprocess_kwargs
                        )

                        if result.returncode == 0:
                            conversion_successful = True
                            logging.info(
                                f"Successfully converted {doc_path.name} to PDF using {cmd}"
                            )
                            break
                        else:
                            logging.warning(
                                f"LibreOffice command '{cmd}' failed: {result.stderr}"
                            )
                    except FileNotFoundError:
                        logging.warning(f"LibreOffice command '{cmd}' not found")
                    except subprocess.TimeoutExpired:
                        logging.warning(f"LibreOffice command '{cmd}' timed out")
                    except Exception as e:
                        logging.error(
                            f"LibreOffice command '{cmd}' failed with exception: {e}"
                        )

                if not conversion_successful:
                    raise RuntimeError(
                        f"LibreOffice conversion failed for {doc_path.name}. "
                        f"Please ensure LibreOffice is installed:\n"
                        "- Windows: Download from https://www.libreoffice.org/download/download/\n"
                        "- macOS: brew install --cask libreoffice\n"
                        "- Ubuntu/Debian: sudo apt-get install libreoffice\n"
                        "- CentOS/RHEL: sudo yum install libreoffice\n"
                        "Alternatively, convert the document to PDF manually."
                    )

                # Find the generated PDF
                pdf_files = list(temp_path.glob("*.pdf"))
                if not pdf_files:
                    raise RuntimeError(
                        f"PDF conversion failed for {doc_path.name} - no PDF file generated. "
                        f"Please check LibreOffice installation or try manual conversion."
                    )

                pdf_path = pdf_files[0]
                logging.info(
                    f"Generated PDF: {pdf_path.name} ({pdf_path.stat().st_size} bytes)"
                )

                # Validate the generated PDF
                if pdf_path.stat().st_size < 100:  # Very small file, likely empty
                    raise RuntimeError(
                        "Generated PDF appears to be empty or corrupted. "
                        "Original file may have issues or LibreOffice conversion failed."
                    )

                # Copy PDF to final output directory
                final_pdf_path = base_output_dir / f"{name_without_suff}.pdf"
                import shutil

                shutil.copy2(pdf_path, final_pdf_path)

                return final_pdf_path

        except Exception as e:
            logging.error(f"Error in convert_office_to_pdf: {str(e)}")
            raise

    @staticmethod
    def convert_text_to_pdf(
        text_path: Union[str, Path], output_dir: Optional[str] = None
    ) -> Path:
        """
        Convert text file (.txt, .md) to PDF using ReportLab with full markdown support.

        Args:
            text_path: Path to the text file
            output_dir: Output directory for the PDF file

        Returns:
            Path to the generated PDF file
        """
        try:
            text_path = Path(text_path)
            if not text_path.exists():
                raise FileNotFoundError(f"Text file does not exist: {text_path}")

            # Supported text formats
            supported_text_formats = {".txt", ".md"}
            if text_path.suffix.lower() not in supported_text_formats:
                raise ValueError(f"Unsupported text format: {text_path.suffix}")

            # Read the text content
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                for encoding in ["gbk", "latin-1", "cp1252"]:
                    try:
                        with open(text_path, "r", encoding=encoding) as f:
                            text_content = f.read()
                        logging.info(f"Successfully read file with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise RuntimeError(
                        f"Could not decode text file {text_path.name} with any supported encoding"
                    )

            # Prepare output directory
            if output_dir:
                base_output_dir = Path(output_dir)
            else:
                base_output_dir = text_path.parent / "reportlab_output"

            base_output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = base_output_dir / f"{text_path.stem}.pdf"

            # Convert text to PDF
            logging.info(f"Converting {text_path.name} to PDF...")

            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.pdfbase import pdfmetrics

                # Create PDF document
                doc = SimpleDocTemplate(
                    str(pdf_path),
                    pagesize=A4,
                    leftMargin=inch,
                    rightMargin=inch,
                    topMargin=inch,
                    bottomMargin=inch,
                )

                # Get styles
                styles = getSampleStyleSheet()
                normal_style = styles["Normal"]
                heading_style = styles["Heading1"]

                # Try to register a font that supports Chinese characters
                try:
                    # Try to use system fonts that support Chinese
                    import platform

                    system = platform.system()
                    if system == "Windows":
                        # Try common Windows fonts
                        for font_name in ["SimSun", "SimHei", "Microsoft YaHei"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                    elif system == "Darwin":  # macOS
                        for font_name in ["STSong-Light", "STHeiti"]:
                            try:
                                from reportlab.pdfbase.cidfonts import (
                                    UnicodeCIDFont,
                                )

                                pdfmetrics.registerFont(UnicodeCIDFont(font_name))
                                normal_style.fontName = font_name
                                heading_style.fontName = font_name
                                break
                            except Exception:
                                continue
                except Exception:
                    pass  # Use default fonts if Chinese font setup fails

                # Build content
                story = []

                # Handle markdown or plain text
                if text_path.suffix.lower() == ".md":
                    # Handle markdown content - simplified implementation
                    lines = text_content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if not line:
                            story.append(Spacer(1, 12))
                            continue

                        # Headers
                        if line.startswith("#"):
                            level = len(line) - len(line.lstrip("#"))
                            header_text = line.lstrip("#").strip()
                            if header_text:
                                header_style = ParagraphStyle(
                                    name=f"Heading{level}",
                                    parent=heading_style,
                                    fontSize=max(16 - level, 10),
                                    spaceAfter=8,
                                    spaceBefore=16 if level <= 2 else 12,
                                )
                                story.append(Paragraph(header_text, header_style))
                        else:
                            # Regular text
                            story.append(Paragraph(line, normal_style))
                            story.append(Spacer(1, 6))
                else:
                    # Handle plain text files (.txt)
                    logging.info(
                        f"Processing plain text file with {len(text_content)} characters..."
                    )

                    # Split text into lines and process each line
                    lines = text_content.split("\n")
                    line_count = 0

                    for line in lines:
                        line = line.rstrip()
                        line_count += 1

                        # Empty lines
                        if not line.strip():
                            story.append(Spacer(1, 6))
                            continue

                        # Regular text lines
                        # Escape special characters for ReportLab
                        safe_line = (
                            line.replace("&", "&amp;")
                            .replace("<", "&lt;")
                            .replace(">", "&gt;")
                        )

                        # Create paragraph
                        story.append(Paragraph(safe_line, normal_style))
                        story.append(Spacer(1, 3))

                    logging.info(f"Added {line_count} lines to PDF")

                    # If no content was added, add a placeholder
                    if not story:
                        story.append(Paragraph("(Empty text file)", normal_style))

                # Build PDF
                doc.build(story)
                logging.info(
                    f"Successfully converted {text_path.name} to PDF ({pdf_path.stat().st_size / 1024:.1f} KB)"
                )

            except ImportError:
                raise RuntimeError(
                    "reportlab is required for text-to-PDF conversion. "
                    "Please install it using: pip install reportlab"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert text file {text_path.name} to PDF: {str(e)}"
                )

            # Validate the generated PDF
            if not pdf_path.exists() or pdf_path.stat().st_size < 100:
                raise RuntimeError(
                    f"PDF conversion failed for {text_path.name} - generated PDF is empty or corrupted."
                )

            return pdf_path

        except Exception as e:
            logging.error(f"Error in convert_text_to_pdf: {str(e)}")
            raise

    @staticmethod
    def _process_inline_markdown(text: str) -> str:
        """
        Process inline markdown formatting (bold, italic, code, links)

        Args:
            text: Raw text with markdown formatting

        Returns:
            Text with ReportLab markup
        """
        import re

        # Escape special characters for ReportLab
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Bold text: **text** or __text__
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)

        # Italic text: *text* or _text_ (but not in the middle of words)
        text = re.sub(r"(?<!\w)\*([^*\n]+?)\*(?!\w)", r"<i>\1</i>", text)
        text = re.sub(r"(?<!\w)_([^_\n]+?)_(?!\w)", r"<i>\1</i>", text)

        # Inline code: `code`
        text = re.sub(
            r"`([^`]+?)`",
            r'<font name="Courier" size="9" color="darkred">\1</font>',
            text,
        )

        # Links: [text](url) - convert to text with URL annotation
        def link_replacer(match):
            link_text = match.group(1)
            url = match.group(2)
            return f'<link href="{url}" color="blue"><u>{link_text}</u></link>'

        text = re.sub(r"\[([^\]]+?)\]\(([^)]+?)\)", link_replacer, text)

        # Strikethrough: ~~text~~
        text = re.sub(r"~~(.*?)~~", r"<strike>\1</strike>", text)

        return text

    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        output_dir: Optional[str] = None,
        method: str = "auto",
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse PDF document.
        Must be implemented by subclasses.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory path
            method: Parsing method (auto, txt, ocr)
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_pdf must be implemented by subclasses")

    def parse_image(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse image document.
        Must be implemented by subclasses.

        Note: Different parsers may support different image formats.
        Check the specific parser's documentation for supported formats.

        Args:
            image_path: Path to the image file
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_image must be implemented by subclasses")

    def parse_document(
        self,
        file_path: Union[str, Path],
        method: str = "auto",
        output_dir: Optional[str] = None,
        lang: Optional[str] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Abstract method to parse a document.
        Must be implemented by subclasses.

        Args:
            file_path: Path to the file to be parsed
            method: Parsing method (auto, txt, ocr)
            output_dir: Output directory path
            lang: Document language for OCR optimization
            **kwargs: Additional parameters for parser-specific command

        Returns:
            List[Dict[str, Any]]: List of content blocks
        """
        raise NotImplementedError("parse_document must be implemented by subclasses")

    def check_installation(self) -> bool:
        """
        Abstract method to check if the parser is properly installed.
        Must be implemented by subclasses.

        Returns:
            bool: True if installation is valid, False otherwise
        """
        raise NotImplementedError(
            "check_installation must be implemented by subclasses"
        )

