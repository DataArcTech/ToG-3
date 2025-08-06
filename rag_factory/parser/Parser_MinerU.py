
import copy
import json
import os
import logging
import threading
import sys
from pathlib import Path
from io import BytesIO
from typing import List, Optional, Union, Tuple, Any, Dict
from timeit import default_timer as timer

from loguru import logger

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
from equation_detect import better_equation_parse

LOCK_KEY_MINERU = "global_shared_lock_mineru"
if LOCK_KEY_MINERU not in sys.modules:
    sys.modules[LOCK_KEY_MINERU] = threading.Lock()


class MinerUPdfParser:
    """
    Features:
    - Thread-safe processing
    - Modular design with clear separation of concerns
    - Comprehensive error handling
    - Performance monitoring
    - Flexible configuration options
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MinerU PDF Parser.
        
        Args:
            **kwargs: Configuration parameters including:
                - backend: Processing backend ('pipeline' or 'vlm-*')
                - parse_method: Parsing method ('auto', 'txt', 'ocr')
                - formula_enable: Enable formula parsing
                - table_enable: Enable table parsing
                - server_url: Server URL for vlm-sglang-client backend
                - output_options: Dictionary of output format options
        """
        self.backend = kwargs.get('backend', 'pipeline')
        self.parse_method = kwargs.get('parse_method', 'auto')
        self.formula_enable = kwargs.get('formula_enable', True)
        self.table_enable = kwargs.get('table_enable', True)
        self.server_url = kwargs.get('server_url', None)
        
        # Output configuration
        self.output_options = kwargs.get('output_options', {
            'f_draw_layout_bbox': True,
            'f_draw_span_bbox': True,
            'f_dump_md': True,
            'f_dump_middle_json': True,
            'f_dump_model_output': True,
            'f_dump_orig_pdf': True,
            'f_dump_content_list': True,
            'f_make_md_mode': MakeMode.MM_MD，
            'better_equation': True
        })
        
        # Initialize processing state
        self.total_pages = 0
        self.processed_pages = 0
        self.processing_stats = {}
        
        logger.info(f"MinerU PDF Parser initialized with backend: {self.backend}")

    def _validate_inputs(self, pdf_files: List[Union[str, Path, bytes]], 
                        languages: Optional[List[str]] = None) -> Tuple[List[str], List[bytes], List[str]]:
        """
        Validate and normalize input parameters.
        
        Args:
            pdf_files: List of PDF file paths or bytes
            languages: List of languages for each PDF
            
        Returns:
            Tuple of (file_names, pdf_bytes_list, language_list)
        """
        file_names = []
        pdf_bytes_list = []
        language_list = []
        
        if not pdf_files:
            raise ValueError("No PDF files provided")
            
        for i, pdf_file in enumerate(pdf_files):
            if isinstance(pdf_file, (str, Path)):
                # File path
                path = Path(pdf_file)
                if not path.exists():
                    raise FileNotFoundError(f"PDF file not found: {path}")
                file_names.append(path.stem)
                pdf_bytes_list.append(read_fn(path))
            elif isinstance(pdf_file, bytes):
                # Raw bytes
                file_names.append(f"document_{i}")
                pdf_bytes_list.append(pdf_file)
            else:
                raise TypeError(f"Invalid PDF file type: {type(pdf_file)}")
                
            # Set language
            if languages and i < len(languages):
                language_list.append(languages[i])
            else:
                language_list.append('ch')  # Default language
                
        return file_names, pdf_bytes_list, language_list

    def _process_pipeline_backend(self, pdf_bytes_list: List[bytes], 
                                language_list: List[str],
                                start_page: int = 0, 
                                end_page: Optional[int] = None) -> Tuple[List[Any], List[Any], List[Any], List[str], List[bool]]:
        """
        Process PDFs using pipeline backend.
        
        Args:
            pdf_bytes_list: List of PDF bytes
            language_list: List of languages
            start_page: Start page ID
            end_page: End page ID
            
        Returns:
            Tuple of processing results
        """
        start_time = timer()
        
        # Convert PDF pages if needed
        processed_bytes = []
        for pdf_bytes in pdf_bytes_list:
            with sys.modules[LOCK_KEY_MINERU]:  # Thread safety
                new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, start_page, end_page
                )
                processed_bytes.append(new_pdf_bytes)
        
        # Analyze documents
        results = pipeline_doc_analyze(
            processed_bytes, 
            language_list, 
            parse_method=self.parse_method,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable
        )
        
        processing_time = timer() - start_time
        self.processing_stats['pipeline_processing_time'] = processing_time
        logger.info(f"Pipeline processing completed in {processing_time:.2f}s")
        
        return results

    def _process_vlm_backend(self, pdf_bytes: bytes, 
                           image_writer: FileBasedDataWriter) -> Tuple[Dict[str, Any], List[str]]:
        """
        Process PDF using VLM backend.
        
        Args:
            pdf_bytes: PDF bytes
            image_writer: Image writer instance
            
        Returns:
            Tuple of (middle_json, infer_result)
        """
        start_time = timer()
        
        backend = self.backend[4:] if self.backend.startswith("vlm-") else self.backend
        
        with sys.modules[LOCK_KEY_MINERU]:  # Thread safety
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes, 
                image_writer=image_writer, 
                backend=backend, 
                server_url=self.server_url
            )
        
        processing_time = timer() - start_time
        self.processing_stats['vlm_processing_time'] = processing_time
        logger.info(f"VLM processing completed in {processing_time:.2f}s")
        
        return middle_json, infer_result

    def _generate_outputs(self, pdf_info: Dict[str, Any], 
                         pdf_bytes: bytes,
                         pdf_file_name: str,
                         local_image_dir: str,
                         local_md_dir: str,
                         md_writer: FileBasedDataWriter,
                         additional_data: Optional[Dict[str, Any]] = None):
        """
        Generate various output formats based on configuration.
        
        Args:
            pdf_info: PDF information dictionary
            pdf_bytes: Original PDF bytes
            pdf_file_name: PDF file name
            local_image_dir: Local image directory
            local_md_dir: Local markdown directory
            md_writer: Markdown writer instance
            additional_data: Additional data for output generation
        """
        opts = self.output_options
        image_dir = str(os.path.basename(local_image_dir))
        
        try:
            # Draw bounding boxes
            if opts.get('f_draw_layout_bbox', False):
                draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")
                
            if opts.get('f_draw_span_bbox', False) and self.backend == "pipeline":
                draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

            # Dump original PDF
            if opts.get('f_dump_orig_pdf', False):
                md_writer.write(f"{pdf_file_name}_origin.pdf", pdf_bytes)

            # Generate markdown
            if opts.get('f_dump_md', False):
                if self.backend == "pipeline":
                    md_content_str = pipeline_union_make(pdf_info, opts.get('f_make_md_mode', MakeMode.MM_MD), image_dir)
                else:
                    md_content_str = vlm_union_make(pdf_info, opts.get('f_make_md_mode', MakeMode.MM_MD), image_dir)
                md_writer.write_string(f"{pdf_file_name}.md", md_content_str)

            # Generate content list
            if opts.get('f_dump_content_list', False):
                if self.backend == "pipeline":
                    content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                else:
                    content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4)
                )
                if opts.get('better_equation', False):
                    logger.info(f"use dolphin to parse equation")
                    better_content_list = better_equation_parse(pdf_bytes, content_list, pdf_info)
                    md_writer.write_string(
                        f"{pdf_file_name}_better_content_list.json",
                        json.dumps(better_content_list, ensure_ascii=False, indent=4)
                    )

            # Dump middle JSON
            if opts.get('f_dump_middle_json', False) and additional_data and 'middle_json' in additional_data:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(additional_data['middle_json'], ensure_ascii=False, indent=4)
                )

            # Dump model output
            if opts.get('f_dump_model_output', False) and additional_data:
                if 'model_json' in additional_data:
                    md_writer.write_string(
                        f"{pdf_file_name}_model.json",
                        json.dumps(additional_data['model_json'], ensure_ascii=False, indent=4)
                    )
                elif 'infer_result' in additional_data:
                    model_output = ("\n" + "-" * 50 + "\n").join(additional_data['infer_result'])
                    md_writer.write_string(f"{pdf_file_name}_model_output.txt", model_output)

        except Exception as e:
            logger.error(f"Error generating outputs for {pdf_file_name}: {str(e)}")
            raise

    def parse_documents(self, pdf_files: List[Union[str, Path, bytes]],
                       output_dir: Union[str, Path],
                       languages: Optional[List[str]] = None,
                       start_page: int = 0,
                       end_page: Optional[int] = None,
                       callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Main method to parse PDF documents.
        
        Args:
            pdf_files: List of PDF file paths or bytes
            output_dir: Output directory for results
            languages: List of languages for each PDF
            start_page: Start page ID for parsing (0-based)
            end_page: End page ID for parsing (None for all pages)
            callback: Optional callback function for progress updates
            
        Returns:
            Dictionary containing parsing results and statistics
        """
        start_time = timer()
        
        try:
            # Validate inputs
            file_names, pdf_bytes_list, language_list = self._validate_inputs(pdf_files, languages)
            
            results = {
                'processed_files': [],
                'failed_files': [],
                'total_processing_time': 0,
                'statistics': self.processing_stats
            }
            
            if self.backend == "pipeline":
                # Pipeline backend processing
                infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = self._process_pipeline_backend(
                    pdf_bytes_list, language_list, start_page, end_page
                )
                
                for idx, model_list in enumerate(infer_results):
                    try:
                        file_name = file_names[idx]
                        model_json = copy.deepcopy(model_list)
                        
                        # Prepare environment
                        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, self.parse_method)
                        image_writer = FileBasedDataWriter(local_image_dir)
                        md_writer = FileBasedDataWriter(local_md_dir)
                        
                        # Process to middle JSON
                        images_list = all_image_lists[idx]
                        pdf_doc = all_pdf_docs[idx]
                        _lang = lang_list[idx]
                        _ocr_enable = ocr_enabled_list[idx]
                        
                        middle_json = pipeline_result_to_middle_json(
                            model_list, images_list, pdf_doc, image_writer, 
                            _lang, _ocr_enable, self.formula_enable
                        )
                        
                        pdf_info = middle_json["pdf_info"]
                        pdf_bytes = pdf_bytes_list[idx]
                        
                        # Generate outputs
                        self._generate_outputs(
                            pdf_info, pdf_bytes, file_name, local_image_dir, local_md_dir, md_writer,
                            {'middle_json': middle_json, 'model_json': model_json}
                        )
                        
                        results['processed_files'].append({
                            'file_name': file_name,
                            'output_dir': local_md_dir,
                            'pages_processed': len(images_list)
                        })
                        
                        if callback:
                            callback(progress=(idx + 1) / len(pdf_bytes_list), 
                                   message=f"Processed {file_name}")
                            
                        logger.info(f"Successfully processed {file_name}, output saved to {local_md_dir}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_names[idx]}: {str(e)}")
                        results['failed_files'].append({
                            'file_name': file_names[idx],
                            'error': str(e)
                        })
            
            else:
                # VLM backend processing
                for idx, pdf_bytes in enumerate(pdf_bytes_list):
                    try:
                        file_name = file_names[idx]
                        
                        # Convert PDF bytes
                        pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page, end_page)
                        
                        # Prepare environment
                        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, "vlm")
                        image_writer = FileBasedDataWriter(local_image_dir)
                        md_writer = FileBasedDataWriter(local_md_dir)
                        
                        # Process with VLM
                        middle_json, infer_result = self._process_vlm_backend(pdf_bytes, image_writer)
                        pdf_info = middle_json["pdf_info"]
                        
                        # Generate outputs
                        self._generate_outputs(
                            pdf_info, pdf_bytes, file_name, local_image_dir, local_md_dir, md_writer,
                            {'middle_json': middle_json, 'infer_result': infer_result}
                        )
                        
                        results['processed_files'].append({
                            'file_name': file_name,
                            'output_dir': local_md_dir,
                            'backend': self.backend
                        })
                        
                        if callback:
                            callback(progress=(idx + 1) / len(pdf_bytes_list), 
                                   message=f"Processed {file_name}")
                            
                        logger.info(f"Successfully processed {file_name}, output saved to {local_md_dir}")
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_names[idx]}: {str(e)}")
                        results['failed_files'].append({
                            'file_name': file_names[idx],
                            'error': str(e)
                        })
            
            total_time = timer() - start_time
            results['total_processing_time'] = total_time
            self.processing_stats['total_time'] = total_time
            
            logger.info(f"Document parsing completed in {total_time:.2f}s. "
                      f"Processed: {len(results['processed_files'])}, "
                      f"Failed: {len(results['failed_files'])}")
            
            return results
            
        except Exception as e:
            logger.error(f"Document parsing failed: {str(e)}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.processing_stats.copy()

    @staticmethod
    def get_supported_backends() -> List[str]:
        """Get list of supported backends."""
        return ["pipeline", "vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client"]

    @staticmethod
    def get_supported_languages() -> List[str]:
        """Get list of supported languages."""
        return ['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']





# Example usage with  MinerU parser
if __name__ == '__main__':
    
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "pdfs")
    output_dir = os.path.join(__dir__, "output")
    
    # Find PDF and image files
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]
    
    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob('*'):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)
    
    if doc_path_list:
        parser_config = {
            'backend': 'pipeline',
            'parse_method': 'auto',
            'formula_enable': True,
            'table_enable': True,
            'output_options': {
                'f_draw_layout_bbox': False,
                'f_draw_span_bbox': False,
                'f_dump_md': True,
                'f_dump_middle_json': False,
                'f_dump_model_output': False,
                'f_dump_orig_pdf': False,
                'f_dump_content_list': False,
                'f_make_md_mode': MakeMode.MM_MD,
                'better_equation': True
            }
        }
        
        parser = MinerUPdfParser(**parser_config)
        
        def progress_callback(progress, message):
            print(f"Progress: {progress*100:.1f}% - {message}")
        
        results = parser.parse_documents(
            pdf_files=doc_path_list,
            output_dir=output_dir,
            callback=progress_callback
        )
        
    else:
        print(f"No PDF files found in {pdf_files_dir}")
