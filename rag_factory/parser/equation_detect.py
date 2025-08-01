import pypdfium2 as pdfium
import os
import json
from tqdm import tqdm 
from PIL import Image

import io
from typing import Dict, Any,Union

import torch
from transformers import AutoProcessor, VisionEncoderDecoderModel
from loguru import logger

class DOLPHIN:
    def __init__(self, model_id_or_path= 'ByteDance/Dophin'):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        # Load model from local path or Hugging Face hub
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()
        
        # Set device and precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model = self.model.half()  # Always use half precision by default
        
        # set tokenizer
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
        """Process an image with the given prompt
        
        Args:
            prompt: Text prompt to guide the model
            image: PIL Image to process
            
        Returns:
            Generated text from the model
        """
        # Prepare image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.half()
            
        # Prepare prompt
        prompt = f"<s>{prompt} <Answer/>"
        prompt_ids = self.tokenizer(
            prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids.to(self.device)
        
        decoder_attention_mask = torch.ones_like(prompt_ids)
        
        # Generate text
        outputs = self.model.generate(
            pixel_values=pixel_values.to(self.device),
            decoder_input_ids=prompt_ids,
            decoder_attention_mask=decoder_attention_mask,
            min_length=1,
            max_length=4096,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            do_sample=False,
            num_beams=1,
        )
        
        # Process the output
        sequence = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
        sequence = sequence.replace(prompt, "").replace("<pad>", "").replace("</s>", "").strip()
        
        return sequence

def equation_detect(lines: list) -> bool:
    for line in lines:
        for span in line['spans']:
            if "equation" in span['type']:
                return True


def add_index_by_page(content_json: list):
    """
    对content_json添加index标签
    """
    data = content_json.copy()
    page_index_counters = {}
    for item in data:
        page_idx = item["page_idx"]
        
        if page_idx not in page_index_counters:
            page_index_counters[page_idx] = 0
        else:
            page_index_counters[page_idx] += 1
        
        item["index"] = page_index_counters[page_idx]
    return data


def use_dophin_analysis_equation(pdf_data: Union[str, bytes], page_id: int, index: int, bbox: list, model:DOLPHIN) -> str:
    """   
    使用dolphin模型对公式区域进行更好的解析
    参数:
        pdf_data: PDF二进制数据
        page_id: 页面索引
        bbox: 区域坐标 [x0, y0, x1, y1]
    输出:
        解析结果
    """
    doc = pdfium.PdfDocument(io.BytesIO(pdf_data))
    page = doc.get_page(page_id)
    x0, y0_top, x1, y1_top = bbox

    # 获取页面高度并翻转 Y 轴
    page_height = page.get_size()[1]
    y0_pdf = page_height - y1_top
    y1_pdf = page_height - y0_top
    page.set_cropbox(*[x0, y0_pdf, x1, y1_pdf])

    bitmap = page.render(scale=2.0)
    pil_image = bitmap.to_pil()
    result = model.chat("Read text in the image.", pil_image).strip()
    return result

def better_equation_parse(pdf_data: Union[str, bytes], content_list: list, pdf_info: Dict[str, Any]):
    """
    参数:
        pdf_data: PDF二进制数据
        content_json: minerU 的content_list结果
        middle_json: minerU 的middle结果

    输出:
        在content_list 更新解析数据
    """
    # load dolphin model
    logger.info(f"Load dophin model")
    try:
        # model = DOLPHIN(model_id_or_path='/home/yangcehao/doc_analysis/model/ByteDance_Dophin')
        model = DOLPHIN()
    except Exception as e:
        logger.error(e)
    # doc = pdfium.PdfDocument(io.BytesIO(pdf_data))
    
    content_json = add_index_by_page(content_list)
    
    for page in tqdm(pdf_info, total=len(pdf_info), desc='Processing pages'):
        page_id = page['page_idx']
        para_blocks = page['para_blocks']
        for index, para_block in enumerate(para_blocks):
            try:
                bbox = para_block['bbox_fs']
            except:
                bbox = para_block['bbox']
            # block_type = para_block['type']
            # pare_block 有两种不同的结构
            try:
                blocks = para_block['blocks']
                for block in blocks:
                    lines = block['lines']
                    if equation_detect(lines):
                        result = use_dophin_analysis_equation(pdf_data=pdf_data, page_id=page_id,index=index, bbox=bbox, model=model)
                        continue
                    else:
                        result = None
            except:
                lines = para_block['lines']
                if equation_detect(lines):
                    result = use_dophin_analysis_equation(pdf_data=pdf_data, page_id=page_id,index=index, bbox=bbox, model=model)
                else:
                    result = None

            if result is not None:
                for row in content_json:
                    if page_id == row['page_idx'] and index == row['index'] and row['type'] =='text':
                        row['text'] = result

    return content_json

if __name__ == '__main__':

    pdf_file = '/home/yangcehao/edu_project/new_edu_vlm_result/2026国考公务员行测-资料部分/vlm/2026国考公务员行测-资料部分_origin.pdf'
    model = DOLPHIN()
