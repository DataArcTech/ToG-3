# dots.ocr parser

This parser is based on the [dots.ocr](https://github.com/rednote-hilab/dots.ocr) model. See [dots.ocr](https://github.com/rednote-hilab/dots.ocr) for details.

## 1. Installation

```
python>=3.10 
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# for GLIBC 2.31, please use flash-attn==2.7.4.post1 instead of flash-attn==2.8.0.post2

```

```
# model download
from huggingface_hub import snapshot_download
snapshot_download(repo_id="rednote-hilab/dots.ocr", local_dir="weights/DotsOCR", local_dir_use_symlinks=False, resume_download=True)

or

from modelscope import snapshot_download
snapshot_download(repo_id="rednote-hilab/dots.ocr", local_dir="weights/DotsOCR")
```

## 2. vLLM inference

Using vLLM for faster paser speed  ( based on vllm==0.9.1 )

```
python vllm_launch.py --model_path dots_model_path
```

## 3. Document Parse

```
python parser.py pdf_path.pdf 
```

Iif you want to parse document with transformers，add `--use_hf=True`
