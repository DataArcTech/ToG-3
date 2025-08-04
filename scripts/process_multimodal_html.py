import os
from bs4 import BeautifulSoup, NavigableString
import tiktoken
import seaborn as sns
import requests
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from llama_index.core import Document
from llama_index.core.schema import ImageDocument

from tqdm import tqdm

def process_html_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the required section
    content_section = soup.find("section", {"data-field": "body", "class": "e-content"})

    if not content_section:
        return "Section not found."

    sections = []
    current_section = {"header": "", "content": "", "source": file_path.split("/")[-1]}
    images = []
    images_metadata = []
    header_found = False

    for element in content_section.find_all(recursive=True):
        if element.name in ["h1", "h2", "h3", "h4"]:
            if header_found and (current_section["content"].strip()):
                sections.append(current_section)
            current_section = {
                "header": element.get_text(),
                "content": "",
                "source": file_path.split("/")[-1],
            }
            header_found = True
        elif header_found:
            if element.name == "pre":
                current_section["content"] += f"```{element.get_text().strip()}```\n"
            elif element.name == "img":
                img_src = element.get("src")
                img_caption = element.find_next("figcaption")
                # img_caption_2是图片url的split[-1]
                img_caption_2 = img_src.split("/")[-1].split(".")[0].replace("_", " ")
                if img_caption:
                    caption_text = img_caption.get_text().strip() or img_caption_2
                else:
                    caption_text = img_caption_2

                # Download the image
                try:
                    response = requests.get(img_src)
                    img = Image.open(BytesIO(response.content))
                    img.save(f"./data/multimodal_test_samples/images/{img_caption_2}.png")
                except Exception as e:
                    print(f"Error downloading image {img_src}: {e}")
                    continue

                images_metadata.append({
                    "image_url": img_src,
                    "caption": caption_text,
                    "file_name": img_caption_2,
                })
                

                images.append(ImageDocument(image_url=img_src))

            elif element.name in ["p", "span", "a"]:
                current_section["content"] += element.get_text().strip() + "\n"

    if current_section["content"].strip():
        sections.append(current_section)

    return images, sections, images_metadata

all_documents = []
all_images = []
all_images_metadata = []

# Directory to search in (current working directory)
# directory = os.getcwd()
directory = "./data/multimodal_test_samples/source_html_files"

# Walking through the directory
files = []
for root, dirs, file_list in os.walk(directory):
    for file in file_list:
        if file.endswith(".html"):
            files.append(file)

for file in tqdm(files):
    if file.endswith(".html"):
        # Update the file path to be relative to the current directory
        images, documents, images_metadata = process_html_file(os.path.join(root, file))
        all_documents.extend(documents)
        all_images.extend(images)
        all_images_metadata.extend(images_metadata)

text_docs = [Document(text=el.pop("content"), metadata=el) for el in all_documents]
print(f"Text document count: {len(text_docs)}")
print(f"Image document count: {len(all_images)}")

# save text documents
save_path = "./data/multimodal_test_samples"
if not os.path.exists(save_path):
    os.makedirs(save_path)
# save to json
import json
with open(os.path.join(save_path, "documents.json"), "w", encoding="utf-8") as f:
    json.dump([doc.to_dict() for doc in text_docs], f, indent=4)

# save images metadata
image_metadata_path = "./data/multimodal_test_samples"
if not os.path.exists(image_metadata_path):
    os.makedirs(image_metadata_path)
with open(os.path.join(image_metadata_path, "images_metadata.json"), "w", encoding="utf-8") as f:
    json.dump(all_images_metadata, f, indent=4)
