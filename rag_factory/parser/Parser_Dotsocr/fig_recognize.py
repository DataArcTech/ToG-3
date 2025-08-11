import os 
import glob
import json
import re
import fitz
from PIL import Image
from tqdm import tqdm
from dashscope import MultiModalConversation
import argparse
from pathlib import Path


def fig_understand(fig_path):
    # prompt = '请给出图像中具体内容信息，并用json格式输出,仅输出json格式数据，其中，图片类型请从["chart","knowladge_map","other"]中选择'
    prompt = '''
你是一个图像内容理解专家，任务是读取图像内容并生成结构化 JSON 数据。请遵循以下规则：

1. **仅输出 JSON 数据**，不要添加任何解释、前缀或后缀文字。
2. JSON 格式中必须包含两个字段：
   - "type": 图像类型，只能从 ["chart", "knowladge_map", "other"] 中选择。
   - "content": 图像的具体结构化内容描述。
3. 如果图像类型是：
   - "chart": 请提取图表的标题、坐标轴标签、图例、系列等结构信息。
   - "knowladge_map": 输出树状结构，所有节点使用 {"name": xxx, "children": [...]} 格式。
   - "other": 尽可能准确描述图像的主要元素。

以下是几个示例，请模仿格式输出。

---

### 示例1（chart）：
输入图像：柱状图，标题为“年度销售统计”，X轴为月份，Y轴为销售额，图例为“产品A”和“产品B”。

输出：
```json
{
  "type": "chart",
  "content": {
  
    "title": "年度销售统计",
    "x_axis": "月份",
    "y_axis": "销售额",
    "legend": ["产品A", "产品B"],
    "series": [
      {"name": "产品A", "data": [100, 120, 130]},
      {"name": "产品B", "data": [80, 90, 100]}
    ]
  }
}
示例2（knowladge_map）：
输入图像：知识图谱，核心为“机器学习”，子节点有“监督学习”和“无监督学习”，监督学习下有“回归”和“分类”。

输出：
{
  "type": "knowladge_map",
  "content": {
    "name": "机器学习",
    "children": [
      {
        "name": "监督学习",
        "children": [
          {"name": "回归"},
          {"name": "分类"}
        ]
      },
      {
        "name": "无监督学习"
      }
    ]
  }
}
示例3（other）：
输入图像：一张会议室内多个人开会的场景。

输出：
{
  "type": "other",
  "content": "一个会议室中有5个人正在围绕会议桌讨论，桌上有笔记本电脑和文件。"
}

请根据上面的示例输出格式，严格输出图像的内容识别结果，只返回符合格式的 JSON 数据。

'''
    messages=    [
    {
        "role": "user",
        "content": [
            {"image": f"file://{fig_path}"},
            {"text": prompt}
        ]
    }
]

    response = MultiModalConversation.call(
        api_key=os.environ.get('DASHSCOPE_API_KEY'),
        model="qwen-vl-plus",  
        messages=messages,
    )

    # print(response)
    return response["output"]["choices"][0]["message"].content[0]["text"].replace("```json",'').replace("```",'').strip()

def save_fig(file_path, page_no, index, bbox, scale):

    file_name, file_ext = os.path.splitext(os.path.basename(file_path))
    file_name = file_name.replace('_layout', '')
    base_dir = os.path.dirname(file_path)
    pdf_file = os.path.join(base_dir, f"{file_name}_original.pdf")
    doc =  fitz.open(pdf_file)
    page = doc.load_page(page_no)

    pdf_width = page.rect.width
    pdf_height = page.rect.height

    scale_x = scale[1] / pdf_width
    scale_y = scale[0] / pdf_height
    x1 = bbox[0] / scale_x
    y1 = bbox[1] / scale_y
    x2 = bbox[2] / scale_x
    y2 = bbox[3] / scale_y
    pdf_bbox = fitz.Rect(x1, y1, x2, y2) 
    zoom = 300 / 72  # 输出300 DPI
    matrix = fitz.Matrix(zoom, zoom)
    img = page.get_pixmap(matrix=matrix, clip=pdf_bbox, alpha=False)

    save_dir = os.path.join(base_dir,f"{file_name}/image")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    
    text = ''
    save_path = os.path.join(save_dir, f'page_{page_no}_{index}.png')
    if img is not None:
        img.save(save_path)  
        
        text = fig_understand(save_path)

    return save_path, text

def process_one_file(json_file):
    file_name = os.path.basename(json_file)
    base_dir = os.path.dirname(json_file)
    output_path = str(json_file).replace("layout", "img_content")
    data = []
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    print(f"Processing file: {file_name}")
    for row in tqdm(json_data):
        if row.get('category','') == 'Picture':
            bbox = row['bbox']
            page_no = row['page_no']
            if (bbox[2]-bbox[0])*(bbox[3]-bbox[1]) < 52000:
                row['text'] = ""
            else:
                fig_path, text = save_fig(json_file, page_no=page_no, index=row['index'], bbox=bbox, scale=row['scale'])
                # print(text)
                row['text'] = json.loads(text)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
    return json_data

def main():
    parser = argparse.ArgumentParser(description="Use vlm to get parsed figure content.")
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output parsed directory  (default: output)"
    )
    args = parser.parse_args()


    if os.path.isdir(args.output):
        for file in sorted(Path(args.output).glob('*_layout.json')):
            data = process_one_file(file)
    elif os.path.isfile(args.output):
        data = process_one_file(args.output)
    else:
        print(f"'{args.output}' no exist")

if __name__ == "__main__":
    os.environ["DASHSCOPE_API_KEY"] = "your api key"
    main()


