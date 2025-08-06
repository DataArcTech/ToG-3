# import json
# import os 
# from openai import OpenAI
# import random
# import time

# def set_openai_key():
#     with open('/home/zhangliyu/tog3/graph_construct/openai.key', 'r') as f:
#         os.environ["OPENAI_API_KEY"] = f.read().strip()


# def gptqa(messages,
#           openai_model_name: str = "gpt-4o-mini",
#           json_format: bool = True,
#           temp: float = 1.0,
#           log_file: str = "gpt_token_stats.jsonl"):
#     start_time = time.time()
#     if openai_model_name and "gpt" in openai_model_name:
#         # set_openai_key()
#         os.environ["OPENAI_API_KEY"] = "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"

#         client = OpenAI(base_url="https://api.gptsapi.net/v1")
#         if json_format:

#             completion = client.chat.completions.create(
#                 model=openai_model_name,
#                 temperature=temp,
#                 response_format={"type": "json_object"},
#                 messages=messages
#             )
#         else:
#             completion = client.chat.completions.create(
#                 model=openai_model_name,
#                 temperature=temp,
#                 messages=messages
#             )
#     else:
#         with open('/home/zhangliyu/tog3/graph_construct/deepseek.key', 'r') as f:
#             deepseek_key = f.read().strip()  
#         client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com/v1")
#         if json_format:

#             completion = client.chat.completions.create(
#                 model="deepseek-chat",
#                 response_format={"type": "json_object"},
#                 messages=messages,
#                 stream=False
#             )
#         else:
#             completion = client.chat.completions.create(
#                 model="deepseek-chat",
#                 messages=messages,
#                 stream=False
#             )
#     print(f"一个api call所花时间: {time.time() - start_time}")
#     output_content = completion.choices[0].message.content
    
#     # 获取准确的 token 使用情况
#     input_tokens = completion.usage.prompt_tokens  # 通过属性访问
#     output_tokens = completion.usage.completion_tokens  # 通过属性访问

#     # 记录输入和输出 token 数量
#     if log_file:
#         with open(log_file, "a") as f:
#             log_data = {
#                 "input_tokens": input_tokens,
#                 "output_tokens": output_tokens,
#                 "model": openai_model_name        
#             }
#             f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
    
#     print(f"response: {output_content}\n")
#     return output_content

# count = 0
# from tqdm import tqdm

# def label_entity(
#                 text: list[str],
#                 choices = None,
#                 area: str = "None",
#                 language: str = "Chinese",
#                 openai_model_name: str = "deepseek",
#                 json_format: bool = True,
#                 temp: float = 1.0,
#                 log_file: str = "gpt_token_stats.jsonl",
#                 batch_size: int = 10
#             ) -> list[dict]:
#     """
#     针对一组文本，批量标注实体。
#     返回每个文本对应的实体识别结果。
#     """
#     try:
#         if language == "Chinese":
#             if choices is None:
#                 system_message = "你是一个智能的NER系统。给定特定的实体类型和一些文本，你需要识别并标注这些文本中的实体。"
#             else:
#                 system_message = (
#                     "你是一个智能的NER系统。给定特定的实体类型和一些文本，你需要识别并标注这些文本中的实体。"
#                     f"entity_type可选类型:{choices}"
#                 ) 
#             # 如果有领域，可以将其添加到 system_message
#             if area in one_shot_example:
#                 system_message = (system_message + f"\n领域:{area}" + f"\n一些例子:{one_shot_example[area]}") if area else system_message
#             else:
#                 system_message = (system_message + f"\n领域:{area}") if area else system_message
#             prompt_template = """        请为以下多个句子进行实体识别，并为每个句子返回实体标注结果。
#             每个句子的结果应独立，格式为JSON，包含句子编号、句子文本、和识别到的实体列表（包括实体文本、类型、和位置）。
#             返回格式为一个dictionary, key只有一个'gpt_labeled_data'，value为一个list，每个元素为一个dictionary，包含句子文本和实体列表。
#             {{
#                 'gpt_labeled_data':             
#                 [
#                     {{"text":"句子1", "entity_list": [{{"entity_text": "", "entity_type": "", "entity_index": [[start, end]]}}]}}, 
#                     {{"text":"句子2", "entity_list": [{{"entity_text": "", "entity_type": "", "entity_index": [[start, end]]}}]}}, 
#                     ...
#                 ]
#                 注意："text"应该只是句子，而不是整篇文章。
#             }}
#             """
#         elif language == "English":
#             if choices is None:
#                 system_message = (
#                     "You are an intelligent NER system. Please extract all entities that are important for solving the text."
#                     "Don't include the whole article as a sentence in the return json format."
#                 )

#             else:
#                 system_message = (
#                     "You are an intelligent NER system. Given specific entity types and some text, you need to identify and label the entities in the text."
#                     f"entity_type choices:{choices}"
#                     f"Don't include the whole article as a sentence in the return json format."
#                 )
#             # 如果有领域，可以将其添加到 system_message
#             if area in one_shot_example:
#                 system_message = (system_message + f"\nArea:{area}" + f"\nSome examples:{one_shot_example[area]}") if area else system_message
#             else:
#                 system_message = (system_message + f"\nArea:{area}") if area else system_message
#             prompt_template = """        
#                 Please identify and label the entities in the following multiple sentences, and return the entity labeling results for each sentence.
#                 The results for each sentence should be independent, in JSON format, containing the sentence number, sentence text, and the list of recognized entities (including entity text, type, and position).
#                 Return format is a dictionary, with only one key 'gpt_labeled_data', and the value is a list, each element is a dictionary containing the sentence text and the entity list.
#                 {{
#                     'gpt_labeled_data':             
#                     [
#                         {{"text":"Sentence 1", "entity_list": [{{"entity_text": "", "entity_type": "", "entity_index": [[start, end]]}}]}}, 
#                         {{"text":"Sentence 2", "entity_list": [{{"entity_text": "", "entity_type": "", "entity_index": [[start, end]]}}]}}, 
#                         ...
#                     ]
#                 }}
#                 Notice that "text" should be only the sentence, not the whole article.
#             """

#         else:
#             raise ValueError(f"language {language} not yet handled")

#         labeled_text_all = []
#         print(f"label_entity的len(text): {len(text)}, batch_size: {batch_size}")
#         # 逐个批次传递给 GPT 进行标注
#         for i in tqdm(range(0, len(text), batch_size), desc="标注实体"):
#             batch_text = text[i:i + batch_size]  # 获取当前批次的句子

#             # 构造 GPT 的 Prompt，要求返回每个句子的标注结果
#             batch_prompt = "\n\n".join([f"句子{i+1}: {txt}" for i, txt in enumerate(batch_text)])  # 将每个句子编号并拼接
#             if language == "Chinese":
#                 prompt = prompt_template+f"""
#                 entity_type可选类型:{choices}
#                 句子列表:
#                 {batch_prompt}
#                 """
#             elif language == "English":
#                 prompt = prompt_template+f"""
#                 entity_type choices:{choices}
#                 Sentence list:
#                 {batch_prompt}
#                 """
#             else:
#                 raise ValueError("language not yet handled")

#             messages = [
#                 {"role": "system", "content": system_message},
#                 {"role": "user", "content": prompt}
#             ]

#             # 调用 GPT 进行标注
#             labeled_text = gptqa(messages, openai_model_name, json_format, temp, log_file)
            
#             try:
#                 # 处理返回的 JSON 格式结果
#                 if 'gpt_labeled_data' in labeled_text:
#                     labeled_text = json.loads(labeled_text)['gpt_labeled_data']
#                 else:
#                     if not labeled_text.startswith("["):
#                         labeled_text = "[" + labeled_text
#                     if not labeled_text.endswith("]"):
#                         labeled_text = labeled_text + "]"
#                     labeled_text = json.loads(labeled_text)
#             except Exception as e:
#                 print(f"Exception during json.loads: {e}")
#                 labeled_text = []

#             # 拼接结果到总结果
#             labeled_text_all.extend(labeled_text)
#             print(f"labeled_text:{labeled_text}")
#             # 更新处理文本计数
#             global count
#             count += len(batch_text)
#             print(f"当前处理文本数: {count}")

#         return labeled_text_all
#     except Exception as e:
#         print(f"Exception during label_entity: {e}")
#         return []


# one_shot_example = {
#     "财务会计":         
#         (
#             """Positive example:                 
#                 {
#                     "text": "除另有规定外，纳税人提供应税劳务（如旅游服务）应当向其机构所在地或者居住地的主管税务机关申报纳税。",
#                     "entity_list": [
#                         {
#                             "entity_text": "纳税人",
#                             "entity_type": "专有名词",
#                             "entity_index": [[7, 9]]
#                         },
#                         {
#                             "entity_text": "应税劳务",
#                             "entity_type": "专有名词",
#                             "entity_index": [[19, 23]]
#                         },
#                         {
#                             "entity_text": "旅游服务",
#                             "entity_type": "专有名词",
#                             "entity_index": [[26, 30]]
#                         },
#                         {
#                             "entity_text": "机构所在地",
#                             "entity_type": "专有名词",
#                             "entity_index": [[36, 40]]
#                         },
#                         {
#                             "entity_text": "居住地",
#                             "entity_type": "专有名词",
#                             "entity_index": [[43, 45]]
#                         },
#                         {
#                             "entity_text": "主管税务机关",
#                             "entity_type": "机构",
#                             "entity_index": [[47, 52]]
#                         }
#                     ]
#                 }
#                 这个例子正确的识别了entity_text, entity_type
#             """
#             # """Positive example:
#             #     {
#             #         "text": "发展综合运输的关键（）。\nA、尽可能的均衡使用各种运输方式\nB、尽可能的提高各种运输方式的利用效率\nC、尽可能发挥各种运输方式的优势\nD、尽可能的限制成本高的运输方式发展",
#             #         "entity_list": [
#             #             {
#             #                 "entity_text": "综合运输",
#             #                 "entity_type": "专有名词",
#             #                 "entity_index": [[2, 5]]
#             #             },
#             #             {
#             #                 "entity_text": "运输方式",
#             #                 "entity_type": "专有名词",
#             #                 "entity_index": [[20, 23], [46, 49], [69, 72], [92, 95]]
#             #             }
#             #         ]
#             #     }
#             # """

#             # """Positive example:         
#             #     {
#             #     "text": "【知识点】汇率制度【考点】汇率制度的含义与划分【考察方向】概念释义【难易程度】中",
#             #     "entity_list": [
#             #         {
#             #             "entity_text": "汇率制度",
#             #             "entity_type": "专有名词",
#             #             "entity_index": [[5, 8]]
#             #         }
#             #     ]}
#             #     这个例子正确的识别了entity_text, entity_type
#             # """
#             # """Negative example:         
#             # {
#             # "text": "下列属于免征增值税吗？农业生产者销售的自产农产品",
#             # "entity_list": [{"entity_text": "农业生产者","entity_type": "专有名词","entity_index": [[11, 15]]}]
#             # }
#             #     这个例子错误的识别了entity_type, 应为"人物"。
#             # """
#             # """Negative example:         
#             # {
#             #     "text": "该企业直接进口供残疾人专用的物品",
#             #     "entity_list": [{"entity_text": "企业直接进口供残疾人专用的物品", "entity_type": "组织", "entity_index": [[1, 9]]}]
#             # }
#             #     这个例子错误的识别了entity_text, 不应该将整个句子作为一个实体。
#             # """
#         ),

        
#     "法律": (
#         """Positive example: 
#         {
#             "text": "2012 年 5 月，郑泽贤通过林东焕非法运输海鲜 141 箱入境海南，偷逃应缴税款人民币 84459 元。于 2012 年 9 月向侦查机关投案，处理以刑法，七年以上有期徒刑。",
#             "entity_list": [
#                 {"entity_text": "郑泽贤", "entity_type": "人物", "entity_index": []},
#                 {"entity_text": "林东焕", "entity_type": "人物", "entity_index": []},
#                 {"entity_text": "非法运输", "entity_type": "事件", "entity_index": []},
#                 {"entity_text": "海南", "entity_type": "地点", "entity_index": []},
#                 {"entity_text": "偷逃应缴税款", "entity_type": "事件", "entity_index": []},
#                 {"entity_text": "侦查机关", "entity_type": "组织", "entity_index": []},
#                 {"entity_text": "投案", "entity_type": "事件", "entity_index": []},
#                 {"entity_text": "刑法", "entity_type": "法规", "entity_index": []},
#                 {"entity_text": "七年以上有期徒刑", "entity_type": "专有名词", "entity_index": []}
#             ]
#         }
#     """
#     )

# }











# main.py

import json
import os
import time
from openai import OpenAI
from tqdm import tqdm
from prompt_temple import get_prompt_template, get_system_message

count = 0

def gptqa(messages, openai_model_name="gpt-4.1-mini", json_format=True, temp=1.0, log_file="gpt_token_stats.jsonl"):
    start_time = time.time()

    if "gpt" in openai_model_name:
        os.environ["OPENAI_API_KEY"] = "sk-2T06b7c7f9c3870049fbf8fada596b0f8ef908d1e233KLY2"
        client = OpenAI(base_url="https://api.gptsapi.net/v1")
    else:
        os.environ["OPENAI_API_KEY"] = "sk-e696b88412204f6f8b747afe92c6e45a"
        client = OpenAI(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    kwargs = {
        "model": openai_model_name,
        "temperature": temp,
        "messages": messages,
        "stream": False
    }
    if json_format:
        kwargs["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**kwargs)
    duration = time.time() - start_time
    print(f"API 调用时间: {duration:.2f}秒")

    result = completion.choices[0].message.content
    usage = completion.usage

    if log_file:
        with open(log_file, "a") as f:
            json.dump({
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "model": openai_model_name
            }, f, ensure_ascii=False)
            f.write("\n")

    print(f"response: {result}\n")
    return result


def label_entity(text: list[str], language="Chinese",
                 openai_model_name="deepseek", json_format=True, temp=1.0,
                 log_file="gpt_token_stats.jsonl", batch_size=10) -> list[dict]:
    
    try:
        prompt_template = get_prompt_template(language)
        system_message = get_system_message(language)

        all_results = []
        print(f"总文本数: {len(text)}, 每批: {batch_size}")

        for i in tqdm(range(0, len(text), batch_size), desc="标注实体"):
            batch = text[i:i + batch_size]
            # 定义语言相关前缀
            if language == "Chinese":
                sentence_prefix = "句子"
                sentence_list_intro = "句子列表"
            else:
                sentence_prefix = "Sentence"
                sentence_list_intro = "Sentence list"

            # 构建句子列表字符串
            formatted_sentences = [
                f"{sentence_prefix}{i+1}: {text}" for i, text in enumerate(batch)
            ]
            batch_prompt = "\n\n".join(formatted_sentences)

            # 构建最终用户提示
            user_prompt = (
                f"{prompt_template}\n\n"
                f"{sentence_list_intro}:\n{batch_prompt}"
            )

            # print(f"当前批次的用户提示: {user_prompt}")

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]


            response = gptqa(messages, openai_model_name, json_format, temp, log_file)

            try:
                data = json.loads(response)
                result = data.get("gpt_labeled_data", [])
            except Exception as e:
                print(f"解析出错: {e}")
                result = []

            all_results.extend(result)
            global count
            count += len(batch)
            print(f"已处理: {count}")

        return all_results

    except Exception as e:
        print(f"标注错误: {e}")
        return []
