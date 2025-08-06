

# CHINESE_PROMPT = """
# 请对以下多个句子进行实体识别，并为每个句子返回实体标注结果。
# 每个句子的结果应独立，格式为JSON，包含句子编号、句子文本、和识别到的实体列表（包括实体文本、类型、和位置）。
# 返回格式为一个dictionary, key只有一个'gpt_labeled_data'，value为一个list，每个元素为一个dictionary，包含句子文本和实体列表，格式如下：

# {{
#     'gpt_labeled_data':             
#     [
#         {{"text":"句子1", "entity_list": [{{"entity_text": "", "entity_type": "", "entity_index": [[start, end]]}}]}}, 
#         {{"text":"句子2", "entity_list": [{{"entity_text": "", "entity_type": "", "entity_index": [[start, end]]}}]}},
#     ]
# }}
# 注意："text" 是输入的原始句子；
# """

# CHINESE_PROMPT = """
# 你是一个专注于马克思主义哲学领域的命名实体识别系统，请对以下句子进行实体识别任务。
# 每个句子的结果应彼此独立，返回格式为一个JSON字典，包含句子编号、原始句子文本、和识别到的实体列表（包括实体文本、类型、和字符级位置区间[start, end)）。

# 输出格式如下：
# {{
#     "gpt_labeled_data": [
#         {{
#             "text": "句子1",
#             "entity_list": [
#                 {{"entity_text": "实体内容", "entity_type": "实体类型", "entity_index": [[start, end]]}}
#             ]
#         }},
#         {{
#             "text": "句子2",
#             "entity_list": [...]
#         }}
#     ]
# }}

# 请严格按照以下七类实体类型进行标注，并参考下面提供的few-shot示例来理解每类实体的抽取标准：

# 实体类型包括：
# 1. 哲学概念（如：唯物主义、辩证法、意识形态、历史唯物主义）
# 2. 理论流派与思潮（如：辩证唯物主义、历史唯物主义、空想社会主义）
# 3. 核心人物（如：马克思、恩格斯、列宁、毛泽东）
# 4. 经典著作与文献（如：《资本论》、《共产党宣言》、《实践论》）
# 5. 历史事件与运动（如：巴黎公社、俄国十月革命、文化大革命）
# 6. 方法论术语（如：对立统一规律、质量互变规律、否定之否定规律）
# 7. 哲学特征/属性（实践性、革命性、科学性、阶级性）
# 8. 哲学原理/原理术语：抽象出的原则性表达，表征方法论或世界观层面的基本立场（如：对立统一规律、质量互变规律、否定之否定规律）

# 示例：

# 输入句子1：
# “《资本论》是马克思的代表作，集中体现了历史唯物主义的基本原理。”
# 标注结果：
# {{
#     "text": "《资本论》是马克思的代表作，集中体现了历史唯物主义的基本原理。",
#     "entity_list": [
#         {{"entity_text": "《资本论》", "entity_type": "经典著作与文献", "entity_index": [[0, 5]]}},
#         {{"entity_text": "马克思", "entity_type": "核心人物", "entity_index": [[6, 9]]}},
#         {{"entity_text": "历史唯物主义", "entity_type": "哲学概念", "entity_index": [[18, 25]]}}
#     ]
# }}

# 输入句子2：
# “辩证唯物主义强调物质世界的客观存在。”
# 标注结果：
# {{
#     "text": "辩证唯物主义强调物质世界的客观存在。",
#     "entity_list": [
#         {{"entity_text": "辩证唯物主义", "entity_type": "理论流派与思潮", "entity_index": [[0, 7]]}}
#     ]
# }}

# 输入句子3：
# “巴黎公社是无产阶级夺取政权的一次伟大尝试。”
# 标注结果：
# {{
#     "text": "巴黎公社是无产阶级夺取政权的一次伟大尝试。",
#     "entity_list": [
#         {{"entity_text": "巴黎公社", "entity_type": "历史事件与运动", "entity_index": [[0, 4]]}}
#     ]
# }}

# 请根据以上示例对下面的句子进行实体识别任务。若句子中不存在实体，也请返回空的 entity_list。
# """


# CHINESE_PROMPT = """
# 你是一个专注于马克思主义哲学领域的命名实体识别系统，请对以下句子进行实体识别任务。
# 每个句子的结果应彼此独立，返回格式为一个JSON字典，包含句子编号、原始句子文本、和识别到的实体列表（包括实体文本、类型、和简要的实体描述信息entity_description）。

# 输出格式如下：
# {
#     "gpt_labeled_data": [
#         {
#             "text": "句子1",
#             "entity_list": [
#                 {"entity_text": "实体内容", "entity_type": "实体类型", "entity_description": "简要描述该实体的含义或背景"}
#             ]
#         },
#         {
#             "text": "句子2",
#             "entity_list": [...]
#         }
#     ]
# }

# 请严格按照以下八类实体类型进行标注，并参考下面提供的few-shot示例来理解每类实体的抽取标准：

# 实体类型包括：
# 1. 哲学概念（如：唯物主义、辩证法、意识形态、历史唯物主义）
# 2. 理论流派与思潮（如：辩证唯物主义、历史唯物主义、空想社会主义）
# 3. 核心人物（如：马克思、恩格斯、列宁、毛泽东）
# 4. 经典著作与文献（如：《资本论》、《共产党宣言》、《实践论》）
# 5. 历史事件与运动（如：巴黎公社、俄国十月革命、文化大革命）
# 6. 方法论术语（如：对立统一规律、质量互变规律、否定之否定规律）
# 7. 哲学特征/属性（如：实践性、革命性、科学性、阶级性）
# 8. 哲学原理/原理术语：抽象出的原则性表达，表征方法论或世界观层面的基本立场（如：对立统一规律、质量互变规律、否定之否定规律）

# 示例：

# 输入句子1：
# “《资本论》是马克思的代表作，集中体现了历史唯物主义的基本原理。”
# 标注结果：
# {
#     "text": "《资本论》是马克思的代表作，集中体现了历史唯物主义的基本原理。",
#     "entity_list": [
#         {"entity_text": "《资本论》", "entity_type": "经典著作与文献", "entity_description": "马克思的代表作，系统阐述资本主义运行机制"},
#         {"entity_text": "马克思", "entity_type": "核心人物", "entity_description": "德国哲学家，马克思主义的创立者"},
#         {"entity_text": "历史唯物主义", "entity_type": "哲学概念", "entity_description": "马克思主义关于社会历史发展规律的基本观点"}
#     ]
# }

# 输入句子2：
# “辩证唯物主义强调物质世界的客观存在。”
# 标注结果：
# {
#     "text": "辩证唯物主义强调物质世界的客观存在。",
#     "entity_list": [
#         {"entity_text": "辩证唯物主义", "entity_type": "理论流派与思潮", "entity_description": "马克思主义哲学的基本理论之一，主张物质第一性和事物发展变化的辩证性"}
#     ]
# }

# 输入句子3：
# “巴黎公社是无产阶级夺取政权的一次伟大尝试。”
# 标注结果：
# {
#     "text": "巴黎公社是无产阶级夺取政权的一次伟大尝试。",
#     "entity_list": [
#         {"entity_text": "巴黎公社", "entity_type": "历史事件与运动", "entity_description": "1871年法国工人建立的政权，是世界上第一个无产阶级政权"}
#     ]
# }

# 请根据以上示例对下面的句子进行实体识别任务。若句子中不存在实体，也请返回空的 entity_list。
# """





# CHINESE_PROMPT = """
# 你是一个面向“行测（行政职业能力测验）”考试内容的命名实体识别系统，请对以下句子进行实体识别任务。

# 请识别以下四类实体：
# 1. 知识点：行测中考察的具体知识内容，如“语义理解”“奇数偶数性质”“主旨判断”等；
# 2. 题型/模块：考试中题目的分类或模块，如“数量关系”“资料分析”“判断推理”“类比推理”等；
# 3. 例题：引用的示例题目或部分题干、选项内容等具有题目特征的文本片段；
# 4. 考频：与出题频率相关的信息，如“常考”“高频”“近五年考过3次”“几乎年年考”等。

# 输出格式为一个JSON字典，包含句子编号、原始句子文本、和识别到的实体列表（包括实体文本、类型、和简要的实体描述信息 entity_description）。  
# 即使句中无实体，也请返回空的 entity_list。

# 输出格式如下：
# {
#     "gpt_labeled_data": [
#         {
#             "text": "句子1",
#             "entity_list": [
#                 {
#                     "entity_text": "实体内容",
#                     "entity_type": "实体类型（知识点/题型或模块/例题/考频）",
#                     "entity_description": "简要描述该实体在行测考试中的作用或含义"
#                 }
#             ]
#         },
#         ...
#     ]
# }

# 示例：

# 输入句子1：
# “数量关系模块中，奇数偶数性质是高频考点，近五年几乎年年出现。”
# 标注结果：
# {
#     "text": "数量关系模块中，奇数偶数性质是高频考点，近五年几乎年年出现。",
#     "entity_list": [
#         {"entity_text": "数量关系", "entity_type": "题型/模块", "entity_description": "行测考试中主要考察数学计算与数理逻辑能力的模块"},
#         {"entity_text": "奇数偶数性质", "entity_type": "知识点", "entity_description": "数量关系中涉及奇偶数运算和性质判断的基本内容"},
#         {"entity_text": "高频考点", "entity_type": "考频", "entity_description": "表示该知识点在考试中多次出现，出题频率较高"},
#         {"entity_text": "近五年几乎年年出现", "entity_type": "考频", "entity_description": "说明该知识点在近五年中频繁出现在考试中"}
#     ]
# }

# 输入句子2：
# “以下是类比推理的一道例题：‘刀：削——笔：写’。”
# 标注结果：
# {
#     "text": "以下是类比推理的一道例题：‘刀：削——笔：写’。",
#     "entity_list": [
#         {"entity_text": "类比推理", "entity_type": "题型/模块", "entity_description": "判断推理模块中的子题型，通过词语之间的逻辑关系选出最匹配选项"},
#         {"entity_text": "‘刀：削——笔：写’", "entity_type": "例题", "entity_description": "类比推理中的典型题干形式，考查词语间的功能关系"}
#     ]
# }

# 输入句子3：
# “行测资料分析部分经常考查增长率、比重等指标的计算。”
# 标注结果：
# {
#     "text": "行测资料分析部分经常考查增长率、比重等指标的计算。",
#     "entity_list": [
#         {"entity_text": "资料分析", "entity_type": "题型/模块", "entity_description": "通过图表数据考查受试者的计算、比较和推理能力"},
#         {"entity_text": "增长率", "entity_type": "知识点", "entity_description": "资料分析中常见的数值指标，表示数据的变化幅度"},
#         {"entity_text": "比重", "entity_type": "知识点", "entity_description": "表示某部分在总体中所占的比例"},
#         {"entity_text": "经常考查", "entity_type": "考频", "entity_description": "指某一知识点在考试中出现频率较高"}
#     ]
# }

# 请根据以上规范对下列句子进行实体识别任务。
# """


CHINESE_PROMPT = """
你是一个面向"行测（行政职业能力测验）"考试内容的命名实体识别系统，请对以下句子进行实体识别任务。

## 核心实体类型定义

请识别以下七大核心实体类型：

### 1. 题型体系
**定义**：题目所属的类型名称或可识别的形式特征，反映考查方式。
**典型样例**："主旨概括题"、"细节判断题"、"增长率变化题"、"削弱型推理题"、"语句排序题"、"比重计算题"、"图形位置类题"

### 2. 考点体系  
**定义**：题目考查的具体知识点、能力或内容范畴。
**典型样例**："增长率计算"、"基期量与现期量比较"、"加强论证"、"词语搭配辨析"、"宪法中的公民权利"、"平均增长速度"、"逻辑矛盾关系识别"

### 3. 解题模型
**定义**：解题所依赖的方法、步骤、技巧或策略。
**典型样例**："代入排除法"、"截位直除法"、"先找论点，再找削弱项"、"关键词定位法"、"赋值法"、"尾数估算法"、"主体一致原则"、"跳读抓首尾句"

### 4. 题目结构
**定义**：题干、选项、解析等组成部分的组织形式与结构特征。
**典型样例**："题干包含一段文字和一个表格"、"选项为并列四选一结构"、"干扰项使用绝对化表述"、"解析分为三步：定位、比对、排除"

### 5. 逻辑与关系结构
**定义**：信息之间的语言逻辑或数量关系，包括推理链条与数学表达。
**典型样例**："因为A，所以B"、"虽然……但是……"、"同比增长8.7%"、"A是B的1.8倍"、"比重下降3.2个百分点"、"现期量 = 基期量 × (1 + r)"

### 6. 试卷与模块结构
**定义**：考试整体的组织形式、模块分布或试卷基本信息。
**典型样例**："国考行测（地市级）"、"2024年省考联考"、"资料分析模块共20题"、"全卷共135题，限时120分钟"、"判断推理部分包含图形推理、定义判断等四类题"

### 7. 背景与语境信息
**定义**：题目所依赖的时间、地域或政策背景信息。
**典型样例**："2023年上半年"、"'十四五'期间"、"长江流域"、"东部地区"、"省级行政区"、"乡村振兴战略"、"碳达峰碳中和政策"

## 输出格式要求

输出格式为一个JSON字典，包含句子编号、原始句子文本、和识别到的实体列表（包括实体文本、类型、和简要的实体描述信息 entity_description）。即使句中无实体，也请返回空的 entity_list。

输出格式如下：
{
    "gpt_labeled_data": [
        {
            "text": "句子1",
            "entity_list": [
                {
                    "entity_text": "实体内容",
                    "entity_type": "实体类型",
                    "entity_description": "简要描述该实体在行测考试中的作用或含义"
                }
            ]
        },
        ...
    ]
}

## 标注示例

### 示例1：
**输入句子：**
"数量关系模块中，奇数偶数性质是高频考点，近五年几乎年年出现。"

**标注结果：**
{
    "text": "数量关系模块中，奇数偶数性质是高频考点，近五年几乎年年出现。",
    "entity_list": [
        {"entity_text": "数量关系", "entity_type": "试卷与模块结构", "entity_description": "行测考试中主要考察数学计算与数理逻辑能力的模块"},
        {"entity_text": "奇数偶数性质", "entity_type": "考点体系", "entity_description": "数量关系中涉及奇偶数运算和性质判断的基本考点"},
        {"entity_text": "近五年几乎年年出现", "entity_type": "背景与语境信息", "entity_description": "说明该知识点在近五年时间范围内的出现频率"}
    ]
}

### 示例2：
**输入句子：**
"以下是类比推理的一道例题：'刀：削——笔：写'，可用功能关系分析法解题。"

**标注结果：**
{
    "text": "以下是类比推理的一道例题：'刀：削——笔：写'，可用功能关系分析法解题。",
    "entity_list": [
        {"entity_text": "类比推理", "entity_type": "题型体系", "entity_description": "判断推理模块中的子题型，通过词语之间的逻辑关系选出最匹配选项"},
        {"entity_text": "'刀：削——笔：写'", "entity_type": "题目结构", "entity_description": "类比推理中的典型题干形式，展示词语间的功能关系"},
        {"entity_text": "功能关系分析法", "entity_type": "解题模型", "entity_description": "类比推理题中通过分析词语功能关系来找到正确答案的解题方法"}
    ]
}

### 示例3：
**输入句子：**
"资料分析部分经常考查增长率计算，公式为现期量 = 基期量 × (1 + r)。"

**标注结果：**
{
    "text": "资料分析部分经常考查增长率计算，公式为现期量 = 基期量 × (1 + r)。",
    "entity_list": [
        {"entity_text": "资料分析", "entity_type": "试卷与模块结构", "entity_description": "通过图表数据考查受试者的计算、比较和推理能力的模块"},
        {"entity_text": "增长率计算", "entity_type": "考点体系", "entity_description": "资料分析中常见的计算类考点，涉及数据变化幅度的计算"},
        {"entity_text": "现期量 = 基期量 × (1 + r)", "entity_type": "逻辑与关系结构", "entity_description": "增长率计算的基本数学公式表达"}
    ]
}

## 标注要求

- 仅提取原文中明确出现的内容，不推断、不扩展
- 每个实体保持原句片段形式，不改写  
- 若某类无内容，标注"无"
- 重点关注对考生、教师学习过程中有用的实体
- 实体描述要结合行测考试特点，说明该实体的实际作用或含义

请根据以上规范对下列句子进行实体识别任务。
"""


# CHINESE_SYSTEM_MESSAGE = """
# 你是一个高性能的命名实体识别（NER）系统，专注于马克思主义哲学和政治哲学领域。你的任务是从给定文本中，基于预定义的实体类型集合，准确识别并提取出相关实体，并根据输入文本的上下文，为每个实体提供一个简短描述。

# 请遵循以下要求完成任务：
# 1. 实体必须是文本中明确出现的内容，不进行引申推理；
# 2. 保持中文输出，尊重专业术语表达。
# 3. 实体描述应简洁明了，突出实体的核心特征或背景信息；
# """

CHINESE_SYSTEM_MESSAGE = """
你是一个高性能的命名实体识别（NER）系统。你的任务是从给定文本中，基于预定义的实体类型集合，准确识别并提取出相关实体，并根据输入文本的上下文，为每个实体提供一个简短描述。

请遵循以下要求完成任务：
1. 实体必须是文本中明确出现的内容，不进行引申推理；
2. 保持中文输出，尊重专业术语表达。
3. 实体描述应简洁明了，突出实体的核心特征或背景信息；
"""


# ENGLISH_SYSTEM_MESSAGE = """
# You are an intelligent Named Entity Recognition (NER) system specializing in the medical and pharmaceutical domain. Given a piece of text and a specific set of entity types, your task is to accurately identify and annotate the medical entities in the text that belong to these types.

# Entity types include: ["Disease and Pathological Condition", "Symptom and Sign", "Examination and Test Indicator", "Treatment Method", "Drug and Therapeutic Product", "Biochemical Substance", "Anatomical Site", "Microorganism and Pathogen", "Gene and Molecule", "Medical Device and Equipment", "Lifestyle and Behavioral Factor", "Time and Frequency"]
# """

def get_prompt_template(language: str) -> str:
    if language == "Chinese":
        return CHINESE_PROMPT
    
    elif language == "English":
        return CHINESE_PROMPT
    else:
        raise ValueError(f"Unsupported language: {language}")


def get_system_message(language: str) -> str:
    if language == "Chinese":
        base = CHINESE_SYSTEM_MESSAGE
    else:
        base = CHINESE_SYSTEM_MESSAGE

    return base
