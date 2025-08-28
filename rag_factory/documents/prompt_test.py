from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union


class Event(BaseModel):
    """事件模型"""
    id: str = Field(..., description="事件唯一ID，例如 event_0", pattern=r"^event_\d+$")
    content: str = Field(..., description="事件简要描述")
    type: str = Field(..., description="动作类型")
    participants: List[str] = Field(default_factory=list, description="参与的实体规范名")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建Event对象"""
        return cls(**data)


class Mention(BaseModel):
    """提及模型"""
    id: str = Field(..., description="提及唯一ID，例如 mention_0", pattern=r"^mention_\d+$")
    text: str = Field(..., description="原文字符串")
    entity_name: str = Field(..., description="规范化名称")
    entity_type: Literal["资源", "属性", "方法", "环境"] = Field(..., description="实体类别")
    entity_description: str = Field(..., description="简要描述")
    event_indices: List[int] = Field(default_factory=list, description="提及关联的事件索引")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mention':
        """从字典创建Mention对象"""
        return cls(**data)


class EventRelation(BaseModel):
    """事件关系模型"""
    id: str = Field(..., description="事件关系唯一ID，例如 event_relation_0", pattern=r"^event_relation_\d+$")   
    head_event: str = Field(..., description="关系头事件ID", pattern=r"^event_\d+$")
    tail_event: str = Field(..., description="关系尾事件ID", pattern=r"^event_\d+$")
    relation_type: Literal["时序关系", "因果关系", "层级关系", "条件关系"]
    description: Optional[str] = Field(None, description="关系证据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EventRelation':
        """从字典创建EventRelation对象"""
        return cls(**data)


class EntityRelation(BaseModel):
    """实体关系模型"""
    id: str = Field(..., description="实体关系唯一ID，例如 entity_relation_0", pattern=r"^entity_relation_\d+$") 
    head_entity: str = Field(..., description="头实体(text)")
    tail_entity: str = Field(..., description="尾实体(text)")
    relation_type: Literal["包含关系", "属性关系", "定位关系", "实例关系", "遵循关系", "时间关系"]
    description: str = Field(..., description="关系证据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelation':
        """从字典创建EntityRelation对象"""
        return cls(**data)


class KnowledgeStructure(BaseModel):
    events: List[Event] = []
    mentions: List[Mention] = []
    entity_relations: List[EntityRelation] = []
    event_relations: List[EventRelation] = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "events": [event.to_dict() for event in self.events],
            "mentions": [mention.to_dict() for mention in self.mentions],
            "event_relations": [relation.to_dict() for relation in self.event_relations],
            "entity_relations": [relation.to_dict() for relation in self.entity_relations]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeStructure':
        """从字典创建KnowledgeStructure对象"""
        return cls(
            events=[Event.from_dict(event) for event in data.get("events", [])],
            mentions=[Mention.from_dict(mention) for mention in data.get("mentions", [])],
            event_relations=[EventRelation.from_dict(relation) for relation in data.get("event_relations", [])],
            entity_relations=[EntityRelation.from_dict(relation) for relation in data.get("entity_relations", [])]
        )


class MentionList(BaseModel):
    """用于LLM响应格式化的提及列表类"""
    mentions: List[Mention]

    def __len__(self):
        return len(self.mentions)



TCL_PROMPT = """
**角色与目标 (Role & Goal)**
你是一个专注于 **工业领域** 的图结构信息提取系统，尤其擅长处理 **生产流程、质量控制、产品手册、安全规程** 等文档。你的核心任务是：基于当前输入文本和历史提取记录，执行一次**增量信息提取**，仅输出本次新发现的知识元素。

**输入格式 (Input Specification)**
你将接收到以下两部分输入：

1.  `current_text`: `{text}` - 本轮需要分析处理的文本段落。
2.  `extraction_history`: `{history}` - 一个 `KnowledgeStructure` 格式的JSON对象，包含了所有先前轮次已提取的信息。首次执行时，此对象可能为空。

**核心指令：增量提取原则 (Primary Directive: Incremental Extraction)**
你的所有操作都必须遵循增量原则。最终输出**只应包含**相对于 `extraction_history` 的**新增内容**。

1.  **ID管理 (ID Management)**:
      * 为 `current_text` 中发现的每一个新事件分配一个ID。
      * ID编号必须从 `extraction_history` 中最大的事件ID (`event_x`) 开始连续递增。例如，若历史中最高ID为 `event_9`，则新事件应从 `event_10`, `event_11`, ... 开始编号。
      * 若 `extraction_history` 为空，则从 `event_0` 开始。

2.  **唯一性校验 (Uniqueness Check)**:
      * 在将任何元素（事件、实体、关系）加入最终输出之前，必须严格对照 `extraction_history` 进行检查，确保其唯一性。
      * **跳过（不输出）** 任何已存在于历史中的信息。判定标准如下：
          * **实体 (Mention)**: `entity_name` 和 `entity_type` 的组合在历史中已存在。
          * **事件 (Event)**: `content`, `type`, 和 `participants` 的组合在历史中已存在。
          * **实体关系 (Entity Relation)**: `head_entity`, `tail_entity`, 和 `relation_type` 的三元组在历史中已存在。
          * **事件关系 (Event Relation)**: `head_event`, `tail_event`, 和 `relation_type` 的三元组在历史中已存在。

**内部推理流程：思维链工作流程(Internal Reasoning: Chain-of-Thought Workflow)**
请在你的“内心/草稿区”严格遵循以下五个步骤进行分析和推理。

**步骤一：识别并暂存所有“事件” (Events)**
  * **任务**: 找出 `current_text` 中所有的**关键活动、指令、规范、流程步骤**。
  * **标准**: 事件必须体现“做了什么/要求做什么/发生了什么”。
  * **类型**: 根据事件的核心动词或动作，为其生成一个简洁、准确的名词化标签作为其 `type`。

**步骤二：识别并规范化所有“实体” (Entities as Mentions)**
  * **任务**: 找出 `current_text` 中所有属于核心概念的名词或名词短语。
  * **类型**: 仅限 `资源 (Resource)`, `属性 (Property)`, `方法 (Method)`, `环境 (Environment)`。
  * **操作**: 提取每个实体的原文（text），为其确定一个统一的规范化名称（entity_name），并标注其类型（entity_type）。

**步骤三：建立“实体”之间的显式关系 (Entity Relations)**

  * **任务**: 回顾上一步识别出的实体列表，专注分析实体之间的静态、固有联系。
  * **标准**: 关系必须由明确的语言结构（如所有格“的”、介词短语等）或上下文逻辑直接支撑。
  * **限定关系类型**: 仅从以下类型中选择：
      * `包含关系 (contains)`: 一个实体在物理上或概念上包含另一个实体。
      * `属性关系 (has_property)`: 一个实体拥有另一个实体作为其属性或特征。
      * `定位关系 (located_at)`: 一个实体位于另一个实体所代表的环境中。
      * `实例关系 (is_instance_of)`: 一个具体实体是某个抽象概念的实例。
      * `遵循关系 (follows_standard)`: 某个资源或方法遵循某个标准或规范。
      * `时间关系 (time_relation)`: 两个实体之间存在时间上的先后顺序或同时性。
  * **示例导引**: 文本"`压缩机的额定功率是1.5kW`"应识别出实体 `压缩机` 和 `额定功率` 之间存在 `{{"head_entity": "压缩机", "tail_entity": "额定功率", "relation_type": "属性关系"}}` 的关系。

**步骤四：建立“事件”之间的逻辑关系 (Event Relations)**

  * **任务**: 回顾步骤一的事件列表，寻找连接它们的**明确**逻辑关系。
  * **类型**: 仅限 `层级关系 (hierarchical)`, `时序关系 (sequential)`, `因果关系 (causal)`, `条件关系 (conditional)`。

**步骤五：关联“事件”的参与实体 (Event Participants)**

  * **任务**: 作为最后一步，将所有信息进行组装。为步骤一中识别的每个事件，明确其核心参与者。
  * **操作**: 从步骤二识别的实体列表中，选取相关实体的 `entity_name` 填入每个事件的 `participants` 字段中。


**输出格式 (Output Specification)**
在完成上述所有内部推理后，将所有**相对历史的新增内容**组装成一个结构化的KnowledgeStructure Python Pydantic模型。

    * **你的最终输出必须且只能是一个符合KnowledgeStructure结构的Python Pydantic模型。**
    * **如果 `current_text` 未包含任何新增信息，必须返回一个所有列表均为空的空结构。**
"""



HYPERRAG_MENTION_CLEAN_PROMPT = """
# 🎯 修改后的 Prompt（考公领域专用）

请对以下从 **公务员考试文档** 中提取的实体提及进行清洗和规范化：

文档内容片段：
{document_content}

原始实体提及（已进行预处理过滤）：
{mentions_json}


## 清洗规则

### 1. 删除无用的实体类型

以下类别视为无价值，需删除：

* **纯数字**：如 `"123"`, `"2023年"`, `"第1题"`, `"表1"`，`"2024-01-01"`, `"7.7%"`
* **笼统时间**：如 `"昨天"`, `"上午"`, `"现在"`, `"过去几年"`
* **无意义代词/指代词**：如 `"我"`, `"你"`, `"他们"`, `"这道题"`, `"这些材料"`
* **常见虚词/连接词**：如 `"但是"`, `"因为"`, `"所以"`, `"如果"`, `"如何"`
* **程度词/修饰语**：如 `"非常"`, `"特别"`, `"极其"`
* **无意义量词**：如 `"一些"`, `"大量"`, `"许多"`
* **标点符号/特殊符号**

### 2. 实体名称规范化

* 统一大小写与书写方式（如 `"GDP"` 保持大写）
* 去除冗余标点和空格（如 `"国内 生产 总值"` → `"国内生产总值"`)
* 统一考试领域专用术语（如 `"资料分析题"` 统一为 `"资料分析"`，`"申论试题"` 统一为 `"申论"`)
* 缩写展开（如 `"GDP"` → `"国内生产总值"`，`"CPI"` → `"居民消费价格指数"`)
* 删除冗余信息（如 `"2023年公务员考试"` 规范化为 `"公务员考试"`，"增长率7.7%")
* 专业术语保持一致性（如 `"宏观调控政策"` 不要出现 `"宏观经济调控"` 的分裂）

### 3. 保留有价值的实体

* **考试专有名词**：如 `"行测"`, `"申论"`, `"资料分析"`, `"判断推理"`, `"言语理解与表达"`
* **政策/文件/法规**：如 `"中华人民共和国宪法"`, `"十四五规划"`, `"政府工作报告"`
* **指标与概念**：如 `"GDP"`, `"财政赤字"`, `"就业率"`, `"教育公平"`
* **考试对象与分类**：如 `"省考"`, `"国考"`, `"乡镇公务员"`, `"副省级试卷"`
* **社会领域词汇**：如 `"医疗改革"`, `"环境保护"`, `"社会治理"`
* **地名/机构**：如 `"北京"`, `"国务院"`, `"国家统计局"`

### 4. 特殊处理规则

* **合并重复实体**：如 `"GDP"` 与 `"国内生产总值"` 视为同一实体，保留 `"国内生产总值"`
* **拆分复合实体**：如 `"申论写作能力考查"` → `"申论"`, `"写作能力"`
* **保留有意义数字**：如 `"三农问题"`, `"五大发展理念"`（但 `"2023年"` 删除）
* **中英文并存时优先中文**（如 `"CPI (居民消费价格指数)"` → `"居民消费价格指数"`)

### 5. 实体类型分类优化（考公领域专用）

* **考试要素**：题型、试卷类别、考查科目
* **政策法规**：法律、规划、报告、政策名词
* **社会经济指标**：GDP、就业率、财政收入、生态环境指数
* **公共管理概念**：治理、服务、改革、发展战略
* **机构与组织**：政府机关、事业单位、地名

---

📌 **输出要求**：

* 返回清洗后的实体提及列表，格式与输入相同
* 修改后的实体更新 `entity_name`
* 删除的实体不再输出
* 确保结果实体均有 **实际意义且与公务员考试相关**
"""

# 通用领域的清洗提示词（可扩展）
TCL_CLEAN_PROMPT = """
## 📥 输入数据

你将收到以下两项输入：
* `document_content`: 原始文档内容
{document_content}
* `mentions_json`: 原始实体提及列表
{mentions_json}
---

## 🧹 清洗与规范化规则

###1. 删除无用实体（整条 mention 被移除）

* `text` 或 `entity_name` 为纯数字、日期时间、标点、代词、连接词、无义短语、纯动词、纯形容词
* `entity_name` 含无效信息，如：“版本C”、“型号F”、“编号I”等缺乏识别性的名称
* `entity_description` 无内容或为空泛（如“设备的一种”、“某种装置”）
* `entity_type` 不属于允许的类型（仅保留 `资源`、`属性`、`方法`、`环境`）

---

###2. 实体名称规范化（仅改 `entity_name` 字段）

对每条 mention 的 `entity_name` 执行以下操作：

* 统一大小写、删除空格
* 去除冗余修饰词（如“产品”、“装置”、“一种”、“类型”、“某某”等）
* 修正拼写错误
* **合并指向同一概念的多个 mention**（即多个 mention 项可以拥有相同的 `entity_name`）
    注意：即使两个实体合并为相同概念，也要**保留各自的 mention（包括原始 text 和 ID）**。

---
###3. 实体描述清洗（`entity_description`）

* 删除冗余前缀或泛化表达
* 可参考 `document_content` 进行补充，但必须准确简洁
* 若无法提供有价值描述，允许将其设为空字符串或删除该字段

---

## 输出格式要求

* 输出为一个 JSON 数组（列表），每项为一个实体 mention
* 每个 mention 必须保留字段：`id`, `text`, `entity_name`, `entity_type`, `entity_description`，`event_indices`
* 保留所有清洗合格的 mention
* 删除无效或无意义的 mention 项
* 对于同义实体，仅需统一 `entity_name`，但原始的 `id` 和 `text` 必须各自保留

"""