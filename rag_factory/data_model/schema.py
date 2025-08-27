from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any, Union


class Event(BaseModel):
    """事件模型"""
    id: str = Field(..., description="事件唯一ID，例如 event_0")
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
    text: str = Field(..., description="原文字符串")
    entity_name: str = Field(..., description="规范化名称")
    entity_type: str = Field(..., description="实体类别")
    entity_description: Optional[str] = Field(None, description="简要描述")
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
    head_event: str = Field(..., description="关系头事件ID")
    tail_event: str = Field(..., description="关系尾事件ID")
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
    head_entity: str = Field(..., description="实体A")
    tail_entity: str = Field(..., description="实体B")
    relation_type: Literal["题型", "考点", "解题方法", "公式与规则", "题目结构", "考试模块"] = Field(..., description="关系类型")
    description: Optional[str] = Field(None, description="关系证据")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelation':
        """从字典创建EntityRelation对象"""
        return cls(**data)


class KnowledgeStructure(BaseModel):
    """知识结构模型"""
    events: List[Event] = []
    mentions: List[Mention] = []
    event_relations: List[EventRelation] = []
    entity_relations: List[EntityRelation] = []

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


class Entity(BaseModel):
    """实体模型"""
    name: str = Field(..., description="实体名称")
    type: str = Field(..., description="实体类型")
    description: Optional[str] = Field(None, description="实体描述")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """从字典创建Entity对象"""
        return cls(**data)


class Triple(BaseModel):
    """三元组模型"""
    head: str = Field(..., description="头实体")
    tail: str = Field(..., description="尾实体")
    relation: str = Field(..., description="关系类型")
    description: Optional[str] = Field(None, description="关系描述")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Triple':
        """从字典创建Triple对象"""
        return cls(**data)


class GraphTriples(BaseModel):
    """图三元组模型"""
    entities: List[Entity] = []
    relationships: List[Triple] = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [triple.to_dict() for triple in self.relationships]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphTriples':
        """从字典创建GraphTriples对象"""
        return cls(
            entities=[Entity.from_dict(entity) for entity in data.get("entities", [])],
            relationships=[Triple.from_dict(triple) for triple in data.get("relationships", [])]
        )


# 统一的格式转换工具类
class PydanticUtils:
    """Pydantic工具类，提供统一的格式转换方法"""
    
    @staticmethod
    def to_dict(obj: Union[BaseModel, Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        将Pydantic对象或包含Pydantic对象的列表转换为字典格式
        
        Args:
            obj: Pydantic对象、字典或包含这些对象的列表
            
        Returns:
            转换后的字典或字典列表
        """
        if isinstance(obj, list):
            return [PydanticUtils._convert_item(item) for item in obj]
        else:
            return PydanticUtils._convert_item(obj)
    
    @staticmethod
    def _convert_item(item: Union[BaseModel, Dict[str, Any]]) -> Dict[str, Any]:
        """转换单个项目"""
        if isinstance(item, BaseModel):
            return item.model_dump()
        elif isinstance(item, dict):
            return item
        else:
            return item
    
    @staticmethod
    def from_dict(cls: type, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[BaseModel, List[BaseModel]]:
        """
        从字典创建Pydantic对象
        
        Args:
            cls: Pydantic模型类
            data: 字典数据或字典列表
            
        Returns:
            Pydantic对象或对象列表
        """
        if isinstance(data, list):
            return [cls(**item) for item in data]
        else:
            return cls(**data)
    
    @staticmethod
    def safe_get_attr(obj: Union[BaseModel, Dict[str, Any]], attr_name: str, default: Any = None) -> Any:
        """
        安全获取对象属性，支持Pydantic对象和字典
        
        Args:
            obj: Pydantic对象或字典
            attr_name: 属性名
            default: 默认值
            
        Returns:
            属性值或默认值
        """
        if isinstance(obj, BaseModel):
            return getattr(obj, attr_name, default)
        elif isinstance(obj, dict):
            return obj.get(attr_name, default)
        return default
