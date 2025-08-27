from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """统一文档数据结构
    
    用于在检索器、向量存储、重排序器等组件之间传递文档数据。
    
    Attributes:
        content (str): 文档的文本内容
        metadata (Dict[str, Any]): 文档的元数据信息，如来源、标题等
        id (Optional[str]): 文档的唯一标识符
    
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None