from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Document:
    """文档数据结构"""
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
