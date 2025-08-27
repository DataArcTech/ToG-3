from typing import List, Dict, Any
from dataclasses import dataclass, field

@dataclass
class SeedNode:
    """种子节点"""
    id_: str
    name: str
    type: str  # 'entity' or 'event'
    score: float
    source: str  # 'extracted' or 'linked'


@dataclass
class RetrievalItem:
    """检索项：可以是chunk或event"""
    id_: str
    content: str
    type: str  # 'chunk' or 'event'
    score: float
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PPRResult:
    """PageRank结果"""
    node_scores: Dict[str, float]
    item_scores: Dict[str, float]  # chunk和event的混合得分
    traversal_path: List[str]
    convergence_info: Dict[str, Any]


@dataclass
class RetrievalContext:
    """检索上下文"""
    items: List[RetrievalItem]
    seed_nodes: List[SeedNode]
    ppr_result: PPRResult


@dataclass
class GenerationResult:
    """生成结果"""
    answer: str
    evidence_items: List[RetrievalItem]
    citations: List[str]
    confidence: float
    retrieval_context: RetrievalContext
