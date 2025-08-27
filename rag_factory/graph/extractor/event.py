from rag_factory.graph.extractor.base import GraphExtractorBase
from rag_factory.prompts.prompt import HYPERRAG_EXTRACTION_PROMPT
from rag_factory.data_model.schema import KnowledgeStructure, PydanticUtils
from rag_factory.data_model.document import Document
from rag_factory.llm.base import LLMBase
from typing import Any


class HyperRAGGraphExtractor(GraphExtractorBase):
    """
    HyperRAGGraphExtractor用于提取完整的图结构，包括事件、mention、实体和关系。
    
    使用专门的提示模板来提取层次化的图结构。
    """
    
    def __init__(
        self,
        llm: LLMBase,
        extract_prompt: str = None,
        response_format: Any = None,
        max_concurrent: int = 100
    ) -> None:
        # 调用父类初始化
        extract_prompt = extract_prompt if extract_prompt is not None else HYPERRAG_EXTRACTION_PROMPT
        response_format = response_format if response_format is not None else KnowledgeStructure
        super().__init__(llm, extract_prompt, response_format, max_concurrent)

    async def _aextract(self, document: Document, semaphore) -> Document:
        """从documents中异步提取完整的图结构"""
        async with semaphore:
            content = document.content
            if not content:
                return document
    
            try:
                prompt = self.extract_prompt.format(text=content)
                messages = [{"role": "user", "content": prompt}]
                llm_response = await self.llm.aparse_chat(messages, response_format=self.response_format)
                
                if llm_response is None:
                    print("错误：LLM 返回空响应！")
                    result = {"events": [], "mentions": [], "event_relations": [], "entity_relations": []}
                else:
                    # 统一转换为字典格式
                    result = llm_response.to_dict()
                    print(f"HyperRAGGraphExtractor 解析结果: {len(result['events'])} 个事件, "
                          f"{len(result['mentions'])} 个mentions, "
                          f"{len(result.get('event_relations', []))} 个事件关系")
                    
            except Exception as e:
                print(f"提取图结构时出错: {e}")
                result = {"events": [], "mentions": [], "event_relations": [], "entity_relations": []}

            # 将结果存储到document metadata中（统一使用字典格式）
            document.metadata["events"] = result["events"]
            document.metadata["mentions"] = result["mentions"]
            
            # 处理事件关系：将索引转换为具体的事件内容
            processed_event_relations = self._process_event_relations(
                result.get("event_relations", []), 
                result["events"]
            )
            
            document.metadata["event_relations"] = processed_event_relations
            document.metadata["entity_relations"] = result.get("entity_relations", [])
            
            return document

    def _process_event_relations(self, event_relations, events):
        """处理事件关系，将索引转换为具体的事件内容"""
        processed_relations = []
        
        for relation in event_relations:
            # 使用统一的属性获取方法
            head_event_id = PydanticUtils.safe_get_attr(relation, 'head_event', '')
            tail_event_id = PydanticUtils.safe_get_attr(relation, 'tail_event', '')
            relation_type = PydanticUtils.safe_get_attr(relation, 'relation_type', '')
            description = PydanticUtils.safe_get_attr(relation, 'description', '')
            
            # 提取事件索引
            try:
                head_event_idx = int(head_event_id.replace("event_", ""))
                tail_event_idx = int(tail_event_id.replace("event_", ""))
            except (ValueError, AttributeError):
                continue
            
            # 获取事件内容
            head_content = self._get_event_content(events, head_event_idx)
            tail_content = self._get_event_content(events, tail_event_idx)
            
            if head_content and tail_content:
                processed_relation = {
                    "head_event_content": head_content,
                    "tail_event_content": tail_content,
                    "relation_type": relation_type,
                    "description": description
                }
                processed_relations.append(processed_relation)
        
        return processed_relations
    
    def _get_event_content(self, events, event_idx):
        """安全获取事件内容"""
        if 0 <= event_idx < len(events):
            event = events[event_idx]
            return PydanticUtils.safe_get_attr(event, 'content', '')
        return ""

    @classmethod
    def class_name(cls) -> str:
        return "HyperRAGGraphExtractor"