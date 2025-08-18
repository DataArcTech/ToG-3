import json

def parse_extraction_result(llm_response: str):
    """
    解析LLM的响应，返回实体和关系列表
    """
    try:
        data = json.loads(llm_response)  # 假设 LLM 输出 JSON
        entities = []
        relationships = []

        for e in data.get("entity_list", []):
            entities.append((
                e.get("entity_text", ""),
                e.get("entity_type", ""),
                e.get("entity_description", "")
            ))

        for r in data.get("relation_list", []):
            relationships.append((
                r.get("head", ""),
                r.get("tail", ""),
                r.get("relation_type", ""),
                r.get("relation_description", "")
            ))

        return entities, relationships
    except Exception as e:
        print(f"parse_extraction_result 解析失败: {e}")
        return [], []