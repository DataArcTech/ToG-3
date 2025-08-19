from typing import Dict, Type, Any, Optional, List

from rag_factory.llms.openai import OpenAILLM
from rag_factory.llms.base import LLMBase

import logging

logging.basicConfig(
    level=logging.INFO,  # 设置最低输出级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LLMRegistry:
    """LLM模型注册表，支持智能创建逻辑"""
    _llms: Dict[str, Type[LLMBase]] = {}

    @classmethod
    def register(cls, name: str, llm_class: Type[LLMBase]):
        """注册LLM模型类"""
        cls._llms[name] = llm_class

    @classmethod
    def create(cls, name: str, **kwargs) -> LLMBase:
        """智能创建LLM实例"""
        if name not in cls._llms:
            available_llms = list(cls._llms.keys())
            raise ValueError(f"LLM模型 '{name}' 未注册。可用的模型: {available_llms}")
        
        try:
            # 根据LLM类型选择创建方式
            if name == "openai":
                return cls._create_openai(**kwargs)
            else:
                # 默认构造函数
                llm_class = cls._llms[name]
                return llm_class(**kwargs)
        except Exception as e:
            logger.error(f"创建LLM '{name}' 失败: {str(e)}")
            raise Exception(f"无法创建LLM '{name}': {str(e)}") from e
    
    @classmethod
    def _create_openai(cls, **kwargs) -> OpenAILLM:
        """创建OpenAI LLM"""
        return OpenAILLM(**kwargs)

    @classmethod
    def list_llms(cls) -> List[str]:
        """列出所有已注册的LLM模型名称
        
        Returns:
            已注册的模型名称列表
        """
        return list(cls._llms.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查模型是否已注册
        
        Args:
            name: 模型名称
            
        Returns:
            如果已注册返回True，否则返回False
        """
        return name in cls._llms

    @classmethod
    def unregister(cls, name: str) -> bool:
        """取消注册模型
        
        Args:
            name: 模型名称
            
        Returns:
            成功取消注册返回True，模型不存在返回False
        """
        if name in cls._llms:
            del cls._llms[name]
            return True
        return False


# 注册默认的LLM模型
LLMRegistry.register("openai", OpenAILLM)