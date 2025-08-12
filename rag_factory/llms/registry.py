from .openai_llm import OpenAILLM
from .llm_base import LLMBase
from typing import Dict, Type, Any, Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,  # 设置最低输出级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class LLMRegistry:
    """LLM模型注册表，用于管理和创建不同类型的LLM模型"""
    _llms: Dict[str, Type[LLMBase]] = {}

    @classmethod
    def register(cls, name: str, llm_class: Type[LLMBase]):
        """注册LLM模型类
        
        Args:
            name: 模型名称
            llm_class: LLM模型类
        """
        cls._llms[name] = llm_class

    @classmethod
    def create(cls, name: str, **kwargs) -> LLMBase:
        """创建LLM实例
        
        Args:
            name: 模型名称
            **kwargs: 模型初始化参数
            
        Returns:
            LLM实例
            
        Raises:
            ValueError: 当模型名称不存在时
        """
        if name not in cls._llms:
            available_llms = list(cls._llms.keys())
            raise ValueError(f"LLM模型 '{name}' 未注册。可用的模型: {available_llms}")
        
        llm_class = cls._llms[name]
        
        return llm_class(**kwargs)

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