from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class LLMBase(ABC):
    """
    大语言模型基类，定义了所有LLM实现必须遵循的接口
    """
    
    def __init__(self, model_name: str, **kwargs):
        """
        初始化LLM基类
        
        Args:
            model_name: 模型名称
            **kwargs: 其他配置参数
        """
        self.model_name = model_name
        self.config = kwargs
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
 
    @abstractmethod
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        对话式生成
        
        Args:
            messages: 对话消息列表，格式如[{"role": "user", "content": "问题"}]
            max_tokens: 最大生成token数
            temperature: 生成温度参数
            **kwargs: 其他生成参数
            
        Returns:
            生成的回复文本
        """
        pass

    @abstractmethod
    def stream_chat(
        self,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ):
        """
        流式对话生成
        """
        pass
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        文本嵌入生成
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型名称和配置的字典
        """
        return {
            "model_name": self.model_name,
            "config": self.config,
            "class_name": self.__class__.__name__
        }
    
    def validate_input(self, input_text: str, max_length: Optional[int] = None) -> bool:
        """
        验证输入文本
        
        Args:
            input_text: 输入文本
            max_length: 最大长度限制
            
        Returns:
            是否验证通过
        """
        if not isinstance(input_text, str):
            self.logger.error("输入必须是字符串类型")
            return False
        
        if not input_text.strip():
            self.logger.error("输入文本不能为空")
            return False
        
        if max_length and len(input_text) > max_length:
            self.logger.error(f"输入文本长度超过限制: {len(input_text)} > {max_length}")
            return False
        
        return True
    
    def format_messages(self, user_message: str, system_message: Optional[str] = None) -> List[Dict[str, str]]:
        """
        格式化对话消息
        
        Args:
            user_message: 用户消息
            system_message: 系统消息（可选）
            
        Returns:
            格式化后的消息列表
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return f"{self.__class__.__name__}(model_name='{self.model_name}', config={self.config})"
