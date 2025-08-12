from typing import Dict, Type, Any, Optional, List
import logging
from .Reranker_Base import RerankerBase
from .Reranker_Qwen3 import Qwen3Reranker

class RerankerRegistry:
    _rerankers: Dict[str, Type[RerankerBase]] = {}

    @classmethod
    def register(cls, name: str, reranker_class: Type[RerankerBase]):
        """注册重排序器类到注册表中
        
        Args:
            name: 重排序器的名称
            reranker_class: 重排序器类，必须继承自RerankerBase
        """
        if not issubclass(reranker_class, RerankerBase):
            raise ValueError(f"重排序器类 {reranker_class} 必须继承自 RerankerBase")
        
        if name in cls._rerankers:
            logging.warning(f"重排序器 '{name}' 已存在，将被覆盖")
            
        cls._rerankers[name] = reranker_class
        logging.info(f"成功注册重排序器: {name}")

    @classmethod
    def create(cls, name: str, **kwargs) -> RerankerBase:
        """根据名称获取重排序器实例
        
        Args:
            name: 重排序器名称
            **kwargs: 传递给重排序器构造函数的参数
            
        Returns:
            RerankerBase: 重排序器实例
            
        Raises:
            ValueError: 当重排序器未注册时抛出
        """
        if name not in cls._rerankers:
            available = list(cls._rerankers.keys())
            raise ValueError(f"未找到重排序器 '{name}'。可用的重排序器: {available}")
        
        reranker_class = cls._rerankers[name]
        return reranker_class(**kwargs)

    @classmethod
    def list_rerankers(cls) -> List[str]:
        """获取所有已注册的重排序器名称列表
        
        Returns:
            List[str]: 重排序器名称列表
        """
        return list(cls._rerankers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """检查重排序器是否已注册
        
        Args:
            name: 重排序器名称
            
        Returns:
            bool: 如果已注册返回True，否则返回False
        """
        return name in cls._rerankers

    @classmethod
    def unregister(cls, name: str) -> bool:
        """注销重排序器
        
        Args:
            name: 要注销的重排序器名称
            
        Returns:
            bool: 如果成功注销返回True，如果重排序器不存在返回False
        """
        if name in cls._rerankers:
            del cls._rerankers[name]
            logging.info(f"成功注销重排序器: {name}")
            return True
        else:
            logging.warning(f"尝试注销不存在的重排序器: {name}")
            return False

    @classmethod
    def clear_all(cls):
        """清除所有已注册的重排序器"""
        cls._rerankers.clear()
        logging.info("已清除所有重排序器")

# 注册默认的重排序器
RerankerRegistry.register("qwen3", Qwen3Reranker)