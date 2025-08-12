import openai
from typing import Dict, Any, List, Optional, Union, Tuple
from .llm_base import LLMBase

class OpenAILLM(LLMBase):
    """
    OpenAI LLM对话模型
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        **kwargs
    ):
        """
        初始化OpenAI LLM
        
        Args:
            model_name: 模型名称，如 gpt-3.5-turbo, gpt-4 等
            api_key: OpenAI API密钥
            base_url: API基础URL
            organization: 组织ID（可选）
            max_retries: 最大重试次数
            timeout: 请求超时时间
            **kwargs: 其他配置参数
        """
        super().__init__(model_name, **kwargs)
        
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout
        )
        
        # 默认参数
        self.default_max_tokens = kwargs.get('max_tokens', 2000)
        self.default_temperature = kwargs.get('temperature', 0.7)
        
        self.logger.info(f"OpenAI LLM初始化完成，模型: {model_name}")
    
    
    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        """
        对话式生成
        
        Args:
            messages: 对话消息列表，格式如[{"role": "user", "content": "问题"}]
            max_tokens: 最大生成token数
            temperature: 生成温度参数
            return_token_count: 是否返回token统计信息
            **kwargs: 其他生成参数
            
        Returns:
            生成的回复文本，如果return_token_count为True则返回(文本, token统计)
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("消息列表不能为空且必须是列表格式")
        
        # 验证消息格式
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("消息格式错误，必须包含role和content字段")
            if not self.validate_input(msg['content']):
                raise ValueError(f"消息内容验证失败: {msg['content']}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                **kwargs
            )
            
            result = response.choices[0].message.content.strip()
            
            if return_token_count:
                # 获取输出token数
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0
                
                token_stats = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens
                }
                
                self.logger.debug(f"对话生成成功，长度: {len(result)}, 输入tokens: {input_tokens}, 输出tokens: {output_tokens}")
                return result, token_stats
            else:
                self.logger.debug(f"对话生成成功，长度: {len(result)}")
                return result
            
        except Exception as e:
            self.logger.error(f"对话生成失败: {str(e)}")
            raise


    def stream_chat(
        self,
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ):
        """
        流式对话生成
        
        Args:
            messages: 对话消息列表
            max_tokens: 最大生成token数
            temperature: 生成温度参数
            return_token_count: 是否返回token统计信息
            **kwargs: 其他生成参数
            
        Yields:
            生成的文本片段，如果return_token_count为True则在流式输出结束后yield token统计
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("消息列表不能为空且必须是列表格式")
        
        # 验证消息格式
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("消息格式错误，必须包含role和content字段")
            if not self.validate_input(msg['content']):
                raise ValueError(f"消息内容验证失败: {msg['content']}")
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stream=True,
                stream_options={"include_usage": True} if return_token_count else None,
                **kwargs
            )
            
            full_response = ""
            
            for chunk in stream:
                # 检查choices是否存在以及是否有内容
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        content = delta.content
                        full_response += content
                        yield content
                
                # 检查是否有usage信息（在流的最后一个chunk中）
                if return_token_count and hasattr(chunk, 'usage') and chunk.usage is not None:
                    input_tokens = chunk.usage.prompt_tokens if chunk.usage else 0
                    output_tokens = chunk.usage.completion_tokens if chunk.usage else 0
                    total_tokens = chunk.usage.total_tokens if chunk.usage else 0
                    
                    token_stats = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens
                    }
                    
                    self.logger.debug(f"流式对话生成成功，长度: {len(full_response)}, 输入tokens: {input_tokens}, 输出tokens: {output_tokens}")
                    yield token_stats
                    
        except Exception as e:
            self.logger.error(f"流式对话生成失败: {str(e)}")
            raise    
    
    
    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        文本嵌入生成
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        # 检查当前模型是否为嵌入模型
        embedding_models = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
        if self.model_name not in embedding_models:
            warning_msg = f"警告：当前模型'{self.model_name}'不是嵌入模型，建议使用嵌入专用模型如 text-embedding-ada-002"
            self.logger.warning(warning_msg)
            raise ValueError(f"当前模型'{self.model_name}'不支持嵌入生成，请使用嵌入专用模型")
        
        # 统一处理为列表格式
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        # 验证输入
        for text in text_list:
            if not self.validate_input(text):
                raise ValueError(f"文本内容验证失败: {text}")
        
        try:
            # 使用当前嵌入模型生成嵌入
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text_list
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # 根据输入格式返回结果
            result = embeddings[0] if is_single else embeddings
            self.logger.debug(f"嵌入生成成功，文本数量: {len(text_list)}")
            return result
            
        except Exception as e:
            self.logger.error(f"嵌入生成失败: {str(e)}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表
        
        Returns:
            可用模型名称列表
        """
        try:
            models = self.client.models.list()
            model_names = [model.id for model in models.data]
            self.logger.debug(f"获取到{len(model_names)}个可用模型")
            return model_names
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            包含模型名称和配置的字典
        """
        info = super().get_model_info()
        info.update({
            "api_base": getattr(self.client, 'base_url', None),
            "organization": getattr(self.client, 'organization', None),
            "max_retries": getattr(self.client, 'max_retries', None),
            "timeout": getattr(self.client, 'timeout', None),
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature
        })
        return info
