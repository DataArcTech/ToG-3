from httpx import options
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
        max_tokens: Optional[int] = None,
        temperature: float = 0.1,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            timeout=timeout
        )

        self.default_max_tokens = max_tokens
        self.default_temperature = temperature

        self.logger.info(f"OpenAI LLM初始化完成，模型: {model_name}")


    def _validate_messages(self, messages: List[Dict[str, str]]):
        if not messages or not isinstance(messages, list):
            raise ValueError("消息列表不能为空且必须是列表格式")
        for msg in messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError("消息格式错误，必须包含role和content字段")
            if not self.validate_input(msg['content']):
                raise ValueError(f"消息内容验证失败: {msg['content']}")

    def _get_token_stats(self, usage) -> Dict[str, int]:
        return {
            "input_tokens": getattr(usage, "prompt_tokens", 0),
            "output_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0)
        }

    def chat(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        self._validate_messages(messages)

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
                token_stats = self._get_token_stats(response.usage)
                return result, token_stats
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
        self._validate_messages(messages)

        try:
            params = {}
            if return_token_count:
                params["stream_options"] = {"include_usage": True}

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stream=True,
                **params,
                **kwargs
            )

            full_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

                if return_token_count and getattr(chunk, "usage", None):
                    yield self._get_token_stats(chunk.usage)

        except Exception as e:
            self.logger.error(f"流式对话生成失败: {str(e)}")
            raise

    def embed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not self.model_name.startswith("text-embedding"):
            raise ValueError(f"当前模型'{self.model_name}'不支持嵌入生成，请使用嵌入专用模型")

        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        for text in text_list:
            if not self.validate_input(text):
                raise ValueError(f"文本内容验证失败: {text}")

        try:
            embeddings = []
            batch_size = 100
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i:i+batch_size]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings.extend(data.embedding for data in response.data)

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            self.logger.error(f"嵌入生成失败: {str(e)}")
            raise

    def get_available_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"获取模型列表失败: {str(e)}")
            return []

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info.update({
            "api_base": getattr(self.client, 'base_url', None),
            "organization": getattr(self.client, 'organization', None),
            "max_retries": getattr(self.client, 'max_retries', None),
            "timeout": getattr(self.client, 'timeout', None),
            "max_tokens": self.default_max_tokens,
            "temperature": self.default_temperature
        })
        return info

    # ---------------------- 异步接口 ----------------------
    async def achat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict[str, int]]]:
        self._validate_messages(messages)

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                **kwargs
            )
            result = response.choices[0].message.content
            if return_token_count:
                return result, self._get_token_stats(response.usage)
            return result

        except Exception as e:
            self.logger.error(f"异步对话生成失败: {str(e)}")
            raise

    async def astream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_token_count: bool = False,
        **kwargs
    ):
        self._validate_messages(messages)

        try:
            params = {}
            if return_token_count:
                params["stream_options"] = {"include_usage": True}

            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature or self.default_temperature,
                stream=True,
                **params,
                **kwargs
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

                if return_token_count and getattr(chunk, "usage", None):
                    yield self._get_token_stats(chunk.usage)

        except Exception as e:
            self.logger.error(f"异步流式对话生成失败: {str(e)}")
            raise

    async def aembed(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if not self.model_name.startswith("text-embedding"):
            raise ValueError(f"当前模型'{self.model_name}'不支持嵌入生成，请使用嵌入专用模型")

        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        for text in text_list:
            if not self.validate_input(text):
                raise ValueError(f"无效输入文本: {text}")

        try:
            embeddings = []
            batch_size = 100
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i:i+batch_size]
                response = await self.client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                embeddings.extend(data.embedding for data in response.data)

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            self.logger.error(f"异步嵌入生成失败: {str(e)}")
            raise
