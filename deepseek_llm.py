"""
DeepSeek LLM wrapper for LlamaIndex
绕过 OpenAI 模型名校验，直接调用 DeepSeek API
"""
import os
import json
import logging
from typing import Any, List, Mapping, Optional

import httpx
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback

logger = logging.getLogger(__name__)


class DeepSeekLLM(CustomLLM):
    """DeepSeek Chat LLM 封装"""

    model: str = "deepseek-chat"
    temperature: float = 0.3
    max_tokens: int = 2048
    api_key: Optional[str] = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        api_key = self.api_key or os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return CompletionResponse(text="错误：未配置 DEEPSEEK_API_KEY")

        resp = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError("流式响应暂不支持")

    @classmethod
    def class_name(cls) -> str:
        return "DeepSeekLLM"
