"""
whaleink RAG — 自定义 Embedding（LlamaIndex 兼容）
先用 jieba + TF-IDF，等有 API key 后一键替换
"""

import os
import logging
from typing import Any, List, Optional

from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field

logger = logging.getLogger(__name__)


class WhaleinkEmbedding(BaseEmbedding):
    """基于 jieba + TF-IDF 的轻量 Embedding（LlamaIndex 兼容）

    占位方案：等有 DASHSCOPE_API_KEY 或其他 embedding API 后，
    只需改这一行：
        Settings.embed_model = DashScopeEmbedding(...)
    """

    embed_dim: int = Field(default=512, description="向量维度")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_backend()

    @classmethod
    def class_name(cls) -> str:
        return "WhaleinkEmbedding"

    def _init_backend(self):
        """初始化 TF-IDF + jieba"""
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer

        def jieba_tokenizer(text: str) -> List[str]:
            return list(jieba.cut(text))

        self._vectorizer = TfidfVectorizer(
            max_features=self.embed_dim,
            lowercase=False,
            tokenizer=jieba_tokenizer,
            analyzer="word",
        )
        # 用一些默认文本初始化，确保 fit 过
        self._vectorizer.fit(["默认文本", "whaleink 博客"])
        logger.info("⚙️  WhaleinkEmbedding: jieba + TF-IDF (512维)")

    def _get_text_embedding(self, text: str) -> List[float]:
        import numpy as np

        vec = self._vectorizer.transform([text]).toarray()
        result = np.zeros((1, self.embed_dim))
        dim = min(vec.shape[1], self.embed_dim)
        result[0, :dim] = vec[0, :dim] if vec.shape[1] <= self.embed_dim else vec[0, :self.embed_dim]
        return result[0].tolist()

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self._get_text_embedding(t) for t in texts]
