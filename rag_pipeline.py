"""
whaleink RAG Pipeline
博客知识库：加载 → 分割 → 向量化 → 检索 → 生成
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv #加载.env的环境变量

load_dotenv()

logger = logging.getLogger(__name__)    #高性能日志器


# ─── LLM 调用 ────────────────────────────────────────────

def call_llm(prompt: str, system: str = "") -> str:
    """调用 DeepSeek API 生成回答"""
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return "错误：未配置 DEEPSEEK_API_KEY"

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


# ─── 文档加载 ────────────────────────────────────────────

def load_blog_posts(posts_dir: str = None) -> List[Dict[str, str]]: 
    if posts_dir is None:
        posts_dir = os.getenv("BLOG_SOURCE_DIR", "/root/whaleink-blog/source/_posts")

    docs = []
    posts_path = Path(posts_dir)

    if not posts_path.exists():
        logger.warning(f"博客目录不存在: {posts_path}")
        return docs

    for md_file in sorted(posts_path.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")

        # 提取 frontmatter 标题 ：frontmatter 就是写在 Markdown 文章最开头的一段 “元数据字典”，夹在两行 --- 之间，用 YAML 格式写，用来存标题、日期、标签这些信息。
        title = md_file.stem
        lines = content.split("\n")
        for line in lines[:20]:
            if line.startswith("title:"):
                title = line.replace("title:", "").strip().strip('"').strip("'")
                break

        # 去掉 frontmatter 部分
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                body = parts[2].strip()

        docs.append({
            "title": title,
            "content": body,
            "source": f"/2026/01/18/{md_file.stem}/",
            "filename": md_file.name,
        })

    logger.info(f"加载了 {len(docs)} 篇博客文章")
    return docs


# ─── 文本分割 ────────────────────────────────────────────

def split_documents(
    docs: List[Dict[str, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """将文档分割成块"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["content"])
        for i, text in enumerate(texts):
            chunks.append({
                "id": f"{doc['filename']}#{i}",     # 唯一ID：文件名+块号
                "text": text,                       # 文本
                "metadata": {                       
                    "title": doc["title"],
                    "source": doc["source"],
                    "chunk": i,
                },
            })

    logger.info(f"分割为 {len(chunks)} 个文本块")
    return chunks


# ─── Embedding（自动选择后端） ────────────────────────────

class EmbeddingFunction:
    """文本转向量——自动选择可用的后端"""

    def __init__(self):
        self._backend = None
        self._init_backend()

    def _init_backend(self):
        """探测可用后端并初始化"""
        # 优先: HuggingFace Inference API (需要 HF_TOKEN + 外网可达)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            import requests
            try:
                r = requests.get("https://api-inference.huggingface.co", timeout=5)
                if r.status_code < 500:
                    logger.info("🌐 使用 HuggingFace Inference API (BAAI/bge-small-zh-v1.5)")
                    self._backend = "hf_api"
                    return
                else:
                    logger.warning(f"HuggingFace API 不可达 (HTTP {r.status_code})")
            except requests.ConnectionError:
                logger.warning("🌐 HuggingFace API 不可达（网络受限），降级到本地方案")

        # 兜底: TF-IDF + jieba 中文分词
        logger.info("⚙️  使用 TF-IDF + jieba 中文分词（轻量免模型）")
        self._backend = "tfidf"
        # 预加载 ChromaDB 中的文档，确保向量空间一致
        self._preload_tfidf()

    def _preload_tfidf(self):
        """从 ChromaDB 加载已有文档，预训练 TF-IDF 词典"""
        try:
            import chromadb
            persist_dir = os.getenv("CHROMA_DB_PATH", "/root/whaleink-rag/chroma_db")
            client = chromadb.PersistentClient(path=persist_dir)
            collection = client.get_or_create_collection(name="whaleink_blog")
            count = collection.count()
            if count > 0:
                all_docs = collection.get(include=["documents"])
                if all_docs and all_docs.get("documents"):
                    logger.info(f"预加载 {len(all_docs['documents'])} 个文档，训练 TF-IDF 词典")
                    self._tfidf_embed(all_docs["documents"])
                    logger.info(f"TF-IDF 词典已就绪 ({self._tfidf_dim} 个特征词)")
        except Exception as e:
            logger.warning(f"预加载 TF-IDF 失败: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._backend == "hf_api":
            return self._hf_embed(texts)
        else:
            return self._tfidf_embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embed_documents(input)

    def _hf_embed(self, texts: List[str]) -> List[List[float]]:
        """使用 HuggingFace Inference API"""
        import requests

        api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/BAAI/bge-small-zh-v1.5"
        headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

        all_embeddings = []
        for text in texts:
            resp = requests.post(api_url, headers=headers, json={"inputs": text, "options": {"wait_for_model": True}}, timeout=30)
            if resp.status_code == 200:
                emb = resp.json()
                # HF API returns list of token embeddings, take the mean
                if emb and isinstance(emb, list) and isinstance(emb[0], list):
                    import numpy as np
                    mean_emb = np.mean(emb, axis=0).tolist()
                    all_embeddings.append(mean_emb)
                else:
                    all_embeddings.append([0.0] * 512)
            else:
                logger.warning(f"HF API error: {resp.status_code}")
                all_embeddings.append([0.0] * 512)

        return all_embeddings

    def _tfidf_embed(self, texts: List[str]) -> List[List[float]]:
        """TF-IDF + jieba 中文分词（轻量免模型）"""
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        # jieba 中文分词器
        def jieba_tokenizer(text: str) -> List[str]:
            return list(jieba.cut(text))

        # 缓存 vectorizer
        if not hasattr(self, "_tfidf"):
            self._tfidf = TfidfVectorizer(
                max_features=512,
                lowercase=False,
                tokenizer=jieba_tokenizer,
                analyzer='word',
            )
            self._tfidf.fit(texts + ["默认文本"])
            self._tfidf_dim = len(self._tfidf.get_feature_names_out())

        vec = self._tfidf.transform(texts).toarray()

        # 填充到 512 维
        result = np.zeros((len(texts), 512))
        dim = min(vec.shape[1], 512)
        result[:, :dim] = vec[:, :dim] if vec.shape[1] <= 512 else vec[:, :512]

        return result.tolist()


# ─── ChromaDB 向量存储 ────────────────────────────────────

class VectorStore:
    """基于 ChromaDB 的向量存储和检索"""

    def __init__(self, embedding_fn: EmbeddingFunction):
        import chromadb

        persist_dir = os.getenv("CHROMA_DB_PATH", "/root/whaleink-rag/chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_fn

        self.collection = self.client.get_or_create_collection(
            name="whaleink_blog",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def ingest(self, chunks: List[Dict[str, Any]]):
        """导入文档块到向量库"""
        if not chunks:
            logger.warning("没有文档可导入")
            return

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        batch_size = 50
        total = len(chunks)
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_texts = texts[i:batch_end]
            embeddings = self.embedding_fn.embed_documents(batch_texts)

            self.collection.add(
                ids=ids[i:batch_end],
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=metadatas[i:batch_end],
            )
            logger.info(f"  导入进度: {batch_end}/{total}")

        logger.info(f"✅ 已导入 {total} 个文档块到向量库")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """检索最相关的文档块"""
        query_embedding = self.embedding_fn.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                hits.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i] if results.get("distances") else 0,
                })

        return hits

    def clear(self):
        """清空向量库"""
        self.client.delete_collection("whaleink_blog")
        self.collection = self.client.get_or_create_collection(name="whaleink_blog")
        logger.info("向量库已清空")


# ─── RAG 查询 ────────────────────────────────────────────

def build_rag_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    """构建 RAG 提示词"""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        context_parts.append(
            f"[{i}] 来自《{meta['title']}》\n{chunk['text']}\n"
        )

    context_str = "\n---\n".join(context_parts)

    prompt = f"""你是一个基于博客内容的 AI 助手。请根据以下博客文章内容，回答用户的问题。

如果问题与博客内容无关，请礼貌地说明无法回答。回答请使用中文。

博客内容：
{context_str}

用户问题：{query}

请给出详细、准确的回答，并在引用博客内容时标注来源文章标题。"""

    return prompt


def query_rag(query: str, vector_store: VectorStore, k: int = 3) -> Dict[str, Any]:
    """执行 RAG 查询：检索 → 生成"""
    chunks = vector_store.search(query, k=k)

    if not chunks:
        return {
            "query": query,
            "answer": "抱歉，知识库中还没有相关内容。",
            "sources": [],
        }

    prompt = build_rag_prompt(query, chunks)
    answer = call_llm(prompt, system="你是 whaleink 博客的 AI 助手，基于博客内容回答问题。")

    sources = list(set(c["metadata"]["title"] for c in chunks))

    return {
        "query": query,
        "answer": answer,
        "sources": sources,
    }
