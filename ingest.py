"""
whaleink RAG — LlamaIndex 知识库导入
运行: python ingest.py
"""

import os
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from embedding import WhaleinkEmbedding

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def strip_frontmatter(content: str) -> tuple[str, dict]:
    """去掉 Markdown frontmatter"""
    meta = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
            body = parts[2].strip()
            return body, meta
    return content, meta


def load_blog_posts(posts_dir: str = None) -> list[Document]:
    """加载博客文章为 LlamaIndex Document"""
    if posts_dir is None:
        posts_dir = os.getenv("BLOG_SOURCE_DIR", "/root/whaleink-blog/source/_posts")

    posts_path = Path(posts_dir)
    if not posts_path.exists():
        logger.warning(f"博客目录不存在: {posts_path}")
        return []

    documents = []
    for md_file in sorted(posts_path.glob("*.md")):
        raw = md_file.read_text(encoding="utf-8")
        body, meta = strip_frontmatter(raw)
        title = meta.get("title", md_file.stem)
        source = f"/2026/01/18/{md_file.stem}/"

        doc = Document(
            text=body,
            metadata={
                "title": title,
                "source": source,
                "filename": md_file.name,
            },
        )
        documents.append(doc)
        logger.info(f"  ✓ {title}")

    logger.info(f"加载了 {len(documents)} 篇博客文章")
    return documents


def main():
    print("=" * 50)
    print("  whaleink RAG — LlamaIndex 知识库导入")
    print("=" * 50)

    # 1. 加载文章
    print("\n📄 加载博客文章...")
    docs = load_blog_posts()
    if not docs:
        print("❌ 未找到博客文章")
        return

    # 2. 配置 LlamaIndex
    print("\n🔧 配置 LlamaIndex...")
    Settings.embed_model = WhaleinkEmbedding()
    Settings.node_parser = SentenceSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    # 3. 初始化 ChromaDB
    print("\n🗄️  连接 ChromaDB...")
    persist_dir = os.getenv("CHROMA_DB_PATH", "/root/whaleink-rag/chroma_db")
    db = chromadb.PersistentClient(path=persist_dir)

    try:
        db.delete_collection("whaleink_blog")
        print("   已清空旧向量库")
    except Exception:
        pass

    chroma_collection = db.create_collection("whaleink_blog")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. 构建索引
    print("\n📥 构建向量索引...")
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        show_progress=True,
    )

    count = chroma_collection.count()
    print(f"\n✅ 导入完成！向量库共 {count} 个文档块")

    # 5. 简单验证
    print("\n🔍 验证检索...")
    results = chroma_collection.query(query_texts=["Spring Boot 注解"], n_results=2)
    if results["ids"] and results["ids"][0]:
        for i, rid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            print(f"   → [{meta.get('title', '?')}] (距离: {results['distances'][0][i]:.4f})")

    print("\n🎉 知识库就绪！运行 python app.py 启动服务")


if __name__ == "__main__":
    main()
