"""
whaleink RAG — 博客知识库导入脚本
运行: python ingest.py
"""

import logging
import sys
from rag_pipeline import (
    load_blog_posts,
    split_documents,
    EmbeddingFunction,
    VectorStore,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 50)
    print("  whaleink RAG — 博客知识库导入")
    print("=" * 50)

    # 1. 加载文章
    print("\n📄 加载博客文章...")
    docs = load_blog_posts()
    if not docs:
        print("❌ 未找到博客文章，请检查 BLOG_SOURCE_DIR")
        sys.exit(1)

    for d in docs:
        print(f"   ✓ {d['title']} ({len(d['content'])} 字符)")

    # 2. 分割
    print("\n✂️  分割文档...")
    chunks = split_documents(docs)
    print(f"   → {len(chunks)} 个文本块")

    # 3. 初始化 embedding
    print("\n🧠 初始化 Embedding...")
    emb_fn = EmbeddingFunction()
    print(f"   后端: {emb_fn._backend}")

    # 4. 初始化向量库
    print("\n🗄️  初始化 ChromaDB...")
    store = VectorStore(embedding_fn=emb_fn)

    if store.count > 0:
        print(f"   ⚠️  向量库已有 {store.count} 条数据，重新导入...")
        store.clear()

    # 5. 导入
    print("\n📥 导入向量库...")
    store.ingest(chunks)

    # 6. 验证
    count = store.count
    print(f"\n✅ 导入完成！向量库共 {count} 个文档块")

    # 测试检索
    print("\n🔍 测试检索...")
    test_query = "Spring Boot 常用注解有哪些？"
    results = store.search(test_query, k=2)
    print(f"   查询: {test_query}")
    for r in results:
        print(f"   → [{r['metadata']['title']}] ({r['score']:.4f})")

    print("\n🎉 知识库就绪！运行 python app.py 启动服务")


if __name__ == "__main__":
    main()
