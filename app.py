"""
whaleink RAG — FastAPI + LlamaIndex
"""

import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from embedding import WhaleinkEmbedding

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── 全局状态 ────────────────────────────────────────────

class AppState:
    query_engine = None
    ready = False
    docs_count = 0

state = AppState()


# ─── 启动/关闭 ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载索引"""
    logger.info("正在启动 whaleink RAG (LlamaIndex)...")

    try:
        # 阻止 LlamaIndex 内部校验 OpenAI LLM
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "sk-placeholder-llamaindex"
        
        # 配置 Embedding
        Settings.embed_model = WhaleinkEmbedding()
        
        # 配默认 LLM 避免 LlamaIndex 自动 fallback 到 OpenAI
        from deepseek_llm import DeepSeekLLM
        Settings.llm = DeepSeekLLM(api_key=os.getenv("DEEPSEEK_API_KEY", "sk-placeholder"))

        # 加载已有 ChromaDB 索引
        persist_dir = os.getenv("CHROMA_DB_PATH", "/root/whaleink-rag/chroma_db")
        db = chromadb.PersistentClient(path=persist_dir)

        try:
            chroma_collection = db.get_collection("whaleink_blog")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            index = VectorStoreIndex.from_vector_store(vector_store)
            state.query_engine = index.as_query_engine(
                similarity_top_k=3,
                response_mode="compact",
            )
            state.docs_count = chroma_collection.count()
            state.ready = state.docs_count > 0
            logger.info(f"✅ 加载索引: {state.docs_count} 个文档块")
        except Exception as e:
            logger.warning(f"⚠️  加载索引失败: {e}")

    except Exception as e:
        logger.error(f"启动失败: {e}")

    yield
    logger.info("whaleink RAG 已关闭")


# ─── FastAPI 应用 ────────────────────────────────────────

app = FastAPI(
    title="whaleink RAG API (LlamaIndex)",
    description="whaleink.top 博客 AI 问答助手 — 基于 LlamaIndex",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API 模型 ────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str] = []

class StatusResponse(BaseModel):
    status: str
    docs_count: int
    version: str
    framework: str = "LlamaIndex"


# ─── API 路由 ────────────────────────────────────────────

@app.get("/", tags=["系统"])
async def root():
    return {
        "name": "whaleink RAG API",
        "version": "2.0.0",
        "framework": "LlamaIndex",
        "status": "ready" if state.ready else "degraded",
        "docs_count": state.docs_count,
    }


@app.get("/api/status", response_model=StatusResponse, tags=["系统"])
async def get_status():
    return StatusResponse(
        status="ready" if state.ready else "degraded",
        docs_count=state.docs_count,
        version="2.0.0",
    )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    if not state.ready or not state.query_engine:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪，请先运行 python ingest.py",
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    try:
        from deepseek_llm import DeepSeekLLM
        
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key:
            Settings.llm = DeepSeekLLM(
                api_key=deepseek_key,
            )
        
        response = state.query_engine.query(req.question)

        # 提取来源
        sources = []
        if hasattr(response, "source_nodes") and response.source_nodes:
            for sn in response.source_nodes:
                meta = sn.node.metadata
                title = meta.get("title", "")
                if title and title not in sources:
                    sources.append(title)

        return QueryResponse(
            answer=str(response),
            sources=sources,
        )
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/api/reload", tags=["管理"])
async def reload_knowledge():
    """重新加载知识库（需要先运行 ingest.py）"""
    try:
        persist_dir = os.getenv("CHROMA_DB_PATH", "/root/whaleink-rag/chroma_db")
        db = chromadb.PersistentClient(path=persist_dir)
        chroma_collection = db.get_collection("whaleink_blog")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        state.query_engine = index.as_query_engine(similarity_top_k=3)
        state.docs_count = chroma_collection.count()
        state.ready = state.docs_count > 0
        return {
            "status": "reloaded",
            "docs_count": state.docs_count,
            "framework": "LlamaIndex",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 静态文件 ────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ─── 启动 ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
