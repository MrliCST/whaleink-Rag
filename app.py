"""
whaleink RAG — FastAPI 服务
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

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── 全局状态 ────────────────────────────────────────────

class AppState:
    vector_store = None
    embedding_fn = None
    ready = False


state = AppState()


# ─── 启动/关闭 ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载向量库"""
    logger.info("正在启动 whaleink RAG...")

    try:
        from rag_pipeline import EmbeddingFunction, VectorStore

        logger.info("加载 Embedding...")
        state.embedding_fn = EmbeddingFunction()

        logger.info("连接 ChromaDB...")
        state.vector_store = VectorStore(embedding_fn=state.embedding_fn)

        count = state.vector_store.count
        logger.info(f"向量库状态: {count} 个文档块")

        if count == 0:
            logger.warning("⚠️  向量库为空！请先运行: python ingest.py")
        else:
            state.ready = True
            logger.info("✅ whaleink RAG 就绪")

    except Exception as e:
        logger.error(f"启动失败: {e}")
        logger.warning("服务将以降级模式运行（仅返回错误信息）")

    yield

    logger.info("whaleink RAG 已关闭")


# ─── FastAPI 应用 ────────────────────────────────────────

app = FastAPI(
    title="whaleink RAG API",
    description="whaleink.top 博客 AI 问答助手",
    version="1.0.0",
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


# ─── API 路由 ────────────────────────────────────────────

@app.get("/", tags=["系统"])
async def root():
    """欢迎页"""
    return {
        "name": "whaleink RAG API",
        "status": "ready" if state.ready else "degraded",
        "docs_count": state.vector_store.count if state.vector_store else 0,
    }


@app.get("/api/status", response_model=StatusResponse, tags=["系统"])
async def get_status():
    """获取服务状态"""
    return StatusResponse(
        status="ready" if state.ready else "degraded",
        docs_count=state.vector_store.count if state.vector_store else 0,
        version="1.0.0",
    )


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    """RAG 问答"""
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="知识库未就绪，请先运行 python ingest.py 导入数据",
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")

    from rag_pipeline import query_rag

    try:
        result = query_rag(req.question, state.vector_store)
        return QueryResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/api/reload", tags=["管理"])
async def reload_knowledge():
    """重新加载知识库（需要先运行 ingest.py）"""
    from rag_pipeline import EmbeddingFunction, VectorStore

    try:
        state.embedding_fn = EmbeddingFunction()
        state.vector_store = VectorStore(embedding_fn=state.embedding_fn)
        state.ready = state.vector_store.count > 0
        return {
            "status": "reloaded",
            "docs_count": state.vector_store.count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── 静态文件（前端） ──────────────────────────────────────

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")


# ─── 启动 ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
