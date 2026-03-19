"""
AI 微服务入口

启动方式:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import article, event, chat, search
from app.api import indexing
from app.api import agent as agent_api
from app.config import settings
import os
from dotenv import load_dotenv

# 强制加载环境变量并注入 LangSmith 以保证 Tracing 生效
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "ai-service")
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
from prometheus_fastapi_instrumentator import Instrumentator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化 RAG 引擎
    print("正在初始化 RAG 引擎...")
    try:
        from app.core.rag_engine import RAGEngine
        app.state.rag_engine = RAGEngine()
        status = app.state.rag_engine.get_status()
        print(f"RAG 引擎初始化完成 - 文章: {status['articles_count']} 条, 活动: {status['events_count']} 条")
    except Exception as e:
        print(f"RAG 引擎初始化失败: {e}")
        print("语义搜索和索引功能将不可用，其他功能正常运行")
        app.state.rag_engine = None

    yield

    # 关闭时清理
    print("AI 微服务已关闭")


app = FastAPI(
    title="AI 微服务",
    description="为 UniApp 新闻资讯应用提供 AI 能力：文章解读、活动助手、语义搜索、智能对话、Adaptive Agent",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(article.router, prefix="/api/ai/article", tags=["文章 AI"])
app.include_router(event.router, prefix="/api/ai/event", tags=["活动 AI"])
app.include_router(chat.router, prefix="/api/ai/chat", tags=["智能对话"])
app.include_router(search.router, prefix="/api/ai/search", tags=["语义搜索"])
app.include_router(indexing.router, prefix="/api/ai/index", tags=["索引管理"])
app.include_router(agent_api.router, tags=["Adaptive Agent"])

# 挂载 Prometheus 监控指标 (将会在 /metrics 暴露数据)
Instrumentator().instrument(app).expose(app)


@app.get("/", tags=["健康检查"])
async def root():
    """服务健康检查"""
    rag_status = "unavailable"
    if hasattr(app.state, "rag_engine") and app.state.rag_engine:
        rag_status = "ready"

    return {
        "service": "AI 微服务",
        "version": "3.0.0",
        "status": "running",
        "model": settings.DASHSCOPE_MODEL,
        "rag": rag_status,
        "agent": "adaptive-engine-v1",
    }


@app.get("/health", tags=["健康检查"])
async def health():
    """服务健康状态"""
    return {"status": "ok"}
