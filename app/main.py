"""
AI 微服务入口

启动方式:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import article, event, chat, search
from app.config import settings

app = FastAPI(
    title="AI 微服务",
    description="为 UniApp 新闻资讯应用提供 AI 能力：文章解读、活动助手、语义搜索、智能对话",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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


@app.get("/", tags=["健康检查"])
async def root():
    """服务健康检查"""
    return {
        "service": "AI 微服务",
        "version": "1.0.0",
        "status": "running",
        "model": settings.DASHSCOPE_MODEL,
    }


@app.get("/health", tags=["健康检查"])
async def health():
    """服务健康状态"""
    return {"status": "ok"}
