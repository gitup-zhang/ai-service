"""
索引管理 API - 供管理后台推送数据到向量库

使用方式：管理后台在创建/更新/删除文章或活动时，调用对应接口将数据写入或删除向量库。
"""

import re
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


# ============ 请求/响应模型 ============

class ArticleIndexRequest(BaseModel):
    """文章索引请求"""
    article_id: int
    article_title: str
    article_content: str = ""  # HTML 格式的文章正文
    brief_content: str = ""
    article_type: str = ""
    field_name: str = ""


class EventIndexRequest(BaseModel):
    """活动索引请求"""
    id: int
    title: str
    detail: str = ""  # HTML 格式的活动详情
    event_address: str = ""
    event_start_time: str = ""
    event_end_time: str = ""
    registration_fee: float | str = ""


class BatchIndexRequest(BaseModel):
    """批量索引请求"""
    articles: list[ArticleIndexRequest] = []
    events: list[EventIndexRequest] = []


class IndexResponse(BaseModel):
    """索引操作响应"""
    success: bool
    message: str
    count: int = 0


class IndexStatusResponse(BaseModel):
    """向量库状态响应"""
    articles_count: int
    events_count: int
    chroma_db_path: str


# ============ 文章索引接口 ============

@router.post("/article", response_model=IndexResponse, summary="写入/更新文章")
async def index_article(req: ArticleIndexRequest, request: Request):
    """将文章写入向量库（如已存在则更新）"""
    try:
        rag = request.app.state.rag_engine
        rag.add_article(req.model_dump())
        return IndexResponse(
            success=True,
            message=f"文章 [{req.article_title}] 已写入向量库",
            count=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"写入文章失败: {str(e)}")


@router.delete("/article/{article_id}", response_model=IndexResponse, summary="删除文章")
async def delete_article(article_id: str, request: Request):
    """从向量库中删除文章"""
    try:
        rag = request.app.state.rag_engine
        rag.delete_article(article_id)
        return IndexResponse(
            success=True,
            message=f"文章 {article_id} 已从向量库删除",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文章失败: {str(e)}")


# ============ 活动索引接口 ============

@router.post("/event", response_model=IndexResponse, summary="写入/更新活动")
async def index_event(req: EventIndexRequest, request: Request):
    """将活动写入向量库（如已存在则更新）"""
    try:
        rag = request.app.state.rag_engine
        rag.add_event(req.model_dump())
        return IndexResponse(
            success=True,
            message=f"活动 [{req.title}] 已写入向量库",
            count=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"写入活动失败: {str(e)}")


@router.delete("/event/{event_id}", response_model=IndexResponse, summary="删除活动")
async def delete_event(event_id: str, request: Request):
    """从向量库中删除活动"""
    try:
        rag = request.app.state.rag_engine
        rag.delete_event(event_id)
        return IndexResponse(
            success=True,
            message=f"活动 {event_id} 已从向量库删除",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除活动失败: {str(e)}")


# ============ 批量索引接口 ============

@router.post("/batch", response_model=IndexResponse, summary="批量写入")
async def batch_index(req: BatchIndexRequest, request: Request):
    """批量将文章和活动写入向量库（适用于首次初始化）"""
    try:
        rag = request.app.state.rag_engine
        count = 0

        for article in req.articles:
            rag.add_article(article.model_dump())
            count += 1

        for event in req.events:
            rag.add_event(event.model_dump())
            count += 1

        return IndexResponse(
            success=True,
            message=f"批量写入完成：{len(req.articles)} 篇文章，{len(req.events)} 个活动",
            count=count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量写入失败: {str(e)}")


# ============ 状态查询 ============

@router.get("/status", response_model=IndexStatusResponse, summary="向量库状态")
async def index_status(request: Request):
    """查看向量库当前状态"""
    try:
        rag = request.app.state.rag_engine
        status = rag.get_status()
        return IndexStatusResponse(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


# ============ 列表查询 ============

@router.get("/articles", summary="已索引文章列表")
async def list_articles(request: Request):
    """列出向量库中所有已索引的文章"""
    try:
        rag = request.app.state.rag_engine
        items = rag.list_articles()
        return {"items": items, "total": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取文章列表失败: {str(e)}")


@router.get("/events", summary="已索引活动列表")
async def list_events(request: Request):
    """列出向量库中所有已索引的活动"""
    try:
        rag = request.app.state.rag_engine
        items = rag.list_events()
        return {"items": items, "total": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取活动列表失败: {str(e)}")


# ============ 清空操作 ============

@router.delete("/clear/{content_type}", response_model=IndexResponse, summary="清空集合")
async def clear_collection(content_type: str, request: Request):
    """清空某个类型的所有索引（article 或 event）"""
    if content_type not in ("article", "event"):
        raise HTTPException(status_code=400, detail="content_type 必须是 article 或 event")
    try:
        rag = request.app.state.rag_engine
        count = rag.clear_collection(content_type)
        return IndexResponse(
            success=True,
            message=f"已清空 {content_type} 集合，共删除 {count} 条记录",
            count=count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空失败: {str(e)}")

