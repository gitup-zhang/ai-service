"""
语义搜索接口 - RAG 驱动 (Phase 2 实现)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SearchRequest(BaseModel):
    """搜索请求"""
    query: str
    top_k: int = 3
    content_type: str = "all"  # "all" | "event" | "article"


class SearchResult(BaseModel):
    """搜索结果项"""
    title: str
    content: str
    content_type: str
    score: float
    metadata: dict = {}


class SearchResponse(BaseModel):
    """搜索响应"""
    results: list[SearchResult]
    answer: str = ""  # RAG 生成的回答


@router.post("/", response_model=SearchResponse, summary="语义搜索")
async def semantic_search(req: SearchRequest):
    """
    语义搜索 - 基于 RAG 的智能检索

    Phase 2 实现，当前返回占位响应。
    """
    # TODO: Phase 2 - 集成 ChromaDB + LlamaIndex
    return SearchResponse(
        results=[],
        answer="语义搜索功能即将上线，敬请期待！"
    )
