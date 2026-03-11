"""
语义搜索接口 - RAG 驱动
"""

from fastapi import APIRouter, HTTPException, Request
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
async def semantic_search(req: SearchRequest, request: Request):
    """
    语义搜索 - 基于 RAG 的智能检索

    根据用户问题在向量库中检索相关文章和活动，并由 AI 生成回答。
    """
    try:
        rag = request.app.state.rag_engine
        result = rag.query(
            question=req.query,
            content_type=req.content_type,
            top_k=req.top_k,
        )
        return SearchResponse(
            results=[SearchResult(**r) for r in result["results"]],
            answer=result["answer"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
