"""
文章 AI 接口 - 摘要、解读、影响分析
"""

import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.llm import call_llm
from app.prompts.article_prompts import (
    ARTICLE_SUMMARY_PROMPT,
    ARTICLE_EXPLANATION_PROMPT,
    ARTICLE_IMPACT_PROMPT,
)

router = APIRouter()


class ArticleRequest(BaseModel):
    """文章请求体"""
    content: str


class ArticleResponse(BaseModel):
    """文章响应体"""
    result: str


def clean_html(html_content: str) -> str:
    """清理 HTML 标签，提取纯文本"""
    clean = re.sub(r"<[^>]+>", "", html_content)
    return clean[:3000]  # 限制长度，避免 token 超限


@router.post("/summary", response_model=ArticleResponse, summary="生成文章摘要")
async def article_summary(req: ArticleRequest):
    """提取文章核心要点，生成摘要"""
    try:
        content = clean_html(req.content)
        result = await call_llm(ARTICLE_SUMMARY_PROMPT, content)
        return ArticleResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成摘要失败: {str(e)}")


@router.post("/explanation", response_model=ArticleResponse, summary="通俗解读文章")
async def article_explanation(req: ArticleRequest):
    """将政策/专业文章翻译成通俗易懂的语言"""
    try:
        content = clean_html(req.content)
        result = await call_llm(ARTICLE_EXPLANATION_PROMPT, content)
        return ArticleResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成解读失败: {str(e)}")


@router.post("/impact", response_model=ArticleResponse, summary="分析文章影响")
async def article_impact(req: ArticleRequest):
    """分析政策/新闻对个人、企业、行业的影响"""
    try:
        content = clean_html(req.content)
        result = await call_llm(ARTICLE_IMPACT_PROMPT, content)
        return ArticleResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成分析失败: {str(e)}")
