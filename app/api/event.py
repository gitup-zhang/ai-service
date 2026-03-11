"""
活动 AI 接口 - 摘要、问答、报名指南
"""

import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.llm import call_llm
from app.prompts.event_prompts import (
    EVENT_SUMMARY_PROMPT,
    EVENT_QA_PROMPT,
    EVENT_GUIDE_PROMPT,
)

router = APIRouter()


# ============ 请求/响应模型 ============

class EventData(BaseModel):
    """活动数据"""
    title: str = ""
    detail: str = ""
    event_address: str = ""
    event_start_time: str = ""
    event_end_time: str = ""
    registration_start_time: str = ""
    registration_end_time: str = ""
    registration_fee: str | float = ""


class EventSummaryRequest(BaseModel):
    """活动摘要请求"""
    event: EventData


class EventQARequest(BaseModel):
    """活动问答请求"""
    question: str
    event: EventData


class EventGuideRequest(BaseModel):
    """报名指南请求"""
    event: EventData


class EventResponse(BaseModel):
    """活动 AI 响应"""
    result: str


# ============ 辅助函数 ============

def format_event_info(event: EventData) -> str:
    """将活动数据格式化为文本"""
    detail_clean = re.sub(r"<[^>]+>", "", event.detail)[:2000]
    return f"""活动名称：{event.title}
活动地点：{event.event_address}
活动时间：{event.event_start_time} 至 {event.event_end_time}
报名时间：{event.registration_start_time} 至 {event.registration_end_time}
报名费用：{event.registration_fee}
活动详情：{detail_clean}"""


# ============ 接口 ============

@router.post("/summary", response_model=EventResponse, summary="生成活动摘要")
async def event_summary(req: EventSummaryRequest):
    """提取活动核心亮点，生成摘要"""
    try:
        event_info = format_event_info(req.event)
        result = await call_llm(EVENT_SUMMARY_PROMPT, event_info)
        return EventResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成活动摘要失败: {str(e)}")


@router.post("/qa", response_model=EventResponse, summary="活动智能问答")
async def event_qa(req: EventQARequest):
    """根据活动信息回答用户问题"""
    try:
        event_info = format_event_info(req.event)
        prompt = EVENT_QA_PROMPT.format(event_info=event_info)
        result = await call_llm(prompt, req.question)
        return EventResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"活动问答失败: {str(e)}")


@router.post("/guide", response_model=EventResponse, summary="生成报名指南")
async def event_guide(req: EventGuideRequest):
    """根据活动信息生成报名参会指南"""
    try:
        event_info = format_event_info(req.event)
        result = await call_llm(EVENT_GUIDE_PROMPT, event_info)
        return EventResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成报名指南失败: {str(e)}")
