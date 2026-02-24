"""
通用聊天接口 - 政策问答助手
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.llm import call_llm, call_llm_with_history
from app.prompts.chat_prompts import POLICY_CHAT_PROMPT, SUGGESTED_QUESTIONS

router = APIRouter()


# ============ 请求/响应模型 ============

class ChatMessage(BaseModel):
    """对话消息"""
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    question: str
    history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str


class SuggestionsResponse(BaseModel):
    """推荐问题响应"""
    questions: list[str]


# ============ 接口 ============

@router.post("/ask", response_model=ChatResponse, summary="政策问答")
async def chat_ask(req: ChatRequest):
    """回答用户的政策相关问题，支持多轮对话"""
    try:
        if req.history:
            # 多轮对话模式
            messages = [msg.model_dump() for msg in req.history]
            messages.append({"role": "user", "content": req.question})
            result = await call_llm_with_history(
                POLICY_CHAT_PROMPT, messages
            )
        else:
            # 单轮对话模式
            result = await call_llm(POLICY_CHAT_PROMPT, req.question)

        return ChatResponse(answer=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取回答失败: {str(e)}")


@router.get("/suggestions", response_model=SuggestionsResponse, summary="获取推荐问题")
async def get_suggestions():
    """获取推荐问题列表"""
    return SuggestionsResponse(questions=SUGGESTED_QUESTIONS)
