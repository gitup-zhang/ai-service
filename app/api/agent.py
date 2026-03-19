"""
Agent API — 对话、工具、Skills、记忆的 HTTP 接口
"""

from fastapi import APIRouter
from app.agent.schemas import AgentChatRequest, AgentChatResponse

router = APIRouter(prefix="/api/ai/agent", tags=["Agent"])

# Agent 引擎 (延迟初始化, 在 main.py 中赋值)
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        from app.agent.engine import AdaptiveAgentEngine
        _engine = AdaptiveAgentEngine()
    return _engine


@router.post("/chat", summary="Agent 对话")
async def agent_chat(req: AgentChatRequest):
    """
    Agent 智能对话

    - 自动加载最匹配的 Skill
    - ReAct 循环执行工具调用
    - 质量评估 + Reflexion 反思闭环
    - 返回完整 trace 和评估结果
    """
    engine = get_engine()
    result = await engine.run(
        question=req.question,
        history=req.history,
    )

    return {
        "answer": result.answer,
        "active_skill": result.active_skill,
        "trace": {
            "steps": [s.model_dump() for s in result.trace],
            "evaluation": result.evaluation.model_dump() if result.evaluation else None,
            "reflections": result.reflections,
            "attempts": result.attempts,
            "tools_used": result.tools_used,
            "total_tokens": result.total_tokens,
            "total_time_ms": result.total_time_ms,
        },
    }


@router.get("/tools", summary="可用工具列表")
async def list_tools():
    """列出 Agent 可以使用的所有工具"""
    engine = get_engine()
    return {"tools": engine.get_tools_info()}


@router.get("/skills", summary="可用 Skills 列表")
async def list_skills():
    """列出 Agent 可以使用的所有领域 Skills"""
    engine = get_engine()
    return {"skills": engine.get_skills_info()}


@router.get("/memory/status", summary="情景记忆状态")
async def memory_status():
    """查看 Agent 情景记忆的状态"""
    engine = get_engine()
    return {
        "memory_count": engine.get_memory_count(),
    }


@router.delete("/memory", summary="清空情景记忆")
async def clear_memory():
    """清空 Agent 的所有情景记忆"""
    engine = get_engine()
    count = engine.memory.clear()
    return {"cleared": count}
