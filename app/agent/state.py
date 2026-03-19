"""
Agent 状态定义 — LangGraph StateGraph 的核心状态
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

from app.agent.schemas import EvaluationResult, TraceStep


class AgentState(TypedDict):
    """
    Adaptive Agent 执行状态

    作为 LangGraph StateGraph 的状态容器,
    每个节点读取和写入这个状态来推进 Agent 执行流程.

    流程: load_skill → plan → react ⇄ tools → evaluate → (reflect → react) → finalize
    """

    # ===== 对话 =====
    messages: Annotated[list, add_messages]       # LangGraph 管理的消息列表

    # ===== Skill =====
    active_skill: str                              # 当前激活的 Skill 名称
    skill_context: str                             # Skill 加载的领域知识/指令

    # ===== 规划 =====
    plan: str                                      # LLM 生成的执行计划
    similar_experiences: list[str]                  # 从情景记忆检索的相似经验

    # ===== 评估与反思 =====
    evaluation: EvaluationResult | None            # 质量评估结果
    reflection: str                                # 反思内容
    attempt: int                                   # 当前尝试次数 (防无限循环)

    # ===== 追踪 =====
    trace: list[TraceStep]                         # 执行链路追踪记录

    # ===== 输出 =====
    final_answer: str                              # 最终答案
