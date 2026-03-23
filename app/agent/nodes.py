"""
Agent 节点实现 — LangGraph 状态图中的各个处理节点

流程: load_skill → plan → react ⇄ tools → evaluate → (reflect → react) → finalize

核心创新:
- load_skill_node: 按需加载领域 Skill
- evaluate_node: 多维度质量评估
- reflect_node: Reflexion 自我反思
- confidence_router: 置信度路由决策
"""

import time
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from dashscope import Generation

from app.config import settings
from app.agent.state import AgentState
from app.agent.schemas import TraceStep, EvaluationResult
from app.agent.evaluator import QualityEvaluator
from app.agent.memory import EpisodicMemory
from app.agent.skill_loader import SkillLoader

# ============ 常量 ============
MAX_ATTEMPTS = 3
CONFIDENCE_THRESHOLD = 0.7


# ============ Prompt 模板 ============

SYSTEM_PROMPT = """你是智源资讯平台的 AI 助手。你可以使用工具来搜索文章、活动并回答用户问题。

{skill_context}

当前执行计划: {plan}

{reflection_hint}

要求:
- 需要信息时主动使用工具获取
- 基于工具返回的真实数据回答, 不要编造
- 回答要有条理, 使用 emoji 和格式化
- 如果工具返回为空, 如实告知用户"""

PLAN_PROMPT = """分析以下用户问题，制定一个简短的执行计划。

用户问题: {question}

{skill_hint}

{memory_hint}

请直接给出 2-3 步的简要计划（中文，一行一步）:"""

REFLECT_PROMPT = """你是一个 AI 反思分析专家。请分析以下对话中 Agent 回答的不足之处。

用户问题: {question}
Agent 的回答: {answer}
评估结果: 置信度={confidence}, 反馈={feedback}

请简要分析:
1. 回答的主要不足是什么？
2. 应该如何改进？

给出 2-3 句话的改进策略:"""


# ============ 辅助函数 ============

def _call_llm(messages: list[dict], max_tokens: int = 1000, temperature: float = 0.7) -> str:
    """调用 DashScope LLM"""
    response = Generation.call(
        api_key=settings.DASHSCOPE_API_KEY,
        model=settings.DASHSCOPE_MODEL,
        messages=messages,
        result_format="message",
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content
    return ""


def _call_llm_with_tools(messages: list[dict], tools_schema: list[dict]) -> dict:
    """调用 DashScope LLM (带工具调用能力)"""
    response = Generation.call(
        api_key=settings.DASHSCOPE_API_KEY,
        model=settings.DASHSCOPE_MODEL,
        messages=messages,
        tools=tools_schema,
        result_format="message",
        max_tokens=2000,
        temperature=0.7,
    )
    if response.status_code == 200:
        return response.output.choices[0].message
    return {"role": "assistant", "content": "抱歉，处理时出现异常。"}


def _extract_tools_used(state: AgentState) -> list[str]:
    """从 AgentState 消息中提取所有已调用工具的名称"""
    tools_used = set()
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in (msg.tool_calls or []):
                tools_used.add(tc.get("name", "unknown"))
    return list(tools_used)

def _get_tool_schemas(tools) -> list[dict]:
    """将 LangChain tool 转为 DashScope 工具 schema"""
    schemas = []
    for t in tools:
        schema = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.args_schema.model_json_schema() if hasattr(t, 'args_schema') and t.args_schema else {
                    "type": "object",
                    "properties": {},
                },
            },
        }
        schemas.append(schema)
    return schemas


def _extract_last_answer(state: AgentState) -> str:
    """提取最后一条 AI 消息内容"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return ""


def _extract_tools_used(state: AgentState) -> list[str]:
    """提取所有使用过的工具名"""
    tools = []
    for msg in state["messages"]:
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") and tc["name"] not in tools:
                    tools.append(tc["name"])
    return tools


# ============ 节点工厂 ============

def create_nodes(tools: list, skill_loader: SkillLoader, memory: EpisodicMemory, evaluator: QualityEvaluator):
    """
    创建并返回所有节点函数

    使用工厂模式, 将依赖注入到闭包中, 避免全局状态
    """

    tool_schemas = _get_tool_schemas(tools)
    tool_map = {t.name: t for t in tools}

    # ===== 0. Skill 加载节点 =====
    async def load_skill_node(state: AgentState) -> dict:
        """根据用户意图, 动态选择并加载最匹配的 Skill"""
        start = time.time()

        # 获取用户最新消息
        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        # 选择 Skill
        skill = skill_loader.select(user_query)
        skill_context = skill_loader.load(skill.name) if skill.name != "general" else ""

        trace_step = TraceStep(
            step=1,
            node="load_skill",
            thought=f"用户意图分析 → 加载 {skill.name} Skill",
            time_ms=int((time.time() - start) * 1000),
        )

        return {
            "active_skill": skill.name,
            "skill_context": skill_context,
            "trace": [trace_step],
        }

    # ===== 1. 规划节点 =====
    async def plan_node(state: AgentState) -> dict:
        """基于 Skill 知识 + 历史经验, 制定执行计划"""
        start = time.time()

        user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        # 检索相似历史经验
        similar = memory.search(user_query) if memory else []

        # 构建规划 prompt
        skill_hint = ""
        if state.get("skill_context"):
            skill_hint = f"可用的领域知识:\n{state['skill_context'][:500]}"

        memory_hint = ""
        if similar:
            memory_hint = f"参考历史经验:\n" + "\n".join(f"- {s[:200]}" for s in similar[:2])

        plan = _call_llm(
            [{"role": "user", "content": PLAN_PROMPT.format(
                question=user_query,
                skill_hint=skill_hint,
                memory_hint=memory_hint,
            )}],
            max_tokens=300,
            temperature=0.3,
        )

        trace_step = TraceStep(
            step=len(state.get("trace", [])) + 1,
            node="plan",
            thought=f"制定执行计划:\n{plan}",
            time_ms=int((time.time() - start) * 1000),
        )

        return {
            "plan": plan,
            "similar_experiences": similar,
            "attempt": state.get("attempt", 0) + 1 if state.get("reflection") else 1,
            "trace": state.get("trace", []) + [trace_step],
        }

    # ===== 2. ReAct 节点 =====
    async def react_node(state: AgentState) -> dict:
        """ReAct 推理 — LLM 根据计划和上下文决定下一步"""
        start = time.time()

        # 构建 system prompt
        reflection_hint = ""
        if state.get("reflection"):
            reflection_hint = f"⚠️ 前一次回答质量不足, 请参考以下反思改进:\n{state['reflection']}"

        system_content = SYSTEM_PROMPT.format(
            skill_context=state.get("skill_context", ""),
            plan=state.get("plan", ""),
            reflection_hint=reflection_hint,
        )

        # 将 LangGraph messages 转为 DashScope 格式
        dash_messages = [{"role": "system", "content": system_content}]
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                dash_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                entry = {"role": "assistant"}
                if msg.content:
                    entry["content"] = msg.content
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    if not entry.get("content"):
                        entry["content"] = ""
                dash_messages.append(entry)
            elif isinstance(msg, ToolMessage):
                dash_messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id if hasattr(msg, 'tool_call_id') else "",
                })

        # 调用 LLM (带工具)
        result = _call_llm_with_tools(dash_messages, tool_schemas)

        # 构建 AIMessage
        tool_calls = []
        if hasattr(result, 'tool_calls') and result.tool_calls:
            for tc in result.tool_calls:
                tool_calls.append({
                    "name": tc.get("function", {}).get("name", ""),
                    "args": json.loads(tc.get("function", {}).get("arguments", "{}")),
                    "id": tc.get("id", ""),
                })

        content = getattr(result, 'content', '') or ""
        ai_msg = AIMessage(content=content, tool_calls=tool_calls)

        # 记录 trace
        trace_step = TraceStep(
            step=len(state.get("trace", [])) + 1,
            node="react",
            thought=content[:200] if content else "(准备调用工具)",
            action=tool_calls[0]["name"] if tool_calls else "",
            action_args=tool_calls[0]["args"] if tool_calls else {},
            time_ms=int((time.time() - start) * 1000),
        )

        return {
            "messages": [ai_msg],
            "trace": state.get("trace", []) + [trace_step],
        }

    # ===== 3. 工具节点 =====
    async def tool_node(state: AgentState) -> dict:
        """执行工具调用, 返回结果"""
        start = time.time()

        # 获取最新的 AI 消息中的工具调用
        last_msg = state["messages"][-1]
        tool_messages = []

        if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")

                if tool_name in tool_map:
                    try:
                        result = tool_map[tool_name].invoke(tool_args)
                    except Exception as e:
                        result = f"工具执行出错: {str(e)}"
                else:
                    result = f"未知工具: {tool_name}"

                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id)
                )

                # 记录 trace
                trace_step = TraceStep(
                    step=len(state.get("trace", [])) + 1,
                    node="tools",
                    action=tool_name,
                    action_args=tool_args,
                    observation=str(result)[:300],
                    time_ms=int((time.time() - start) * 1000),
                )
                state_trace = state.get("trace", []) + [trace_step]

        return {
            "messages": tool_messages,
            "trace": state_trace if tool_messages else state.get("trace", []),
        }

    # ===== 4. 评估节点 (自研核心) =====
    async def evaluate_node(state: AgentState) -> dict:
        """多维度质量评估 + 置信度打分"""
        start = time.time()

        answer = _extract_last_answer(state)
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        evaluation = await evaluator.evaluate(
            question=question,
            answer=answer,
            tools_used=_extract_tools_used(state),
            plan=state.get("plan", ""),
        )

        trace_step = TraceStep(
            step=len(state.get("trace", [])) + 1,
            node="evaluate",
            thought=f"质量评估: 置信度={evaluation.confidence:.2f}, "
                    f"相关性={evaluation.relevance:.2f}, "
                    f"完整度={evaluation.completeness:.2f}",
            time_ms=int((time.time() - start) * 1000),
        )

        return {
            "evaluation": evaluation,
            "trace": state.get("trace", []) + [trace_step],
        }

    # ===== 5. 反思节点 (MAR: Multi-Agent Reflexion) =====
    async def reflect_node(state: AgentState) -> dict:
        """
        多智体反思 — 基于 MAR (Multi-Agent Reflexion, 2025.12) 论文

        将原始的单 Agent 自我反思升级为:
        1. 多角色 Critic 并发诊断 (FactChecker + CompletenessAuditor + ClarityReviewer)
        2. Judge 裁判合成统一反思策略
        3. 存入情景记忆

        降级策略: MAR 流程异常时退回到单 LLM 反思
        """
        start = time.time()

        answer = _extract_last_answer(state)
        question = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                question = msg.content
                break

        eval_result = state.get("evaluation") or EvaluationResult()
        tools_used = _extract_tools_used(state)

        # ---- MAR 多角色反思 ----
        reflection = ""
        reflection_detail = {}
        try:
            from app.agent.multi_agent_reflect import MAROrchestrator

            mar = MAROrchestrator()
            verdict = await mar.reflect(
                question=question,
                answer=answer[:800],
                plan=state.get("plan", ""),
                tools_used=tools_used,
                confidence=eval_result.confidence,
                eval_feedback=eval_result.feedback,
            )

            reflection = verdict.synthesized_reflection

            # 保存详细信息供 trace 和 API 返回
            reflection_detail = {
                "method": "MAR",
                "critic_count": verdict.critic_count,
                "consensus_issues": verdict.consensus_issues,
                "prioritized_actions": verdict.prioritized_actions,
                "aggregate_confidence": verdict.aggregate_confidence,
            }

        except Exception as e:
            # ---- 降级: 单 LLM 反思 ----
            reflection = _call_llm(
                [{"role": "user", "content": REFLECT_PROMPT.format(
                    question=question,
                    answer=answer[:500],
                    confidence=eval_result.confidence,
                    feedback=eval_result.feedback,
                )}],
                max_tokens=300,
                temperature=0.3,
            )
            reflection_detail = {"method": "single_agent_fallback", "error": str(e)[:100]}

        # 存入情景记忆
        if memory:
            memory.store(
                question=question,
                reflection=reflection,
                evaluation_feedback=eval_result.feedback,
                skill_used=state.get("active_skill", ""),
            )

        # 构建 trace
        thought_summary = f"MAR 多角色反思:\n{reflection}"
        if reflection_detail.get("consensus_issues"):
            thought_summary += (
                f"\n共识问题: {', '.join(reflection_detail['consensus_issues'][:3])}"
            )

        trace_step = TraceStep(
            step=len(state.get("trace", [])) + 1,
            node="reflect",
            thought=thought_summary,
            time_ms=int((time.time() - start) * 1000),
        )

        return {
            "reflection": reflection,
            "attempt": state.get("attempt", 1) + 1,
            "trace": state.get("trace", []) + [trace_step],
        }

    # ===== 6. 输出整理节点 =====
    async def finalize_node(state: AgentState) -> dict:
        """整理最终输出"""
        return {
            "final_answer": _extract_last_answer(state),
        }

    # ===== 路由函数 =====

    def should_use_tool(state: AgentState) -> str:
        """ReAct 路由: 判断是否需要调用工具"""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
            return "tool_call"
        return "answer_ready"

    def confidence_router(state: AgentState) -> str:
        """置信度路由: 根据评估结果决定下一步"""
        attempt = state.get("attempt", 1)
        if attempt >= MAX_ATTEMPTS:
            return "max_attempts"

        eval_result = state.get("evaluation")
        if eval_result and eval_result.confidence >= CONFIDENCE_THRESHOLD:
            return "pass"

        return "reflect"

    return {
        "load_skill": load_skill_node,
        "plan": plan_node,
        "react": react_node,
        "tools": tool_node,
        "evaluate": evaluate_node,
        "reflect": reflect_node,
        "finalize": finalize_node,
        "should_use_tool": should_use_tool,
        "confidence_router": confidence_router,
    }
