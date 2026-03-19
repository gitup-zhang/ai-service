"""
Adaptive Agent Engine — LangGraph 状态图定义

核心架构: 在 LangGraph StateGraph 上构建 ReAct + Reflexion 混合引擎

流程:
  START → load_skill → plan → react → [tool_call → tools → react | answer_ready → evaluate]
  evaluate → [pass → finalize | reflect → react | max_attempts → finalize]
  finalize → END

参考论文:
- ReAct (Yao 2023): Thought→Action→Observation 循环推理
- Reflexion (Shinn 2023): Actor+Evaluator+Self-Reflection + 情景记忆
"""

import time
from langgraph.graph import StateGraph, START, END

from app.agent.state import AgentState
from app.agent.schemas import AgentResult, TraceStep
from app.agent.nodes import create_nodes
from app.agent.evaluator import QualityEvaluator
from app.agent.memory import EpisodicMemory
from app.agent.skill_loader import SkillLoader

from app.tools.event_tools import search_events, get_event_detail
from app.tools.article_tools import search_articles, get_article_detail
from app.tools.rag_tools import semantic_search


class AdaptiveAgentEngine:
    """
    自适应 Agent 引擎

    在 LangGraph 框架上融合 ReAct + Reflexion 架构,
    实现 Load Skill → Plan → ReAct → Evaluate → Reflect 闭环.
    """

    def __init__(self):
        # ===== 工具集 =====
        self.tools = [
            search_events,
            get_event_detail,
            search_articles,
            get_article_detail,
            semantic_search,
        ]

        # ===== 组件 =====
        self.skill_loader = SkillLoader()
        self.memory = EpisodicMemory()
        self.evaluator = QualityEvaluator()

        # ===== 构建状态图 =====
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建 Adaptive Agent 的 LangGraph StateGraph"""

        nodes = create_nodes(
            tools=self.tools,
            skill_loader=self.skill_loader,
            memory=self.memory,
            evaluator=self.evaluator,
        )

        graph = StateGraph(AgentState)

        # ========== 注册节点 ==========
        graph.add_node("load_skill", nodes["load_skill"])   # 0. 加载 Skill
        graph.add_node("plan", nodes["plan"])                 # 1. 规划
        graph.add_node("react", nodes["react"])               # 2. ReAct 推理
        graph.add_node("tools", nodes["tools"])               # 3. 工具执行
        graph.add_node("evaluate", nodes["evaluate"])         # 4. 质量评估
        graph.add_node("reflect", nodes["reflect"])           # 5. 自我反思
        graph.add_node("finalize", nodes["finalize"])         # 6. 整理输出

        # ========== 定义边 ==========
        graph.add_edge(START, "load_skill")
        graph.add_edge("load_skill", "plan")
        graph.add_edge("plan", "react")

        # ReAct 内部循环: LLM → 是否调用工具
        graph.add_conditional_edges("react", nodes["should_use_tool"], {
            "tool_call": "tools",
            "answer_ready": "evaluate",
        })
        graph.add_edge("tools", "react")  # 工具结果 → 回到 ReAct

        # ========== 核心创新: 置信度路由 ==========
        graph.add_conditional_edges("evaluate", nodes["confidence_router"], {
            "pass": "finalize",             # 置信度 ≥ 阈值 → 通过
            "reflect": "reflect",           # 置信度 < 阈值 → 反思
            "max_attempts": "finalize",     # 超过最大尝试次数 → 强制输出
        })
        graph.add_edge("reflect", "react")  # 反思后 → 重新 ReAct

        graph.add_edge("finalize", END)

        return graph.compile()

    async def run(self, question: str, history: list[dict] = None) -> AgentResult:
        """
        执行 Agent

        Args:
            question: 用户问题
            history: 历史对话 [{"role": "user/assistant", "content": "..."}]

        Returns:
            AgentResult 包含答案、trace、评估等完整信息
        """
        start_time = time.time()

        # 构建初始消息
        from langchain_core.messages import HumanMessage, AIMessage
        messages = []
        if history:
            for msg in history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=question))

        # 初始状态
        initial_state = {
            "messages": messages,
            "active_skill": "",
            "skill_context": "",
            "plan": "",
            "similar_experiences": [],
            "evaluation": None,
            "reflection": "",
            "attempt": 0,
            "trace": [],
            "final_answer": "",
        }

        # 执行状态图
        try:
            final_state = await self.graph.ainvoke(initial_state)
        except Exception as e:
            return AgentResult(
                answer=f"Agent 执行异常: {str(e)}",
                attempts=1,
            )

        # 整理结果
        total_time = int((time.time() - start_time) * 1000)
        trace = final_state.get("trace", [])
        evaluation = final_state.get("evaluation")
        tools_used = []
        for step in trace:
            if step.action and step.action not in tools_used:
                tools_used.append(step.action)

        # 收集反思内容
        reflections = []
        for step in trace:
            if step.node == "reflect" and step.thought:
                reflections.append(step.thought)

        return AgentResult(
            answer=final_state.get("final_answer", ""),
            active_skill=final_state.get("active_skill", ""),
            trace=trace,
            evaluation=evaluation,
            reflections=reflections,
            attempts=final_state.get("attempt", 1),
            tools_used=tools_used,
            total_tokens=sum(s.tokens for s in trace),
            total_time_ms=total_time,
        )

    def get_tools_info(self) -> list[dict]:
        """获取所有可用工具信息"""
        return [
            {
                "name": t.name,
                "description": t.description,
            }
            for t in self.tools
        ]

    def get_skills_info(self) -> list[dict]:
        """获取所有可用 Skills 信息"""
        return self.skill_loader.list_skills()

    def get_memory_count(self) -> int:
        """获取情景记忆条数"""
        return self.memory.get_count()
