"""
MAR — Multi-Agent Reflexion

基于论文: MAR: Multi-Agent Reflexion Improves Reasoning Abilities in LLMs (arXiv, 2025.12)

核心架构:
  Actor(Agent 生成回答) → 多角色 Critic(各自独立诊断) → Judge(合成统一反思)

解决的问题:
  单 Agent 自我反思 (Reflexion) 存在 "思维退化" (degeneration-of-thought):
  - 同一个 LLM 既生成又反思 → 确认偏误
  - 反思视角单一 → 重复相同的改进建议
  MAR 通过多角色分离 (acting/diagnosing/critiquing/aggregating) 克服上述问题.

参考: https://arxiv.org/abs/2412.17781
"""

import asyncio
import json
import logging
import re
import time
from typing import Optional

from pydantic import BaseModel, Field
from dashscope import Generation

from app.config import settings

logger = logging.getLogger(__name__)


# ============ 数据模型 ============

class CriticPersona(BaseModel):
    """
    批评家角色定义

    每个角色有独立的视角和评分标准,
    论文指出多样化的推理 persona 能产生更丰富、更少盲点的反馈.
    """
    name: str
    role_description: str
    system_prompt: str
    focus_dimensions: list[str]
    scoring_rubric: str


class CriticReport(BaseModel):
    """
    单个批评家的诊断报告

    包含结构化的问题诊断、具体 issues 列表、改进建议和各维度评分.
    """
    persona_name: str = ""
    diagnosis: str = ""
    specific_issues: list[str] = []
    improvement_suggestions: list[str] = []
    dimension_scores: dict[str, float] = {}
    overall_score: float = 0.5
    raw_response: str = ""


class JudgeVerdict(BaseModel):
    """
    裁判合成结果

    裁判模型综合所有批评家报告, 进行:
    1. 共识识别 — 多个批评家都指出的问题 → 高优先级
    2. 冲突仲裁 — 意见矛盾时基于证据强度决定
    3. 策略合成 — 输出可操作的改进指令
    """
    consensus_issues: list[str] = []
    resolved_conflicts: list[str] = []
    prioritized_actions: list[str] = []
    synthesized_reflection: str = ""
    aggregate_confidence: float = 0.5
    critic_count: int = 0


# ============ 内置批评家角色 ============

BUILTIN_CRITICS = [
    CriticPersona(
        name="FactChecker",
        role_description="事实核查员 — 验证回答中每个事实声明的数据来源",
        system_prompt=(
            "你是一个严谨的事实核查员。你的任务是审查 AI Agent 的回答，"
            "检查以下方面:\n"
            "1. 回答中的每个事实声明是否有工具返回的数据支撑\n"
            "2. 是否存在编造或夸大的信息\n"
            "3. 引用的数据（数字、日期、名称）是否与工具返回一致\n"
            "4. 是否存在「看起来正确但实际无来源」的断言\n\n"
            "你必须逐条审查，标注出所有无数据支撑的声明。"
        ),
        focus_dimensions=["factual_accuracy", "source_grounding", "hallucination_detection"],
        scoring_rubric=(
            "1.0=所有事实均有工具数据支撑; "
            "0.7=大部分有支撑,个别细节未验证; "
            "0.4=多处事实无来源; "
            "0.1=明显编造或严重不准确"
        ),
    ),
    CriticPersona(
        name="CompletenessAuditor",
        role_description="完整性审计员 — 检查回答是否覆盖了问题的各个方面",
        system_prompt=(
            "你是一个完整性审计员。你的任务是对比用户问题和 Agent 回答，"
            "审查以下方面:\n"
            "1. 用户问题的每个子问题/方面是否都有对应回答\n"
            "2. 执行计划中的每个步骤是否都有对应的输出\n"
            "3. 是否有需要但未调用的工具（遗漏的信息源）\n"
            "4. 回答的深度是否足够（是否停留在表面而非深入分析）\n\n"
            "你必须列出所有遗漏的信息点和未充分展开的方面。"
        ),
        focus_dimensions=["information_coverage", "missing_aspects", "tool_utilization"],
        scoring_rubric=(
            "1.0=完整覆盖所有方面且深度足够; "
            "0.7=覆盖大部分方面,个别细节欠缺; "
            "0.4=明显遗漏重要信息; "
            "0.1=回答严重不完整"
        ),
    ),
    CriticPersona(
        name="ClarityReviewer",
        role_description="清晰度评审员 — 评估回答的结构、可读性和用户友好度",
        system_prompt=(
            "你是一个清晰度评审员。你的任务是从终端用户的视角评估 Agent 回答，"
            "审查以下方面:\n"
            "1. 回答的组织结构是否清晰（是否有分段、标题、要点）\n"
            "2. 语言是否通俗易懂（是否避免了不必要的专业术语）\n"
            "3. 是否直接回应了用户的核心疑问（而非答非所问）\n"
            "4. 信息的呈现顺序是否合理（重要信息是否置顶）\n\n"
            "你必须指出所有影响用户理解的表达问题。"
        ),
        focus_dimensions=["structure", "readability", "user_alignment"],
        scoring_rubric=(
            "1.0=结构清晰、语言通俗、完全切题; "
            "0.7=整体清楚,个别表述可优化; "
            "0.4=结构混乱或大量术语; "
            "0.1=答非所问或完全无法理解"
        ),
    ),
]


# ============ Prompt 模板 ============

CRITIC_PROMPT = """请以 {role_description} 的身份，严格审查以下 Agent 对话。

## 诊断上下文

**用户问题**: {question}

**执行计划**: {plan}

**使用的工具**: {tools_used}

**Agent 的回答**:
{answer}

**自动评估结果**: 置信度={confidence}, 反馈={eval_feedback}

## 你的审查标准

{scoring_rubric}

## 输出要求

请严格返回 JSON:
{{
    "diagnosis": "整体问题诊断（2-3句话描述核心问题）",
    "specific_issues": ["具体问题1", "具体问题2", ...],
    "improvement_suggestions": ["改进建议1", "改进建议2", ...],
    "dimension_scores": {{"维度1": 0.8, "维度2": 0.6}},
    "overall_score": 0.7
}}"""


JUDGE_PROMPT = """你是反思合成裁判。以下是 {critic_count} 位不同角色批评家对同一 Agent 回答的独立审查报告。

## 用户原始问题
{question}

## 各批评家报告

{critic_reports}

## 你的任务

1. **共识识别**: 找出多位批评家都指出的共同问题（这些是高优先级问题）
2. **冲突仲裁**: 如果批评家之间有矛盾观点，根据论据的充分性和合理性做出裁决
3. **优先级排序**: 将所有改进建议按对回答质量的影响程度排序
4. **策略合成**: 综合以上分析，生成 2-3 条**具体的、可操作的**改进指令，Agent 可以直接参照执行

请严格返回 JSON:
{{
    "consensus_issues": ["共识问题1", "共识问题2"],
    "resolved_conflicts": ["冲突1的裁决结果"],
    "prioritized_actions": ["最重要的改进1", "次重要的改进2", "其他改进3"],
    "synthesized_reflection": "综合反思策略（3-5句话，Agent 可直接参照改进的指令）",
    "aggregate_confidence": 0.6
}}"""


# ============ 核心组件 ============

class CriticEngine:
    """
    批评执行引擎

    为每个 CriticPersona 构建独立的诊断 prompt, 使用 asyncio.gather
    并发执行所有批评, 然后解析结构化输出.

    论文核心: 将 diagnosing/critiquing 从 acting 中分离,
    通过多样化 persona 减少共同盲点.
    """

    def __init__(self, personas: list[CriticPersona] | None = None):
        self.personas = personas or BUILTIN_CRITICS

    async def criticize(
        self,
        question: str,
        answer: str,
        plan: str = "",
        tools_used: list[str] | None = None,
        confidence: float = 0.5,
        eval_feedback: str = "",
    ) -> list[CriticReport]:
        """
        并发执行所有批评家的诊断

        Args:
            question: 用户问题
            answer: Agent 的回答
            plan: 执行计划
            tools_used: 使用的工具列表
            confidence: 自动评估的置信度
            eval_feedback: 自动评估的反馈

        Returns:
            各批评家的结构化报告列表
        """
        tools_str = ", ".join(tools_used) if tools_used else "无"

        # 为每个 persona 构建独立任务
        tasks = []
        for persona in self.personas:
            prompt = CRITIC_PROMPT.format(
                role_description=persona.role_description,
                question=question,
                plan=plan,
                tools_used=tools_str,
                answer=answer[:1000],  # 截断避免 token 过多
                confidence=f"{confidence:.2f}",
                eval_feedback=eval_feedback,
                scoring_rubric=persona.scoring_rubric,
            )
            tasks.append(self._single_critic(persona, prompt))

        # 并发执行所有批评
        reports = await asyncio.gather(*tasks, return_exceptions=True)

        # 过滤异常结果
        valid_reports = []
        for i, report in enumerate(reports):
            if isinstance(report, Exception):
                logger.error(
                    f"[MAR] Critic '{self.personas[i].name}' 执行异常: {report}"
                )
                # 生成兜底报告
                valid_reports.append(CriticReport(
                    persona_name=self.personas[i].name,
                    diagnosis="批评执行异常，无法完成诊断",
                    overall_score=0.5,
                ))
            else:
                valid_reports.append(report)

        logger.info(
            f"[MAR] CriticEngine: {len(valid_reports)}/{len(self.personas)} "
            f"批评家完成诊断, "
            f"scores={[f'{r.overall_score:.2f}' for r in valid_reports]}"
        )
        return valid_reports

    async def _single_critic(
        self, persona: CriticPersona, prompt: str
    ) -> CriticReport:
        """单个批评家的诊断执行"""
        start = time.time()

        # 使用 asyncio 在线程池中运行同步 SDK
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, self._call_llm_sync, persona.system_prompt, prompt
        )

        report = self._parse_critic_response(raw, persona.name)
        report.raw_response = raw

        elapsed = int((time.time() - start) * 1000)
        logger.debug(
            f"[MAR] Critic '{persona.name}' 完成: "
            f"score={report.overall_score:.2f}, "
            f"issues={len(report.specific_issues)}, "
            f"time={elapsed}ms"
        )
        return report

    def _call_llm_sync(self, system_prompt: str, user_prompt: str) -> str:
        """同步 LLM 调用"""
        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                result_format="message",
                max_tokens=600,
                temperature=0.2,  # 低温度保证评估一致性
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
        except Exception as e:
            logger.error(f"[MAR] Critic LLM 调用失败: {e}")
        return "{}"

    def _parse_critic_response(self, raw: str, persona_name: str) -> CriticReport:
        """解析批评家输出，包含多层兜底"""
        report = CriticReport(persona_name=persona_name)

        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                report.diagnosis = parsed.get("diagnosis", "")
                report.specific_issues = parsed.get("specific_issues", [])
                report.improvement_suggestions = parsed.get("improvement_suggestions", [])
                report.dimension_scores = parsed.get("dimension_scores", {})
                report.overall_score = min(1.0, max(0.0, float(
                    parsed.get("overall_score", 0.5)
                )))
                return report
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[MAR] Critic '{persona_name}' JSON 解析失败: {e}")

        # 兜底: 从原始文本提取关键信息
        report.diagnosis = raw[:200] if raw else "无法解析诊断结果"
        report.overall_score = 0.5
        return report


class ReflectionJudge:
    """
    反思裁判 — 合成多个批评家的报告

    论文核心: 用 Judge 模型进行 aggregating, 将多角色的多样化反馈
    合成为统一的、无矛盾的反思策略.

    合成逻辑:
    1. 共识识别 — 多家都提到的问题 → 高优先级
    2. 冲突仲裁 — 矛盾观点 → 按证据强度裁决
    3. 策略合成 → 2-3 条具体可操作的改进指令
    """

    async def synthesize(
        self,
        question: str,
        critic_reports: list[CriticReport],
    ) -> JudgeVerdict:
        """
        合成所有批评家报告为统一反思

        Args:
            question: 用户原始问题
            critic_reports: 各批评家的结构化报告

        Returns:
            JudgeVerdict 包含合成的反思策略
        """
        if not critic_reports:
            return JudgeVerdict(
                synthesized_reflection="无批评家报告可供合成",
                aggregate_confidence=0.5,
            )

        # 格式化批评家报告
        reports_text_parts = []
        for report in critic_reports:
            issues_str = "\n".join(
                f"  - {issue}" for issue in report.specific_issues
            ) if report.specific_issues else "  (无具体问题)"

            suggestions_str = "\n".join(
                f"  - {sug}" for sug in report.improvement_suggestions
            ) if report.improvement_suggestions else "  (无改进建议)"

            scores_str = ", ".join(
                f"{k}={v:.2f}" for k, v in report.dimension_scores.items()
            ) if report.dimension_scores else "无"

            reports_text_parts.append(
                f"### {report.persona_name} (综合评分: {report.overall_score:.2f})\n"
                f"**诊断**: {report.diagnosis}\n"
                f"**具体问题**:\n{issues_str}\n"
                f"**改进建议**:\n{suggestions_str}\n"
                f"**维度评分**: {scores_str}"
            )

        reports_text = "\n\n".join(reports_text_parts)

        prompt = JUDGE_PROMPT.format(
            critic_count=len(critic_reports),
            question=question,
            critic_reports=reports_text,
        )

        # 调用 Judge LLM
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None, self._call_judge_llm, prompt
        )

        verdict = self._parse_judge_response(raw, len(critic_reports))

        logger.info(
            f"[MAR] Judge 合成完成: "
            f"consensus={len(verdict.consensus_issues)}, "
            f"actions={len(verdict.prioritized_actions)}, "
            f"confidence={verdict.aggregate_confidence:.2f}"
        )
        return verdict

    def _call_judge_llm(self, prompt: str) -> str:
        """调用裁判 LLM"""
        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个公正的反思合成裁判。你的任务是综合多位批评家的意见，"
                            "形成统一的改进策略。只返回 JSON 格式。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=800,
                temperature=0.2,
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
        except Exception as e:
            logger.error(f"[MAR] Judge LLM 调用失败: {e}")
        return "{}"

    def _parse_judge_response(
        self, raw: str, critic_count: int
    ) -> JudgeVerdict:
        """解析裁判输出"""
        verdict = JudgeVerdict(critic_count=critic_count)

        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                verdict.consensus_issues = parsed.get("consensus_issues", [])
                verdict.resolved_conflicts = parsed.get("resolved_conflicts", [])
                verdict.prioritized_actions = parsed.get("prioritized_actions", [])
                verdict.synthesized_reflection = parsed.get(
                    "synthesized_reflection", ""
                )
                verdict.aggregate_confidence = min(1.0, max(0.0, float(
                    parsed.get("aggregate_confidence", 0.5)
                )))
                return verdict
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[MAR] Judge 输出解析失败: {e}")

        # 兜底: 从原文提取
        verdict.synthesized_reflection = raw[:300] if raw else "裁判合成失败"
        verdict.aggregate_confidence = 0.5
        return verdict


class MAROrchestrator:
    """
    MAR 编排器 — 组合 CriticEngine + ReflectionJudge 的完整流程

    完整流程:
    1. 从 AgentState 提取诊断上下文
    2. CriticEngine 并发执行多角色批评
    3. ReflectionJudge 合成统一反思
    4. 返回 JudgeVerdict 供 reflect_node 使用

    降级策略:
    - 如果 MAR 完整流程异常, 降级为原始单 Agent 反思
    """

    def __init__(
        self,
        personas: list[CriticPersona] | None = None,
    ):
        self.critic_engine = CriticEngine(personas=personas)
        self.judge = ReflectionJudge()

    async def reflect(
        self,
        question: str,
        answer: str,
        plan: str = "",
        tools_used: list[str] | None = None,
        confidence: float = 0.5,
        eval_feedback: str = "",
    ) -> JudgeVerdict:
        """
        MAR 完整反思流程

        Args:
            question: 用户问题
            answer: Agent 回答
            plan: 执行计划
            tools_used: 使用的工具
            confidence: 评估置信度
            eval_feedback: 评估反馈

        Returns:
            JudgeVerdict — 合成后的反思策略
        """
        start = time.time()

        try:
            # Phase 1: 多角色并发批评
            critic_reports = await self.critic_engine.criticize(
                question=question,
                answer=answer,
                plan=plan,
                tools_used=tools_used,
                confidence=confidence,
                eval_feedback=eval_feedback,
            )

            # Phase 2: 裁判合成
            verdict = await self.judge.synthesize(
                question=question,
                critic_reports=critic_reports,
            )

            elapsed = int((time.time() - start) * 1000)
            logger.info(
                f"[MAR] 完整流程完成: "
                f"critics={len(critic_reports)}, "
                f"consensus={len(verdict.consensus_issues)}, "
                f"confidence={verdict.aggregate_confidence:.2f}, "
                f"time={elapsed}ms"
            )
            return verdict

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            logger.error(f"[MAR] 流程异常 ({elapsed}ms): {e}")

            # 降级: 返回带错误信息的 verdict
            return JudgeVerdict(
                synthesized_reflection=(
                    f"MAR 多角色反思流程异常，降级为基础反思。"
                    f"建议: 重新审视回答是否充分回应了用户问题，"
                    f"检查是否有遗漏的工具调用。\n"
                    f"(错误: {str(e)[:100]})"
                ),
                aggregate_confidence=0.4,
            )

    def get_critic_names(self) -> list[str]:
        """获取所有批评家角色名"""
        return [p.name for p in self.critic_engine.personas]
