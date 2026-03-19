"""
质量评估器 — Agent 输出的多维度质量评估 + 置信度打分

核心创新: 用 LLM-as-Judge 模式对 Agent 回答进行自动评估,
根据置信度决定是否触发 Reflexion 反思循环.
"""

import json
from dashscope import Generation
from app.config import settings
from app.agent.schemas import EvaluationResult


EVAL_PROMPT = """你是一个 AI 回答质量评估专家。请对以下问答进行多维度评分。

用户问题: {question}
Agent 执行计划: {plan}
Agent 使用的工具: {tools_used}
Agent 的回答: {answer}

请从以下三个维度评分 (0.0 ~ 1.0):

1. **relevance** (相关性): 回答是否切题、是否回应了用户的核心需求
2. **completeness** (完整度): 信息是否充分、是否遗漏了重要内容
3. **tool_accuracy** (工具准确率): 使用的工具是否合理、是否有多余或遗漏的工具调用

同时给出一段简短的改进建议 (feedback)。

请严格以 JSON 格式返回:
{{"relevance": 0.0, "completeness": 0.0, "tool_accuracy": 0.0, "feedback": "..."}}

注意:
- 如果回答基本满足需求, 分数应在 0.7 以上
- 只有回答明显有问题时才给低分
- feedback 要具体指出不足之处和改进方向"""


class QualityEvaluator:
    """Agent 输出质量评估器 — 多维度置信度打分"""

    def __init__(self):
        self.weights = {
            "relevance": 0.4,
            "completeness": 0.35,
            "tool_accuracy": 0.25,
        }

    async def evaluate(
        self,
        question: str,
        answer: str,
        tools_used: list[str],
        plan: str,
    ) -> EvaluationResult:
        """
        对 Agent 输出进行多维度评估

        Returns:
            EvaluationResult with confidence score and feedback
        """
        prompt = EVAL_PROMPT.format(
            question=question,
            plan=plan,
            tools_used=", ".join(tools_used) if tools_used else "无",
            answer=answer[:1000],  # 截断避免 token 过多
        )

        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个严谨的评估专家，只返回 JSON 格式。"},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=500,
                temperature=0.1,  # 低温度保证评估稳定性
            )

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                scores = self._parse_scores(content)
                # 计算加权置信度
                confidence = sum(
                    scores.get(k, 0.5) * w
                    for k, w in self.weights.items()
                )
                return EvaluationResult(
                    relevance=scores.get("relevance", 0.5),
                    completeness=scores.get("completeness", 0.5),
                    tool_accuracy=scores.get("tool_accuracy", 0.5),
                    confidence=round(confidence, 4),
                    feedback=scores.get("feedback", ""),
                )
            else:
                # LLM 调用失败, 给默认通过分数
                return EvaluationResult(
                    confidence=0.75,
                    feedback="评估器 LLM 调用失败，已使用默认分数",
                )
        except Exception as e:
            return EvaluationResult(
                confidence=0.75,
                feedback=f"评估异常: {str(e)}",
            )

    def _parse_scores(self, content: str) -> dict:
        """从 LLM 输出中解析评分 JSON"""
        try:
            # 尝试直接解析
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块
        import re
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # 兜底: 返回中等分数
        return {
            "relevance": 0.6,
            "completeness": 0.6,
            "tool_accuracy": 0.6,
            "feedback": "无法解析评估结果",
        }
