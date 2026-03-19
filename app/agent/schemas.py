"""
Agent 数据模型 — 定义 Agent 运行过程中的所有结构化数据
"""

from pydantic import BaseModel, Field
from typing import Any


# ============ 执行追踪 ============

class TraceStep(BaseModel):
    """单步执行追踪记录"""
    step: int = 0
    node: str = ""                          # 节点名称 (plan/react/evaluate/reflect)
    thought: str = ""                       # LLM 的思考过程
    action: str = ""                        # 工具调用名称
    action_args: dict[str, Any] = {}        # 工具调用参数
    observation: str = ""                   # 工具返回结果 / 观察
    tokens: int = 0                         # 本步 token 消耗
    time_ms: int = 0                        # 本步耗时 (ms)


class ReactStep(BaseModel):
    """ReAct 推理步骤"""
    thought: str = ""
    action: str = ""
    action_args: dict[str, Any] = {}
    observation: str = ""


# ============ 质量评估 ============

class EvaluationResult(BaseModel):
    """Agent 输出质量评估结果"""
    relevance: float = Field(0.0, ge=0, le=1, description="答案相关性")
    completeness: float = Field(0.0, ge=0, le=1, description="信息完整度")
    tool_accuracy: float = Field(0.0, ge=0, le=1, description="工具使用合理性")
    confidence: float = Field(0.0, ge=0, le=1, description="综合置信度")
    feedback: str = Field("", description="改进建议")


# ============ Skill 元数据 ============

class SkillMeta(BaseModel):
    """Agent Skill 元数据"""
    name: str                                # Skill 名称
    description: str = ""                    # Skill 描述
    trigger_keywords: list[str] = []         # 触发关键词
    recommended_tools: list[str] = []        # 推荐工具列表
    file_path: str = ""                      # SKILL.md 文件路径


# ============ Agent 最终结果 ============

class AgentResult(BaseModel):
    """Agent 执行最终结果"""
    answer: str = ""                         # 最终回答
    active_skill: str = ""                   # 使用的 Skill
    trace: list[TraceStep] = []              # 执行追踪
    evaluation: EvaluationResult | None = None  # 质量评估
    reflections: list[str] = []              # 反思记录
    attempts: int = 1                        # 尝试次数
    tools_used: list[str] = []               # 使用的工具列表
    total_tokens: int = 0                    # 总 token 消耗
    total_time_ms: int = 0                   # 总耗时 (ms)
    total_cost: str = ""                     # 估算成本


# ============ API 请求/响应 ============

class AgentChatRequest(BaseModel):
    """Agent 对话请求"""
    question: str
    history: list[dict] = []
    content_type: str = "all"                # 搜索范围: all/article/event


class AgentChatResponse(BaseModel):
    """Agent 对话响应"""
    answer: str
    trace: dict = {}
    eval: dict = {}


class AgentToolInfo(BaseModel):
    """工具信息"""
    name: str
    description: str
    parameters: dict = {}
