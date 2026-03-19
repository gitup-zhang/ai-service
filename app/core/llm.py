"""
LLM 调用层 - 通义千问 (DashScope) 封装
"""

from dashscope import Generation
from app.config import settings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from langsmith import traceable


def _fallback_msg(retry_state):
    print(f"LLM 调用失败，触发降级兜底方案。错误详情: {retry_state.outcome.exception()}")
    return "抱歉，由于模型服务器当前响应异常，暂时无法处理您的请求。请稍后再试或换一个问题表达。"

@traceable(name="call_llm_qwen", run_type="llm")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    retry_error_callback=_fallback_msg
)
async def call_llm(
    prompt: str,
    content: str,
    max_tokens: int = 1500,
    temperature: float = 0.7,
) -> str:
    """
    调用通义千问生成文本

    Args:
        prompt: 系统提示词
        content: 用户输入内容
        max_tokens: 最大生成 token 数
        temperature: 温度参数

    Returns:
        AI 生成的文本
    """
    response = Generation.call(
        api_key=settings.DASHSCOPE_API_KEY,
        model=settings.DASHSCOPE_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
        result_format="message",
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(
            f"LLM 调用失败: {response.code} - {response.message}"
        )


@traceable(name="call_llm_with_history_qwen", run_type="llm")
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    retry_error_callback=_fallback_msg
)
async def call_llm_with_history(
    system_prompt: str,
    messages: list[dict],
    max_tokens: int = 1500,
    temperature: float = 0.7,
) -> str:
    """
    带对话历史的 LLM 调用（用于多轮对话）

    Args:
        system_prompt: 系统提示词
        messages: 对话历史 [{"role": "user"|"assistant", "content": "..."}]
        max_tokens: 最大生成 token 数
        temperature: 温度参数

    Returns:
        AI 生成的文本
    """
    full_messages = [{"role": "system", "content": system_prompt}]
    full_messages.extend(messages)

    response = Generation.call(
        api_key=settings.DASHSCOPE_API_KEY,
        model=settings.DASHSCOPE_MODEL,
        messages=full_messages,
        result_format="message",
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(
            f"LLM 调用失败: {response.code} - {response.message}"
        )
