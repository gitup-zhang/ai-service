"""
RAG 语义搜索工具 — 封装 RAG 引擎为 Agent 可调用的工具 (Agentic RAG)

Agent 自主决定是否需要语义检索, 而非固定流程触发.
"""

from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

def _rag_fallback_msg(retry_state):
    err = retry_state.outcome.exception()
    print(f"[RAG 工具] 向量检索发生错误: {err}")
    return f"[系统提示] 向量检索服务当前不可用 (错误信息: {str(err)})，请勿再次尝试检索知识库，径直根据用户的上下文或自身知识给出答复并安抚用户。"


def _get_rag_engine():
    """延迟获取 RAG 引擎实例 (避免循环导入)"""
    from app.main import app
    if hasattr(app.state, "rag_engine") and app.state.rag_engine:
        return app.state.rag_engine
    return None


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=3),
    retry=retry_if_exception_type(Exception),
    retry_error_callback=_rag_fallback_msg
)
def semantic_search(query: str, content_type: str = "all") -> str:
    """对知识库进行语义搜索，返回最相关的文档内容。
    content_type 可选: all(全部), article(文章), event(活动)"""
    engine = _get_rag_engine()
    if not engine:
        return "RAG 引擎未初始化，语义搜索不可用"

    try:
        result = engine.query(question=query, content_type=content_type, top_k=3)
        search_results = result.get("results", [])
        if not search_results:
            return "未找到相关内容"

        output = f"语义搜索找到 {len(search_results)} 条结果:\n\n"
        for i, r in enumerate(search_results, 1):
            output += (
                f"[{i}] {r.get('title', '未知')} "
                f"(类型: {r.get('content_type', '')}, "
                f"相关度: {r.get('score', 0):.2f})\n"
                f"    {r.get('content', '')[:200]}\n\n"
            )

        # 附上 LLM 的综合回答
        answer = result.get("answer", "")
        if answer and answer != "未找到相关内容":
            output += f"\n综合分析: {answer}"

        return output
    except Exception as e:
        raise e # Let tenacity handle it
