"""
文章相关工具 — 供 Agent 调用的文章/政策搜索工具
"""

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

def _tool_fallback_msg(retry_state):
    err = retry_state.outcome.exception()
    print(f"[文章工具] 调用后台服务完全失败: {err}")
    return f"检索后台文章或政策服务暂时不可用 (错误信息: {str(err)})，请凭借自身已有知识尽力回答用户的问题并告知服务异常。"


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    retry_error_callback=_tool_fallback_msg
)
def search_articles(keyword: str, field: str = "", limit: int = 5) -> str:
    """搜索文章/政策，支持按领域过滤。返回文章标题和摘要列表"""
    try:
        params = {"keyword": keyword, "page": 1, "page_size": limit}
        if field:
            params["field_name"] = field
        resp = httpx.get(
            f"{settings.BACKEND_API_URL}/article",
            params=params,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            articles = data.get("data", {}).get("records", data.get("data", []))
            if not articles:
                return "未找到相关文章"
            result = f"找到 {len(articles)} 篇文章:\n"
            for a in articles:
                title = a.get("article_title", "未知")
                brief = a.get("brief_content", "")[:100]
                article_type = a.get("article_type", "")
                result += f"- 【{title}】{article_type}\n"
                if brief:
                    result += f"  摘要: {brief}\n"
                result += f"  文章ID: {a.get('article_id', '')}\n"
            return result
        resp.raise_for_status()
        return f"搜索失败: HTTP {resp.status_code}"
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise e # Let tenacity handle it
    except Exception as e:
        return f"搜索文章解析出错: {str(e)}"


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    retry_error_callback=_tool_fallback_msg
)
def get_article_detail(article_id: str) -> str:
    """获取文章全文详情，包含标题、摘要、正文内容"""
    try:
        resp = httpx.get(
            f"{settings.BACKEND_API_URL}/article/{article_id}",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            import re
            content = re.sub(r"<[^>]+>", "", data.get("article_content", ""))[:800]
            return (
                f"标题: {data.get('article_title', '未知')}\n"
                f"类型: {data.get('article_type', '')}\n"
                f"领域: {data.get('field_name', '')}\n"
                f"摘要: {data.get('brief_content', '')}\n"
                f"正文: {content}"
            )
        resp.raise_for_status()
        return f"获取失败: HTTP {resp.status_code}"
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise e
    except Exception as e:
        return f"获取文章详情解析出错: {str(e)}"
