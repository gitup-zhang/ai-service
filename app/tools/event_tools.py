"""
活动相关工具 — 供 Agent 调用的活动搜索和管理工具
"""

import httpx
from langchain_core.tools import tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

def _tool_fallback_msg(retry_state):
    err = retry_state.outcome.exception()
    print(f"[活动工具] 调用后台服务完全失败: {err}")
    return f"检索后台活动服务暂时不可用 (错误信息: {str(err)})，请向用户致歉并告知服务异常。"


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    retry_error_callback=_tool_fallback_msg
)
def search_events(keyword: str, limit: int = 5) -> str:
    """根据关键词搜索活动，返回活动列表（标题、地点、时间）"""
    try:
        resp = httpx.get(
            f"{settings.BACKEND_API_URL}/event",
            params={"keyword": keyword, "page": 1, "page_size": limit},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            events = data.get("data", {}).get("records", data.get("data", []))
            if not events:
                return "未找到相关活动"
            result = f"找到 {len(events)} 个活动:\n"
            for e in events:
                title = e.get("title", "未知")
                address = e.get("event_address", "未知")
                start = e.get("event_start_time", "")
                fee = e.get("registration_fee", "免费")
                result += f"- 【{title}】地点: {address} | 时间: {start} | 费用: {fee}\n"
                result += f"  活动ID: {e.get('id', '')}\n"
            return result
        resp.raise_for_status()
        return f"搜索失败: HTTP {resp.status_code}"
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise e
    except Exception as e:
        return f"搜索活动解析出错: {str(e)}"


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    retry_error_callback=_tool_fallback_msg
)
def get_event_detail(event_id: str) -> str:
    """获取活动的详细信息，包括标题、时间、地点、费用、详情描述"""
    try:
        resp = httpx.get(
            f"{settings.BACKEND_API_URL}/event/{event_id}",
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return (
                f"活动: {data.get('title', '未知')}\n"
                f"时间: {data.get('event_start_time', '')} 至 {data.get('event_end_time', '')}\n"
                f"地点: {data.get('event_address', '未知')}\n"
                f"报名时间: {data.get('registration_start_time', '')} 至 {data.get('registration_end_time', '')}\n"
                f"费用: {data.get('registration_fee', '未知')}\n"
                f"详情: {str(data.get('detail', ''))[:500]}"
            )
        resp.raise_for_status()
        return f"获取失败: HTTP {resp.status_code}"
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise e
    except Exception as e:
        return f"获取活动详情解析出错: {str(e)}"


@tool
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
    retry_error_callback=lambda rs: f"报名动作不可用 (错误: {str(rs.outcome.exception())})，请告知用户。"
)
def register_event(event_id: str, user_token: str) -> str:
    """报名参加活动（需要用户登录 token）"""
    try:
        resp = httpx.post(
            f"{settings.BACKEND_API_URL}/event/registration",
            json={"event_id": event_id},
            headers={"Authorization": f"Bearer {user_token}"},
            timeout=10,
        )
        if resp.status_code == 200:
            return "报名成功！"
        resp.raise_for_status()
        return f"报名失败: {resp.text}"
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        raise e
    except Exception as e:
        return f"报名解析出错: {str(e)}"
