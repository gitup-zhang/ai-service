"""
活动相关 MCP/Agent 工具 (Phase 3 实现)
"""

# TODO: Phase 3 实现内容
#
# import httpx
# from langchain_core.tools import tool
# from app.config import settings
#
#
# @tool
# def search_events(keyword: str) -> str:
#     """根据关键词搜索活动"""
#     resp = httpx.get(
#         f"{settings.BACKEND_API_URL}/event",
#         params={"keyword": keyword, "page": 1, "page_size": 5},
#     )
#     if resp.status_code == 200:
#         data = resp.json()
#         events = data.get("data", [])
#         if not events:
#             return "未找到相关活动"
#         result = "找到以下活动:\n"
#         for e in events:
#             result += f"- {e['title']} (地点: {e.get('event_address', '未知')})\n"
#         return result
#     return f"搜索失败: HTTP {resp.status_code}"
#
#
# @tool
# def get_event_detail(event_id: str) -> str:
#     """获取活动详细信息"""
#     resp = httpx.get(f"{settings.BACKEND_API_URL}/event/{event_id}")
#     if resp.status_code == 200:
#         data = resp.json().get("data", {})
#         return (
#             f"活动: {data.get('title', '未知')}\n"
#             f"时间: {data.get('event_start_time', '')} ~ {data.get('event_end_time', '')}\n"
#             f"地点: {data.get('event_address', '未知')}\n"
#             f"费用: {data.get('registration_fee', '未知')}"
#         )
#     return f"获取失败: HTTP {resp.status_code}"
#
#
# @tool
# def register_event(event_id: str, user_token: str) -> str:
#     """报名参加活动"""
#     resp = httpx.post(
#         f"{settings.BACKEND_API_URL}/event/registration",
#         json={"event_id": event_id},
#         headers={"Authorization": f"Bearer {user_token}"},
#     )
#     if resp.status_code == 200:
#         return "报名成功！"
#     return f"报名失败: {resp.text}"
