"""
数据同步服务 - 同步业务数据到向量库 (Phase 2 实现)
"""

# TODO: Phase 2 实现内容
#
# import httpx
# from apscheduler.schedulers.asyncio import AsyncIOScheduler
# from app.config import settings
# from app.core.rag_engine import RAGEngine
#
#
# class DataSyncService:
#     """数据同步服务 - 定时从业务 API 拉取数据并写入向量库"""
#
#     def __init__(self, rag_engine: RAGEngine):
#         self.rag = rag_engine
#         self.scheduler = AsyncIOScheduler()
#
#     async def sync_events(self):
#         """同步活动数据"""
#         try:
#             resp = httpx.get(
#                 f"{settings.BACKEND_API_URL}/event",
#                 params={"page": 1, "page_size": 100},
#             )
#             if resp.status_code == 200:
#                 events = resp.json().get("data", [])
#                 for event in events:
#                     self.rag.add_event(event)
#                 print(f"同步 {len(events)} 条活动数据")
#         except Exception as e:
#             print(f"同步活动数据失败: {e}")
#
#     async def sync_articles(self):
#         """同步文章数据"""
#         try:
#             for article_type in ["POLICY", "NEWS"]:
#                 resp = httpx.get(
#                     f"{settings.BACKEND_API_URL}/article",
#                     params={"article_type": article_type, "page": 1, "page_size": 100},
#                 )
#                 if resp.status_code == 200:
#                     articles = resp.json().get("data", [])
#                     for article in articles:
#                         self.rag.add_article(article)
#                     print(f"同步 {len(articles)} 条{article_type}数据")
#         except Exception as e:
#             print(f"同步文章数据失败: {e}")
#
#     def start(self):
#         """启动定时同步"""
#         # 每小时同步一次
#         self.scheduler.add_job(self.sync_events, "interval", hours=1)
#         self.scheduler.add_job(self.sync_articles, "interval", hours=1)
#         self.scheduler.start()
#         print("数据同步服务已启动")
