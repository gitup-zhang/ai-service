"""
RAG 引擎 - 基于 LlamaIndex + ChromaDB

提供向量化存储和语义搜索能力。
"""

import re
import chromadb
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from app.config import settings


def _clean_html(html: str) -> str:
    """清理 HTML 标签，提取纯文本"""
    text = re.sub(r"<[^>]+>", "", html or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


class RAGEngine:
    """RAG 检索增强生成引擎"""

    def __init__(self):
        # ChromaDB 持久化客户端
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH
        )

        # DashScope Embedding 模型
        self.embed_model = DashScopeEmbedding(
            api_key=settings.DASHSCOPE_API_KEY,
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )

        # DashScope LLM
        self.llm = DashScope(
            api_key=settings.DASHSCOPE_API_KEY,
            model_name=settings.DASHSCOPE_MODEL,
        )

        # 文本分割器
        self.splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

        # 初始化向量集合
        self._init_collections()

    def _init_collections(self):
        """初始化文章和活动两个向量集合"""
        # 文章集合
        self.article_collection = self.chroma_client.get_or_create_collection(
            "articles"
        )
        article_vector_store = ChromaVectorStore(
            chroma_collection=self.article_collection
        )
        self.article_index = VectorStoreIndex.from_vector_store(
            article_vector_store,
            embed_model=self.embed_model,
        )

        # 活动集合
        self.event_collection = self.chroma_client.get_or_create_collection(
            "events"
        )
        event_vector_store = ChromaVectorStore(
            chroma_collection=self.event_collection
        )
        self.event_index = VectorStoreIndex.from_vector_store(
            event_vector_store,
            embed_model=self.embed_model,
        )

    def _get_index(self, content_type: str) -> VectorStoreIndex:
        """根据类型获取对应索引"""
        if content_type == "event":
            return self.event_index
        return self.article_index

    def _get_collection(self, content_type: str):
        """根据类型获取对应集合"""
        if content_type == "event":
            return self.event_collection
        return self.article_collection

    # ============ 写入操作 ============

    def add_article(self, article_data: dict):
        """
        写入文章到向量库

        article_data 需包含: article_id, article_title, article_content,
                              brief_content(可选), article_type(可选), field_name(可选)
        """
        article_id = str(article_data["article_id"])
        title = article_data.get("article_title", "")
        content = _clean_html(article_data.get("article_content", ""))
        brief = article_data.get("brief_content", "")

        # 先删除旧数据（支持更新）
        self._delete_by_doc_id(article_id, "article")

        # 构建文档文本
        text = f"文章标题：{title}\n"
        if brief:
            text += f"摘要：{brief}\n"
        text += f"正文：{content}"

        doc = Document(
            doc_id=f"article_{article_id}",
            text=text,
            metadata={
                "doc_id": article_id,
                "title": title,
                "content_type": "article",
                "article_type": article_data.get("article_type", ""),
                "field_name": article_data.get("field_name", ""),
            },
        )
        self.article_index.insert(doc)

    def add_event(self, event_data: dict):
        """
        写入活动到向量库

        event_data 需包含: id, title, detail,
                           event_address(可选), event_start_time(可选), event_end_time(可选)
        """
        event_id = str(event_data["id"])
        title = event_data.get("title", "")
        detail = _clean_html(event_data.get("detail", ""))

        # 先删除旧数据（支持更新）
        self._delete_by_doc_id(event_id, "event")

        # 构建文档文本
        text = f"活动名称：{title}\n"
        if event_data.get("event_address"):
            text += f"活动地点：{event_data['event_address']}\n"
        if event_data.get("event_start_time"):
            text += f"活动时间：{event_data['event_start_time']} 至 {event_data.get('event_end_time', '')}\n"
        if event_data.get("registration_fee") is not None:
            text += f"报名费用：{event_data['registration_fee']}\n"
        if detail:
            text += f"活动详情：{detail}"

        doc = Document(
            doc_id=f"event_{event_id}",
            text=text,
            metadata={
                "doc_id": event_id,
                "title": title,
                "content_type": "event",
                "address": event_data.get("event_address", ""),
            },
        )
        self.event_index.insert(doc)

    # ============ 删除操作 ============

    def _delete_by_doc_id(self, doc_id: str, content_type: str):
        """根据 doc_id 从 ChromaDB 集合中直接删除文档"""
        collection = self._get_collection(content_type)
        try:
            # 用 metadata 的 doc_id 字段查找
            results = collection.get(
                where={"doc_id": doc_id},
            )
            ids = results.get("ids", [])
            if ids:
                collection.delete(ids=ids)
                print(f"已从 ChromaDB 删除 {content_type} doc_id={doc_id}，共 {len(ids)} 条")
            else:
                # 回退：尝试用 LlamaIndex 的 ref_doc_id 格式查找
                ref_id = f"{content_type}_{doc_id}"
                all_data = collection.get(limit=collection.count())
                matching_ids = []
                for i, cid in enumerate(all_data.get("ids", [])):
                    if cid == ref_id or cid.startswith(ref_id):
                        matching_ids.append(cid)
                if matching_ids:
                    collection.delete(ids=matching_ids)
                    print(f"已通过 ref_id 模式删除 {content_type} doc_id={doc_id}，共 {len(matching_ids)} 条")
                else:
                    print(f"未找到 {content_type} doc_id={doc_id} 的记录")
        except Exception as e:
            print(f"删除文档 doc_id={doc_id} 时出错: {e}")

    def delete_article(self, article_id: int | str):
        """删除文章"""
        self._delete_by_doc_id(str(article_id), "article")

    def delete_event(self, event_id: int | str):
        """删除活动"""
        self._delete_by_doc_id(str(event_id), "event")

    # ============ 查询操作 ============

    def query(
        self,
        question: str,
        content_type: str = "all",
        top_k: int = 3,
    ) -> dict:
        """
        语义搜索

        Args:
            question: 搜索问题
            content_type: "all" | "article" | "event"
            top_k: 返回结果数

        Returns:
            {"results": [...], "answer": "..."}
        """
        all_results = []
        MIN_SCORE = 0.3  # 最低相关性分数阈值

        search_types = (
            [content_type]
            if content_type in ("article", "event")
            else ["article", "event"]
        )

        for st in search_types:
            index = self._get_index(st)
            # 使用 retriever 先检索，再按分数过滤
            retriever = index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(question)

            for node in nodes:
                score = node.score or 0.0
                # 过滤掉相关性太低的结果
                if score < MIN_SCORE:
                    continue
                meta = node.metadata or {}
                all_results.append({
                    "title": meta.get("title", ""),
                    "content": node.text[:300],
                    "content_type": meta.get("content_type", st),
                    "score": round(score, 4),
                    "metadata": {
                        k: v
                        for k, v in meta.items()
                        if k not in ("title", "content_type")
                    },
                })

        # 按 doc_id 去重，保留分数最高的
        seen = {}
        for r in all_results:
            doc_id = r.get("metadata", {}).get("doc_id", "")
            key = f"{r['content_type']}_{doc_id}" if doc_id else id(r)
            if key not in seen or r["score"] > seen[key]["score"]:
                seen[key] = r
        all_results = list(seen.values())

        # 按相关性排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:top_k]

        # 只有存在高质量结果时才让 LLM 生成回答
        answer = "未找到相关内容"
        if top_results:
            # 拼接检索到的上下文给 LLM
            context_parts = []
            for i, r in enumerate(top_results, 1):
                context_parts.append(f"[来源{i}] {r['title']}\n{r['content']}")
            context_text = "\n\n".join(context_parts)

            prompt = (
                "你是一个智能搜索助手。请仅根据以下检索到的内容回答用户的问题。\n"
                "如果检索内容无法回答问题，请如实说明'根据现有资料无法回答该问题'。\n"
                "不要编造内容，不要添加检索内容中没有的信息。\n\n"
                f"检索到的相关内容：\n{context_text}\n\n"
                f"用户问题：{question}\n\n"
                "请用简洁的中文回答："
            )
            try:
                llm_response = self.llm.complete(prompt)
                answer = str(llm_response).strip()
            except Exception as e:
                answer = f"AI 回答生成失败: {str(e)}"

        return {
            "results": top_results,
            "answer": answer,
        }

    # ============ 状态查询 ============

    def get_status(self) -> dict:
        """获取向量库状态"""
        return {
            "articles_count": len(self.list_articles()),
            "events_count": len(self.list_events()),
            "chroma_db_path": settings.CHROMA_DB_PATH,
        }

    def _list_collection(self, content_type: str) -> list[dict]:
        """列出某个集合中的所有文档（按 doc_id 去重合并）"""
        collection = self._get_collection(content_type)
        count = collection.count()
        if count == 0:
            return []

        result = collection.get(
            include=["metadatas", "documents"],
            limit=count,
        )

        ids = result.get("ids", [])
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])

        # 按 doc_id 分组合并 chunks
        grouped = {}
        for i, chroma_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_text = documents[i] if i < len(documents) else ""
            doc_id = meta.get("doc_id", chroma_id)

            if doc_id not in grouped:
                grouped[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "content_type": meta.get("content_type", content_type),
                    "preview": (doc_text or "")[:200],
                    "chunks": 1,
                    "chroma_ids": [chroma_id],
                    "children": [{
                        "chroma_id": chroma_id,
                        "preview": (doc_text or "")[:300],
                    }],
                    "metadata": {
                        k: v for k, v in (meta or {}).items()
                        if k not in ("doc_id", "title", "content_type")
                    },
                }
            else:
                grouped[doc_id]["chunks"] += 1
                grouped[doc_id]["chroma_ids"].append(chroma_id)
                grouped[doc_id]["children"].append({
                    "chroma_id": chroma_id,
                    "preview": (doc_text or "")[:300],
                })

        return list(grouped.values())

    def list_articles(self) -> list[dict]:
        """列出所有已索引文章"""
        return self._list_collection("article")

    def list_events(self) -> list[dict]:
        """列出所有已索引活动"""
        return self._list_collection("event")

    def clear_collection(self, content_type: str) -> int:
        """清空某个集合，返回删除数量"""
        collection = self._get_collection(content_type)
        count = collection.count()
        if count > 0:
            # 获取所有 ID 并删除
            result = collection.get(limit=count)
            ids = result.get("ids", [])
            if ids:
                collection.delete(ids=ids)
        return count

