"""
情景记忆 — 基于 ChromaDB 的 Agent 反思经验存储和检索

参考 Reflexion 论文 (Shinn 2023):
Agent 每次反思的结果存入向量数据库, 下次遇到类似问题时检索参考.
复用项目已有的 ChromaDB 基础设施.
"""

import time
import chromadb
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from app.config import settings


class EpisodicMemory:
    """
    情景记忆 — 存储 Agent 的反思经验

    核心功能:
    - store(): 将反思经验向量化存入 ChromaDB
    - search(): 语义搜索相似的历史经验
    - clear(): 清空记忆
    """

    COLLECTION_NAME = "agent_memory"

    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH
        )
        self.collection = self.chroma_client.get_or_create_collection(
            self.COLLECTION_NAME
        )
        self.embed_model = DashScopeEmbedding(
            api_key=settings.DASHSCOPE_API_KEY,
            model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
            text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
        )

    def store(
        self,
        question: str,
        reflection: str,
        evaluation_feedback: str = "",
        skill_used: str = "",
    ):
        """
        存储一条反思经验

        Args:
            question: 用户原始问题
            reflection: Agent 的反思内容
            evaluation_feedback: 评估器的改进建议
            skill_used: 使用的 Skill
        """
        doc_text = (
            f"问题: {question}\n"
            f"反思: {reflection}\n"
            f"改进建议: {evaluation_feedback}"
        )
        doc_id = f"mem_{int(time.time() * 1000)}"

        try:
            # 获取 embedding
            embedding = self.embed_model.get_text_embedding(doc_text)
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[doc_text],
                metadatas=[{
                    "question": question[:200],
                    "skill": skill_used,
                    "timestamp": str(int(time.time())),
                }],
            )
        except Exception as e:
            print(f"存储情景记忆失败: {e}")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        检索相似的历史经验

        Args:
            query: 当前用户问题
            top_k: 返回数量

        Returns:
            相似经验文本列表
        """
        if self.collection.count() == 0:
            return []

        try:
            embedding = self.embed_model.get_text_embedding(query)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, self.collection.count()),
            )
            documents = results.get("documents", [[]])
            return documents[0] if documents else []
        except Exception as e:
            print(f"检索情景记忆失败: {e}")
            return []

    def get_count(self) -> int:
        """获取记忆条数"""
        return self.collection.count()

    def clear(self):
        """清空所有记忆"""
        count = self.collection.count()
        if count > 0:
            result = self.collection.get(limit=count)
            ids = result.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
        return count
