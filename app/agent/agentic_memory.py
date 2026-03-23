"""
A-MEM — Agentic Memory for LLM Agents

基于论文: A-MEM: Agentic Memory for LLM Agents (arXiv, 2025.02)

核心架构 (受 Zettelkasten 卡片盒笔记法启发):
  NoteConstructor(结构化笔记构建)
  → LinkDiscovery(知识网络链接发现)
  → MemoryEvolution(已有记忆的自适应进化)
  → AgenticMemoryManager(统一管理 + 图扩展检索)

与原始 EpisodicMemory 的关键区别:
  - 扁平向量存储 → 结构化笔记 (context + keywords + tags)
  - 独立记忆条    → 知识网络 (自动发现关联、双向链接)
  - 静态写入      → 动态进化 (新记忆触发旧记忆更新)
  - 纯向量检索    → 图扩展检索 (BFS 沿链接探索关联记忆)

参考: https://arxiv.org/abs/2502.12110
"""

import hashlib
import json
import logging
import re
import time
from collections import deque
from typing import Optional

import chromadb
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from pydantic import BaseModel, Field
from dashscope import Generation

from app.config import settings

logger = logging.getLogger(__name__)


# ============ 数据模型 ============

class MemoryNote(BaseModel):
    """
    结构化记忆笔记 — A-MEM 的核心数据单元

    论文 Section 3.1: 每条记忆不仅包含原始内容,
    还包含 LLM 生成的结构化属性 (上下文、关键词、标签),
    以及与其他记忆的关联链接.
    """
    note_id: str = ""
    raw_content: str = ""
    context_description: str = ""
    keywords: list[str] = []
    tags: list[str] = []
    linked_notes: list[str] = []
    link_reasons: dict[str, str] = {}

    # 生命周期管理
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    importance_score: float = 0.5

    # 进化追踪
    version: int = 1
    evolution_history: list[str] = []


# 预定义语义标签集
PREDEFINED_TAGS = [
    "政策解读", "活动查询", "文章搜索", "多轮对话",
    "工具失败处理", "信息不足应对", "格式优化", "数据整合",
    "用户意图澄清", "多源信息融合",
]


# ============ Prompt 模板 ============

NOTE_CONSTRUCTION_PROMPT = """你是一个知识管理专家。请分析以下 Agent 交互经验，提取结构化属性。

## 交互记录
{raw_content}

## 要求
请提取以下信息并以 JSON 返回:
1. context_description: 对这段经验的语义理解（2-3句话，概括核心教训和适用场景）
2. keywords: 关键词列表（5-8个，包含实体、概念、工具名等）
3. tags: 从以下标签集中选择适用的，也可添加自定义标签
   可选标签: {available_tags}
4. importance_score: 该经验的参考价值（0.0-1.0，重大失败/成功的教训=0.8+，普通交互=0.3-0.5）

严格返回 JSON:
{{"context_description": "...", "keywords": [...], "tags": [...], "importance_score": 0.6}}"""


LINK_ANALYSIS_PROMPT = """你是一个知识关联分析专家。请分析以下两条记忆之间是否存在有意义的关联。

## 新记忆
上下文: {new_context}
关键词: {new_keywords}

## 已有记忆
上下文: {old_context}
关键词: {old_keywords}

## 判断标准
请分析是否存在以下任一关联:
1. 因果关系（如：上次的失败经验 → 这次的改进策略）
2. 同一主题的不同视角
3. 互补信息（组合后比单独更有价值）
4. 策略演化（同类问题的处理策略变迁）

严格返回 JSON:
{{"should_link": true/false, "reason": "关联原因", "link_strength": 0.8}}"""


EVOLUTION_PROMPT = """你是一个知识进化专家。已有一条旧记忆，现在有新的相关经验。
请根据新经验更新旧记忆的上下文描述，使其包含更全面的理解。

## 旧记忆
上下文: {old_context}
关键词: {old_keywords}

## 新的相关经验
上下文: {new_context}
关键词: {new_keywords}
关联原因: {link_reason}

## 要求
请输出更新后的上下文描述（融合新旧信息，2-3句话），以及需要补充的关键词。

严格返回 JSON:
{{"updated_context": "...", "additional_keywords": [...]}}"""


# ============ 核心组件 ============

class NoteConstructor:
    """
    笔记构建器 — 将原始交互转化为结构化记忆笔记

    论文 Section 3.1:
    1. 组装原始内容 (question + answer + reflection)
    2. LLM 抽取结构化属性 (context, keywords, tags, importance)
    3. 生成唯一 ID (timestamp + content hash)
    """

    def construct(
        self,
        question: str,
        reflection: str,
        evaluation_feedback: str = "",
        skill_used: str = "",
        answer: str = "",
    ) -> MemoryNote:
        """构建结构化记忆笔记"""
        # Step 1: 组装原始内容
        raw_parts = [f"问题: {question}"]
        if answer:
            raw_parts.append(f"回答: {answer[:300]}")
        raw_parts.append(f"反思: {reflection}")
        if evaluation_feedback:
            raw_parts.append(f"评估建议: {evaluation_feedback}")
        if skill_used:
            raw_parts.append(f"使用技能: {skill_used}")
        raw_content = "\n".join(raw_parts)

        # Step 2: 生成唯一 ID
        content_hash = hashlib.md5(raw_content.encode()).hexdigest()[:8]
        note_id = f"amem_{int(time.time())}_{content_hash}"

        now = time.time()
        note = MemoryNote(
            note_id=note_id,
            raw_content=raw_content,
            created_at=now,
            last_accessed=now,
        )

        # Step 3: LLM 结构化抽取
        try:
            prompt = NOTE_CONSTRUCTION_PROMPT.format(
                raw_content=raw_content,
                available_tags=", ".join(PREDEFINED_TAGS),
            )
            raw_response = self._call_llm(prompt)
            parsed = self._parse_response(raw_response)

            note.context_description = parsed.get("context_description", raw_content[:200])
            note.keywords = parsed.get("keywords", [])
            note.tags = parsed.get("tags", [])
            note.importance_score = min(1.0, max(0.0, float(
                parsed.get("importance_score", 0.5)
            )))
        except Exception as e:
            logger.warning(f"[A-MEM] NoteConstructor LLM 抽取失败: {e}")
            # 兜底: 规则抽取
            note.context_description = raw_content[:200]
            note.keywords = self._extract_keywords_fallback(raw_content)
            note.importance_score = 0.5

        logger.info(
            f"[A-MEM] Note constructed: id={note_id}, "
            f"keywords={note.keywords[:5]}, tags={note.tags}"
        )
        return note

    def _call_llm(self, prompt: str) -> str:
        response = Generation.call(
            api_key=settings.DASHSCOPE_API_KEY,
            model=settings.DASHSCOPE_MODEL,
            messages=[
                {"role": "system", "content": "你是知识管理专家，只返回 JSON。"},
                {"role": "user", "content": prompt},
            ],
            result_format="message",
            max_tokens=400,
            temperature=0.2,
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        return "{}"

    def _parse_response(self, raw: str) -> dict:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {}

    def _extract_keywords_fallback(self, text: str) -> list[str]:
        """规则兜底关键词提取"""
        # 简单提取中文词和英文词
        words = re.findall(r'[\u4e00-\u9fa5]{2,6}|[a-zA-Z_]{3,}', text)
        # 去重保序
        seen = set()
        result = []
        for w in words:
            if w not in seen and len(result) < 8:
                seen.add(w)
                result.append(w)
        return result


class LinkDiscovery:
    """
    链接发现引擎 — 自动建立记忆间的知识网络

    论文 Section 3.2:
    1. 候选检索: 向量相似度找 top-K 候选
    2. 关键词过滤: Jaccard 相似度初筛
    3. LLM 深度关联分析: 判断因果/互补/演化关系
    4. 双向链接: 在新旧双方都建立链接引用
    """

    # 候选检索数
    CANDIDATE_TOP_K = 8
    # 关键词 Jaccard 最低阈值
    KEYWORD_JACCARD_MIN = 0.05
    # 深度分析最大数
    MAX_DEEP_ANALYSIS = 4
    # 链接强度阈值
    LINK_STRENGTH_THRESHOLD = 0.5

    def discover_links(
        self,
        new_note: MemoryNote,
        collection: chromadb.Collection,
        embed_model: DashScopeEmbedding,
    ) -> list[tuple[str, str, float]]:
        """
        发现新笔记与已有笔记的关联

        Args:
            new_note: 新构建的笔记
            collection: ChromaDB 集合
            embed_model: 嵌入模型

        Returns:
            [(linked_note_id, reason, strength), ...]
        """
        if collection.count() == 0:
            return []

        # Step 1: 向量候选检索
        try:
            query_text = f"{new_note.context_description} {' '.join(new_note.keywords)}"
            embedding = embed_model.get_text_embedding(query_text)

            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(self.CANDIDATE_TOP_K, collection.count()),
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.warning(f"[A-MEM] LinkDiscovery 候选检索失败: {e}")
            return []

        candidates = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for i, note_id in enumerate(ids):
            if note_id == new_note.note_id:
                continue

            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""

            # 解析已有笔记的关键词
            old_keywords = []
            kw_str = meta.get("keywords", "")
            if kw_str:
                try:
                    old_keywords = json.loads(kw_str)
                except (json.JSONDecodeError, TypeError):
                    old_keywords = kw_str.split(",")

            # Step 2: 关键词 Jaccard 过滤
            new_kw_set = set(new_note.keywords)
            old_kw_set = set(old_keywords)
            if new_kw_set and old_kw_set:
                jaccard = len(new_kw_set & old_kw_set) / len(new_kw_set | old_kw_set)
            else:
                jaccard = 0.0

            if jaccard >= self.KEYWORD_JACCARD_MIN:
                old_context = meta.get("context", doc[:200] if doc else "")
                candidates.append({
                    "note_id": note_id,
                    "context": old_context,
                    "keywords": old_keywords,
                    "jaccard": jaccard,
                })

        # 按 Jaccard 排序取前 N
        candidates.sort(key=lambda x: x["jaccard"], reverse=True)
        candidates = candidates[:self.MAX_DEEP_ANALYSIS]

        if not candidates:
            return []

        # Step 3: LLM 深度关联分析
        links = []
        for cand in candidates:
            try:
                link_result = self._analyze_link(new_note, cand)
                if link_result and link_result[2] >= self.LINK_STRENGTH_THRESHOLD:
                    links.append(link_result)
            except Exception as e:
                logger.debug(f"[A-MEM] LinkAnalysis 失败 ({cand['note_id']}): {e}")

        logger.info(
            f"[A-MEM] LinkDiscovery: "
            f"candidates={len(candidates)}, links={len(links)}"
        )
        return links

    def _analyze_link(
        self, new_note: MemoryNote, candidate: dict
    ) -> Optional[tuple[str, str, float]]:
        """LLM 深度分析两条记忆的关联"""
        prompt = LINK_ANALYSIS_PROMPT.format(
            new_context=new_note.context_description,
            new_keywords=", ".join(new_note.keywords),
            old_context=candidate["context"],
            old_keywords=", ".join(candidate["keywords"]),
        )

        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": "你是知识关联分析专家，只返回 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=200,
                temperature=0.1,
            )
            if response.status_code == 200:
                raw = response.output.choices[0].message.content
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if parsed.get("should_link", False):
                        return (
                            candidate["note_id"],
                            parsed.get("reason", ""),
                            min(1.0, max(0.0, float(parsed.get("link_strength", 0.5)))),
                        )
        except Exception as e:
            logger.debug(f"[A-MEM] Link LLM 分析异常: {e}")
        return None


class MemoryEvolution:
    """
    记忆进化引擎 — 新记忆触发已有记忆的上下文更新

    论文 Section 3.3:
    当新笔记与旧笔记建立强链接 (strength > threshold) 时,
    触发旧笔记的上下文描述更新, 融合新信息.
    这使得记忆网络能够随时间不断进化和完善.
    """

    EVOLUTION_STRENGTH_THRESHOLD = 0.65

    def maybe_evolve(
        self,
        new_note: MemoryNote,
        linked_note_id: str,
        link_reason: str,
        link_strength: float,
        collection: chromadb.Collection,
    ) -> bool:
        """
        检查是否需要触发记忆进化, 如需要则执行

        Returns:
            是否执行了进化
        """
        if link_strength < self.EVOLUTION_STRENGTH_THRESHOLD:
            return False

        # 读取旧笔记
        try:
            result = collection.get(
                ids=[linked_note_id],
                include=["documents", "metadatas"],
            )
            if not result.get("ids"):
                return False

            old_meta = result["metadatas"][0] if result.get("metadatas") else {}
            old_context = old_meta.get("context", "")
            old_keywords_str = old_meta.get("keywords", "[]")
            try:
                old_keywords = json.loads(old_keywords_str)
            except (json.JSONDecodeError, TypeError):
                old_keywords = []

        except Exception as e:
            logger.warning(f"[A-MEM] 进化读取旧笔记失败: {e}")
            return False

        # LLM 生成进化后的上下文
        try:
            prompt = EVOLUTION_PROMPT.format(
                old_context=old_context,
                old_keywords=", ".join(old_keywords),
                new_context=new_note.context_description,
                new_keywords=", ".join(new_note.keywords),
                link_reason=link_reason,
            )
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": "你是知识进化专家，只返回 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=300,
                temperature=0.2,
            )

            if response.status_code != 200:
                return False

            raw = response.output.choices[0].message.content
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if not json_match:
                return False

            parsed = json.loads(json_match.group())
            updated_context = parsed.get("updated_context", "")
            additional_keywords = parsed.get("additional_keywords", [])

            if not updated_context:
                return False

            # 更新旧笔记的 metadata
            merged_keywords = list(set(old_keywords + additional_keywords))
            old_version = int(old_meta.get("version", 1))
            evolution_history = old_meta.get("evolution_history", "[]")
            try:
                history = json.loads(evolution_history)
            except (json.JSONDecodeError, TypeError):
                history = []
            history.append(
                f"v{old_version+1}: 因与 {new_note.note_id} 关联而进化"
            )

            # 更新 linked_notes
            old_links = old_meta.get("linked_notes", "[]")
            try:
                links_list = json.loads(old_links)
            except (json.JSONDecodeError, TypeError):
                links_list = []
            if new_note.note_id not in links_list:
                links_list.append(new_note.note_id)

            # 写回 ChromaDB
            collection.update(
                ids=[linked_note_id],
                metadatas=[{
                    **old_meta,
                    "context": updated_context,
                    "keywords": json.dumps(merged_keywords, ensure_ascii=False),
                    "version": str(old_version + 1),
                    "evolution_history": json.dumps(history, ensure_ascii=False),
                    "linked_notes": json.dumps(links_list, ensure_ascii=False),
                }],
            )

            logger.info(
                f"[A-MEM] Evolution: {linked_note_id} v{old_version}→v{old_version+1}, "
                f"reason='{link_reason[:50]}'"
            )
            return True

        except Exception as e:
            logger.warning(f"[A-MEM] 记忆进化执行失败: {e}")
            return False


class AgenticMemoryManager:
    """
    智体记忆统一管理器 — A-MEM 对外接口

    组合 NoteConstructor + LinkDiscovery + MemoryEvolution,
    提供与原 EpisodicMemory 兼容的 API (store/search/get_count/clear).

    检索增强:
    - 图扩展检索: 先向量检索, 再沿链接做 BFS 探索关联记忆
    - 综合排序: similarity × importance × recency
    """

    COLLECTION_NAME = "agent_memory_v2"

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

        # 子组件
        self.note_constructor = NoteConstructor()
        self.link_discovery = LinkDiscovery()
        self.memory_evolution = MemoryEvolution()

    def store(
        self,
        question: str,
        reflection: str,
        evaluation_feedback: str = "",
        skill_used: str = "",
        answer: str = "",
    ):
        """
        存储一条结构化记忆

        完整流程: 构建笔记 → 向量化 → 持久化 → 链接发现 → 记忆进化
        """
        # Phase 1: 构建结构化笔记
        note = self.note_constructor.construct(
            question=question,
            reflection=reflection,
            evaluation_feedback=evaluation_feedback,
            skill_used=skill_used,
            answer=answer,
        )

        # Phase 2: 向量化 + 持久化
        try:
            embedding = self.embed_model.get_text_embedding(
                f"{note.context_description} {' '.join(note.keywords)}"
            )

            metadata = {
                "context": note.context_description,
                "keywords": json.dumps(note.keywords, ensure_ascii=False),
                "tags": json.dumps(note.tags, ensure_ascii=False),
                "importance": str(note.importance_score),
                "skill": skill_used,
                "question": question[:200],
                "created_at": str(int(note.created_at)),
                "access_count": "0",
                "version": "1",
                "linked_notes": "[]",
                "evolution_history": "[]",
            }

            self.collection.add(
                ids=[note.note_id],
                embeddings=[embedding],
                documents=[note.raw_content],
                metadatas=[metadata],
            )
        except Exception as e:
            logger.error(f"[A-MEM] 持久化失败: {e}")
            return

        # Phase 3: 链接发现
        links = self.link_discovery.discover_links(
            new_note=note,
            collection=self.collection,
            embed_model=self.embed_model,
        )

        if links:
            # 更新新笔记的链接信息
            linked_ids = [lid for lid, _, _ in links]
            link_reasons = {lid: reason for lid, reason, _ in links}
            note.linked_notes = linked_ids
            note.link_reasons = link_reasons

            try:
                self.collection.update(
                    ids=[note.note_id],
                    metadatas=[{
                        **metadata,
                        "linked_notes": json.dumps(linked_ids, ensure_ascii=False),
                    }],
                )
            except Exception as e:
                logger.warning(f"[A-MEM] 更新新笔记链接失败: {e}")

            # Phase 4: 尝试触发记忆进化
            for linked_id, reason, strength in links:
                self.memory_evolution.maybe_evolve(
                    new_note=note,
                    linked_note_id=linked_id,
                    link_reason=reason,
                    link_strength=strength,
                    collection=self.collection,
                )

        logger.info(
            f"[A-MEM] Store 完成: id={note.note_id}, "
            f"links={len(links)}, tags={note.tags}"
        )

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        图扩展检索 — 向量检索 + BFS 链接探索

        流程:
        1. 向量相似度检索 top-K 候选
        2. 从候选出发, BFS 探索 1 层链接邻居
        3. 综合排序: relevance × importance × recency
        4. 更新 access_count
        """
        if self.collection.count() == 0:
            return []

        try:
            embedding = self.embed_model.get_text_embedding(query)
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k * 2, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"[A-MEM] 检索失败: {e}")
            return []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # 收集初始候选 + 链接邻居
        all_candidates: dict[str, dict] = {}

        for i, note_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            dist = distances[i] if i < len(distances) else 1.0

            # 计算综合分数
            similarity = max(0, 1.0 - dist) if dist else 0.5
            importance = float(meta.get("importance", "0.5"))
            created_at = float(meta.get("created_at", "0"))
            recency = self._recency_score(created_at)

            score = similarity * 0.5 + importance * 0.3 + recency * 0.2

            all_candidates[note_id] = {
                "note_id": note_id,
                "document": doc,
                "score": score,
                "source": "direct",
            }

            # BFS: 探索链接邻居 (深度=1)
            linked_str = meta.get("linked_notes", "[]")
            try:
                linked_ids = json.loads(linked_str)
            except (json.JSONDecodeError, TypeError):
                linked_ids = []

            for linked_id in linked_ids[:3]:  # 限制展开数
                if linked_id not in all_candidates:
                    try:
                        linked_result = self.collection.get(
                            ids=[linked_id],
                            include=["documents", "metadatas"],
                        )
                        if linked_result.get("ids"):
                            l_doc = linked_result["documents"][0] if linked_result.get("documents") else ""
                            l_meta = linked_result["metadatas"][0] if linked_result.get("metadatas") else {}
                            l_importance = float(l_meta.get("importance", "0.5"))

                            all_candidates[linked_id] = {
                                "note_id": linked_id,
                                "document": l_doc,
                                "score": score * 0.6 + l_importance * 0.4,
                                "source": "linked",
                            }
                    except Exception:
                        pass

        # 排序 + 取 top_k
        sorted_candidates = sorted(
            all_candidates.values(),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]

        # 更新 access_count
        for cand in sorted_candidates:
            self._increment_access(cand["note_id"])

        return [c["document"] for c in sorted_candidates if c["document"]]

    def get_count(self) -> int:
        """获取记忆条数"""
        return self.collection.count()

    def clear(self) -> int:
        """清空所有记忆"""
        count = self.collection.count()
        if count > 0:
            result = self.collection.get(limit=count)
            ids = result.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
        return count

    def get_note(self, note_id: str) -> Optional[MemoryNote]:
        """获取单条笔记详情"""
        try:
            result = self.collection.get(
                ids=[note_id],
                include=["documents", "metadatas"],
            )
            if result.get("ids"):
                meta = result["metadatas"][0]
                return MemoryNote(
                    note_id=note_id,
                    raw_content=result["documents"][0],
                    context_description=meta.get("context", ""),
                    keywords=json.loads(meta.get("keywords", "[]")),
                    tags=json.loads(meta.get("tags", "[]")),
                    linked_notes=json.loads(meta.get("linked_notes", "[]")),
                    importance_score=float(meta.get("importance", "0.5")),
                    access_count=int(meta.get("access_count", "0")),
                    version=int(meta.get("version", "1")),
                )
        except Exception as e:
            logger.warning(f"[A-MEM] 获取笔记失败 ({note_id}): {e}")
        return None

    def _recency_score(self, created_at: float) -> float:
        """时间衰减评分 — 越近的记忆分数越高"""
        if created_at <= 0:
            return 0.5
        age_hours = (time.time() - created_at) / 3600
        # 指数衰减: 24小时内=1.0, 1周≈0.5, 1月≈0.1
        import math
        return math.exp(-age_hours / 168)  # 半衰期 ≈ 1 周

    def _increment_access(self, note_id: str):
        """递增访问计数"""
        try:
            result = self.collection.get(ids=[note_id], include=["metadatas"])
            if result.get("ids"):
                meta = result["metadatas"][0]
                count = int(meta.get("access_count", "0")) + 1
                self.collection.update(
                    ids=[note_id],
                    metadatas=[{**meta, "access_count": str(count)}],
                )
        except Exception:
            pass
