"""
CRAG — Corrective Retrieval Augmented Generation

基于论文: Corrective Retrieval Augmented Generation (Yan et al., ICLR 2025)

核心架构:
  检索文档 → RetrievalEvaluator(质量评估) → 三路分发:
    CORRECT   → KnowledgeRefiner(Decompose-Recompose) → 精炼上下文
    AMBIGUOUS → KnowledgeRefiner + QueryRewriter(重写) → 补充检索 → 合并上下文
    INCORRECT → QueryRewriter(重写) → 重新检索 → 精炼/兜底

参考: https://arxiv.org/abs/2401.15884
"""

import json
import re
import logging
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from dashscope import Generation

from app.config import settings

logger = logging.getLogger(__name__)


# ============ 数据模型 ============

class RetrievalGrade(str, Enum):
    """检索质量评级 — 论文 Section 3.1"""
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


class DocumentScore(BaseModel):
    """单文档评估结果"""
    doc_index: int
    score: float = Field(0.0, ge=0, le=1)
    reason: str = ""
    grade: RetrievalGrade = RetrievalGrade.INCORRECT


class KnowledgeStrip(BaseModel):
    """知识条 — Decompose 阶段的最小信息单元"""
    strip_id: int
    content: str
    source_doc_index: int
    is_relevant: bool = False
    relevance_reason: str = ""


class CRAGResult(BaseModel):
    """CRAG 处理结果"""
    original_query: str
    retrieval_grade: RetrievalGrade
    doc_scores: list[DocumentScore] = []
    refined_context: str = ""
    rewritten_queries: list[str] = []
    total_strips: int = 0
    relevant_strips: int = 0
    processing_path: str = ""  # "correct" / "ambiguous" / "incorrect"


# ============ 评估 Prompt 模板 ============

EVAL_PROMPT = """你是一个检索质量评估专家。请评估以下检索文档是否能够回答用户的问题。

用户问题: {question}

以下是检索到的文档列表，请逐个评分:
{documents}

对每个文档，请判断:
- 该文档是否包含可以直接回答或有助于回答用户问题的信息
- 评分标准: 1.0=完全相关且包含答案, 0.7=部分相关有参考价值, 0.3=弱相关可能有间接帮助, 0.0=完全无关

请严格返回 JSON 数组，每个元素包含 doc_index、score (0.0-1.0)、reason (简短原因):
[{{"doc_index": 0, "score": 0.85, "reason": "包含关于该政策的详细条款"}}]"""


STRIP_FILTER_PROMPT = """你是一个信息过滤专家。以下是从检索文档中拆分出的知识条(knowledge strips)。
请判断每个知识条是否与用户问题**直接相关**(即包含能帮助回答问题的信息)。

用户问题: {question}

知识条列表:
{strips}

对每个知识条，判断 is_relevant (true/false) 和简短原因。
请严格返回 JSON 数组:
[{{"strip_id": 0, "is_relevant": true, "reason": "直接提到了该政策的适用范围"}}]"""


QUERY_REWRITE_PROMPT = """你是一个查询优化专家。用户的原始查询未能检索到满意的结果。
请将原始查询改写为 3 个不同角度的精确搜索查询，以提高检索召回率。

原始查询: {question}

改写策略:
1. 关键词提取: 提取核心实体和概念，去除冗余修饰
2. 同义词替换: 用不同的表达方式描述同一需求
3. 问题分解: 如果是复合问题，拆分为更细粒度的子问题

请严格返回 JSON:
{{"queries": ["改写查询1", "改写查询2", "改写查询3"]}}"""


# ============ 核心组件 ============

class RetrievalEvaluator:
    """
    检索质量评估器

    论文 Section 3.1:
    对检索到的每个文档进行相关性评分, 然后聚合为整体检索质量判定.
    聚合规则:
      - 任一文档 score >= CORRECT_THRESHOLD → 整体 CORRECT
      - 所有文档 score < INCORRECT_THRESHOLD → 整体 INCORRECT
      - 其余情况 → AMBIGUOUS
    """

    CORRECT_THRESHOLD = 0.7
    INCORRECT_THRESHOLD = 0.3

    def evaluate(
        self,
        question: str,
        documents: list[dict],
    ) -> tuple[RetrievalGrade, list[DocumentScore]]:
        """
        评估检索结果质量

        Args:
            question: 用户原始问题
            documents: 检索文档列表 [{"title": ..., "content": ...}]

        Returns:
            (整体 grade, 各文档评分列表)
        """
        if not documents:
            return RetrievalGrade.INCORRECT, []

        # 构建批量评估 prompt
        doc_text_parts = []
        for i, doc in enumerate(documents):
            title = doc.get("title", "")
            content = doc.get("content", "")[:500]
            doc_text_parts.append(f"[文档{i}] 标题: {title}\n内容: {content}")
        documents_text = "\n\n".join(doc_text_parts)

        prompt = EVAL_PROMPT.format(
            question=question,
            documents=documents_text,
        )

        # LLM 批量评估
        raw_scores = self._call_llm_for_eval(prompt)
        doc_scores = self._parse_eval_response(raw_scores, len(documents))

        # 为每个文档打 grade
        for ds in doc_scores:
            if ds.score >= self.CORRECT_THRESHOLD:
                ds.grade = RetrievalGrade.CORRECT
            elif ds.score < self.INCORRECT_THRESHOLD:
                ds.grade = RetrievalGrade.INCORRECT
            else:
                ds.grade = RetrievalGrade.AMBIGUOUS

        # 聚合判定 — 按论文规则
        overall_grade = self._aggregate_grades(doc_scores)
        logger.info(
            f"[CRAG] RetrievalEvaluator: "
            f"overall={overall_grade.value}, "
            f"scores={[f'{ds.score:.2f}' for ds in doc_scores]}"
        )
        return overall_grade, doc_scores

    def _aggregate_grades(self, doc_scores: list[DocumentScore]) -> RetrievalGrade:
        """论文聚合规则: 任一 CORRECT → CORRECT; 全部 INCORRECT → INCORRECT; 否则 AMBIGUOUS"""
        if any(ds.grade == RetrievalGrade.CORRECT for ds in doc_scores):
            return RetrievalGrade.CORRECT
        if all(ds.grade == RetrievalGrade.INCORRECT for ds in doc_scores):
            return RetrievalGrade.INCORRECT
        return RetrievalGrade.AMBIGUOUS

    def _call_llm_for_eval(self, prompt: str) -> str:
        """调用 LLM 进行评估 — 使用低温度保证一致性"""
        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": "你是检索质量评估专家，只返回 JSON 格式。"},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=800,
                temperature=0.1,
            )
            if response.status_code == 200:
                return response.output.choices[0].message.content
        except Exception as e:
            logger.error(f"[CRAG] RetrievalEvaluator LLM 调用失败: {e}")
        return "[]"

    def _parse_eval_response(
        self, raw: str, expected_count: int
    ) -> list[DocumentScore]:
        """解析 LLM 评估输出，包含兜底逻辑"""
        scores = []
        try:
            # 尝试提取 JSON 数组
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for item in parsed:
                    scores.append(DocumentScore(
                        doc_index=item.get("doc_index", len(scores)),
                        score=min(1.0, max(0.0, float(item.get("score", 0.5)))),
                        reason=item.get("reason", ""),
                    ))
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[CRAG] 评估结果解析失败: {e}, 使用默认分数")

        # 兜底: 未解析到足够结果时补充默认分数
        while len(scores) < expected_count:
            scores.append(DocumentScore(
                doc_index=len(scores),
                score=0.5,
                reason="评估解析失败，使用默认分数",
            ))
        return scores[:expected_count]


class KnowledgeRefiner:
    """
    知识精炼器 — Decompose-Recompose 算法

    论文 Section 3.2:
    1. Decompose: 将检索文档拆分为句子级「知识条」(knowledge strips)
    2. Filter:    LLM 逐条判断知识条与查询的相关性
    3. Recompose:  将相关知识条重新组装为精炼上下文

    目的: 即使文档整体相关, 其中也可能有大量与查询无关的冗余段落.
    知识精炼确保仅保留直接有助于回答的信息, 提高上下文信息密度.
    """

    # 知识条最少长度（过短的strip信息价值低）
    MIN_STRIP_LENGTH = 10
    # 单次过滤的最大 strip 数（控制 prompt 长度）
    FILTER_BATCH_SIZE = 15

    def refine(
        self, question: str, documents: list[dict], doc_scores: list[DocumentScore]
    ) -> tuple[str, int, int]:
        """
        Decompose-Recompose 精炼

        Args:
            question: 用户问题
            documents: 原始检索文档
            doc_scores: 文档评分（用于按分数优先排序）

        Returns:
            (refined_context, total_strips_count, relevant_strips_count)
        """
        # ---- Step 1: Decompose — 按句子拆分为知识条 ----
        all_strips: list[KnowledgeStrip] = []
        strip_id_counter = 0

        # 按评分降序处理文档
        scored_docs = sorted(
            zip(documents, doc_scores),
            key=lambda x: x[1].score,
            reverse=True,
        )

        for doc, ds in scored_docs:
            # 跳过评分极低的文档
            if ds.score < 0.15:
                continue

            doc_content = doc.get("content", "")
            title = doc.get("title", "")

            # 句子级拆分: 按中文句号、问号、叹号、分号 + 英文标点
            sentences = re.split(
                r'(?<=[。！？；\.\!\?\;])\s*', doc_content
            )

            for sent in sentences:
                sent = sent.strip()
                if len(sent) < self.MIN_STRIP_LENGTH:
                    continue
                all_strips.append(KnowledgeStrip(
                    strip_id=strip_id_counter,
                    content=sent,
                    source_doc_index=ds.doc_index,
                ))
                strip_id_counter += 1

        total_strips = len(all_strips)
        if total_strips == 0:
            return "", 0, 0

        # ---- Step 2: Filter — LLM 批量判断相关性 ----
        relevant_strips: list[KnowledgeStrip] = []

        for batch_start in range(0, len(all_strips), self.FILTER_BATCH_SIZE):
            batch = all_strips[batch_start:batch_start + self.FILTER_BATCH_SIZE]
            filtered_batch = self._filter_strips_batch(question, batch)
            relevant_strips.extend(
                s for s in filtered_batch if s.is_relevant
            )

        relevant_count = len(relevant_strips)
        logger.info(
            f"[CRAG] KnowledgeRefiner: "
            f"{relevant_count}/{total_strips} strips 判定为相关"
        )

        # ---- Step 3: Recompose — 按原始文档顺序重组 ----
        if not relevant_strips:
            return "", total_strips, 0

        # 按 (source_doc_index, strip_id) 排序保持原文顺序
        relevant_strips.sort(key=lambda s: (s.source_doc_index, s.strip_id))

        # 按源文档分组组装
        refined_parts = []
        current_doc_idx = -1
        for strip in relevant_strips:
            if strip.source_doc_index != current_doc_idx:
                current_doc_idx = strip.source_doc_index
                # 找到对应文档标题
                if current_doc_idx < len(documents):
                    title = documents[current_doc_idx].get("title", "")
                    if title:
                        refined_parts.append(f"\n[来源: {title}]")
            refined_parts.append(strip.content)

        refined_context = "\n".join(refined_parts).strip()
        return refined_context, total_strips, relevant_count

    def _filter_strips_batch(
        self, question: str, strips: list[KnowledgeStrip]
    ) -> list[KnowledgeStrip]:
        """批量过滤知识条的相关性"""
        strips_text_parts = []
        for s in strips:
            strips_text_parts.append(f"[知识条{s.strip_id}] {s.content}")
        strips_text = "\n".join(strips_text_parts)

        prompt = STRIP_FILTER_PROMPT.format(
            question=question,
            strips=strips_text,
        )

        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": "你是信息过滤专家，只返回 JSON 数组。"},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=600,
                temperature=0.1,
            )
            if response.status_code == 200:
                raw = response.output.choices[0].message.content
                return self._parse_filter_response(raw, strips)
        except Exception as e:
            logger.error(f"[CRAG] KnowledgeRefiner 过滤调用失败: {e}")

        # 兜底: 保守策略 — 保留所有 strips
        for s in strips:
            s.is_relevant = True
        return strips

    def _parse_filter_response(
        self, raw: str, strips: list[KnowledgeStrip]
    ) -> list[KnowledgeStrip]:
        """解析过滤结果"""
        strip_map = {s.strip_id: s for s in strips}
        try:
            json_match = re.search(r'\[.*\]', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                for item in parsed:
                    sid = item.get("strip_id")
                    if sid in strip_map:
                        strip_map[sid].is_relevant = bool(item.get("is_relevant", False))
                        strip_map[sid].relevance_reason = item.get("reason", "")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[CRAG] 过滤结果解析失败: {e}, 保留所有 strips")
            for s in strips:
                s.is_relevant = True
        return strips


class QueryRewriter:
    """
    查询重写器

    论文 Section 3.3:
    当检索评估为 AMBIGUOUS 或 INCORRECT 时触发.
    利用 LLM 将用户原始查询改写为多个更精确的搜索查询变体,
    通过不同角度的表述提高召回率.

    策略:
    1. 关键词提取 — 去除冗余修饰, 保留核心概念
    2. 同义词替换 — 用不同表达描述同一需求
    3. 问题分解   — 复合问题拆分为子问题
    """

    def rewrite(self, question: str) -> list[str]:
        """
        将原始查询改写为多个搜索变体

        Args:
            question: 用户原始查询

        Returns:
            改写后的查询列表 (含原始查询)
        """
        prompt = QUERY_REWRITE_PROMPT.format(question=question)

        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[
                    {"role": "system", "content": "你是查询优化专家，只返回 JSON 格式。"},
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                max_tokens=300,
                temperature=0.3,
            )
            if response.status_code == 200:
                raw = response.output.choices[0].message.content
                queries = self._parse_rewrite_response(raw)
                if queries:
                    logger.info(
                        f"[CRAG] QueryRewriter: 原始='{question}' → "
                        f"重写 {len(queries)} 个变体"
                    )
                    return queries
        except Exception as e:
            logger.error(f"[CRAG] QueryRewriter 调用失败: {e}")

        # 兜底: 简单的规则重写
        return self._rule_based_rewrite(question)

    def _parse_rewrite_response(self, raw: str) -> list[str]:
        """解析重写响应"""
        try:
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                queries = parsed.get("queries", [])
                # 过滤空查询和过长查询
                return [q.strip() for q in queries if q.strip() and len(q) < 200]
        except (json.JSONDecodeError, Exception):
            pass
        return []

    def _rule_based_rewrite(self, question: str) -> list[str]:
        """
        规则兜底重写 — 当 LLM 调用失败时使用

        策略: 提取问题中的关键名词和动词, 组成简短查询
        """
        # 去除常见问句词
        cleaned = re.sub(
            r'(请问|请|怎么|如何|什么|哪些|有没有|可以|能不能|吗|呢|的|了|是)',
            '',
            question,
        )
        cleaned = cleaned.strip()
        if cleaned and cleaned != question:
            return [cleaned]
        return []


class CRAGPipeline:
    """
    CRAG 完整流水线编排

    将 RetrievalEvaluator → KnowledgeRefiner → QueryRewriter 组合为
    三条完整的处理路径:

    路径 A (CORRECT):
      检索文档 → KnowledgeRefiner 精炼 → 高质量上下文

    路径 B (AMBIGUOUS):
      检索文档 → KnowledgeRefiner 精炼部分结果
                → QueryRewriter 重写查询
                → 二次检索
                → 合并去重
                → 再次精炼
                → 补充上下文

    路径 C (INCORRECT):
      QueryRewriter 重写查询
      → 重新检索
      → 如有结果 → KnowledgeRefiner 精炼
      → 如无结果 → 返回空上下文 + 标记 no_context
    """

    def __init__(self, retriever_fn=None):
        """
        Args:
            retriever_fn: 检索回调函数 (query, top_k) -> list[dict]
                          由 RAGEngine 注入, 用于 CRAG 内部二次检索
        """
        self.evaluator = RetrievalEvaluator()
        self.refiner = KnowledgeRefiner()
        self.rewriter = QueryRewriter()
        self.retriever_fn = retriever_fn

    def process(
        self,
        question: str,
        documents: list[dict],
        top_k: int = 3,
    ) -> CRAGResult:
        """
        CRAG 主流程

        Args:
            question: 用户问题
            documents: 初始检索到的文档列表
                       每个 dict 需包含 "title" 和 "content" 字段
            top_k: 最终返回的上下文条数上限

        Returns:
            CRAGResult 包含精炼后的上下文和完整处理信息
        """
        result = CRAGResult(original_query=question)

        # ---- Phase 1: 检索质量评估 ----
        overall_grade, doc_scores = self.evaluator.evaluate(question, documents)
        result.retrieval_grade = overall_grade
        result.doc_scores = doc_scores

        # ---- Phase 2: 按评估结果分路处理 ----
        if overall_grade == RetrievalGrade.CORRECT:
            result = self._path_correct(question, documents, doc_scores, result)

        elif overall_grade == RetrievalGrade.AMBIGUOUS:
            result = self._path_ambiguous(
                question, documents, doc_scores, result, top_k
            )

        else:  # INCORRECT
            result = self._path_incorrect(question, result, top_k)

        logger.info(
            f"[CRAG] Pipeline 完成: grade={result.retrieval_grade.value}, "
            f"path={result.processing_path}, "
            f"strips={result.relevant_strips}/{result.total_strips}, "
            f"context_len={len(result.refined_context)}"
        )
        return result

    def _path_correct(
        self,
        question: str,
        documents: list[dict],
        doc_scores: list[DocumentScore],
        result: CRAGResult,
    ) -> CRAGResult:
        """路径 A: 检索质量好 → 精炼后直接使用"""
        result.processing_path = "correct"

        # 只精炼评分 >= threshold 的文档
        good_docs = [
            documents[ds.doc_index]
            for ds in doc_scores
            if ds.score >= RetrievalEvaluator.CORRECT_THRESHOLD
            and ds.doc_index < len(documents)
        ]
        good_scores = [
            ds for ds in doc_scores
            if ds.score >= RetrievalEvaluator.CORRECT_THRESHOLD
        ]

        refined, total, relevant = self.refiner.refine(
            question, good_docs, good_scores
        )
        result.refined_context = refined
        result.total_strips = total
        result.relevant_strips = relevant
        return result

    def _path_ambiguous(
        self,
        question: str,
        documents: list[dict],
        doc_scores: list[DocumentScore],
        result: CRAGResult,
        top_k: int,
    ) -> CRAGResult:
        """路径 B: 检索质量模糊 → 精炼已有 + 重写补充检索"""
        result.processing_path = "ambiguous"

        # Step 1: 精炼当前已有的文档（保留分数不低于 INCORRECT_THRESHOLD 的）
        usable_docs = [
            documents[ds.doc_index]
            for ds in doc_scores
            if ds.score >= RetrievalEvaluator.INCORRECT_THRESHOLD
            and ds.doc_index < len(documents)
        ]
        usable_scores = [
            ds for ds in doc_scores
            if ds.score >= RetrievalEvaluator.INCORRECT_THRESHOLD
        ]

        refined_existing, total_1, relevant_1 = self.refiner.refine(
            question, usable_docs, usable_scores
        )

        # Step 2: 重写查询 + 二次检索
        rewritten_queries = self.rewriter.rewrite(question)
        result.rewritten_queries = rewritten_queries

        supplementary_context = ""
        if rewritten_queries and self.retriever_fn:
            # 用重写查询做补充检索
            all_new_docs = []
            seen_titles = set(d.get("title", "") for d in documents)

            for rq in rewritten_queries:
                try:
                    new_docs = self.retriever_fn(rq, top_k)
                    for nd in new_docs:
                        # 去重: 避免重复文档
                        nd_title = nd.get("title", "")
                        if nd_title and nd_title not in seen_titles:
                            all_new_docs.append(nd)
                            seen_titles.add(nd_title)
                except Exception as e:
                    logger.warning(f"[CRAG] 补充检索失败 ({rq}): {e}")

            if all_new_docs:
                # 对补充文档也做评估和精炼
                _, new_scores = self.evaluator.evaluate(question, all_new_docs)
                supplementary, total_2, relevant_2 = self.refiner.refine(
                    question, all_new_docs, new_scores
                )
                supplementary_context = supplementary
                total_1 += total_2
                relevant_1 += relevant_2

        # Step 3: 合并上下文
        context_parts = []
        if refined_existing:
            context_parts.append(refined_existing)
        if supplementary_context:
            context_parts.append(supplementary_context)

        result.refined_context = "\n\n".join(context_parts)
        result.total_strips = total_1
        result.relevant_strips = relevant_1
        return result

    def _path_incorrect(
        self,
        question: str,
        result: CRAGResult,
        top_k: int,
    ) -> CRAGResult:
        """路径 C: 检索质量差 → 重写查询重新检索"""
        result.processing_path = "incorrect"

        # 重写查询
        rewritten_queries = self.rewriter.rewrite(question)
        result.rewritten_queries = rewritten_queries

        if not rewritten_queries or not self.retriever_fn:
            # 完全兜底: 无可用上下文
            result.refined_context = ""
            logger.warning(
                f"[CRAG] INCORRECT 路径: 无法重写或无检索器, "
                f"将使用 LLM 内部知识兜底"
            )
            return result

        # 用重写查询重新检索
        all_new_docs = []
        seen_titles: set[str] = set()

        for rq in rewritten_queries:
            try:
                new_docs = self.retriever_fn(rq, top_k)
                for nd in new_docs:
                    nd_title = nd.get("title", "")
                    if nd_title not in seen_titles:
                        all_new_docs.append(nd)
                        seen_titles.add(nd_title)
            except Exception as e:
                logger.warning(f"[CRAG] 重新检索失败 ({rq}): {e}")

        if not all_new_docs:
            result.refined_context = ""
            return result

        # 对新检索结果做评估 + 精炼
        _, new_scores = self.evaluator.evaluate(question, all_new_docs)
        refined, total, relevant = self.refiner.refine(
            question, all_new_docs, new_scores
        )
        result.refined_context = refined
        result.total_strips = total
        result.relevant_strips = relevant
        return result
