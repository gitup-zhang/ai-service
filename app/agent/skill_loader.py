"""
Agent Skill 加载器 — 按需加载模块化领域知识

Agent Skills 是 2025 年 Anthropic 推出的开放标准:
- Tools 是 Agent 的"手"（做什么）
- Skills 是 Agent 的"脑"（知道怎么做）

Skill 以 SKILL.md 文件为核心, 包含角色定位、分析框架、工具策略和输出规范.
Agent 执行前根据用户意图动态加载最匹配的 Skill.
"""

import os
import re
import yaml
from dashscope import Generation
from app.config import settings
from app.agent.schemas import SkillMeta


class SkillLoader:
    """Agent Skill 加载器 — 发现、选择、加载领域知识"""

    DEFAULT_SKILL = SkillMeta(
        name="general",
        description="通用助手，提供综合性回答",
        trigger_keywords=[],
        recommended_tools=[],
    )

    def __init__(self, skills_dir: str | None = None):
        if skills_dir is None:
            # 默认在 app/agent/skills 下
            skills_dir = os.path.join(
                os.path.dirname(__file__), "skills"
            )
        self.skills_dir = skills_dir
        self.skills: dict[str, SkillMeta] = {}
        self._discover_skills()

    def _discover_skills(self):
        """扫描 skills 目录, 解析所有 SKILL.md 的 frontmatter 元数据"""
        if not os.path.exists(self.skills_dir):
            return

        for entry in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, entry, "SKILL.md")
            if os.path.isfile(skill_path):
                meta = self._parse_frontmatter(skill_path)
                if meta:
                    meta.file_path = skill_path
                    self.skills[meta.name] = meta

    def _parse_frontmatter(self, filepath: str) -> SkillMeta | None:
        """解析 SKILL.md 的 YAML frontmatter"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 提取 --- 之间的 YAML
            match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
            if not match:
                return None

            meta_dict = yaml.safe_load(match.group(1))
            return SkillMeta(**meta_dict)
        except Exception as e:
            print(f"解析 Skill 失败 [{filepath}]: {e}")
            return None

    def select(self, user_query: str) -> SkillMeta:
        """
        根据用户输入, 选择最匹配的 Skill

        策略:
        1. 关键词匹配 (快速, 零成本)
        2. LLM 意图分类 (兜底, 更准确)
        """
        if not self.skills:
            return self.DEFAULT_SKILL

        # 策略 1: 关键词匹配
        best_match = None
        best_score = 0
        for skill in self.skills.values():
            score = sum(
                1 for kw in skill.trigger_keywords
                if kw in user_query
            )
            if score > best_score:
                best_score = score
                best_match = skill

        if best_match and best_score > 0:
            return best_match

        # 策略 2: LLM 意图分类
        return self._llm_classify(user_query)

    def _llm_classify(self, user_query: str) -> SkillMeta:
        """用 LLM 分析用户意图, 选择最匹配的 Skill"""
        skill_list = "\n".join(
            f"- {s.name}: {s.description}"
            for s in self.skills.values()
        )
        prompt = (
            f"用户说: \"{user_query}\"\n\n"
            f"以下是可用的 Skills:\n{skill_list}\n\n"
            f"请直接返回最匹配的 Skill name（只返回名称，不要其他内容）。"
            f"如果没有匹配的，返回 'general'。"
        )

        try:
            response = Generation.call(
                api_key=settings.DASHSCOPE_API_KEY,
                model=settings.DASHSCOPE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
                max_tokens=50,
                temperature=0,
            )
            if response.status_code == 200:
                name = response.output.choices[0].message.content.strip()
                if name in self.skills:
                    return self.skills[name]
        except Exception:
            pass

        return self.DEFAULT_SKILL

    def load(self, skill_name: str) -> str:
        """
        加载 Skill 的完整内容, 注入 Agent 上下文

        Returns:
            SKILL.md 的 markdown body（不含 frontmatter）
        """
        if skill_name not in self.skills:
            return ""

        filepath = self.skills[skill_name].file_path
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 去掉 frontmatter, 只返回 markdown body
            match = re.match(r'^---\s*\n.*?\n---\s*\n', content, re.DOTALL)
            if match:
                return content[match.end():]
            return content
        except Exception:
            return ""

    def list_skills(self) -> list[dict]:
        """列出所有可用 Skills (供 API 展示)"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "keywords": s.trigger_keywords,
                "tools": s.recommended_tools,
            }
            for s in self.skills.values()
        ]
