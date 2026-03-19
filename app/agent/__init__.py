"""
Adaptive Agent Engine — 基于 LangGraph 的 ReAct + Reflexion 混合架构

核心创新:
- 置信度路由: Agent 输出后自动评估质量, 不达标触发反思
- Reflexion 自我反思: 分析失败原因, 生成改进策略
- 情景记忆: 存储反思经验, 检索历史避免重复犯错
- 模块化 Skills: 按需加载领域专业知识
"""
