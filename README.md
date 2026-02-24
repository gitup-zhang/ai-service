# AI 微服务

为 UniApp 新闻资讯应用提供 AI 能力的独立后端服务。

## 功能

| 模块     | 接口                          | 说明                   |
| -------- | ----------------------------- | ---------------------- |
| 文章 AI  | `/api/ai/article/summary`     | 生成文章摘要           |
| 文章 AI  | `/api/ai/article/explanation` | 通俗解读文章           |
| 文章 AI  | `/api/ai/article/impact`      | 影响分析               |
| 活动 AI  | `/api/ai/event/summary`       | 活动摘要               |
| 活动 AI  | `/api/ai/event/qa`            | 活动智能问答           |
| 活动 AI  | `/api/ai/event/guide`         | 报名指南               |
| 智能对话 | `/api/ai/chat/ask`            | 政策问答               |
| 语义搜索 | `/api/ai/search/`             | RAG 语义搜索 (Phase 2) |

## 快速开始

### 1. 创建 conda 环境

```bash
conda create -n ai-service python=3.10
conda activate ai-service
```

### 2. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 DashScope API Key
```

### 4. 启动服务

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动后访问 http://localhost:8000/docs 查看 API 文档。

## Docker 部署

```bash
docker-compose up -d
```

## 技术栈

- **FastAPI** - Web 框架
- **DashScope** - 通义千问 LLM
- **LlamaIndex** - RAG 框架 (Phase 2)
- **ChromaDB** - 向量数据库 (Phase 2)
- **LangChain** - MCP 工具链 (Phase 3)

## 项目结构

```
ai-service/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py             # 配置管理
│   ├── api/                  # API 路由
│   │   ├── article.py        # 文章 AI 接口
│   │   ├── event.py          # 活动 AI 接口
│   │   ├── chat.py           # 智能对话接口
│   │   └── search.py         # 语义搜索接口
│   ├── core/                 # 核心模块
│   │   ├── llm.py            # LLM 调用封装
│   │   ├── rag_engine.py     # RAG 引擎 (Phase 2)
│   │   └── data_sync.py      # 数据同步 (Phase 2)
│   ├── tools/                # MCP 工具 (Phase 3)
│   │   └── event_tools.py    # 活动相关工具
│   └── prompts/              # Prompt 模板
│       ├── article_prompts.py
│       ├── event_prompts.py
│       └── chat_prompts.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```
