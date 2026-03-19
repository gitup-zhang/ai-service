"""
AI 微服务配置管理
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """应用配置"""

    # LLM 配置
    DASHSCOPE_API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    DASHSCOPE_MODEL: str = os.getenv("DASHSCOPE_MODEL", "qwen-turbo")
    DASHSCOPE_EMBEDDING_MODEL: str = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")

    # 后端业务 API
    BACKEND_API_URL: str = os.getenv("BACKEND_API_URL", "http://47.113.194.28:8080/api")

    # 向量数据库
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "./chroma_db")

    # 服务配置
    AI_SERVICE_HOST: str = os.getenv("AI_SERVICE_HOST", "0.0.0.0")
    AI_SERVICE_PORT: int = int(os.getenv("AI_SERVICE_PORT", "8000"))

    # LangSmith 链路追踪配置 (如需开启请在环境中配置对应的环境变量)
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "ai-service")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")


settings = Settings()
