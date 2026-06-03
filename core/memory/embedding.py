"""
=============================================================================
Embedding 工厂函数 —— 自动检测后端，返回 LangChain 原生 Embeddings 实例
=============================================================================

支持后端：
    1. 本地模型 → langchain_huggingface.HuggingFaceEmbeddings（默认）
    2. API 模型 → langchain_openai.OpenAIEmbeddings（OpenAI 兼容）

配置项（.env）：
    EMBEDDING_MODEL_PATH  — 本地模型路径（如 "D:/AI2test/bge-m3"）
    EMBEDDING_BASE_URL    — API 地址（OpenAI 兼容）
    EMBEDDING_API_KEY     — API Key（可选，默认复用 LLM_API_KEY）
    EMBEDDING_MODEL_NAME  — API 模型名（默认 text-embedding-3-small）
=============================================================================
"""

import os
from dotenv import load_dotenv
from core.logger import logger

load_dotenv()


def create_embedder():
    """
    自动检测后端配置，返回 LangChain Embeddings 实例。

    优先级：本地模型 > API 专用端点 > LLM 端点推测
    """
    model_path = os.getenv("EMBEDDING_MODEL_PATH", "")

    # ── 本地模型 ──
    if model_path and os.path.exists(model_path):
        logger.info(f"[Embedding] 使用本地模型: {model_path}")
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "本地模型需要 langchain_huggingface，请执行: pip install langchain-huggingface"
            )
        return HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # ── API 模式 ──
    base_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("BASE_URL")
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")

    if base_url and api_key:
        model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        logger.info(f"[Embedding] 使用 API 模式: {base_url} (model={model})")
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError:
            raise ImportError("langchain_openai 未安装")
        return OpenAIEmbeddings(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError(
        "未配置 Embedding 模型。请设置 EMBEDDING_MODEL_PATH 或 EMBEDDING_BASE_URL"
    )
