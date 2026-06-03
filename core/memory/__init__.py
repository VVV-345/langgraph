"""
core.memory 包 —— 向量记忆层

模块结构：
    embedding.py — Embedding 工厂函数（LangChain 原生）
    store.py    — Qdrant + MMR Retriever 经验存储

使用方式：
    from core.memory.embedding import create_embedder
    from core.memory.store import ExperienceStore

    embeddings = create_embedder()
    store = ExperienceStore(embeddings)
    store.connect()

    # 写入
    store.store_experience(task_summary="...", planning=..., execution=...)

    # 检索
    results = store.retrieve_similar("新任务描述", top_k=5)
"""

from core.memory.embedding import create_embedder
from core.memory.store import ExperienceStore

__all__ = ["create_embedder", "ExperienceStore"]
