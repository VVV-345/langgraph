"""
=============================================================================
Qdrant 向量存储层 —— 经验读写 + MMR 语义检索
=============================================================================

基于 LangChain 原生 QdrantVectorStore + MMR Retriever。

配置项（.env）：
    QDRANT_PATH         — 本地持久化路径（默认 ./qdrant_data）
    QDRANT_COLLECTION   — Collection 名称（默认 "task_experiences"）
    EMBEDDING_DIMENSION — 向量维度（默认 1024，bge-m3 标准）
=============================================================================
"""

import os
import atexit
import uuid
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from core.logger import logger

load_dotenv()

# ── 全局共享的 QdrantClient（同一进程内复用，避免本地文件锁冲突）──
_shared_clients: dict = {}  # path → QdrantClient


def _cleanup_shared_clients():
    """解释器关闭前释放 QdrantClient，避免 shutdown 时报错"""
    for client in list(_shared_clients.values()):
        try:
            client.close()
        except Exception:
            pass
    _shared_clients.clear()


atexit.register(_cleanup_shared_clients)


class ExperienceStore:
    """
    Qdrant 经验存储（本地文件模式）。

    参数：
        embeddings: LangChain Embeddings 实例（由 create_embedder() 创建）
    """

    def __init__(self, embeddings):
        self._embeddings = embeddings
        self._collection_name = os.getenv("QDRANT_COLLECTION", "task_experiences")
        self._vector_store = None
        self._retriever = None
        self._connected = False

        # 默认路径：魔塔环境自动使用 /data/ 持久存储，本地用项目目录
        _default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "qdrant_data"
        )
        if os.path.isdir("/data") and not os.getenv("QDRANT_PATH"):
            _default_path = "/data/qdrant_data"
        self._qdrant_path = os.getenv("QDRANT_PATH", _default_path)

    # ======================================================================
    # 连接管理
    # ======================================================================

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """建立 Qdrant 连接并初始化 Collection + MMR Retriever。"""
        if self._connected:
            return True

        try:
            from qdrant_client import QdrantClient
            from langchain_qdrant import QdrantVectorStore

            # 本地文件模式：同一路径复用 client，避免文件锁冲突
            if self._qdrant_path not in _shared_clients:
                _shared_clients[self._qdrant_path] = QdrantClient(path=self._qdrant_path)
                logger.info(f"[Qdrant] 创建本地连接: {self._qdrant_path}")

            client = _shared_clients[self._qdrant_path]

            # 首次运行自动创建 collection
            if not client.collection_exists(self._collection_name):
                dim = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
                client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={"size": dim, "distance": "Cosine"},
                )
                logger.info(f"[Qdrant] 自动创建 Collection: {self._collection_name} (dim={dim})")

            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=self._collection_name,
                embedding=self._embeddings,
                distance="Cosine",
            )

            self._retriever = self._vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 5,
                    "fetch_k": 20,
                    "lambda_mult": 0.6,   # 0=最大多样性，1=最大相似度
                }
            )

            self._connected = True
            logger.info(f"[Qdrant] 连接成功，Collection: {self._collection_name}")
            return True

        except ImportError:
            logger.warning("[Qdrant] langchain_qdrant 未安装，无法使用向量存储")
            return False
        except Exception as e:
            logger.warning(f"[Qdrant] 连接失败: {e}")
            return False

    # ======================================================================
    # 写入经验
    # ======================================================================

    def store_experience(
        self,
        task_summary: str,
        task_complexity: str,
        planning_strategy: dict,
        execution_result: dict,
        pitfalls: list,
        tools_used: list,
        session_id: Optional[str] = None,
    ) -> bool:
        """将一次流水线执行的经验写入 Qdrant。失败返回 False，不抛异常。"""
        if not self._connected:
            return False

        try:
            from langchain_core.documents import Document

            page_content = self._build_page_content(
                task_summary, planning_strategy, execution_result, pitfalls
            )

            metadata = {
                "task_summary": task_summary[:500],
                "task_complexity": task_complexity,
                "sub_task_count": len(planning_strategy.get("task_plan", [])),
                "planning_strategy": planning_strategy,
                "execution_result": execution_result,
                "pitfalls": pitfalls,
                "tools_used": tools_used,
                "tags": self._extract_tags(task_summary, planning_strategy),
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id or str(uuid.uuid4())[:8],
            }

            doc = Document(page_content=page_content, metadata=metadata)
            # Qdrant 点 ID 必须是标准 UUID 格式，不能直接使用 session_id
            doc_id = str(uuid.uuid4())

            self._vector_store.add_documents([doc], ids=[doc_id])

            logger.info(
                f"[Qdrant] ✅ 经验已写入 (doc_id={doc_id}, "
                f"session={metadata['session_id']}, "
                f"success_rate={execution_result.get('success_rate', 0):.0%})"
            )
            return True

        except Exception as e:
            logger.warning(f"[Qdrant] 写入失败（非致命）: {e}")
            return False

    def _build_page_content(
        self,
        task_summary: str,
        planning_strategy: dict,
        execution_result: dict,
        pitfalls: list,
    ) -> str:
        """将结构化经验数据格式化为嵌入用文本。"""
        parts = [f"任务: {task_summary}"]

        task_plan = planning_strategy.get("task_plan", [])
        if task_plan:
            parts.append("拆解方案:")
            for t in task_plan:
                parts.append(
                    f"  - {t.get('description', '')}: {t.get('objective', '')} "
                    f"[{t.get('risk_level', 'low')}]"
                )

        sr = execution_result.get("success_rate", 0)
        parts.append(
            f"结果: 成功率 {sr:.0%}, "
            f"完成 {execution_result.get('finished', 0)} 个, "
            f"失败 {execution_result.get('failed', 0)} 个, "
            f"重试 {execution_result.get('retry_total', 0)} 次"
        )

        if pitfalls:
            parts.append("踩坑记录:")
            for p in pitfalls:
                parts.append(
                    f"  - [{p.get('task_id', '?')}] {p.get('root_cause', '')}: "
                    f"{p.get('how_fixed', '')}"
                )

        return "\n".join(parts)

    def _extract_tags(self, task_summary: str, planning_strategy: dict) -> list:
        """从任务描述和拆解方案中提取中文关键词标签。"""
        TAG_MAP = {
            "爬虫": ["爬虫", "网络请求", "数据采集"],
            "爬取": ["爬虫", "数据采集"],
            "数据处理": ["数据处理", "ETL"],
            "数据库": ["数据库", "存储"],
            "SQLite": ["SQLite", "数据库"],
            "MySQL": ["MySQL", "数据库"],
            "API": ["API", "接口"],
            "Web": ["Web", "后端"],
            "Flask": ["Flask", "Web框架"],
            "Django": ["Django", "Web框架"],
            "前端": ["前端", "UI"],
            "React": ["React", "前端"],
            "Vue": ["Vue", "前端"],
            "脚本": ["脚本", "自动化"],
            "批处理": ["批处理", "自动化"],
            "测试": ["测试"],
            "文件": ["文件IO"],
            "分析": ["数据分析"],
            "统计": ["统计分析"],
            "可视化": ["数据可视化"],
            "机器学习": ["ML"],
            "深度学习": ["DL"],
            "图像": ["图像处理"],
            "视频": ["视频处理"],
            "并发": ["并发", "多线程"],
            "异步": ["异步", "协程"],
            "命令行": ["CLI"],
            "GUI": ["GUI"],
        }

        combined = task_summary + " "
        for t in planning_strategy.get("task_plan", []):
            combined += t.get("description", "") + " " + t.get("objective", "") + " "

        tags = set()
        for keyword, tag_list in TAG_MAP.items():
            if keyword.lower() in combined.lower():
                tags.update(tag_list)

        return sorted(tags)[:10]

    # ======================================================================
    # 检索相似经验
    # ======================================================================

    def retrieve_similar(self, query: str, top_k: int = 5) -> list:
        """MMR 语义检索与查询最相似的历史经验。返回 List[Document]。"""
        if not self._connected:
            return []

        try:
            self._retriever.search_kwargs["k"] = top_k
            results = self._retriever.invoke(query)
            logger.info(f"[Qdrant] MMR 检索返回 {len(results)} 条经验 (query=\"{query[:50]}...\")")
            return results
        except Exception as e:
            logger.warning(f"[Qdrant] 检索失败（降级返回空列表）: {e}")
            return []

    def count(self) -> int:
        """返回 Collection 中的经验总数。"""
        if not self._connected or not self._vector_store:
            return 0
        try:
            return self._vector_store.client.count(
                collection_name=self._collection_name
            ).count
        except Exception:
            return 0
