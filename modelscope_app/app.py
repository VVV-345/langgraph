"""
=============================================================================
app.py —— AI 编码代理 Gradio 界面（魔塔社区 ModelScope 可部署）
=============================================================================

【部署方式（魔塔社区）】
  1. 将此仓库上传到魔塔 Space
  2. 在魔塔 Space 设置中配置环境变量（Settings → Environment Variables）：
     - LLM_API_KEY      : LLM API 密钥
     - BASE_URL         : LLM API 地址
     - PROCESS_MODEL    : 分析/规划/整合用的模型（推荐 deepseek-v3 或 qwen-max）
     - CODING_MODEL     : 编码用的模型（推荐 deepseek-v3 或 qwen-coder）
  3. Space SDK 选 Gradio，入口文件指向 modelscope_app/app.py

【本地运行】
  cd Agent_upgrade && python -m modelscope_app.app
  或：python modelscope_app/app.py（需设置 PYTHONPATH 到项目根）

【核心改动——本地沙盒替代 Docker】
  Docker 在魔塔不可用，启动时自动把 core.tools.docker_sandbox 的所有函数
  替换为 subprocess 本地执行版。沙盒测代码、文件读写都在本地工作区完成。
  替换时机在 import core.* 之前，对后续所有节点透明。

【文件结构】
  local_sandbox.py  — 沙盒环境注入
  ui_helpers.py     — 显示格式化 & 辅助函数
  pipeline.py       — 流水线执行核心
  event_handlers.py — Gradio 事件处理
  ui_layout.py      — Gradio 布局 & CSS
  app.py            — 入口 & 组装
=============================================================================
"""

import sys
import os

# 确保项目根在 sys.path 中（魔塔部署时 modelscope_app 可能不在自动路径里）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: 沙盒注入（必须在 import core.* 之前）
# ═══════════════════════════════════════════════════════════════════════════

from modelscope_app.local_sandbox import _LOCAL_WORKSPACE  # noqa: E402, F401 — 触发沙盒替换

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: 环境变量 & 核心模块
# ═══════════════════════════════════════════════════════════════════════════

import gradio as gr  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from core.graph.main import build_master_graph  # noqa: E402
from core.logger import logger  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: 构建图谱
# ═══════════════════════════════════════════════════════════════════════════

checkpointer = MemorySaver()
app_graph = build_master_graph(checkpointer=checkpointer)
print("[UI] ✅ LangGraph 8 阶段流水线已就绪")

# ═══════════════════════════════════════════════════════════════════════════
# Step 4: 导入拆分模块 & 用闭包注入 app_graph
# ═══════════════════════════════════════════════════════════════════════════

from modelscope_app.ui_helpers import list_workspace, get_session_list  # noqa: E402
from modelscope_app.event_handlers import (  # noqa: E402
    on_start as _on_start,
    on_clarify as _on_clarify,
    on_intervene as _on_intervene,
    on_action_change,
)
from modelscope_app.ui_layout import build_ui  # noqa: E402

# 将 app_graph 注入到需要它的 handler 中
# 注意：必须用显式 yield from 包装，不能直接用 lambda。
# Gradio 通过 inspect.isgeneratorfunction() 检测生成器，
# lambda 本身不是生成器函数，会导致 Gradio 不迭代直接把生成器对象当单值返回。
def on_start(ur, rid):
    yield from _on_start(app_graph, ur, rid)

def on_clarify(ans, st, tid):
    yield from _on_clarify(app_graph, ans, st, tid)

def on_intervene(act, cus, st, tid, ii):
    yield from _on_intervene(app_graph, act, cus, st, tid, ii)

def list_sessions():
    """刷新断点续传下拉列表 → 返回 gr.Dropdown.update(choices=...)"""
    items = get_session_list(checkpointer, app_graph)
    return gr.update(choices=items, value=None)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5: 构建 UI 并启动
# ═══════════════════════════════════════════════════════════════════════════

demo = build_ui(
    on_start=on_start,
    on_clarify=on_clarify,
    on_intervene=on_intervene,
    on_action_change=on_action_change,
    list_workspace=list_workspace,
    list_sessions=list_sessions,
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
