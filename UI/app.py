"""
=============================================================================
UI/app.py —— AI 编码代理 Gradio 界面（HuggingFace Spaces 可部署）
=============================================================================

基于 main.py 的 8 阶段流水线：感知→规划→调度→执行→验证→整合→输出→复盘

【功能】
  1. 输入需求 → 启动流水线
  2. 需求澄清 → 对话式追问
  3. 人工介入裁决 → 沙盒报错 + 模型思考 + 决策按钮
  4. 断点续传 → 从 thread_id 恢复
  5. HF Spaces 兼容 → Docker 不可用时自动降级为本地 subprocess 沙盒

【部署到 HuggingFace Spaces】
  1. 将此仓库推送到 HF Space
  2. 在 Space Settings 中设置环境变量：LLM_API_KEY, BASE_URL, PROCESS_MODEL, CODING_MODEL
  3. Space SDK 选 Gradio，入口文件指向 ui/app.py

【本地运行】
  cd Agent_upgrade && python UI/app.py

【文件结构】
  sandbox.py  — 沙盒兼容层（Docker / 本地 fallback）
  helpers.py  — 显示格式化 & 辅助函数
  pipeline.py — 流水线执行核心
  handlers.py — Gradio 事件处理
  layout.py   — Gradio 布局 & CSS
  app.py      — 入口 & 组装
=============================================================================
"""

import sys
import os

# 确保项目根在 sys.path 中（支持直接 python UI/app.py 运行）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import gradio as gr
from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from core.graph.main import build_master_graph

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: 沙盒初始化（sandbox 模块导入即触发 Docker/本地 fallback + monkey-patch）
# ═══════════════════════════════════════════════════════════════════════════

import UI.sandbox  # noqa: E402, F401 — 副作用导入：触发沙盒探测

# ═══════════════════════════════════════════════════════════════════════════
# Step 2: 构建图谱
# ═══════════════════════════════════════════════════════════════════════════

checkpointer = MemorySaver()
app_graph = build_master_graph(checkpointer=checkpointer)
print("[UI] LangGraph 流水线已初始化")

# ═══════════════════════════════════════════════════════════════════════════
# Step 3: 导入拆分模块 & 用闭包注入 app_graph
# ═══════════════════════════════════════════════════════════════════════════

from UI.helpers import list_workspace_files, get_session_list  # noqa: E402
from UI.handlers import (  # noqa: E402
    on_start as _on_start,
    on_clarify as _on_clarify,
    on_intervene as _on_intervene,
    on_action_change,
)
from UI.layout import build_ui  # noqa: E402

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
# Step 4: 构建 UI 并启动
# ═══════════════════════════════════════════════════════════════════════════

demo = build_ui(
    on_start=on_start,
    on_clarify=on_clarify,
    on_intervene=on_intervene,
    on_action_change=on_action_change,
    list_workspace_files=list_workspace_files,
    list_sessions=list_sessions,
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
