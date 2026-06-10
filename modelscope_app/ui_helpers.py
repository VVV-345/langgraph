"""
=============================================================================
ui_helpers.py —— 显示格式化 & 辅助工具函数
=============================================================================

提供 Gradio 界面所需的各种 HTML / Markdown 渲染函数，以及工作区文件列表。
=============================================================================
"""

import os
import shutil

from .local_sandbox import _LOCAL_WORKSPACE


# ==========================================================================
# 断点续传——枚举历史会话
# ==========================================================================

def get_session_list(checkpointer, app_graph, show_completed: bool = False) -> list:
    """
    从 MemorySaver 中枚举所有历史会话，返回 [(label, thread_id), ...]。
    按最近优先排序。默认隐藏已完成任务。
    """
    items = []
    if not hasattr(checkpointer, "storage"):
        return items

    thread_ids = sorted(checkpointer.storage.keys(), reverse=True)

    for tid in thread_ids:
        if not tid:
            continue
        try:
            saved = app_graph.get_state({"configurable": {"thread_id": tid}})
            state = saved.values if saved else {}
        except Exception:
            continue

        if not state:
            continue

        task_desc = _extract_task_summary(state)

        planning = state.get("planning")
        progress = ""
        all_finished = False
        if planning and planning.task_plan:
            done = sum(1 for t in planning.task_plan if t.status == "finished")
            fail = sum(1 for t in planning.task_plan if t.status == "failed")
            total = len(planning.task_plan)
            progress = f"({done}/{total}"
            if fail > 0:
                progress += f", {fail} 失败"
            progress += ")"
            all_finished = (done == total and total > 0)

        blocked = state.get("react_blocked", False)
        need_clarify = planning.need_clarification if planning else False
        tag = ""
        if all_finished:
            tag = " ✅已完成"
        elif blocked:
            tag = " ⚠️待决策"
        elif need_clarify:
            tag = " 🤔待澄清"

        label = f"[{tid[:16]}] {task_desc[:40]} {progress}{tag}"
        items.append((label, tid, all_finished))

    # 默认过滤已完成任务
    if not show_completed:
        items = [(label, tid) for label, tid, finished in items if not finished]
    else:
        items = [(label, tid) for label, tid, _ in items]

    return items


def _extract_task_summary(state: dict) -> str:
    """从 state 中提取用户原始需求简述"""
    msgs = state.get("messages", [])
    for m in msgs:
        if hasattr(m, "type") and m.type == "human":
            return str(m.content)[:60].replace("\n", " ")
    planning = state.get("planning")
    if planning and planning.task_plan:
        return planning.task_plan[0].description[:60]
    return "(未知任务)"


# ==========================================================================
# 沙盒命令简写
# ==========================================================================

def _docker(cmd: str, timeout: int = 5) -> dict:
    """简写：调本地沙盒执行命令"""
    from core.tools.docker_sandbox import docker_exec
    return docker_exec(cmd, timeout=timeout)


# ==========================================================================
# 工作区文件列表
# ==========================================================================

def get_file_list() -> list:
    """返回工作区文件路径列表（纯字符串，供 Dropdown choices）"""
    w = _LOCAL_WORKSPACE
    if not os.path.isdir(w):
        return ["(工作区为空)"]
    choices = []
    for root, dirs, files in os.walk(w):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            fp = os.path.relpath(os.path.join(root, fn), w)
            choices.append(fp)
    return sorted(choices)[:100] if choices else ["(工作区为空)"]


def list_workspace():
    """列出工作区文件 → 返回 gr.Dropdown.update(choices=...)"""
    import gradio as gr
    return gr.update(choices=get_file_list(), value=None)


def preview_file(filename: str) -> str:
    """读取工作区文件内容，返回文本。大文件截断前 3000 行。"""
    if not filename or filename == "(工作区为空)":
        return "# 选择一个文件即可预览"
    w = _LOCAL_WORKSPACE
    filepath = os.path.join(w, filename)
    real = os.path.realpath(filepath)
    if not real.startswith(os.path.realpath(w)):
        return "# ⛔ 不允许访问工作区外的文件"
    if not os.path.isfile(filepath):
        return f"# ⚠️ 文件不存在: {filename}"
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= 3000:
                    lines.append(f"\n... (截断，文件共 {i+1}+ 行)")
                    break
                lines.append(line)
            return "".join(lines)
    except Exception as e:
        return f"# ❌ 读取失败: {e}"


def clear_workspace():
    """清空工作区所有文件，返回更新后的文件列表"""
    w = _LOCAL_WORKSPACE
    if os.path.isdir(w):
        for item in os.listdir(w):
            item_path = os.path.join(w, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception:
                pass
    import gradio as gr
    return gr.update(choices=get_file_list(), value=None)


# ==========================================================================
# 子任务进度 HTML
# ==========================================================================

def task_progress_html(state: dict) -> str:
    """子任务进度 HTML"""
    planning = state.get("planning")
    if not planning or not planning.task_plan:
        return '<p style="color:#888">暂无任务</p>'

    emoji = {
        "finished": "✅", "pending": "🔄", "failed": "❌",
        "testing": "🧪", "doing": "⚙️",
    }
    execution = state.get("execution")
    parts = ['<div style="font-size:13px;line-height:1.8">']
    for t in planning.task_plan:
        e = emoji.get(t.status, "⏳")
        desc = t.description[:50]
        extra = ""
        if execution and execution.task_retry_count.get(t.task_id, 0) > 0:
            extra = f" <span style='color:#f80'>(×{execution.task_retry_count[t.task_id]})</span>"
        parts.append(
            f'<div style="margin:2px 0;padding:4px 8px;'
            f'background:#e8e8e8;color:#1a1a1a;border-radius:4px;'
            f'border-left:3px solid {"#52c41a" if t.status=="finished" else "#faad14" if t.status=="testing" else "#ff4d4f" if t.status=="failed" else "#d9d9d9"}">'
            f'{e} <b>T{t.task_id}</b>: {desc}{extra}</div>'
        )
    parts.append('</div>')
    return "".join(parts)


# ==========================================================================
# 消息格式转换
# ==========================================================================

def messages_to_chat(state: dict) -> list:
    """LangChain 消息 → Gradio chatbot 格式"""
    chat = []
    for m in state.get("messages", []):
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        content = str(getattr(m, "content", ""))[:2000].strip()
        if content:
            chat.append({"role": role, "content": content})
    return chat


# ==========================================================================
# 错误 / 思考 / 摘要渲染
# ==========================================================================

def build_error_html(state: dict) -> str:
    """提取沙盒报错，格式化为 HTML 卡片"""
    execution = state.get("execution")
    if not execution:
        return ""
    error_text = (execution.error_trace or "").strip()
    if not error_text:
        return ""

    error_text = error_text[:2500]
    retry = execution.retry_count
    task_retry = execution.task_retry_count or {}

    html = [
        '<div style="background:#fff2f0;border:1px solid #ff4d4f;'
        'border-radius:8px;padding:12px;margin:8px 0">',
        '<div style="font-weight:bold;color:#cf1322;font-size:15px;margin-bottom:8px">'
        f'🔴 沙盒验证失败（全局重试 {retry} 次）</div>',
    ]
    if task_retry:
        html.append(
            '<div style="font-size:12px;color:#666;margin-bottom:6px">'
            "各任务重试: " +
            ", ".join(f"T{k} ×{v}" for k, v in sorted(task_retry.items())) +
            '</div>'
        )
    html.append(
        f'<pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;'
        f'border-radius:4px;font-size:12px;max-height:260px;'
        f'overflow-y:auto;white-space:pre-wrap;word-break:break-all;'
        f'line-height:1.5">'
        f'{error_text}</pre>'
        f'</div>'
    )
    return "".join(html)


def build_thought_html(state: dict) -> str:
    """提取模型最后思考"""
    # 优先从 react_history 倒查
    react_history = state.get("react_history", [])
    thought = ""
    if react_history:
        for step in reversed(react_history):
            t = step.get("thought", "") if isinstance(step, dict) else getattr(step, "thought", "")
            if t and t.strip():
                thought = t.strip()[:1500]
                break

    # 回退：最后一条 assistant 消息
    if not thought:
        for m in reversed(state.get("messages", [])):
            if getattr(m, "type", "") in ("ai", "assistant"):
                c = str(getattr(m, "content", "")).strip()
                if c:
                    thought = c[:1500]
                    break

    if not thought:
        return ""

    return (
        '<div style="background:#f0f5ff;border:1px solid #2f54eb;'
        'border-radius:8px;padding:12px;margin:8px 0">'
        '<div style="font-weight:bold;color:#1d39c4;font-size:15px;margin-bottom:8px">'
        '🧠 模型最后思考</div>'
        f'<pre style="background:#f0f0f0;color:#1a1a1a;padding:10px;border-radius:4px;'
        f'font-size:12px;max-height:200px;overflow-y:auto;'
        f'white-space:pre-wrap;word-break:break-all;line-height:1.5">'
        f'{thought}</pre>'
        f'</div>'
    )


def build_summary(state: dict) -> str:
    """执行摘要 Markdown"""
    planning = state.get("planning")
    execution = state.get("execution")
    output_ctx = state.get("output")

    if not planning:
        return ""

    total = len(planning.task_plan)
    finished = sum(1 for t in planning.task_plan if t.status == "finished")
    failed = sum(1 for t in planning.task_plan if t.status == "failed")

    lines = [
        "## 📊 执行摘要",
        "",
        f"| 复杂度 | 子任务 | ✅ 通过 | ❌ 失败 |",
        f"|--------|--------|---------|--------|",
        f"| {planning.task_complexity} | {total} | {finished} | {failed} |",
    ]

    if execution and execution.retry_count > 0:
        lines.append(f"\n🔄 全局重试 **{execution.retry_count}** 次")

    if failed > 0:
        lines.append("\n### ❌ 失败任务")
        for t in planning.task_plan:
            if t.status == "failed":
                lines.append(f"- **T{t.task_id}** ({t.description[:60]})")
                reason = (t.result or "无错误信息")[:300]
                lines.append(f"  ```\n  {reason}\n  ```")

    if output_ctx and output_ctx.output_done:
        lines.append(f"\n📁 输出目录: `{output_ctx.output_dir}`")
        for f in output_ctx.files_written[:8]:
            lines.append(f"  - ✅ `{f}`")
        if len(output_ctx.files_written) > 8:
            lines.append(f"  - ... 共 {len(output_ctx.files_written)} 个文件")

    return "\n".join(lines)


# ==========================================================================
# 向量模型（Embedding）管理
# ==========================================================================

_embeddings = None
_embed_state = {"status": "not_loaded", "message": "未加载"}


def get_embed_status() -> str:
    """返回 embedding 模型状态 HTML 卡片"""
    s = _embed_state
    color = {
        "not_loaded": "#888",
        "loading": "#fa8c16",
        "ready": "#52c41a",
        "error": "#ff4d4f",
    }.get(s["status"], "#888")
    icon = {
        "not_loaded": "⚪",
        "loading": "🟠",
        "ready": "🟢",
        "error": "🔴",
    }.get(s["status"], "⚪")
    return (
        f'<div style="font-size:12px;padding:4px 8px;'
        f'border:1px solid {color};border-radius:4px;color:#1a1a1a">'
        f'{icon} <b>向量模型:</b> {s["message"]}</div>'
    )


def load_embedding_model(mode: str, model_path: str = "",
                         base_url: str = "", api_key: str = "",
                         model_name: str = "") -> str:
    """
    mode: "local" | "api"
    加载 embedding 模型，存到模块全局变量，返回状态 HTML。
    """
    global _embeddings, _embed_state

    _embed_state = {"status": "loading", "message": "加载中..."}

    try:
        if mode == "local":
            if not model_path or not os.path.exists(model_path):
                _embed_state = {"status": "error",
                                "message": f"路径不存在: {model_path}"}
                return get_embed_status()
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            _embed_state = {"status": "ready",
                            "message": f"本地: {os.path.basename(model_path)}"}

        elif mode == "api":
            if not base_url or not api_key:
                _embed_state = {"status": "error",
                                "message": "缺少 API 地址或密钥"}
                return get_embed_status()
            from langchain_openai import OpenAIEmbeddings
            _embeddings = OpenAIEmbeddings(
                model=model_name or "text-embedding-3-small",
                base_url=base_url,
                api_key=api_key,
            )
            _embed_state = {"status": "ready",
                            "message": f"API: {model_name or 'text-embedding-3-small'}"}

        else:
            _embed_state = {"status": "error",
                            "message": f"未知模式: {mode}"}

    except ImportError as e:
        _embed_state = {"status": "error",
                        "message": f"缺少依赖: {e}"}
    except Exception as e:
        _embed_state = {"status": "error",
                        "message": f"失败: {str(e)[:80]}"}

    return get_embed_status()


def get_embeddings():
    """供 analyzer/reviewer 获取已加载的 embedder"""
    global _embeddings
    return _embeddings
