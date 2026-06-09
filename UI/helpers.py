"""
=============================================================================
UI/helpers.py —— 显示格式化 & 辅助函数
=============================================================================

为 Gradio 界面提供 HTML/Markdown 渲染函数，以及工作区文件列表。
=============================================================================
"""

import os

from UI.sandbox import docker_exec, _LOCAL_WORKSPACE


# ==========================================================================
# 断点续传——枚举历史会话
# ==========================================================================

def get_session_list(checkpointer, app_graph) -> list:
    """
    从 MemorySaver 中枚举所有历史会话，返回 [(label, thread_id), ...]。
    按最近优先排序。
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

        # 提取用户原始需求
        task_desc = _extract_task_summary(state)

        # 提取进度
        planning = state.get("planning")
        progress = ""
        if planning and planning.task_plan:
            done = sum(1 for t in planning.task_plan if t.status == "finished")
            fail = sum(1 for t in planning.task_plan if t.status == "failed")
            total = len(planning.task_plan)
            progress = f"({done}/{total}"
            if fail > 0:
                progress += f", {fail} 失败"
            progress += ")"

        # 是否有中断
        blocked = state.get("react_blocked", False)
        need_clarify = planning.need_clarification if planning else False
        tag = ""
        if blocked:
            tag = " ⚠️待决策"
        elif need_clarify:
            tag = " 🤔待澄清"

        label = f"[{tid[:16]}] {task_desc[:40]} {progress}{tag}"
        items.append((label, tid))

    return items


def _extract_task_summary(state: dict) -> str:
    """从 state 中提取用户原始需求简述"""
    # 从第一条 HumanMessage 取
    msgs = state.get("messages", [])
    for m in msgs:
        if hasattr(m, "type") and m.type == "human":
            return str(m.content)[:60].replace("\n", " ")
    # 回退：从第一个子任务描述取
    planning = state.get("planning")
    if planning and planning.task_plan:
        return planning.task_plan[0].description[:60]
    return "(未知任务)"


# ==========================================================================
# 工作区文件列表
# ==========================================================================

def list_workspace_files() -> str:
    """列出沙盒工作区文件树"""
    try:
        result = docker_exec(
            "find /workspace -type f 2>/dev/null | head -50",
            timeout=5,
        )
        if result["returncode"] == 0 and result["stdout"].strip():
            files = result["stdout"].strip().split("\n")
            return "\n".join(f"📄 {f}" for f in sorted(files))
    except Exception:
        pass

    # 本地回退
    if _LOCAL_WORKSPACE and os.path.isdir(_LOCAL_WORKSPACE):
        try:
            files = []
            for root, dirs, filenames in os.walk(_LOCAL_WORKSPACE):
                for fn in filenames:
                    fp = os.path.relpath(os.path.join(root, fn), _LOCAL_WORKSPACE)
                    files.append(f"📄 {fp}")
            return "\n".join(sorted(files)[:50]) if files else "(工作区为空)"
        except Exception:
            pass
    return "(工作区为空)"


# ==========================================================================
# 子任务进度 HTML
# ==========================================================================

def get_task_progress_html(state: dict) -> str:
    """生成子任务进度 HTML（右侧面板）"""
    planning = state.get("planning")
    if not planning or not planning.task_plan:
        return "<p style='color:#888'>暂无任务数据</p>"

    emoji_map = {
        "finished": "✅",
        "pending": "🔄",
        "failed": "❌",
        "testing": "🧪",
        "doing": "⚙️",
    }
    lines = ["<div style='font-size:13px;line-height:1.6'>"]
    for t in planning.task_plan:
        emoji = emoji_map.get(t.status, "⏳")
        desc = t.description[:60]
        retry_info = ""
        execution = state.get("execution")
        if execution and t.task_id in execution.task_retry_count:
            cnt = execution.task_retry_count[t.task_id]
            if cnt > 0:
                retry_info = f" <span style='color:#f80'>(重试{cnt})</span>"
        lines.append(
            f"<div style='margin:4px 0;padding:4px 8px;"
            f"background:#e8e8e8;color:#1a1a1a;border-radius:4px'>"
            f"{emoji} <b>T{t.task_id}</b>: {desc}{retry_info}</div>"
        )
    lines.append("</div>")
    return "".join(lines)


# ==========================================================================
# 消息格式转换
# ==========================================================================

def messages_to_chatbot(state: dict) -> list:
    """将 LangChain 消息转为 Gradio Chatbot 格式 [{role, content}]"""
    chat = []
    messages = state.get("messages", [])
    for m in messages:
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        content = str(getattr(m, "content", ""))[:2000]
        if content.strip():
            chat.append({"role": role, "content": content})
    return chat


# ==========================================================================
# 错误 / 思考 / 摘要 HTML 渲染
# ==========================================================================

def get_sandbox_error_html(state: dict) -> str:
    """提取沙盒报错信息，格式化为 HTML"""
    execution = state.get("execution")
    if not execution or not execution.error_trace:
        return ""

    error_text = execution.error_trace[:2000]
    retry_count = execution.retry_count
    task_retry = execution.task_retry_count or {}

    lines = [
        '<div style="background:#fff5f5;border:1px solid #ff4d4f;'
        'border-radius:8px;padding:12px;margin:8px 0">',
        '<div style="font-weight:bold;color:#cf1322;margin-bottom:8px">'
        f'🔴 沙盒验证失败（全局重试 {retry_count} 次）</div>',
    ]

    if task_retry:
        lines.append(
            '<div style="font-size:12px;color:#666;margin-bottom:8px">'
            "各任务重试: " +
            ", ".join(f"T{k}={v}次" for k, v in task_retry.items()) +
            "</div>"
        )

    lines.append(
        f'<pre style="background:#1a1a2e;color:#e0e0e0;padding:10px;'
        f'border-radius:4px;font-size:12px;max-height:300px;'
        f'overflow-y:auto;white-space:pre-wrap;word-break:break-all">'
        f'{error_text}</pre>'
        "</div>"
    )
    return "".join(lines)


def get_last_thought_html(state: dict) -> str:
    """提取模型最后思考"""
    react_history = state.get("react_history", [])
    last_thought = ""
    if react_history:
        for step in reversed(react_history):
            if isinstance(step, dict):
                t = step.get("thought", "")
            elif hasattr(step, "thought"):
                t = step.thought
            else:
                t = ""
            if t and t.strip():
                last_thought = t.strip()[:1500]
                break

    # 如果没有 react_history，回退到最后一条 assistant 消息
    if not last_thought:
        messages = state.get("messages", [])
        for m in reversed(messages):
            if getattr(m, "type", "") in ("ai", "assistant"):
                content = str(getattr(m, "content", ""))
                if content.strip():
                    last_thought = content.strip()[:1500]
                    break

    if not last_thought:
        return ""

    return (
        '<div style="background:#f0f5ff;border:1px solid #2f54eb;'
        'border-radius:8px;padding:12px;margin:8px 0">'
        '<div style="font-weight:bold;color:#1d39c4;margin-bottom:8px">'
        '🧠 模型最后思考</div>'
        f'<pre style="background:#f0f0f0;color:#1a1a1a;padding:8px;border-radius:4px;'
        f'font-size:12px;max-height:200px;overflow-y:auto;'
        f'white-space:pre-wrap;word-break:break-all">'
        f'{last_thought}</pre>'
        "</div>"
    )


def get_summary_markdown(state: dict) -> str:
    """生成执行摘要 Markdown"""
    planning = state.get("planning")
    execution = state.get("execution")
    output_ctx = state.get("output")

    lines = ["## 📊 执行摘要\n"]

    if planning:
        total = len(planning.task_plan)
        finished = sum(1 for t in planning.task_plan if t.status == "finished")
        failed = sum(1 for t in planning.task_plan if t.status == "failed")
        complexity = planning.task_complexity
        lines.append(
            f"| 复杂度 | 子任务 | ✅ 通过 | ❌ 失败 |\n"
            f"|--------|--------|---------|--------|\n"
            f"| {complexity} | {total} | {finished} | {failed} |\n"
        )

        if failed > 0:
            lines.append("\n### ❌ 失败任务\n")
            for t in planning.task_plan:
                if t.status == "failed":
                    reason = (t.result or "(无)")[:300]
                    lines.append(f"- **T{t.task_id}** ({t.description[:50]}): `{reason}`\n")

    if execution and execution.retry_count > 0:
        lines.append(f"\n🔄 全局重试: **{execution.retry_count}** 次\n")

    if output_ctx and output_ctx.output_done:
        lines.append(f"\n📁 输出目录: `{output_ctx.output_dir}`\n")
        for f in output_ctx.files_written[:8]:
            lines.append(f"- ✅ `{f}`\n")
        if len(output_ctx.files_written) > 8:
            lines.append(f"- ... 共 {len(output_ctx.files_written)} 个文件\n")

    return "".join(lines)
