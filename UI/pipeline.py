"""
=============================================================================
UI/pipeline.py —— 流水线执行核心
=============================================================================

封装 app_graph.stream() + 状态检测逻辑，返回统一 9 元组供 UI 渲染。
app_graph 通过闭包注入（从 app.py 传入）。

9 元组: (chatbot, state, status, task_html, file_tree,
         clarify_vis, intervene_vis, intervene_html, progress_html)
=============================================================================
"""

import traceback

import gradio as gr
from langgraph.errors import GraphInterrupt

from core.logger import logger
from UI.helpers import (
    get_task_progress_html,
    list_workspace_files,
    get_sandbox_error_html,
    get_last_thought_html,
    get_summary_markdown,
)


# ==========================================================================
# 进度条
# ==========================================================================

NODE_LABEL = {
    "executor":      "🔍 分析 & 编码",
    "sandbox":       "🧪 沙盒验证",
    "integrator":    "📦 整合交付物",
    "output_writer": "📁 写入输出",
    "reviewer":      "📊 复盘归档",
}

NODE_PCT = {
    "executor":      20,
    "sandbox":       20,
    "integrator":    20,
    "output_writer": 15,
    "reviewer":      10,
}


def _progress_bar(pct: int, label: str) -> str:
    """生成进度条 HTML"""
    color = "#1677ff"
    if "验证" in label:
        color = "#fa8c16"
    elif "整合" in label:
        color = "#722ed1"
    elif "输出" in label:
        color = "#13c2c2"
    elif "复盘" in label:
        color = "#52c41a"

    return (
        f'<div style="margin:6px 0;font-size:12px;color:#555">{label}</div>'
        f'<div style="width:100%;background:#e8e8e8;border-radius:6px;height:14px">'
        f'<div style="width:{min(pct,100)}%;background:{color};height:14px;'
        f'border-radius:6px;transition:width 0.4s ease"></div>'
        f'</div>'
    )


# ==========================================================================
# 阻塞态 / 完成态 返回
# ==========================================================================

def _blocked_return(chat: list, state: dict, status: str,
                    block_type: str, block_reason: str,
                    progress_html: str = "") -> tuple:
    """构建"流水线暂停"的统一返回（9 元组）"""
    parts = []
    parts.append(f"### ⚠️ {block_reason[:500]}\n")
    parts.append(get_sandbox_error_html(state))
    parts.append(get_last_thought_html(state))
    intervene_html = "".join(parts)
    progress_html = progress_html or _progress_bar(100, "⏸️ 暂停等待决策")

    if block_type == "clarify":
        return (
            chat, state, status,
            get_task_progress_html(state), list_workspace_files(),
            gr.update(visible=True), gr.update(visible=False),
            intervene_html, progress_html,
        )
    else:
        return (
            chat, state, status,
            get_task_progress_html(state), list_workspace_files(),
            gr.update(visible=False), gr.update(visible=True),
            intervene_html, progress_html,
        )


def _done_return(chat: list, state: dict) -> tuple:
    """构建"流水线完成"的统一返回（9 元组）"""
    summary = get_summary_markdown(state)
    chat.append({"role": "assistant", "content": summary})
    return (
        chat, state, "✅ 流水线执行完成",
        get_task_progress_html(state), list_workspace_files(),
        gr.update(visible=False), gr.update(visible=False),
        f"### ✅ 执行完成\n\n{summary}",
        _progress_bar(100, "✅ 完成"),
    )


def _empty_progress() -> str:
    return _progress_bar(0, "⏳ 等待启动...")


# ==========================================================================
# 流式执行（逐节点进度）
# ==========================================================================

def stream_and_check(app_graph, state: dict, thread_id: str, chat: list):
    """
    逐节点流式执行流水线，每次 yield 一个 9 元组供 UI 渐进更新。

    流程：
    1. 用 app_graph.stream() 逐节点产出
    2. 每个节点 → 更新进度条 + 聊天 → yield
    3. 流结束后做阻塞检测（need_clarification / react_blocked / sandbox 失败）
    4. 正常完成 → _done_return
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350}

    # ── 阶段 1：流式执行 ──
    new_state = state
    total_pct = 0
    node_count = 0
    nodes_done = set()

    try:
        for chunk in app_graph.stream(state, config, stream_mode="updates"):
            node_count += 1
            for node_name, node_update in chunk.items():
                # 更新累积状态
                if isinstance(node_update, dict):
                    new_state.update(node_update)

                label = NODE_LABEL.get(node_name, node_name)

                # 节点首次出现才推进进度（同一节点重入只显示不推进）
                if node_name not in nodes_done:
                    nodes_done.add(node_name)
                    total_pct = min(total_pct + NODE_PCT.get(node_name, 8), 92)

                progress_html = _progress_bar(total_pct, label)
                chat.append({"role": "assistant", "content": f"⚙️ {label}"})

                logger.info(f"[UI-stream] 节点 {node_count}: {node_name}（进度 {total_pct}%）")

                yield (
                    chat, new_state, f"⚙️ {label}",
                    get_task_progress_html(new_state), list_workspace_files(),
                    gr.update(visible=False), gr.update(visible=False),
                    _progress_bar(total_pct, label),
                    progress_html,
                )

    except GraphInterrupt as e:
        interrupt_vals = e.args[0] if e.args else []
        msg = interrupt_vals[0] if interrupt_vals else "ReAct 循环卡住"
        logger.info(f"[UI-stream] GraphInterrupt: {msg[:120]}")
        chat.append({"role": "assistant", "content": f"⚠️ 流水线中断：{msg[:300]}"})
        new_state["_ui_is_interrupt"] = True
        yield _blocked_return(
            chat, new_state, "⚠️ 等待人工决策（GraphInterrupt）",
            "intervene", f"GraphInterrupt: {msg[:400]}",
            _progress_bar(total_pct, "⚠️ 中断"),
        )
        return

    except Exception as e:
        logger.error(f"[UI-stream] 流水线异常: {e}")
        tb = traceback.format_exc()[:2000]
        chat.append({"role": "assistant", "content": f"❌ 执行异常：{str(e)[:500]}"})
        yield _blocked_return(
            chat, new_state, "❌ 流水线异常",
            "intervene", f"执行异常:\n```\n{tb}\n```",
            _progress_bar(total_pct, "❌ 异常"),
        )
        return

    # ── 阶段 2：流结束后做阻塞检测 ──
    planning = new_state.get("planning")
    execution = new_state.get("execution")

    # 检测 1：需求不清晰
    if planning and planning.need_clarification:
        question = planning.clarification_question
        chat.append({"role": "assistant", "content": f"🤔 {question}"})
        yield _blocked_return(
            chat, new_state, "等待补充信息...",
            "clarify", question,
        )
        return

    # 检测 2：ReAct 阻塞 flag
    if new_state.get("react_blocked", False):
        reason = new_state.get("react_block_reason", "ReAct 循环卡住")
        chat.append({"role": "assistant", "content": f"⚠️ {reason[:500]}"})
        yield _blocked_return(
            chat, new_state, "⚠️ ReAct 循环阻塞",
            "intervene", reason,
        )
        return

    # 检测 3：沙盒验证失败的 pending 任务（安全网）
    if planning:
        pending = [t for t in planning.task_plan if t.status == "pending"]
        testing = [t for t in planning.task_plan if t.status == "testing"]
        doing = [t for t in planning.task_plan if t.status == "doing"]

        if pending and not testing and not doing:
            is_react_internal_block = any(
                "等待人工介入" in (t.result or "") for t in pending
            )
            if not is_react_internal_block:
                error_parts = []
                for t in pending:
                    retry_n = execution.task_retry_count.get(t.task_id, 0) if execution else 0
                    err = (execution.error_trace if execution and execution.error_trace
                           else t.result)[:600]
                    error_parts.append(
                        f"• 子任务 {t.task_id}（{t.description[:50]}）\n"
                        f"  沙盒验证失败（第 {retry_n} 次重试）\n"
                        f"  报错：\n  ```\n  {err}\n  ```"
                    )
                reason = "### ⚠️ 沙盒验证失败，需人工决策\n\n" + "\n".join(error_parts)
                reason += "\n\n请选择：继续执行 / 强制提交 / 跳过任务 / 修改需求"

                chat.append({"role": "assistant", "content": f"⚠️ 沙盒验证失败，{len(pending)} 个任务需人工决策"})
                yield _blocked_return(
                    chat, new_state, f"沙盒验证失败（{len(pending)} 个任务）",
                    "intervene", reason,
                )
                return

    # ── 正常完成 ──
    chat.append({"role": "assistant", "content": "✅ 流水线执行完成"})
    yield _done_return(chat, new_state)


# ==========================================================================
# 保留同步版（供 GraphInterrupt 路径需要 Command 时使用）
# ==========================================================================

def invoke_and_check(app_graph, state: dict, thread_id: str, chat: list) -> tuple:
    """同步版 invoke——作为 stream_and_check 无法覆盖的边缘情况兜底"""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350}

    try:
        new_state = app_graph.invoke(state, config)
    except GraphInterrupt as e:
        vals = e.args[0] if e.args else []
        msg = vals[0] if vals else "ReAct 循环卡住，需要人工决策"
        logger.info(f"[UI] GraphInterrupt: {msg[:120]}")
        chat.append({"role": "assistant", "content": f"⚠️ 流水线中断：{msg[:300]}"})
        state["_ui_is_interrupt"] = True
        return _blocked_return(chat, state, "⚠️ 等待人工决策（GraphInterrupt）",
                               "intervene", f"GraphInterrupt: {msg[:500]}")
    except Exception as e:
        tb = traceback.format_exc()[-3000:]
        logger.error(f"[UI] 流水线异常: {e}")
        chat.append({"role": "assistant", "content": f"❌ 执行异常：{str(e)[:500]}"})
        return _blocked_return(chat, state, "❌ 流水线异常", "intervene",
                               f"异常:\n```\n{tb}\n```")

    planning = new_state.get("planning")
    execution = new_state.get("execution")

    if planning and planning.need_clarification:
        q = planning.clarification_question
        chat.append({"role": "assistant", "content": f"🤔 {q}"})
        return _blocked_return(chat, new_state, "等待补充信息...", "clarify", q)

    if new_state.get("react_blocked", False):
        reason = new_state.get("react_block_reason", "ReAct 循环卡住")
        chat.append({"role": "assistant", "content": f"⚠️ {reason[:500]}"})
        return _blocked_return(chat, new_state, "⚠️ ReAct 循环阻塞", "intervene", reason)

    if planning:
        pending = [t for t in planning.task_plan if t.status == "pending"]
        testing = [t for t in planning.task_plan if t.status == "testing"]
        doing = [t for t in planning.task_plan if t.status == "doing"]

        if pending and not testing and not doing:
            if not any("等待人工介入" in (t.result or "") for t in pending):
                parts = []
                for t in pending:
                    n = execution.task_retry_count.get(t.task_id, 0) if execution else 0
                    err = (execution.error_trace if execution and execution.error_trace
                           else t.result or "无")[:600]
                    parts.append(
                        f"• **T{t.task_id}**（{t.description[:60]}）\n"
                        f"  第 {n} 次重试失败\n"
                        f"  ```\n  {err}\n  ```"
                    )
                reason = "### ⚠️ 沙盒验证失败\n\n" + "\n".join(parts)
                reason += "\n\n请选择操作：继续执行 / 强制提交 / 跳过任务 / 修改需求"

                chat.append({"role": "assistant", "content": f"⚠️ {len(pending)} 个任务沙盒失败，需人工决策"})
                return _blocked_return(chat, new_state,
                                       f"沙盒失败（{len(pending)} 个任务）",
                                       "intervene", reason)

    chat.append({"role": "assistant", "content": "✅ 流水线执行完成"})
    return _done_return(chat, new_state)
