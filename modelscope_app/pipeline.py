"""
=============================================================================
pipeline.py —— 流水线执行核心
=============================================================================

封装 app_graph.stream() + 阻塞检测逻辑，返回统一 9 元组供 UI 渲染。
app_graph 通过闭包注入。

9 元组: (chatbot, state, status, task_html, file_tree,
         clarify_vis, intervene_vis, intervene_html, progress_html)
=============================================================================
"""

import traceback

import gradio as gr
from langgraph.errors import GraphInterrupt

from core.logger import logger
from .ui_helpers import (
    build_error_html,
    build_thought_html,
    build_summary,
    task_progress_html,
    list_workspace,
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
# 阻塞态 / 完成态
# ==========================================================================

def _blocked(chat, state, status, block_type, reason, progress_html="") -> tuple:
    parts = [f"### ⚠️ {reason[:600]}"]
    err = build_error_html(state)
    if err:
        parts.append(err)
    thought = build_thought_html(state)
    if thought:
        parts.append(thought)
    intervene_html = "\n".join(parts)
    progress_html = progress_html or _progress_bar(100, "⏸️ 暂停等待决策")

    return (
        chat, state, status,
        task_progress_html(state), list_workspace(),
        gr.update(visible=(block_type == "clarify")),
        gr.update(visible=(block_type == "intervene")),
        intervene_html, progress_html,
    )


def _done(chat, state) -> tuple:
    summary = build_summary(state)
    chat.append({"role": "assistant", "content": summary})
    return (
        chat, state, "✅ 流水线执行完成",
        task_progress_html(state), list_workspace(),
        gr.update(visible=False), gr.update(visible=False),
        f"### ✅ 执行完成\n\n{summary}",
        _progress_bar(100, "✅ 完成"),
    )


# ==========================================================================
# 流式执行
# ==========================================================================

def stream_and_check(app_graph, state: dict, thread_id: str, chat: list):
    """
    逐节点流式执行流水线，每次 yield 9 元组供 UI 渐进更新。
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350}

    new_state = state
    total_pct = 0
    node_count = 0
    nodes_done = set()

    # ── 阶段 1：流式执行 ──
    try:
        for chunk in app_graph.stream(state, config, stream_mode="updates"):
            node_count += 1
            for node_name, node_update in chunk.items():
                if isinstance(node_update, dict):
                    new_state.update(node_update)

                label = NODE_LABEL.get(node_name, node_name)
                if node_name not in nodes_done:
                    nodes_done.add(node_name)
                    total_pct = min(total_pct + NODE_PCT.get(node_name, 8), 92)

                progress_html = _progress_bar(total_pct, label)
                chat.append({"role": "assistant", "content": f"⚙️ {label}"})

                logger.info(f"[stream] 节点 {node_count}: {node_name}（进度 {total_pct}%）")

                yield (
                    chat, new_state, f"⚙️ {label}",
                    task_progress_html(new_state), list_workspace(),
                    gr.update(visible=False), gr.update(visible=False),
                    _progress_bar(total_pct, label),
                    progress_html,
                )

    except GraphInterrupt as e:
        vals = e.args[0] if e.args else []
        msg = vals[0] if vals else "ReAct 循环卡住，需要人工决策"
        logger.info(f"[stream] GraphInterrupt: {msg[:120]}")
        chat.append({"role": "assistant", "content": f"⚠️ 流水线中断：{msg[:300]}"})
        new_state["_ui_is_interrupt"] = True
        yield _blocked(chat, new_state, "⚠️ 等待人工决策（GraphInterrupt）",
                       "intervene", f"GraphInterrupt: {msg[:500]}",
                       _progress_bar(total_pct, "⚠️ 中断"))
        return

    except Exception as e:
        tb = traceback.format_exc()[-3000:]
        logger.error(f"[stream] 流水线异常: {e}")
        chat.append({"role": "assistant", "content": f"❌ 执行异常：{str(e)[:500]}"})
        yield _blocked(chat, new_state, "❌ 流水线异常", "intervene",
                       f"异常:\n```\n{tb}\n```",
                       _progress_bar(total_pct, "❌ 异常"))
        return

    # ── 阶段 2: 阻塞检测 ──
    planning = new_state.get("planning")
    execution = new_state.get("execution")

    if planning and planning.need_clarification:
        q = planning.clarification_question
        chat.append({"role": "assistant", "content": f"🤔 {q}"})
        yield _blocked(chat, new_state, "等待补充信息...", "clarify", q)
        return

    if new_state.get("react_blocked", False):
        reason = new_state.get("react_block_reason", "ReAct 循环卡住")
        chat.append({"role": "assistant", "content": f"⚠️ {reason[:500]}"})
        yield _blocked(chat, new_state, "⚠️ ReAct 循环阻塞", "intervene", reason)
        return

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
                chat.append({"role": "assistant",
                             "content": f"⚠️ {len(pending)} 个任务沙盒失败，需人工决策"})
                yield _blocked(chat, new_state,
                               f"沙盒失败（{len(pending)} 个任务）",
                               "intervene", reason)
                return

    chat.append({"role": "assistant", "content": "✅ 流水线执行完成"})
    yield _done(chat, new_state)


# ==========================================================================
# 同步版（兜底）
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
        return _blocked(chat, state, "⚠️ 等待人工决策（GraphInterrupt）",
                        "intervene", f"GraphInterrupt: {msg[:500]}")
    except Exception as e:
        tb = traceback.format_exc()[-3000:]
        logger.error(f"[UI] 流水线异常: {e}")
        chat.append({"role": "assistant", "content": f"❌ 执行异常：{str(e)[:500]}"})
        return _blocked(chat, state, "❌ 流水线异常", "intervene",
                        f"异常:\n```\n{tb}\n```")

    planning = new_state.get("planning")
    execution = new_state.get("execution")

    if planning and planning.need_clarification:
        q = planning.clarification_question
        chat.append({"role": "assistant", "content": f"🤔 {q}"})
        return _blocked(chat, new_state, "等待补充信息...", "clarify", q)

    if new_state.get("react_blocked", False):
        reason = new_state.get("react_block_reason", "ReAct 循环卡住")
        chat.append({"role": "assistant", "content": f"⚠️ {reason[:500]}"})
        return _blocked(chat, new_state, "⚠️ ReAct 循环阻塞", "intervene", reason)

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
                return _blocked(chat, new_state,
                                f"沙盒失败（{len(pending)} 个任务）",
                                "intervene", reason)

    chat.append({"role": "assistant", "content": "✅ 流水线执行完成"})
    return _done(chat, new_state)
