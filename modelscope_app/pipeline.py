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

# 合法顶层业务节点（仅这些节点允许更新进度）
VALID_TOP_NODES = set(NODE_LABEL.keys())


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
# 通用工具函数（修复：支持元组/列表/字符串，剥离UUID）
# ==========================================================================
def _short(name) -> str:
    """
    通用节点名裁剪：兼容 字符串/元组/列表，移除 LangGraph :uuid 后缀
    """
    # 处理元组/列表：取最后一级节点
    if isinstance(name, (tuple, list)):
        if not name:
            return ""
        name = name[-1]
    # 转为字符串
    name_str = str(name)
    # 移除 UUID 后缀
    return name_str.split(":")[0] if ":" in name_str else name_str


def _get_all_namespace(path_tuple) -> list:
    """将路径元组全部元素去 hash 后返回（用于判断当前处于哪个子图）"""
    if not isinstance(path_tuple, (tuple, list)):
        return []
    return [_short(p) for p in path_tuple if p]


# ==========================================================================
# 流式执行
# ==========================================================================

def stream_and_check(app_graph, state: dict, thread_id: str, chat: list):
    """
    逐节点流式执行流水线，每次 yield 9 元组供 UI 渐进更新。
    已适配 LangGraph subgraphs=True + stream_mode="updates" 标准格式
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350}

    new_state = state
    total_pct = 0
    node_count = 0
    nodes_done = set()
    shown_react_steps = 0   # 已展示的 ReAct 历史步数（防重复显示）
    _last_chat_msg = ""    # 防重复：同上一条不追加

    def _format_react_msgs(step: dict) -> list:
        """格式化单步 ReAct 思考为 chatbot 消息"""
        msgs = []
        thought = (step.get("thought", "") or "").strip()
        action = step.get("action", {}) or {}
        if isinstance(action, dict):
            tool_name = action.get("tool_name", "") or ""
            tool_input = action.get("tool_input", {}) or {}
        else:
            tool_name = getattr(action, "tool_name", "") or ""
            tool_input = getattr(action, "tool_input", {}) or {}
        obs = (step.get("observation", "") or "").strip()

        if thought:
            msgs.append({"role": "assistant", "content": f"💭 {thought[:300]}"})
        if tool_name and tool_name != "FINISH":
            args_brief = str(tool_input)[:120]
            msgs.append({"role": "assistant", "content": f"🔧 `{tool_name}({args_brief})`"})
        if obs and tool_name != "FINISH":
            msgs.append({"role": "assistant", "content": f"📋 {obs[:300]}"})
        return msgs

    # ── 阶段 1：流式执行 ──
    try:
        for chunk in app_graph.stream(state, config, stream_mode="updates", subgraphs=True):
            node_count += 1
            # LangGraph 0.5.x subgraphs=True 实际格式:
            #   (namespace_tuple, update_dict)   ← 2 元组：节点状态更新
            #   (namespace_tuple,)                ← 1 元组：子图边界事件
            #   ()                                ← 0 元组：根图边界
            # 向下兼容 3 元组 (namespace, node_name, update)
            namespace = ()
            node_update = {}

            if len(chunk) == 2:
                namespace, second = chunk
                if isinstance(second, dict):
                    node_update = second
                else:
                    # (namespace, node_name) 无更新 → 跳过
                    continue
            elif len(chunk) == 3:
                namespace, _, node_update = chunk  # 3 元组兼容（旧版 LangGraph）
                if not isinstance(node_update, dict):
                    continue
            else:
                # 0~1 元组：边界事件 → 跳过
                continue

            # 更新 state
            if isinstance(node_update, dict):
                new_state.update(node_update)

            # 所有 namespace 元素去 hash
            all_ns = _get_all_namespace(namespace)

            # ========== 精准识别 worker（ReAct）子图内部节点 ==========
            # 注意：execution.py 注册名为 "worker"，不是 "react_worker"
            if "worker" in all_ns:
                # ReAct 内部节点的 node_name 不在 chunk 中，从 update_dict 的 key 推断阶段
                if "_pending_thought" in node_update:
                    thought = (node_update.get("_pending_thought", "") or "").strip()
                    tool_calls = node_update.get("_pending_tool_calls", []) or []
                    if thought:
                        chat.append({"role": "assistant", "content": f"💭 {thought[:300]}"})
                    if tool_calls:
                        names = [tc.get("name", "?") for tc in tool_calls if isinstance(tc, dict)]
                        if names:
                            chat.append({"role": "assistant", "content": f"🔧 准备调用: {', '.join(names)}"})

                elif "react_history" in node_update:
                    rh = node_update.get("react_history", []) or []
                    for step in rh[shown_react_steps:]:
                        chat.extend(_format_react_msgs(step))
                    shown_react_steps = len(rh)

                elif "react_finished" in node_update:
                    if node_update.get("react_finished") and not node_update.get("react_blocked"):
                        chat.append({"role": "assistant", "content": "✅ 子任务编码完成 → 进入沙盒验证"})

                # 子图节点：不更新进度、仅轻量刷新 UI
                yield (
                    chat, new_state, "⚙️ 执行中...",
                    task_progress_html(new_state), list_workspace(),
                    gr.update(visible=False), gr.update(visible=False),
                    "",  # 介入面板空（ReAct 执行中不需要）
                    _progress_bar(total_pct, "⚙️ 执行中..."),
                )
                continue

            # ========== 顶层业务节点（根据 namespace 推断当前阶段） ==========
            # all_ns 为空 → 主图级别节点（integrator / output_writer / reviewer）
            # all_ns 第一层 → executor / sandbox 子图内部的汇总更新
            if not all_ns:
                # 主图级别：从 update_dict 推断节点
                if "integration" in node_update:
                    base_name = "integrator"
                elif "output" in node_update:
                    base_name = "output_writer"
                else:
                    base_name = "reviewer"  # 默认（含 messages 等通用更新）
            elif all_ns[0] == "executor":
                # executor 子图内部，但不是 worker（worker 已被上面拦截）
                # analyzer 输出 planning，scheduler 输出 execution
                base_name = "executor"
            elif all_ns[0] == "sandbox":
                base_name = "sandbox"
            else:
                base_name = all_ns[0] if all_ns else ""

            # 过滤空节点
            if not base_name:
                continue

            label = NODE_LABEL.get(base_name, base_name)
            # 仅五大合法节点才累加进度
            if base_name in VALID_TOP_NODES and base_name not in nodes_done:
                nodes_done.add(base_name)
                add_pct = NODE_PCT[base_name]
                total_pct = min(total_pct + add_pct, 92)

            progress_html = _progress_bar(total_pct, label)
            # 防重复消息
            msg = f"⚙️ {label}"
            if msg != _last_chat_msg:
                chat.append({"role": "assistant", "content": msg})
                _last_chat_msg = msg

            logger.info(f"[stream] 节点 {node_count}: {base_name}（进度 {total_pct}%）")

            yield (
                chat, new_state, f"⚙️ {label}",
                task_progress_html(new_state), list_workspace(),
                gr.update(visible=False), gr.update(visible=False),
                "",  # 介入面板空
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
    # 兜底：从 LangGraph checkpoint 读取权威最终状态，
    # 防止流式 chunk 累积遗漏导致漏检 need_clarification / react_blocked
    try:
        checkpoint_state = app_graph.get_state(config).values
        if checkpoint_state:
            new_state = checkpoint_state
    except Exception:
        pass  # 降级：使用流式累积的 new_state

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