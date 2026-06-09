"""
=============================================================================
UI/handlers.py —— Gradio 事件处理函数
=============================================================================

处理启动 / 澄清 / 人工干预等用户交互。

所有需要 app_graph 的函数将其作为显式第一参数，
由 app.py 通过闭包注入。
=============================================================================
"""

import uuid

import gradio as gr
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from core.logger import logger
from UI.sandbox import docker_exec
from UI.helpers import (
    get_task_progress_html,
    list_workspace_files,
    messages_to_chatbot,
)
from UI.pipeline import stream_and_check, _progress_bar, _empty_progress


# ==========================================================================
# Session 数据（跨 Gradio 回调共享 is_interrupt 标记）
# ==========================================================================

SESSION_DATA: dict = {}  # thread_id → {"is_interrupt": bool, "state": dict}


# ==========================================================================
# 9 元组构建辅助
# ==========================================================================

def _make_outputs(chat, state, status, task_html, file_tree,
                  clarify_vis, intervene_vis, intervene_html, progress_html) -> tuple:
    return (chat, state, status, task_html, file_tree,
            clarify_vis, intervene_vis, intervene_html, progress_html)


def _initial_progress(label: str) -> str:
    return _progress_bar(5, f"⏳ {label}")


# ==========================================================================
# 启动 / 续传
# ==========================================================================

def on_start(app_graph, user_request: str, resume_thread_id: str):
    """启动新任务 / 断点续传（生成器，逐节点流式更新 UI）"""
    user_request = (user_request or "").strip()
    resume_thread_id = (resume_thread_id or "").strip()

    if not user_request and not resume_thread_id:
        empty_chat = [{"role": "assistant", "content": "👋 请输入任务需求或断点续传 ID"}]
        yield _make_outputs(
            empty_chat, {}, "请输入需求或续传 ID",
            get_task_progress_html({}), list_workspace_files(),
            gr.update(visible=False), gr.update(visible=False),
            "### 🤖 AI 编码代理\n输入需求后点击启动按钮",
            _empty_progress(),
        )
        return

    # 断点续传
    if resume_thread_id:
        config = {"configurable": {"thread_id": resume_thread_id}}
        try:
            saved = app_graph.get_state(config)
        except Exception:
            saved = None

        if saved is None or not saved.values:
            yield _make_outputs(
                [{"role": "assistant", "content": f"❌ 无效的 thread_id: {resume_thread_id}"}],
                {}, "无效 ID",
                get_task_progress_html({}), list_workspace_files(),
                gr.update(visible=False), gr.update(visible=False),
                f"### ❌ thread_id 无效\n`{resume_thread_id}` 无对应存档",
                _empty_progress(),
            )
            return

        state = saved.values
        thread_id = resume_thread_id

        exec_box = state.get("execution")
        if exec_box:
            exec_box.retry_count = 0
            exec_box.error_trace = ""
            exec_box.all_tasks_completed = False
        plan_box = state.get("planning")
        if plan_box:
            plan_box.need_clarification = False
            plan_box.clarification_question = ""
            for t in plan_box.task_plan:
                if t.status == "failed":
                    t.status = "pending"
                    t.result = f"[续传] 上次失败: {t.result}"

        chat = [{"role": "assistant", "content": f"🔄 从断点恢复 (thread_id: {thread_id})"}]
        logger.info(f"[UI] 断点续传启动: {thread_id}")
    else:
        # 全新任务
        state = {"messages": [HumanMessage(content=user_request)]}
        thread_id = f"ui_{uuid.uuid4().hex[:12]}"

        try:
            docker_exec("rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null; echo ok", timeout=5)
        except Exception:
            pass

        chat = [
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": "🚀 流水线启动，正在分析需求..."},
        ]
        logger.info(f"[UI] 新任务启动: thread_id={thread_id}")

    SESSION_DATA[thread_id] = {"is_interrupt": False, "state": state}

    # 先 yield 初始状态（让用户看到启动确认）
    yield _make_outputs(
        chat, state, "🔄 流水线启动中...",
        get_task_progress_html(state), list_workspace_files(),
        gr.update(visible=False), gr.update(visible=False),
        "### 🚀 流水线启动",
        _initial_progress("流水线启动"),
    )

    # 然后流式执行
    yield from stream_and_check(app_graph, state, thread_id, chat)


# ==========================================================================
# 需求澄清
# ==========================================================================

def on_clarify(app_graph, answer: str, state: dict, thread_id: str):
    """用户回答需求澄清问题（生成器，与 main.py 对齐：最多 4 轮）"""
    answer = (answer or "").strip()
    if not answer:
        yield _make_outputs(
            messages_to_chatbot(state or {}), state, "请输入补充信息！",
            get_task_progress_html(state or {}), list_workspace_files(),
            gr.update(visible=True), gr.update(visible=False),
            "请输入补充信息后再提交",
            _progress_bar(0, "⏸️ 等待输入"),
        )
        return

    planning = state.get("planning") if state else None
    if planning:
        planning.need_clarification = False

    clarify_count = state.get("_clarify_count", 0) + 1
    state["_clarify_count"] = clarify_count

    if clarify_count >= 4:
        enriched = (
            f"[系统指令] 已追问 {clarify_count} 轮，信息足够。"
            f"请基于以下补充直接生成执行计划，绝对不要再提问。\n\n"
            f"用户补充：{answer}"
        )
    else:
        enriched = answer

    state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=enriched)]
    thread_id = thread_id or f"ui_{uuid.uuid4().hex[:12]}"

    chat = messages_to_chatbot(state)
    chat.append({"role": "assistant", "content": f"📝 已收到补充信息（第 {clarify_count} 轮），重新分析需求..."})

    SESSION_DATA[thread_id] = {"is_interrupt": False, "state": state}

    yield _make_outputs(
        chat, state, "🔄 重新分析需求...",
        get_task_progress_html(state), list_workspace_files(),
        gr.update(visible=False), gr.update(visible=False),
        f"📝 已收集第 {clarify_count} 轮补充",
        _initial_progress("重新分析需求"),
    )

    yield from stream_and_check(app_graph, state, thread_id, chat)


# ==========================================================================
# 人工干预裁决
# ==========================================================================

def on_intervene(app_graph, action: str, custom_input: str, state: dict,
                 thread_id: str, is_interrupt: bool):
    """
    人工干预裁决（生成器）。

    - GraphInterrupt 路径：Command(resume=...)
    - react_blocked 路径：修改 state 后重新 stream
    """
    action = action or "继续执行"
    custom_input = (custom_input or "").strip()

    if not state:
        yield _make_outputs(
            [{"role": "assistant", "content": "❌ 状态丢失，请重新启动任务"}],
            {}, "状态丢失",
            get_task_progress_html({}), list_workspace_files(),
            gr.update(visible=False), gr.update(visible=False),
            "### ❌ 状态丢失\n请重新提交需求",
            _empty_progress(),
        )
        return

    # pipeline 在捕获 GraphInterrupt 时会将 _ui_is_interrupt 写入 state
    if state and state.pop("_ui_is_interrupt", False):
        is_interrupt = True

    logger.info(f"[UI] 人工干预: action={action}, is_interrupt={is_interrupt}")

    if is_interrupt:
        # GraphInterrupt → 用 Command 恢复
        if "强制提交" in action:
            next_input = Command(
                resume=action,
                update={"force_submit": True, "react_blocked": False, "react_round": 0},
            )
        else:
            next_input = Command(resume=action)
        chat = messages_to_chatbot(state)
    else:
        # react_blocked flag / sandbox failure → 修改 state 后重新 invoke
        plan_box = state.get("planning")
        exec_box = state.get("execution")

        stuck_tasks = []
        if plan_box:
            for t in plan_box.task_plan:
                if t.status == "pending":
                    stuck_tasks.append(t)

        if "跳过" in action:
            for t in stuck_tasks:
                t.status = "failed"
                t.result = "用户跳过"
            logger.info(f"[UI] 跳过 {len(stuck_tasks)} 个子任务")

        elif "强制提交" in action:
            state["force_submit"] = True
            for t in stuck_tasks:
                t.status = "pending"
            logger.info(f"[UI] 强制提交 {len(stuck_tasks)} 个子任务")

        elif "修改需求" in action and custom_input:
            state["messages"] = list(state.get("messages", [])) + [
                HumanMessage(content=custom_input)
            ]
            for t in stuck_tasks:
                t.description = f"{t.description}\n[用户补充] {custom_input}"
                t.result = f"用户修改需求：{custom_input}"
            logger.info(f"[UI] 修改 {len(stuck_tasks)} 个子任务方向")

        else:
            # 继续执行：只重置 ReAct 状态位，保留 error_trace / retry_count
            if exec_box:
                exec_box.all_tasks_completed = False
            logger.info(f"[UI] 继续执行，保留错误上下文供修复参考（{len(stuck_tasks)} 个任务）")

        state["react_blocked"] = False
        state["react_block_reason"] = ""
        state["react_round"] = 0
        state["react_finished"] = False
        next_input = state

        chat = messages_to_chatbot(state)

    action_desc = action
    if "修改需求" in action and custom_input:
        action_desc = f"修改需求 → {custom_input[:100]}"
    chat.append({"role": "assistant", "content": f"🛠️ 人工干预：{action_desc}"})

    thread_id = thread_id or f"ui_{uuid.uuid4().hex[:12]}"
    SESSION_DATA[thread_id] = {"is_interrupt": False, "state": state}

    # 先 yield 干预确认
    yield _make_outputs(
        chat, state, f"🛠️ {action_desc}",
        get_task_progress_html(state), list_workspace_files(),
        gr.update(visible=False), gr.update(visible=False),
        f"🛠️ 人工干预：{action_desc}",
        _initial_progress("人工干预后继续"),
    )

    # 流式继续
    yield from stream_and_check(app_graph, next_input, thread_id, chat)


# ==========================================================================
# 辅助
# ==========================================================================

def on_action_change(action: str) -> dict:
    """当用户选择「修改需求」时显示额外输入框"""
    return gr.update(visible=(action == "修改需求"))
