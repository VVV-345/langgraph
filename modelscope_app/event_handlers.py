"""
=============================================================================
event_handlers.py —— Gradio 事件处理函数
=============================================================================

处理启动 / 澄清 / 人工干预等用户交互事件。

所有需要 app_graph 的函数将其作为显式第一参数，
由 app.py 通过闭包注入。
=============================================================================
"""

import uuid

import gradio as gr
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from core.logger import logger
from .ui_helpers import (
    _docker,
    list_workspace,
    task_progress_html,
    messages_to_chat,
    build_error_html,
    build_thought_html,
)
from .pipeline import stream_and_check, _blocked, _progress_bar as _pb


# ==========================================================================
# 9 元组辅助
# ==========================================================================

def _make_outputs(chat, state, status, task_html, file_tree,
                  clarify_vis, intervene_vis, intervene_html, progress_html) -> tuple:
    return (chat, state, status, task_html, file_tree,
            clarify_vis, intervene_vis, intervene_html, progress_html)


def _empty_progress() -> str:
    return _pb(0, "⏳ 等待启动...")


def _initial_progress(label: str) -> str:
    return _pb(5, f"⏳ {label}")


# ==========================================================================
# 启动 / 续传
# ==========================================================================

def on_start(app_graph, user_request: str, resume_id: str):
    """启动新任务 / 断点续传（生成器，逐节点流式更新 UI）"""
    user_request = (user_request or "").strip()
    resume_id = (resume_id or "").strip()

    if not user_request and not resume_id:
        yield _make_outputs(
            [{"role": "assistant", "content": "👋 请输入任务需求或断点续传 ID"}],
            {}, "请输入需求",
            task_progress_html({}), list_workspace(),
            gr.update(visible=False), gr.update(visible=False),
            "### 🤖 AI 编码代理\n8 阶段流水线：感知→规划→调度→执行→验证→整合→输出→复盘\n\n输入需求后点击启动按钮",
            _empty_progress(),
        )
        return

    if resume_id:
        config = {"configurable": {"thread_id": resume_id}}
        try:
            saved = app_graph.get_state(config)
        except Exception:
            saved = None

        if saved is None or not saved.values:
            yield _make_outputs(
                [{"role": "assistant", "content": f"❌ thread_id 无效: `{resume_id}`"}],
                {}, "无效 ID",
                task_progress_html({}), list_workspace(),
                gr.update(visible=False), gr.update(visible=False),
                f"### ❌ 存档不存在\n`{resume_id}` 无对应存档",
                _empty_progress(),
            )
            return

        state = saved.values
        thread_id = resume_id

        # ── 守卫：已完成任务不可续传 ──
        planning_check = state.get("planning")
        exec_check = state.get("execution")
        if planning_check and planning_check.task_plan:
            tasks = planning_check.task_plan
            all_done = all(t.status == "finished" for t in tasks)
            if all_done and exec_check and getattr(exec_check, "all_tasks_completed", False):
                yield _make_outputs(
                    [{"role": "assistant",
                      "content": "✅ 该任务已执行完毕，无需续传。请直接提交新任务。"}],
                    state, "✅ 任务已完成",
                    task_progress_html(state), list_workspace(),
                    gr.update(visible=False), gr.update(visible=False),
                    "### ✅ 任务已完成\n\n所有子任务均已成功执行，无需断点续传。"
                    "\n\n想开新任务请在输入框填写需求后点击启动。",
                    _empty_progress(),
                )
                return
        # ── 守卫结束 ──

        # ── 检测：如果上次是因需求不清晰而暂停，直接展示澄清面板 ──
        plan_box = state.get("planning")
        if plan_box and plan_box.need_clarification:
            q = plan_box.clarification_question or "请补充更多需求细节"
            logger.info(f"[UI续传] 检测到待澄清状态，直接展示澄清面板: {q[:80]}")
            chat = [
                {"role": "assistant", "content": f"🔄 从断点恢复 (thread_id: `{thread_id}`)"},
                {"role": "assistant", "content": f"🤔 {q}"},
            ]
            yield _make_outputs(
                chat, state, "等待补充信息...",
                task_progress_html(state), list_workspace(),
                gr.update(visible=False), gr.update(visible=False),
                f"### 🚀 流水线断点恢复\n\n需要补充信息",
                _pb(5, "⏳ 等待补充"),
            )
            yield _blocked(chat, state, "等待补充信息...", "clarify", q)
            return

        exec_box = state.get("execution")
        if exec_box:
            exec_box.retry_count = 0
            exec_box.error_trace = ""
            exec_box.all_tasks_completed = False
        if plan_box:
            plan_box.need_clarification = False
            plan_box.clarification_question = ""
            for t in plan_box.task_plan:
                if t.status == "failed":
                    t.status = "pending"
                    t.result = f"[续传] 上次失败: {t.result}"

        chat = [{"role": "assistant", "content": f"🔄 从断点恢复 (thread_id: `{thread_id}`)"}]
    else:
        state = {"messages": [HumanMessage(content=user_request)]}
        thread_id = f"ui_{uuid.uuid4().hex[:12]}"
        _docker("rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null; echo ok")
        chat = [
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": "🚀 流水线启动，正在分析需求..."},
        ]

    yield _make_outputs(
        chat, state, "🔄 流水线启动中...",
        task_progress_html(state), list_workspace(),
        gr.update(visible=False), gr.update(visible=False),
        "### 🚀 流水线启动",
        _initial_progress("流水线启动"),
    )

    yield from stream_and_check(app_graph, state, thread_id, chat)


# ==========================================================================
# 需求澄清
# ==========================================================================

def on_clarify(app_graph, answer: str, state: dict, thread_id: str):
    """回答需求澄清问题（生成器，与 main.py 对齐：最多 4 轮）"""
    answer = (answer or "").strip()
    if not answer or not state:
        yield _make_outputs(
            messages_to_chat(state or {}), state, "请输入补充信息",
            task_progress_html(state or {}), list_workspace(),
            gr.update(visible=True), gr.update(visible=False),
            "请输入补充后再提交",
            _pb(0, "⏸️ 等待输入"),
        )
        return

    planning = state.get("planning")
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

    chat = messages_to_chat(state)
    chat.append({"role": "assistant", "content": f"📝 已收到补充信息（第 {clarify_count} 轮），重新分析需求..."})

    yield _make_outputs(
        chat, state, "🔄 重新分析需求...",
        task_progress_html(state), list_workspace(),
        gr.update(visible=False), gr.update(visible=False),
        f"📝 已收集第 {clarify_count} 轮补充",
        _initial_progress("重新分析需求"),
    )

    yield from stream_and_check(app_graph, state, thread_id, chat)


# ==========================================================================
# 人工干预裁决
# ==========================================================================

def on_intervene(app_graph, action: str, custom: str, state: dict,
                 thread_id: str, is_interrupt: bool):
    """人工干预裁决（生成器）"""
    action = action or "继续执行"
    custom = (custom or "").strip()

    if not state:
        yield _make_outputs(
            [{"role": "assistant", "content": "❌ 状态丢失，请重新提交需求"}],
            {}, "状态丢失",
            task_progress_html({}), list_workspace(),
            gr.update(visible=False), gr.update(visible=False),
            "### ❌ 状态丢失\n请重新启动任务",
            _empty_progress(),
        )
        return

    if state and state.pop("_ui_is_interrupt", False):
        is_interrupt = True

    logger.info(f"[UI] 人工干预: action={action}, is_interrupt={is_interrupt}")

    if is_interrupt:
        if "强制提交" in action:
            nxt = Command(resume=action,
                          update={"force_submit": True, "react_blocked": False, "react_round": 0})
        else:
            nxt = Command(resume=action)
        chat = messages_to_chat(state)
    else:
        plan_box = state.get("planning")
        exec_box = state.get("execution")

        stuck = []
        if plan_box:
            stuck = [t for t in plan_box.task_plan if t.status == "pending"]

        if "跳过" in action:
            for t in stuck:
                t.status = "failed"
                t.result = "用户跳过"
        elif "强制提交" in action:
            state["force_submit"] = True
        elif "修改需求" in action and custom:
            state["messages"] = list(state.get("messages", [])) + [
                HumanMessage(content=custom)
            ]
            for t in stuck:
                t.description = f"{t.description}\n[用户补充] {custom}"
                t.result = f"用户修改需求: {custom}"
        else:
            if exec_box:
                exec_box.all_tasks_completed = False
            logger.info(f"[UI] 继续执行，保留错误上下文供修复参考（{len(stuck)} 个任务）")

        state["react_blocked"] = False
        state["react_block_reason"] = ""
        state["react_round"] = 0
        state["react_finished"] = False
        nxt = state
        chat = messages_to_chat(state)

    desc = action if not ("修改需求" in action and custom) else f"修改需求 → {custom[:80]}"
    chat.append({"role": "assistant", "content": f"🛠️ 人工干预：{desc}"})

    thread_id = thread_id or f"ui_{uuid.uuid4().hex[:12]}"

    yield _make_outputs(
        chat, state, f"🛠️ {desc}",
        task_progress_html(state), list_workspace(),
        gr.update(visible=False), gr.update(visible=False),
        f"🛠️ 人工干预：{desc}",
        _initial_progress("人工干预后继续"),
    )

    yield from stream_and_check(app_graph, nxt, thread_id, chat)


# ==========================================================================
# 辅助
# ==========================================================================

def on_action_change(action: str):
    return gr.update(visible=(action == "修改需求"))
