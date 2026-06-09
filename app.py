"""
=============================================================================
app.py —— AI 编码代理 Gradio 界面（本地运行版）
=============================================================================

基于 main.py 的 8 阶段流水线，提供可视化 Gradio 交互界面。

【与 main.py 对齐的核心设计】
  - Docker 沙盒（init_container + docker_exec），同 main.py
  - SqliteSaver 持久化断点（output/checkpoints.db），同 main.py
  - 全文拷贝 main.py 的 save_checkpoint / load_checkpoint / print_summary 逻辑

【本地运行】
  python app.py

【与魔塔版（modelscope_app/）的区别】
  - 不用本地 subprocess 替代沙盒，直接用 Docker
  - 用 SqliteSaver 替代 MemorySaver（重启不丢状态）
  - 不含 modelscope_app/ 的拆分模块
=============================================================================
"""

import os
import sys
import uuid
import json
import sqlite3
import traceback

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from langchain_core.messages import HumanMessage

from core.graph.main import build_master_graph
from core.state import PlanningContext, ExecutionContext, IntegrationContext, OutputContext
from core.tools.docker_sandbox import init_container, docker_exec
from core.logger import logger


# ==========================================================================
# 全局初始化
# ==========================================================================

# Docker 沙盒
if not init_container():
    logger.error("Docker 容器初始化失败，请确认 Docker Desktop 已启动")
    sys.exit(1)

# 持久化断点
os.makedirs("output", exist_ok=True)
db_conn = sqlite3.connect("output/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(db_conn)
app_graph = build_master_graph(checkpointer=checkpointer)
print("[UI] ✅ LangGraph 8 阶段流水线已就绪（Docker + SqliteSaver）")

# 断点文件路径（对齐 main.py）
CHECKPOINT_STATE_FILE = "output/.pipeline_state.json"


# ==========================================================================
# 辅助函数
# ==========================================================================

def list_workspace() -> str:
    """列出 Docker 沙盒工作区文件"""
    result = docker_exec("find /workspace -type f 2>/dev/null | head -50")
    if result["returncode"] == 0 and result["stdout"].strip():
        return result["stdout"].strip()
    return "(工作区为空)"


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
            f'background:#fafafa;border-radius:4px;'
            f'border-left:3px solid {"#52c41a" if t.status=="finished" else "#faad14" if t.status=="testing" else "#ff4d4f" if t.status=="failed" else "#d9d9d9"}">'
            f'{e} <b>T{t.task_id}</b>: {desc}{extra}</div>'
        )
    parts.append('</div>')
    return "".join(parts)


def messages_to_chat(state: dict) -> list:
    """LangChain 消息 → Gradio chatbot 格式"""
    chat = []
    for m in state.get("messages", []):
        role = "user" if getattr(m, "type", "") == "human" else "assistant"
        content = str(getattr(m, "content", ""))[:2000].strip()
        if content:
            chat.append({"role": role, "content": content})
    return chat


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
    react_history = state.get("react_history", [])
    thought = ""
    if react_history:
        for step in reversed(react_history):
            t = step.get("thought", "") if isinstance(step, dict) else getattr(step, "thought", "")
            if t and t.strip():
                thought = t.strip()[:1500]
                break

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
        f'<pre style="background:#fafafa;padding:10px;border-radius:4px;'
        f'font-size:12px;max-height:200px;overflow-y:auto;'
        f'white-space:pre-wrap;word-break:break-all;line-height:1.5">'
        f'{thought}</pre>'
        f'</div>'
    )


def build_summary(state: dict) -> str:
    """执行摘要 Markdown（对齐 main.py 的 print_summary）"""
    planning = state.get("planning")
    execution = state.get("execution")
    output_ctx = state.get("output")
    review = state.get("_review_summary")

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

    if review:
        sr = review.get("success_rate", 0)
        pit = review.get("pitfall_count", 0)
        total_exp = review.get("total_experiences_in_db", 0)
        lines.append(
            f"\n📚 经验归档: 成功率 {sr:.0%}  |  "
            f"踩坑 {pit} 条  |  向量库累计 {total_exp} 条"
        )

    return "\n".join(lines)


# ==========================================================================
# 断点保存/恢复（全文拷贝 main.py）
# ==========================================================================

def save_checkpoint(state: dict, thread_id: str) -> None:
    """保存管道状态快照到 JSON 文件，供断点续传使用。"""
    planning = state.get("planning")
    execution = state.get("execution")
    integration = state.get("integration")
    output_ctx = state.get("output")

    snapshot = {
        "thread_id": thread_id,
        "planning": planning.model_dump() if planning else {},
        "execution": execution.model_dump() if execution else {},
        "integration": integration.model_dump() if integration else {},
        "output": output_ctx.model_dump() if output_ctx else {},
        "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": str(m.content)}
            for m in state.get("messages", [])
        ],
    }

    os.makedirs("output", exist_ok=True)
    with open(CHECKPOINT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"[断点] 状态已保存 → {CHECKPOINT_STATE_FILE}")


def load_checkpoint() -> tuple:
    """从 JSON 文件加载管道状态快照，返回 (initial_state_dict, thread_id)。"""
    if not os.path.exists(CHECKPOINT_STATE_FILE):
        raise FileNotFoundError(f"断点文件不存在: {CHECKPOINT_STATE_FILE}")

    with open(CHECKPOINT_STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    thread_id = data.get("thread_id", "")

    task_plan = data.get("planning", {}).get("task_plan", [])
    for t in task_plan:
        if t["status"] in ("failed",):
            t["status"] = "pending"
            t["result"] = f"[续传] 上次失败: {t.get('result', '')}"

    planning = PlanningContext(**data.get("planning", {}))
    execution = ExecutionContext(**data.get("execution", {}))
    execution.retry_count = 0
    execution.error_trace = ""
    execution.all_tasks_completed = False
    planning.need_clarification = False
    planning.clarification_question = ""

    messages = []
    for m in data.get("messages", []):
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))

    return {
        "messages": messages,
        "planning": planning,
        "execution": execution,
        "integration": IntegrationContext(),
        "output": OutputContext(),
    }, thread_id


# ==========================================================================
# 流水线执行核心
# ==========================================================================

def _blocked(chat, state, status, block_type, reason) -> tuple:
    """构建暂停态返回"""
    parts = [f"### ⚠️ {reason[:600]}"]
    err = build_error_html(state)
    if err:
        parts.append(err)
    thought = build_thought_html(state)
    if thought:
        parts.append(thought)
    intervene_html = "\n".join(parts)

    return (
        chat, state, status,
        task_progress_html(state), list_workspace(),
        gr.update(visible=(block_type == "clarify")),
        gr.update(visible=(block_type == "intervene")),
        intervene_html,
    )


def _done(chat, state) -> tuple:
    """构建完成态返回"""
    summary = build_summary(state)
    chat.append({"role": "assistant", "content": summary})
    return (
        chat, state, "✅ 流水线执行完成",
        task_progress_html(state), list_workspace(),
        gr.update(visible=False), gr.update(visible=False),
        f"### ✅ 执行完成\n\n{summary}",
    )


def invoke_and_check(state: dict, thread_id: str, chat: list) -> tuple:
    """调用流水线 → 检测阻塞/完成 → 返回 UI 更新元组"""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 350}

    try:
        new_state = app_graph.invoke(state, config)
    except GraphInterrupt as e:
        vals = e.args[0] if e.args else []
        msg = vals[0] if vals else "ReAct 循环卡住，需要人工决策"
        logger.info(f"[UI] GraphInterrupt: {msg[:120]}")
        chat.append({"role": "assistant", "content": f"⚠️ 流水线中断：{msg[:300]}"})
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

    # 保存断点
    try:
        save_checkpoint(new_state, thread_id)
    except Exception:
        pass

    # 检测 1: 需求不清晰
    if planning and planning.need_clarification:
        q = planning.clarification_question
        chat.append({"role": "assistant", "content": f"🤔 {q}"})
        return _blocked(chat, new_state, "等待补充信息...", "clarify", q)

    # 检测 2: ReAct 阻塞 flag
    if new_state.get("react_blocked", False):
        reason = new_state.get("react_block_reason", "ReAct 循环卡住")
        chat.append({"role": "assistant", "content": f"⚠️ {reason[:500]}"})
        return _blocked(chat, new_state, "⚠️ ReAct 循环阻塞", "intervene", reason)

    # 检测 3: 沙盒验证失败的 pending
    if planning:
        pending = [t for t in planning.task_plan if t.status == "pending"]
        testing = [t for t in planning.task_plan if t.status == "testing"]
        doing   = [t for t in planning.task_plan if t.status == "doing"]

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

    # 正常完成
    chat.append({"role": "assistant", "content": "✅ 流水线执行完成"})
    return _done(chat, new_state)


# ==========================================================================
# Gradio 事件处理
# ==========================================================================

import gradio as gr


def on_start(user_request: str, resume_id: str):
    """启动新任务 / 断点续传"""
    user_request = (user_request or "").strip()
    resume_id = (resume_id or "").strip()

    if not user_request and not resume_id:
        return (
            [{"role": "assistant", "content": "👋 请输入任务需求或断点续传 ID"}],
            {}, "请输入需求",
            task_progress_html({}), list_workspace(),
            gr.update(visible=False), gr.update(visible=False),
            "### 🤖 AI 编码代理\n8 阶段流水线：感知→规划→调度→执行→验证→整合→输出→复盘\n\n输入需求后点击启动按钮",
        )

    if resume_id:
        config = {"configurable": {"thread_id": resume_id}}
        try:
            saved = app_graph.get_state(config)
        except Exception:
            saved = None

        if saved is None or not saved.values:
            return (
                [{"role": "assistant", "content": f"❌ thread_id 无效: `{resume_id}`"}],
                {}, "无效 ID",
                task_progress_html({}), list_workspace(),
                gr.update(visible=False), gr.update(visible=False),
                f"### ❌ 存档不存在\n`{resume_id}` 无对应存档",
            )

        state = saved.values
        thread_id = resume_id

        # 重置续传状态
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

        chat = [{"role": "assistant", "content": f"🔄 从断点恢复 (thread_id: `{thread_id}`)"}]
    else:
        # 全新任务 → 清空工作区（对齐 main.py）
        state = {"messages": [HumanMessage(content=user_request)]}
        thread_id = f"ui_{uuid.uuid4().hex[:12]}"

        docker_exec("rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null; echo ok")
        logger.info("[沙盒] 新任务——已清空工作区")

        chat = [
            {"role": "user", "content": user_request},
            {"role": "assistant", "content": "🚀 流水线启动，正在分析需求..."},
        ]

    return invoke_and_check(state, thread_id, chat)


def on_clarify(answer: str, state: dict, thread_id: str):
    """回答需求澄清问题"""
    answer = (answer or "").strip()
    if not answer or not state:
        return (
            messages_to_chat(state or {}), state, "请输入补充信息",
            task_progress_html(state or {}), list_workspace(),
            gr.update(visible=True), gr.update(visible=False),
            "请输入补充后再提交",
        )

    state["messages"] = list(state.get("messages", [])) + [HumanMessage(content=answer)]
    thread_id = thread_id or f"ui_{uuid.uuid4().hex[:12]}"

    chat = messages_to_chat(state)
    chat.append({"role": "assistant", "content": "📝 已收到补充信息，重新分析需求..."})

    return invoke_and_check(state, thread_id, chat)


def on_intervene(action: str, custom: str, state: dict,
                 thread_id: str, is_interrupt: bool):
    """
    人工干预裁决。
    两条路径：GraphInterrupt（用 Command.resume） / react_blocked（改 state 重跑）
    """
    action = action or "继续执行"
    custom = (custom or "").strip()

    if not state:
        return (
            [{"role": "assistant", "content": "❌ 状态丢失，请重新提交需求"}],
            {}, "状态丢失",
            task_progress_html({}), list_workspace(),
            gr.update(visible=False), gr.update(visible=False),
            "### ❌ 状态丢失\n请重新启动任务",
        )

    logger.info(f"[UI] 人工干预: action={action}, is_interrupt={is_interrupt}")

    if is_interrupt:
        # GraphInterrupt 路径 → Command.resume（同 main.py）
        if "强制提交" in action:
            nxt = Command(resume=action,
                          update={"force_submit": True, "react_blocked": False, "react_round": 0})
        else:
            nxt = Command(resume=action)
        chat = messages_to_chat(state)
    else:
        # react_blocked / sandbox 失败路径 → 修改 state 重跑（同 main.py）
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
            # 继续执行：重置计数，保留 pending
            if exec_box:
                for t in stuck:
                    exec_box.task_retry_count[t.task_id] = 0
                exec_box.retry_count = 0
                exec_box.error_trace = ""

        # 清除阻塞状态
        state["react_blocked"] = False
        state["react_block_reason"] = ""
        state["react_round"] = 0
        state["react_finished"] = False
        nxt = state

        chat = messages_to_chat(state)

    desc = action if not ("修改需求" in action and custom) else f"修改需求 → {custom[:80]}"
    chat.append({"role": "assistant", "content": f"🛠️ 人工干预：{desc}"})

    thread_id = thread_id or f"ui_{uuid.uuid4().hex[:12]}"
    return invoke_and_check(nxt, thread_id, chat)


def on_action_change(action: str):
    return gr.update(visible=(action == "修改需求"))


# ==========================================================================
# CSS
# ==========================================================================

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
#agent-chatbot { border-radius: 8px !important; border: 1px solid #e0e0e0 !important; }
#agent-chatbot .message-row { font-size: 14px !important; }
#left-panel { background: #fafbfc; border-right: 1px solid #e0e0e0; padding: 8px; }
#control-panel { background: #fafbfc; border-left: 1px solid #e0e0e0; padding: 8px; }
#intervene-panel {
    border: 2px solid #ff4d4f !important;
    box-shadow: 0 0 12px rgba(255,77,79,0.3) !important;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 4px rgba(255,77,79,0.15); }
    50% { box-shadow: 0 0 16px rgba(255,77,79,0.45); }
    100% { box-shadow: 0 0 4px rgba(255,77,79,0.15); }
}
pre, code { font-family: 'Fira Code','Consolas',monospace !important; font-size: 12px !important; }
#start-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important; color: white !important; font-weight: bold !important;
}
#start-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102,126,234,0.4) !important; }
#status-text textarea { font-weight: bold !important; font-size: 13px !important; }
footer { display: none !important; }
"""


# ==========================================================================
# Gradio 布局
# ==========================================================================

def build_ui():
    with gr.Blocks(
        title="AI 编码代理控制台（本地 Docker 版）",
        css=CSS,
        theme=gr.themes.Soft(),
    ) as demo:
        agent_state = gr.State({})
        thread_id_state = gr.State("")
        is_interrupt_state = gr.State(False)

        gr.Markdown(
            "# 🤖 AI 编码代理控制台\n"
            "8 阶段流水线：感知 → 规划 → 调度 → 执行 → 验证 → 整合 → 输出 → 复盘\n"
            "> 本地 Docker 版 · SqliteSaver 持久化断点 · 重启不丢进度"
        )

        with gr.Row(equal_height=True):
            # 左栏：任务进度 + 工作区文件
            with gr.Column(scale=1, elem_id="left-panel"):
                gr.Markdown("### 📋 任务进度")
                task_html = gr.HTML(
                    value="<p style='color:#888'>暂无任务</p>",
                    elem_id="task-progress",
                )
                gr.Markdown("---\n### 📂 工作区文件")
                file_tree = gr.Textbox(
                    value="(工作区为空)",
                    lines=14,
                    interactive=False,
                    show_label=False,
                )
                refresh_btn = gr.Button("🔄 刷新", size="sm")

            # 中栏：对话日志
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[{
                        "role": "assistant",
                        "content": "👋 欢迎使用 AI 编码代理！输入需求后点击 ▶️ 启动任务。"
                    }],
                    label="执行日志",
                    height=520,
                    elem_id="agent-chatbot",
                    type="messages",
                )
                status_text = gr.Textbox(
                    value="🟢 就绪",
                    label="状态",
                    interactive=False,
                    elem_id="status-text",
                )
                result_md = gr.Markdown(
                    "### 🤖 AI 编码代理\n"
                    "输入需求后点击 **▶️ 启动任务** 按钮开始"
                )

            # 右栏：控制台
            with gr.Column(scale=1, elem_id="control-panel"):
                gr.Markdown("### ⚙️ 控制台")
                user_input = gr.Textbox(
                    label="📝 任务需求",
                    lines=3,
                    placeholder="例如：用 Python 写一个数独游戏\n或：写一个 Flask 学生成绩管理系统",
                )
                resume_id = gr.Textbox(
                    label="🆔 断点续传 ID（可选）",
                    placeholder="输入之前的 thread_id",
                )
                start_btn = gr.Button(
                    "▶️ 启动任务",
                    variant="primary",
                    size="lg",
                    elem_id="start-btn",
                )

                gr.Markdown("---")

                # 需求澄清面板
                with gr.Column(visible=False) as clarify_panel:
                    gr.Markdown("### 🤔 需求澄清")
                    clarify_question_md = gr.Markdown("")
                    clarify_input = gr.Textbox(
                        label="请补充信息",
                        lines=2,
                        placeholder="输入补充说明...",
                    )
                    clarify_btn = gr.Button("📤 提交补充", variant="secondary")

                # 人工介入面板
                with gr.Column(visible=False, elem_id="intervene-panel") as intervene_panel:
                    gr.Markdown("### ⚠️ 人工决策")
                    intervene_content = gr.HTML(value="")

                    intervene_action = gr.Radio(
                        choices=["继续执行", "强制提交", "跳过任务", "修改需求"],
                        label="选择操作",
                        value="继续执行",
                    )
                    customize_input = gr.Textbox(
                        label="自定义修改内容（仅「修改需求」时填写）",
                        lines=3,
                        visible=False,
                        placeholder="输入修改后的需求或补充指示...",
                    )
                    intervene_btn = gr.Button(
                        "🛠️ 执行决策",
                        variant="secondary",
                        size="lg",
                    )

        # ── 统一输出 ──
        OUTPUTS = [
            chatbot, agent_state, status_text,
            task_html, file_tree,
            clarify_panel, intervene_panel, intervene_content,
        ]

        # ── 事件绑定 ──
        start_btn.click(
            fn=on_start,
            inputs=[user_input, resume_id],
            outputs=OUTPUTS,
        )
        clarify_btn.click(
            fn=on_clarify,
            inputs=[clarify_input, agent_state, thread_id_state],
            outputs=OUTPUTS,
        )
        intervene_action.change(
            fn=on_action_change,
            inputs=[intervene_action],
            outputs=[customize_input],
        )
        intervene_btn.click(
            fn=on_intervene,
            inputs=[intervene_action, customize_input,
                    agent_state, thread_id_state, is_interrupt_state],
            outputs=OUTPUTS,
        )
        refresh_btn.click(
            fn=lambda: list_workspace(),
            inputs=[],
            outputs=[file_tree],
        )

    return demo


# ==========================================================================
# 入口
# ==========================================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
