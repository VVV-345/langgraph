"""
=============================================================================
ReAct 执行子图 —— 思考 → 行动 → 观察 循环 (工业级抗死锁安全重构版)
=============================================================================

【定位】
    替换原有的 worker_node，为每个子任务提供 ReAct 循环执行能力。
    LLM 可以调用真实工具（读文件、写文件、执行命令、网络搜索），
    观察结果后决定下一步，直到任务完成或卡住。

【子图拓扑】

    START
      ↓
    start_react (初始化：选任务 + 重置轮数)
      ↓
    think (LLM 思考 → 决定下一步行动)
      ↓
    act (执行工具调用)
      ↓
    judge (判断：继续 / 完成 / 卡住)
      ↓ (continue)
    think ← (循环，最多 10 轮)
      ↓ (done/blocked)
    END

【卡住处理】
    当 ReAct 达到 10 轮仍无法完成，judge 节点会设置 react_blocked=True，
    向上传播到 main.py 等待人工介入。

【使用方式】
    from core.nodes.react_worker import build_react_worker_subgraph

    react = build_react_worker_subgraph()
    workflow.add_node("react_worker", react)
=============================================================================
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.state import AgentState, ExecutionContext, ReActStep, ToolCall
from core.tools import ALL_TOOLS, execute_tool
from core.logger import logger

load_dotenv()

MAX_REACT_ROUNDS = 10

# ReAct 专用 LLM（需要复杂推理，temperature 稍高）
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),
    temperature=0.3,
    max_tokens=16384,
    max_retries=5,
    timeout=90
)


def _build_tool_description(allowed_tools: list = None) -> str:
    """构建工具列表描述文本，支持动态工具缩减"""
    lines = []
    tools = allowed_tools if allowed_tools is not None else ALL_TOOLS
    for t in tools:
        params_desc = ", ".join(
            f"{k}: {v.get('description', '')}"
            for k, v in t.parameters.get("properties", {}).items()
        )
        lines.append(f"- {t.name}({params_desc})")
    return "\n".join(lines)


def _build_current_files_snapshot(react_history: list) -> str:
    """
    从 ReAct 历史中重建本子任务当前文件状态。
    不扫描磁盘 —— 只追踪 write_file 和 edit_file 调用，
    按时间线正向演进文件内容，为模型提供一个"自己刚做了什么"的精确视图。
    防止模型因为遗忘自己写过的代码而反复调用 read_file 空转。
    """
    file_states = {}  # path -> {content, source, chars, lines}

    for i, step in enumerate(react_history):
        # 解析 action
        if isinstance(step, dict):
            action = step.get("action", {})
            if not isinstance(action, dict):
                continue
            tool_name = action.get("tool_name", "")
            params = action.get("tool_input", {})
        elif hasattr(step, "action"):
            tool_name = step.action.tool_name
            params = step.action.tool_input
        else:
            continue

        if not isinstance(params, dict):
            continue

        path = params.get("path", "")
        if not path:
            continue

        if tool_name == "write_file":
            content = params.get("content", "")
            file_states[path] = {
                "content": content,
                "source": f"第{i+1}轮写入",
                "chars": len(content),
                "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
            }
        elif tool_name == "edit_file":
            old_str = params.get("old_string", "")
            new_str = params.get("new_string", "")
            if path in file_states:
                current = file_states[path]["content"]
                if old_str in current:
                    updated = current.replace(old_str, new_str, 1)
                    file_states[path] = {
                        "content": updated,
                        "source": file_states[path]["source"] + f"→第{i+1}轮编辑",
                        "chars": len(updated),
                        "lines": updated.count("\n") + (1 if updated and not updated.endswith("\n") else 0),
                    }

    if not file_states:
        return "【📂 本子任务文件状态】\n（尚未创建或修改任何文件）\n"

    lines = ["【📂 本子任务当前文件状态（根据操作历史重建，不含前置子任务文件）】"]
    for path, info in file_states.items():
        preview = info["content"][:25000]
        if len(info["content"]) > 25000:
            preview += f"\n... (共 {info['chars']} 字符, {info['lines']} 行, 已截断前 25000 字符)"
        lines.append(f"\n── {path}  ({info['source']}, {info['chars']}字符, {info['lines']}行) ──")
        lines.append(preview)

    return "\n".join(lines) + "\n"


def _build_react_prompt(
    task_description: str,
    task_objective: str,
    previous_outputs: str,
    error_feedback: str,
    react_history: list,
    current_round: int,
    allowed_tools: list = None
) -> str:
    """构建经过过滤压缩的读写分离提示词"""
    tool_desc = _build_tool_description(allowed_tools)
    total_steps = len(react_history)

    # ── 本子任务文件状态快照（从操作历史重建，防止模型遗忘）──
    current_files = _build_current_files_snapshot(react_history)

    # 历史步骤
    history_text = ""
    if react_history:
        history_text = "【已执行的步骤历史（注意：请勿重复执行相同动作）】\n"
        for i, step in enumerate(react_history):
            if isinstance(step, dict):
                s = ReActStep(**step)
            else:
                s = step

            obs = s.observation
            tool_name = s.action.tool_name

            # 距离感应指针化：距离当前超过 2 步的老历史做折叠，防止源码墙刷屏
            is_old_step = (total_steps - i) > 2
            if tool_name in ["read_file", "search_content", "list_directory"] and is_old_step and len(obs) > 400:
                file_path = s.action.tool_input.get("path", "未知文件")
                obs = f"[系统回执] 成功读取了 {file_path} 的内容。由于该视图在后续轮次中可能已有增量修改，为防止注意力干扰，该历史老长文本已被折叠。最新代码视图请查阅上方的【📂 本子任务当前文件状态】和【已完成的前置任务成果】。"
            elif tool_name == "run_command" and len(obs) > 1000:
                obs = f"...(前部大量控制台输出已由系统折叠)...\n" + obs[-800:]
            else:
                obs = obs[:2000]

            lines = []
            t_str = s.thought.strip() if s.thought else ""
            if t_str:
                lines.append(f"  Thought: {t_str}")
            if tool_name:
                lines.append(f"  Action: {tool_name}({s.action.tool_input})")
            if obs:
                lines.append(f"  Observation: {obs}")

            history_text += f"步骤 {i + 1}:\n" + "\n".join(lines) + "\n\n"

    return f"""你是一个顶级的软件工程师，正在执行一个编程任务。你可以使用以下工具来完成工作：

{tool_desc}

【当前任务】
描述：{task_description}
目标：{task_objective}

{previous_outputs}
{current_files}
{error_feedback}
{history_text}
【当前是第 {current_round}/{MAX_REACT_ROUNDS} 轮】

任务完成时，必须调用 submit_task(summary="完成摘要") 提交结果并结束任务。"""


# ==========================================================================
# ReAct 子图节点
# ==========================================================================

def start_react_node(state: AgentState):
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    if exec_box is None:
        exec_box = ExecutionContext()

    finished_ids = {t.task_id for t in plan_box.task_plan if t.status in ("testing", "finished")}
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    ready = [t for t in pending_tasks if not t.dependencies or all(dep in finished_ids for dep in t.dependencies)]

    if not ready:
        exec_box.all_tasks_completed = True
        return {"execution": exec_box, "react_finished": True}

    task = ready[0]
    logger.info(f"[ReAct入口] 开始处理子任务 {task.task_id}: {task.description}")

    return {
        "execution": exec_box,
        "react_round": 0,
        "react_history": [],
        "react_blocked": False,
        "react_block_reason": "",
        "react_finished": False
    }


def think_node(state: AgentState):
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    round_num = state.get("react_round", 0) + 1

    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    if not pending_tasks:
        return {"react_finished": True}

    task = pending_tasks[0]

    # ── 恢复精简的结构化 previous_outputs（来自前置子任务成果）──
    previous_outputs = ""
    if exec_box.stage_outputs:
        previous_outputs = "【已完成的前置任务成果（只读全局参考视图）】\n"
        for t_id, output in exec_box.stage_outputs.items():
            if isinstance(output, dict):
                files = output.get("files", {})
                for path, code in files.items():
                    previous_outputs += f"--- 子任务 {t_id} 产出 {path} ---\n{code[:25000]}\n\n"

    error_feedback = ""
    if exec_box.error_trace:
        error_feedback = f"【⚠️ 上次沙盒验证报错——需要修复】\n{exec_box.error_trace[:25000]}\n"

    react_history = state.get("react_history", [])

    # ── 第 8 轮起物理裁剪工具列表：只允许写入 + 修改 + 提交 ──
    allowed_tools = ALL_TOOLS
    if round_num >= 8:
        logger.warning(f"[ReAct-Think] 🚨 第 {round_num} 轮触发工具强制裁剪！仅保留 write/edit/submit")
        allowed_tools = [t for t in ALL_TOOLS if t.name in ["write_file", "edit_file", "submit_task"]]

    prompt = _build_react_prompt(
        task_description=task.description, task_objective=task.objective,
        previous_outputs=previous_outputs, error_feedback=error_feedback,
        react_history=react_history, current_round=round_num,
        allowed_tools=allowed_tools
    )

    force_submit = state.get("force_submit", False)
    tool_schemas = [t.to_openai_schema() for t in allowed_tools]
    llm_with_tools = llm.bind_tools(tool_schemas)

    if force_submit:
        logger.info(f"[ReAct-Think] 第 {round_num}/{MAX_REACT_ROUNDS} 轮 ⚡强制提交模式")
        response = llm_with_tools.invoke([
            SystemMessage(content="⛔ 停止一切操作。现在必须立即调用 submit_task。"),
            HumanMessage(content="代码已足够完整。请直接执行 submit_task 工具提交任务。")
        ])
    else:
        logger.info(f"[ReAct-Think] 第 {round_num}/{MAX_REACT_ROUNDS} 轮思考中...")

        # ── 死循环检测：4 步滑动窗口检测 A→B→A→B 交替 + 相邻复读 ──
        is_repeating = False
        if len(react_history) >= 4:
            actions = []
            for step in react_history[-4:]:
                act = step.get("action", {}) if isinstance(step, dict) else getattr(step, "action", None)
                if act:
                    name = act.get("tool_name", "") if isinstance(act, dict) else getattr(act, "tool_name", "")
                    t_input = act.get("tool_input", {}) if isinstance(act, dict) else getattr(act, "tool_input", {})
                    actions.append((name, str(sorted(t_input.items())) if isinstance(t_input, dict) else str(t_input)))

            if len(actions) == 4 and actions[0] == actions[2] and actions[1] == actions[3]:
                is_repeating = True

        if not is_repeating and len(react_history) >= 2:
            l_act = react_history[-1].get("action", {}) if isinstance(react_history[-1], dict) else getattr(react_history[-1], "action", None)
            p_act = react_history[-2].get("action", {}) if isinstance(react_history[-2], dict) else getattr(react_history[-2], "action", None)
            if l_act and p_act:
                l_name = l_act.get("tool_name", "") if isinstance(l_act, dict) else getattr(l_act, "tool_name", "")
                p_name = p_act.get("tool_name", "") if isinstance(p_act, dict) else getattr(p_act, "tool_name", "")
                l_in = l_act.get("tool_input", {}) if isinstance(l_act, dict) else getattr(l_act, "tool_input", {})
                p_in = p_act.get("tool_input", {}) if isinstance(p_act, dict) else getattr(p_act, "tool_input", {})
                if l_name == p_name and l_in == p_in and l_name in ["read_file", "list_directory"]:
                    is_repeating = True

        base_system_content = (
            "你是一个编程助手，运行在 Linux Docker 容器中。\n"
            "工作目录是 /workspace，所有文件操作都相对于此目录。\n"
            "⚡ 核心规则：\n"
            "  1. 【严禁过度验证】：一旦你修改完毕，绝对不要再调用 read_file 去重复检查！\n"
            "    你刚写入的代码已在上方的【📂 本子任务当前文件状态】中精确展示，直接看顶部即可！\n"
            "  2. 【严禁自己测试】：严禁启动 web 服务或运行脚本自行验证！\n"
            "    代码的运行测试、校验和服务启动将由后续的【沙盒节点】自动完成，你只负责写代码。\n"
            "  3. 【立刻交卷】：只要核心逻辑落盘，在同一轮或下一轮立刻调用 submit_task 交付！\n"
        )
        system_messages = [SystemMessage(content=base_system_content)]

        if round_num >= 7:
            system_messages.append(SystemMessage(content=(
                f"⚠️ 核心警告：当前子任务执行已达第 {round_num}/{MAX_REACT_ROUNDS} 轮！"
                "非核心读取工具（如 read_file/list_directory/web_search）已被系统物理禁用！"
                "请立刻、无条件地调用 edit_file/write_file 写入代码并执行 submit_task 交卷！"
            )))

        if is_repeating:
            system_messages.append(SystemMessage(content=(
                "🚨 严重死循环红牌警告：系统检测到你连续多轮在重复、交替地调用完全相同的读取操作！"
                "你已经陷入了思维盲区。请立即停止调用 read_file，立刻转去调用 edit_file/write_file，"
                "或者认为代码已完整，直接调用 submit_task 结束任务！"
            )))

        response = llm_with_tools.invoke([*system_messages, HumanMessage(content=prompt)])

    # ── Thought 自动补全：Function Calling 下 content 可能为空 ──
    thought = (response.content or "").strip()
    if not thought and response.tool_calls:
        called_names = [tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '') for tc in response.tool_calls]
        thought = f"[工具意图] 决定批量调用工具: {', '.join(called_names)} 来向前推进核心任务。"

    return {
        "react_round": round_num,
        "_pending_thought": thought[:300],
        "_pending_tool_calls": [
            {
                "name": tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", ""),
                "args": tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            }
            for tc in response.tool_calls
        ]
    }


def act_node(state: AgentState):
    thought = state.get("_pending_thought", "")
    tool_calls = state.get("_pending_tool_calls", [])
    react_history = state.get("react_history", [])

    if not tool_calls:
        step = ReActStep(
            thought=thought,
            action=ToolCall(tool_name="FINISH", tool_input={}, tool_output="任务完成"),
            observation="任务标记为完成"
        )
        react_history.append(step.model_dump())
        return {
            "react_history": react_history,
            "react_finished": True,
            "_pending_thought": "",
            "_pending_tool_calls": [],
        }

    has_submit_task = any(tc["name"] == "submit_task" for tc in tool_calls)
    observations = []

    for i, tc in enumerate(tool_calls):
        name = tc["name"]
        args = tc["args"]
        logger.info(f"[ReAct-Act] [{i+1}/{len(tool_calls)}] {name}({str(args)[:150]})")

        result = execute_tool(name, **args)
        tool_call = ToolCall(
            tool_name=name, tool_input=args,
            tool_output=result[:25000], timestamp=datetime.now().isoformat()
        )

        step = ReActStep(thought=thought if i == 0 else "", action=tool_call, observation=result)
        react_history.append(step.model_dump())
        observations.append(f"[{name}]: {result[:300]}")

    merged = "\n".join(observations)
    logger.info(f"[ReAct-Act] 批量完成 {len(tool_calls)} 个工具，总输出 {len(merged)} 字符")

    result = {"react_history": react_history, "_pending_thought": "", "_pending_tool_calls": []}
    if has_submit_task:
        result["react_finished"] = True

    return result


def judge_node(state: AgentState):
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    round_num = state.get("react_round", 0)
    react_finished = state.get("react_finished", False)
    react_history = state.get("react_history", [])

    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    if not pending_tasks:
        return {}

    task = pending_tasks[0]

    if react_finished:
        logger.info(f"[ReAct-Judge] 子任务 {task.task_id} ReAct 完成")
        final_files = _extract_files(react_history, exec_box.stage_outputs.get(task.task_id, {}))
        if final_files.get("files"):
            exec_box.stage_outputs[task.task_id] = final_files

        if state.get("force_submit", False):
            task.status = "finished"
            task.result = "【人工干预】验证阶段被跳过，直接进入整合。"
        else:
            task.status = "testing"
            task.result = f"ReAct 循环完成（{round_num} 轮）"

        exec_box.current_task_index += 1
        return {
            "planning": plan_box,
            "execution": exec_box,
            "current_code": final_files,
            "react_finished": True,
            "react_blocked": False,
            "force_submit": False,
        }

    if round_num >= MAX_REACT_ROUNDS:
        logger.warning(f"[ReAct-Judge] 子任务 {task.task_id} 达到 {MAX_REACT_ROUNDS} 轮上限，暂停等待人工介入")
        task.result = f"ReAct 循环达上限（{MAX_REACT_ROUNDS} 轮），等待人工介入"

        partial_files = _extract_files(react_history, exec_box.stage_outputs.get(task.task_id, {}))
        if partial_files.get("files"):
            exec_box.stage_outputs[task.task_id] = partial_files

        # 修补 .get() 空字符串陷阱：向前回溯找有内容的 thought
        last_thought = "N/A"
        if react_history:
            last_step = react_history[-1]
            raw_t = last_step.get('thought', '') if isinstance(last_step, dict) else getattr(last_step, 'thought', '')
            if not raw_t and len(react_history) >= 2:
                for step in reversed(react_history):
                    t = step.get('thought', '') if isinstance(step, dict) else getattr(step, 'thought', '')
                    if t:
                        raw_t = t
                        break
            last_thought = raw_t if raw_t else "批量执行工具中，未留白文本思考"

        block_reason = (
            f"子任务 {task.task_id}（{task.description}）在 {MAX_REACT_ROUNDS} 轮 ReAct 循环后仍未完成。\n"
            f"最后思考：{last_thought}\n"
            f"沙盒报错：{exec_box.error_trace[:500] if exec_box.error_trace else '无'}\n"
            f"请选择：继续执行 / 跳过此任务 / 修改需求"
        )

        return {
            "planning": plan_box,
            "execution": exec_box,
            "react_finished": True,
            "react_blocked": True,
            "react_block_reason": block_reason,
        }

    logger.info(f"[ReAct-Judge] 继续循环，当前 {round_num}/{MAX_REACT_ROUNDS} 轮")
    return {}


def _extract_files(react_history: list, base_output: dict = None) -> dict:
    """
    完美文件演进提取器。
    正序遍历历史：write_file 创建/覆盖，edit_file 就地替换。
    从 base_output（前置子任务成果）继承初始文件状态，
    按时间线正向演进，确保 edit_file 的修改不丢失。
    """
    if not react_history:
        return base_output or {}

    files = dict(base_output.get("files", {})) if base_output and isinstance(base_output, dict) else {}
    main = base_output.get("main", "") if base_output and isinstance(base_output, dict) else ""

    for step in react_history:
        if isinstance(step, dict):
            action = step.get("action", {})
            if not isinstance(action, dict):
                continue
            tool_name = action.get("tool_name", "")
            params = action.get("tool_input", {})
        else:
            tool_name = step.action.tool_name
            params = step.action.tool_input

        if not isinstance(params, dict):
            continue

        path = params.get("path", "")
        if not path:
            continue

        if tool_name == "write_file":
            content = params.get("content", "")
            files[path] = content
            if not main:
                main = path
        elif tool_name == "edit_file":
            old_str = params.get("old_string", "")
            new_str = params.get("new_string", "")
            if path in files and old_str in files[path]:
                files[path] = files[path].replace(old_str, new_str, 1)

    return {"files": files, "main": main}


# ==========================================================================
# 路由函数
# ==========================================================================

def route_after_think(state: AgentState) -> str:
    return "act"


def route_after_act(state: AgentState) -> str:
    return "judge"


def route_after_judge(state: AgentState) -> str:
    if state.get("react_finished", False) or state.get("react_blocked", False):
        logger.info("[ReAct路由] 退出 ReAct 循环" +
                    ("（任务完成）" if not state.get("react_blocked") else "（需要人工介入）"))
        return "end"
    logger.info("[ReAct路由] 继续下一轮")
    return "think"


# ==========================================================================
# 构建 ReAct 执行子图
# ==========================================================================

def build_react_worker_subgraph(checkpointer=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("start_react", start_react_node)
    workflow.add_node("think", think_node)
    workflow.add_node("act", act_node)
    workflow.add_node("judge", judge_node)
    workflow.set_entry_point("start_react")
    workflow.add_edge("start_react", "think")
    workflow.add_edge("think", "act")
    workflow.add_edge("act", "judge")
    workflow.add_conditional_edges("judge", route_after_judge, {"think": "think", "end": END})
    return workflow.compile(checkpointer=checkpointer)
