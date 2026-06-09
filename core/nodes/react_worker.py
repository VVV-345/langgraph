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

MAX_REACT_ROUNDS = 7

# ── 截断常量（防止 LLM 上下文窗口被大文件撑爆）────────────────────
SNAPSHOT_HEAD = 15000       # 文件快照：头部保留字符数
SNAPSHOT_TAIL = 10000       # 文件快照：尾部保留字符数
SNAPSHOT_LIMIT = SNAPSHOT_HEAD + SNAPSHOT_TAIL  # 25000
PREVIOUS_OUTPUT_LIMIT = 20000   # 前置任务代码截断阈值
ERROR_TRACE_LIMIT = 25000       # 沙盒报错截断阈值
TOOL_OUTPUT_LIMIT = 25000       # 工具输出截断阈值

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


def _build_current_files_snapshot(react_history: list, base_output: dict = None) -> str:
    """
    从 ReAct 历史 + 上一轮遗留文件（修复模式）中重建本子任务当前文件状态。
    不扫描磁盘 —— 只追踪 write_file / edit_file / delete_file / move_file 调用，
    按时间线正向演进文件内容，为模型提供一个"自己刚做了什么"的精确视图。

    修复模式下 base_output 继承上一轮的 stage_outputs[task_id]，
    打通记忆：模型能看到上轮自己写了什么，再叠加本轮改动，避免失忆。
    """
    # ── 初始化：修复模式从 base_output 继承上一轮的文件底子 ──
    file_states: dict[str, dict] = {}
    inherited_count = 0
    if base_output and isinstance(base_output, dict):
        base_files = base_output.get("files", {})
        for path, content in base_files.items():
            if content:
                file_states[path] = {
                    "content": content,
                    "source": "[上一轮遗留]",
                    "chars": len(content),
                    "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
                }
        inherited_count = len(file_states)

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

        # ── write_file ──
        if tool_name == "write_file":
            path = params.get("path", "")
            if not path:
                continue
            content = params.get("content", "")
            file_states[path] = {
                "content": content,
                "source": f"第{i+1}轮写入",
                "chars": len(content),
                "lines": content.count("\n") + (1 if content and not content.endswith("\n") else 0),
            }

        # ── edit_file ──
        elif tool_name == "edit_file":
            path = params.get("path", "")
            if not path:
                continue
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

        # ── delete_file ──
        elif tool_name == "delete_file":
            path = params.get("path", "")
            if path and path in file_states:
                del file_states[path]

        # ── move_file (rename / relocate) ──
        elif tool_name == "move_file":
            src = params.get("source", "")
            tgt = params.get("target", "")
            if src and tgt and src in file_states:
                info = file_states.pop(src)
                info["source"] = info["source"] + f"→第{i+1}轮移动至 {tgt}"
                file_states[tgt] = info

    if not file_states:
        if inherited_count > 0:
            return "【📂 本子任务文件状态】\n（上轮遗留文件已在修复轮中被全部删除，尚未创建新文件）\n"
        return "【📂 本子任务文件状态】\n（尚未创建或修改任何文件）\n"

    # ── 构建输出：头尾双截断（防止大文件后半截完全不可见）──
    lines = ["【📂 本子任务当前文件状态（根据操作历史重建，不含前置子任务文件）】"]
    if inherited_count > 0:
        lines.append(f"（已从上一轮继承 {inherited_count} 个文件，本轮操作在此基础上增量演进）")

    for path, info in file_states.items():
        content = info["content"]
        if len(content) > SNAPSHOT_LIMIT:
            head = content[:SNAPSHOT_HEAD]
            tail = content[-SNAPSHOT_TAIL:]
            omitted = len(content) - SNAPSHOT_HEAD - SNAPSHOT_TAIL
            preview = (
                f"{head}\n\n"
                f"⸻ [省略中间 {omitted} 字符] ⸻\n\n"
                f"{tail}\n"
                f"(文件共 {info['chars']} 字符, {info['lines']} 行，已展示头部 {SNAPSHOT_HEAD} + 尾部 {SNAPSHOT_TAIL} 字符)"
            )
        else:
            preview = content
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
    allowed_tools: list = None,
    base_output: dict = None
) -> str:
    """构建经过过滤压缩的读写分离提示词"""
    tool_desc = _build_tool_description(allowed_tools)
    total_steps = len(react_history)

    # ── 本子任务文件状态快照（从操作历史 + 上轮遗留重建，防止模型遗忘）──
    current_files = _build_current_files_snapshot(react_history, base_output)

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

    # 防御：重置残留的 "doing" 任务（进程崩溃/中断后可能残留），避免永久死锁
    stale_doing = [t for t in plan_box.task_plan if t.status == "doing"]
    if stale_doing:
        logger.warning(f"[ReAct入口] 检测到残留 {len(stale_doing)} 个 'doing' 任务，重置为 pending")
        for t in stale_doing:
            t.status = "pending"

    # 仅 "finished" 满足依赖——"testing" 尚未验证通过，下游不得基于它编码
    finished_ids = {t.task_id for t in plan_box.task_plan if t.status == "finished"}
    failed_ids = {t.task_id for t in plan_box.task_plan if t.status == "failed"}
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    ready = [t for t in pending_tasks if not t.dependencies or all(
        dep in finished_ids for dep in t.dependencies
    )]

    # 依赖断裂降级：依赖了失败任务的 pending 任务自动标记为失败
    for t in pending_tasks:
        if t in ready:
            continue
        broken_deps = [d for d in t.dependencies if d in failed_ids]
        if broken_deps:
            logger.warning(
                f"[ReAct入口] 子任务 {t.task_id} 依赖了失败的任务 {broken_deps}，"
                f"自动标记为失败（依赖断裂）"
            )
            t.status = "failed"
            t.result = (
                f"依赖断裂：前置子任务 {broken_deps} 已失败，"
                f"当前任务无法执行"
            )

    # 重新计算 ready（排除刚被标记为 failed 的任务）
    ready = [t for t in plan_box.task_plan if t.status == "pending" and (
        not t.dependencies or all(dep in finished_ids for dep in t.dependencies)
    )]

    if not ready:
        exec_box.all_tasks_completed = True
        return {"execution": exec_box, "react_finished": True}

    task = ready[0]

    # ── 区分「全新编码」还是「修复重写」──
    is_repair = bool(task.result)  # result 非空 = 之前沙盒失败过，带着报错回来

    # 全新编码 → 清除上一任务的残留报错，防止误注入
    if not is_repair:
        exec_box.error_trace = ""

    # 标记为 "doing" 并锁定 _current_task_id，保证 think/judge 操作的是同一个任务
    task.status = "doing"
    logger.info(
        f"[ReAct入口] 开始处理子任务 {task.task_id}: {task.description}"
        f"{' [修复模式]' if is_repair else ''}"
    )

    return {
        "execution": exec_box,
        "react_round": 0,
        "react_history": [],
        "react_blocked": False,
        "react_block_reason": "",
        "react_finished": False,
        "_current_task_id": task.task_id,
    }


def think_node(state: AgentState):
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    round_num = state.get("react_round", 0) + 1

    # 用 _current_task_id 锁定任务——防止与 start_react 选的不是同一个
    task_id = state.get("_current_task_id", 0)
    task = None
    if task_id:
        for t in plan_box.task_plan:
            if t.task_id == task_id:
                task = t
                break
    # 兜底：_current_task_id 未设置（不应发生，但做个防御）
    if task is None:
        logger.warning(f"[ReAct-Think] _current_task_id={task_id} 未找到对应任务，回退到 pending[0]")
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
                    if len(code) > PREVIOUS_OUTPUT_LIMIT:
                        truncated = code[:PREVIOUS_OUTPUT_LIMIT]
                        truncated += (
                            f"\n\n⚠️ [系统提示] 此文件共 {len(code)} 字符，"
                            f"以上仅展示前 {PREVIOUS_OUTPUT_LIMIT} 字符。"
                            f"如需查看完整内容，请使用 read_file 读取 {path}。"
                        )
                        previous_outputs += f"--- 子任务 {t_id} 产出 {path} ---\n{truncated}\n\n"
                    else:
                        previous_outputs += f"--- 子任务 {t_id} 产出 {path} ---\n{code}\n\n"

    # ── 仅修复模式注入 error_trace（全新编码不注入，防止读到其他任务的残留报错）──
    error_feedback = ""
    is_repair = bool(task.result)  # result 非空 = 沙盒失败后回来修复
    if is_repair and exec_box.error_trace:
        error_feedback = f"【⚠️ 上次沙盒验证报错——需要修复】\n{exec_box.error_trace[:ERROR_TRACE_LIMIT]}\n"

    react_history = state.get("react_history", [])

    # ── 第 6 轮起物理裁剪工具列表：只允许写入 + 修改 + 提交 ──
    allowed_tools = ALL_TOOLS
    if round_num >= 6:
        logger.warning(f"[ReAct-Think] 🚨 第 {round_num} 轮触发工具强制裁剪！仅保留 write/edit/submit")
        allowed_tools = [t for t in ALL_TOOLS if t.name in ["write_file", "edit_file", "submit_task"]]

    # ── 修复模式下继承上轮遗留文件，让快照"不失忆" ──
    base_output = exec_box.stage_outputs.get(task.task_id) if exec_box and exec_box.stage_outputs else None

    prompt = _build_react_prompt(
        task_description=task.description, task_objective=task.objective,
        previous_outputs=previous_outputs, error_feedback=error_feedback,
        react_history=react_history, current_round=round_num,
        allowed_tools=allowed_tools, base_output=base_output
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
            "  1. 【沙盒非交互】：代码在无标准输入环境下运行，严禁使用 input()！\n"
            "    获取用户参数请用 argparse 解析命令行参数或 sys.argv。\n"
            "  2. 【严禁过度验证】：一旦你修改完毕，绝对不要再调用 read_file 去重复检查！\n"
            "    你刚写入的代码已在上方的【📂 本子任务当前文件状态】中精确展示，直接看顶部即可！\n"
            "  3. 【严禁自己测试】：严禁启动 web 服务或运行脚本自行验证！\n"
            "    代码的运行测试、校验和服务启动将由后续的【沙盒节点】自动完成，你只负责写代码。\n"
            "  4. 【立刻交卷】：只要核心逻辑落盘，在同一轮或下一轮立刻调用 submit_task 交付！\n"
        )
        system_messages = [SystemMessage(content=base_system_content)]

        if round_num >= 5:
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
            tool_output=result[:TOOL_OUTPUT_LIMIT], timestamp=datetime.now().isoformat()
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

    # 用 _current_task_id 锁定任务——与 start_react / think 保持一致
    task_id = state.get("_current_task_id", 0)
    task = None
    if task_id:
        for t in plan_box.task_plan:
            if t.task_id == task_id:
                task = t
                break
    if task is None:
        logger.warning(f"[ReAct-Judge] _current_task_id={task_id} 未找到对应任务，跳过裁决")
        return {}

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
        task.status = "pending"  # "doing" → "pending"，等待人工决策
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
    正序遍历历史：write_file 创建/覆盖，edit_file 就地替换，
    delete_file 删除，move_file 移动/重命名。
    从 base_output（前置子任务成果）继承初始文件状态，
    按时间线正向演进，确保增量修改不丢失。
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

        if tool_name == "write_file":
            path = params.get("path", "")
            if not path:
                continue
            content = params.get("content", "")
            files[path] = content
            if not main:
                main = path

        elif tool_name == "edit_file":
            path = params.get("path", "")
            if not path:
                continue
            old_str = params.get("old_string", "")
            new_str = params.get("new_string", "")
            if path in files and old_str in files[path]:
                files[path] = files[path].replace(old_str, new_str, 1)

        elif tool_name == "delete_file":
            path = params.get("path", "")
            if path and path in files:
                del files[path]
                if main == path:
                    # 入口文件被删，选取剩余第一个作为新的 main
                    remaining = list(files.keys())
                    main = remaining[0] if remaining else ""

        elif tool_name == "move_file":
            src = params.get("source", "")
            tgt = params.get("target", "")
            if src and tgt and src in files:
                files[tgt] = files.pop(src)
                if main == src:
                    main = tgt

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
