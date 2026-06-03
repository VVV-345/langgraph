"""
=============================================================================
ReAct 执行子图 —— 思考 → 行动 → 观察 循环
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
    think ← (循环，最多 7 轮)
      ↓ (done/blocked)
    END

【卡住处理】
    当 ReAct 达到 7 轮仍无法完成，judge 节点会设置 react_blocked=True，
    通过 LangGraph interrupt() 暂停等待人工介入。

【使用方式】
    from core.nodes.react_worker import build_react_worker_subgraph

    react = build_react_worker_subgraph()
    workflow.add_node("react_worker", react)
=============================================================================
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.state import AgentState, ExecutionContext, ReActStep, ToolCall
from core.tools import ALL_TOOLS, execute_tool, get_tool_schemas
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


def _build_tool_description() -> str:
    """构建工具列表描述文本，注入 prompt"""
    lines = []
    for t in ALL_TOOLS:
        params_desc = ", ".join(
            f"{k}: {v.get('description', '')}"
            for k, v in t.parameters.get("properties", {}).items()
        )
        lines.append(f"- {t.name}({params_desc})")
    return "\n".join(lines)


def _build_react_prompt(
    task_description: str,
    task_objective: str,
    previous_outputs: str,
    error_feedback: str,
    react_history: list,
    current_round: int,
) -> str:
    """构建 ReAct 提示词"""
    tool_desc = _build_tool_description()

    # 历史步骤
    history_text = ""
    if react_history:
        history_text = "【已执行的步骤】\n"
        for i, step in enumerate(react_history):
            if isinstance(step, dict):
                s = ReActStep(**step)
            else:
                s = step
            history_text += f"步骤 {i + 1}:\n{s.to_prompt_str()}\n\n"

    return f"""你是一个顶级的软件工程师，正在执行一个编程任务。你可以使用以下工具来完成工作：

{tool_desc}

【当前任务】
描述：{task_description}
目标：{task_objective}

{previous_outputs}
{error_feedback}
{history_text}
【当前是第 {current_round}/{MAX_REACT_ROUNDS} 轮】

任务完成时，必须调用 submit_task(summary="完成摘要") 提交结果并结束。"""


# ==========================================================================
# ReAct 子图节点
# ==========================================================================

def start_react_node(state: AgentState):
    """
    入口节点：选择当前待处理的子任务，初始化 ReAct 循环状态。
    """
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    if exec_box is None:
        exec_box = ExecutionContext()

    # 找到第一个依赖已满足的 pending 任务
    finished_ids = {
        t.task_id for t in plan_box.task_plan
        if t.status in ("testing", "finished")
    }
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    ready = [
        t for t in pending_tasks
        if not t.dependencies or all(dep in finished_ids for dep in t.dependencies)
    ]

    if not ready:
        blocked = [t for t in pending_tasks if t.dependencies]
        if blocked:
            missing = set()
            for t in blocked:
                for dep in t.dependencies:
                    if dep not in finished_ids:
                        missing.add(dep)
            logger.warning(
                f"[ReAct入口] 所有待处理任务依赖未满足，"
                f"缺: {sorted(missing)}，已就绪: {sorted(finished_ids)}"
            )
        else:
            logger.info("[ReAct入口] 没有待处理任务，跳过")
        exec_box.all_tasks_completed = True
        return {
            "execution": exec_box,
            "react_finished": True
        }

    task = ready[0]
    logger.info(f"[ReAct入口] 开始处理子任务 {task.task_id}: {task.description}")

    # 初始化 ReAct 状态
    return {
        "execution": exec_box,
        "react_round": 0,
        "react_history": [],
        "react_blocked": False,
        "react_block_reason": "",
        "react_finished": False
    }


def think_node(state: AgentState):
    """
    思考节点：LLM 分析当前状态和已有历史，决定下一步行动。
    使用 OpenAI function calling（bind_tools）替代文本解析。
    """
    plan_box = state.get("planning")
    exec_box = state.get("execution")

    round_num = state.get("react_round", 0) + 1

    # 找到当前正在处理的 pending 任务
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    if not pending_tasks:
        return {"react_finished": True}

    task = pending_tasks[0]

    # 收集前置任务成果
    previous_outputs = ""
    if exec_box.stage_outputs:
        previous_outputs = "【已完成的前置任务成果】\n"
        for t_id, output in exec_box.stage_outputs.items():
            if isinstance(output, dict):
                files = output.get("files", {})
                for path, code in files.items():
                    previous_outputs += f"--- 子任务 {t_id} 产出 {path} ---\n{code[:2000]}\n\n"
            else:
                # 兼容旧格式
                previous_outputs += f"--- 子任务 {t_id} 的产出 ---\n{str(output)[:3000]}\n\n"

    # 沙盒报错反馈
    error_feedback = ""
    if exec_box.error_trace:
        error_feedback = (
            "【⚠️ 上次沙盒验证报错——需要修复】\n"
            f"{exec_box.error_trace[:2000]}\n"
        )

    # 历史 ReAct 步骤
    react_history = state.get("react_history", [])

    prompt = _build_react_prompt(
        task_description=task.description,
        task_objective=task.objective,
        previous_outputs=previous_outputs,
        error_feedback=error_feedback,
        react_history=react_history,
        current_round=round_num,
    )

    force_submit = state.get("force_submit", False)

    # ── 使用 OpenAI function calling 替代文本解析 ──
    tool_schemas = get_tool_schemas()
    llm_with_tools = llm.bind_tools(tool_schemas)

    if force_submit:
        logger.info(f"[ReAct-Think] 第 {round_num}/{MAX_REACT_ROUNDS} 轮 ⚡强制提交模式")
        response = llm_with_tools.invoke([
            SystemMessage(content=(
                "⛔ 停止一切操作。\n"
                "你的代码已经完整交付，项目已通过测试验证。\n"
                "现在必须立即调用 submit_task(summary=\"代码已完整交付，任务完成\")。\n"
                "不要再读文件、不要列目录、不要做任何其他操作。只调用 submit_task。"
            )),
            HumanMessage(content="代码已足够完整，无需再阅读任何文件。请直接执行 submit_task 工具提交任务。")
        ])
    else:
        logger.info(f"[ReAct-Think] 第 {round_num}/{MAX_REACT_ROUNDS} 轮思考中...")
        response = llm_with_tools.invoke([
            SystemMessage(content=(
                "你是一个编程助手，运行在 Linux Docker 容器中。\n"
                "工作目录是 /workspace，所有文件操作都相对于此目录。\n"
                "📁 工具说明：\n"
                "  • list_directory — 查看目录结构\n"
                "  • read_file — 读取文件\n"
                "  • write_file — 创建或覆盖文件\n"
                "  • edit_file — 局部修改文件\n"
                "  • search_content — 搜索代码\n"
                "  • run_command — 执行普通 shell 命令（如安装依赖或查看环境）\n"
                "  • submit_task — 任务完成时调用，提交并结束当前任务\n"
                "⚡ 核心规则（必须绝对遵守）：\n"
                "  1. 【职责边界】：你的唯一任务是分析需求并编写/保存代码。\n"
                "  2. 【严禁过度验证】：一旦你调用 write_file 写完了任务要求的代码，绝对不要再调用 read_file 去检查你刚写的代码！写完即视为成功。\n"
                "  3. 【严禁自己测试】：代码的运行测试、校验和服务启动将由后续的【沙盒节点】自动完成。你严禁为了测试而启动 web 服务或无限循环运行脚本。\n"
                "  4. 【立刻交卷】：只要核心代码文件落盘，必须在同一轮或下一轮立刻调用 submit_task(summary=\"...\") 提交结果，不要做任何拖延！\n"
            )),
            HumanMessage(content=prompt)
        ])

    # 提取 LLM 文本输出作为思考过程
    thought = (response.content or "")[:300]

    # 判断是否有工具调用
    if not response.tool_calls:
        # 无工具调用 = 任务完成
        logger.debug(f"[ReAct-Think] 思考: {thought[:100]}")
        logger.info("[ReAct-Think] 行动: FINISH（模型未调用工具，判定完成）")
        return {
            "react_round": round_num,
            "_pending_thought": thought or "任务完成",
            "_pending_tool_calls": [],  # 空列表 = FINISH
        }

    # 收集全部工具调用（不再只取第一个）
    tool_calls = []
    for tc in response.tool_calls:
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args", {})
        else:
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
        tool_calls.append({"name": name, "args": args})

    names = [t["name"] for t in tool_calls]
    logger.info(f"[ReAct-Think] 第 {round_num} 轮计划执行 {len(tool_calls)} 个工具: {names}")
    logger.debug(f"[ReAct-Think] 思考: {thought[:100]}")

    return {
        "react_round": round_num,
        "_pending_thought": thought,
        "_pending_tool_calls": tool_calls,
    }


def act_node(state: AgentState):
    """
    执行节点：执行 think 节点返回的全部工具调用（批量执行）。
    如果调用中包含 submit_task，所有工具执行完毕后标记任务完成（显式退出）。
    """
    thought = state.get("_pending_thought", "")
    tool_calls = state.get("_pending_tool_calls", [])
    react_history = state.get("react_history", [])

    # 空列表 = LLM 判定任务完成（隐式退出，保留为 fallback）
    if not tool_calls:
        step = ReActStep(
            thought=thought,
            action=ToolCall(tool_name="FINISH", tool_input={}, tool_output="任务完成"),
            observation="任务标记为完成"
        )
        react_history.append(step.model_dump())
        logger.info("[ReAct-Act] LLM 判定任务完成（隐式退出）")
        return {
            "react_history": react_history,
            "react_finished": True,
            "_pending_thought": "",
            "_pending_tool_calls": [],
        }

    # 检查本轮批量中是否包含 submit_task（后置检测，不中断批量执行）
    has_submit_task = any(tc["name"] == "submit_task" for tc in tool_calls)

    # 批量执行全部工具调用（包括 submit_task 本身）
    observations = []
    for i, tc in enumerate(tool_calls):
        name = tc["name"]
        args = tc["args"]
        logger.info(f"[ReAct-Act] [{i+1}/{len(tool_calls)}] {name}({str(args)[:150]})")

        result = execute_tool(name, **args)

        tool_call = ToolCall(
            tool_name=name,
            tool_input=args,
            tool_output=result[:2000],
            timestamp=datetime.now().isoformat()
        )

        step = ReActStep(
            thought=thought if i == 0 else "",  # 只在第一步保留思考过程
            action=tool_call,
            observation=result
        )
        react_history.append(step.model_dump())
        observations.append(f"[{name}]: {result[:300]}")

    # 合并观察结果
    merged = "\n".join(observations)
    logger.info(f"[ReAct-Act] 批量完成 {len(tool_calls)} 个工具，总输出 {len(merged)} 字符")

    result = {
        "react_history": react_history,
        "_pending_thought": "",
        "_pending_tool_calls": [],
    }

    # 检测到 submit_task → 所有工具已执行完毕，标记任务完成
    if has_submit_task:
        logger.info("[ReAct-Act] 检测到 submit_task，标记子任务完成（显式退出）")
        result["react_finished"] = True

    return result


def judge_node(state: AgentState):
    """
    判断节点：根据当前状态决定下一步。
    - 任务完成 → 归档成果，退出
    - 达到最大轮数 → 中断等人
    - 否则 → 继续循环
    """
    plan_box = state.get("planning")
    exec_box = state.get("execution")

    round_num = state.get("react_round", 0)
    react_finished = state.get("react_finished", False)
    react_history = state.get("react_history", [])

    # 找当前任务
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    if not pending_tasks:
        return {}

    task = pending_tasks[0]

    if react_finished:
        # 任务完成——从历史中提取所有文件产出
        logger.info(f"[ReAct-Judge] 子任务 {task.task_id} ReAct 完成")
        final_files = _extract_files(react_history)
        if final_files.get("files"):
            exec_box.stage_outputs[task.task_id] = final_files

        if state.get("force_submit", False):
            # ⚡ 强制提交：跳过沙盒验证，直接标记完成
            task.status = "finished"
            task.result = "【人工干预】验证阶段被跳过，直接进入整合。"
            logger.info(f"[ReAct-Judge] ⚡ 强制提交：子任务 {task.task_id} 跳过沙盒验证")
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
            "force_submit": False,  # 清除标志位，不影响后续任务
        }

    if round_num >= MAX_REACT_ROUNDS:
        # 达到上限——设置阻塞标志，向上传播到 main.py 处理人工介入
        logger.warning(f"[ReAct-Judge] 子任务 {task.task_id} 达到 {MAX_REACT_ROUNDS} 轮上限，暂停等待人工介入")
        task.result = f"ReAct 循环达上限（{MAX_REACT_ROUNDS} 轮），等待人工介入"

        # 尝试从历史中提取已有的代码（暂存，等人工介入后再决定状态）
        partial_files = _extract_files(react_history)
        if partial_files.get("files"):
            exec_box.stage_outputs[task.task_id] = partial_files

        # ⚠️ 保持 task.status = "pending"，不要改 testing
        # 否则上层路由检测不到需要处理的卡住状态

        block_reason = (
            f"子任务 {task.task_id}（{task.description}）在 {MAX_REACT_ROUNDS} 轮 "
            f"ReAct 循环后仍未完成。\n"
            f"最后思考：{react_history[-1].get('thought', 'N/A') if react_history else 'N/A'}\n"
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

    # 继续循环——什么都不做，回到 think
    logger.info(f"[ReAct-Judge] 继续循环，当前 {round_num}/{MAX_REACT_ROUNDS} 轮")
    return {}


def _extract_files(react_history: list) -> dict:
    """从 ReAct 历史中提取所有 write_file 调用，返回 {"files": {path: content}, "main": entry}"""
    if not react_history:
        return {}
    files = {}
    main = ""
    # 倒序遍历：后写的文件覆盖先写的（同一文件取最后一次内容）
    for step in reversed(react_history):
        action = None
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

        if tool_name != "write_file":
            continue

        path = params.get("path", "") if isinstance(params, dict) else ""
        content = params.get("content", "") if isinstance(params, dict) else ""
        if not path:
            continue
        # 不覆盖已记录的（倒序遍历，先遇到的即最后一次写入）
        if path not in files:
            files[path] = content
            if not main:
                main = path
    return {"files": files, "main": main}


# ==========================================================================
# 路由函数
# ==========================================================================

def route_after_think(state: AgentState) -> str:
    """think 之后直接进入 act"""
    return "act"


def route_after_act(state: AgentState) -> str:
    """act 之后进入 judge"""
    return "judge"


def route_after_judge(state: AgentState) -> str:
    """judge 决定：继续循环 / 退出 / 阻塞等人"""
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
    """
    构建 ReAct 执行子图。

    参数：
        checkpointer : SqliteSaver 实例，由上层传入，用于持久化检查点。

    返回值：
        已编译的 LangGraph 子图，可直接嵌入执行子图作为 worker 节点使用。

    与原有 worker_node 的兼容性：
        - 原有 worker_node 只负责生成代码
        - ReAct worker 子图负责 思考→工具调用→观察 的完整循环
        - 两者对外接口一致：输入 AgentState，输出更新后的 AgentState
        - 子任务状态最终都标记为 "testing"，交给沙盒验证
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("start_react", start_react_node)
    workflow.add_node("think", think_node)
    workflow.add_node("act", act_node)
    workflow.add_node("judge", judge_node)

    workflow.set_entry_point("start_react")

    workflow.add_edge("start_react", "think")
    workflow.add_edge("think", "act")
    workflow.add_edge("act", "judge")

    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "think": "think",
            "end": END
        }
    )

    return workflow.compile(checkpointer=checkpointer)
