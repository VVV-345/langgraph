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
from langgraph.types import interrupt
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.state import AgentState, ExecutionContext, ReActStep, ToolCall
from core.tools import ALL_TOOLS, execute_tool, get_tool_schemas
from core.logger import logger

load_dotenv()

MAX_REACT_ROUNDS = 7

# ReAct 专用 LLM（需要复杂推理，temperature 稍高）
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),
    temperature=0.3,
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

请按以下格式回复（严格遵循，不要输出其他内容）：

Thought: <你的思考过程——分析当前状态，决定下一步做什么>
Action: <工具名>(<参数1>=<值1>, <参数2>=<值2>, ...)

或者，如果任务已经完成：

Thought: <总结确认任务完成的思考>
Action: FINISH

可用的 Action 示例：
- Action: read_file(path="src/main.py")
- Action: write_file(path="src/main.py", content="print('hello')")
- Action: run_command(command="python src/main.py")
- Action: web_search(query="python argparse example")
- Action: FINISH"""


def _parse_react_response(response: str) -> tuple:
    """解析 LLM 的 ReAct 格式输出，返回 (thought, action_name, action_params)"""
    thought = ""
    action_name = ""
    action_params = {}

    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("thought:"):
            thought = line[len("thought:"):].strip()
        elif line.lower().startswith("action:"):
            action_text = line[len("action:"):].strip()

            if action_text.upper() == "FINISH":
                action_name = "FINISH"
                break

            # 解析 Action: tool_name(key=value, ...)
            paren_idx = action_text.find("(")
            if paren_idx == -1:
                action_name = action_text.strip()
                break

            action_name = action_text[:paren_idx].strip()
            params_str = action_text[paren_idx + 1:].rstrip(")")

            # 简单的 key=value 解析
            for part in _split_params(params_str):
                if "=" in part:
                    key, _, value = part.partition("=")
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    action_params[key] = value

            break

    if not action_name:
        action_name = "FINISH"
        if not thought:
            thought = "无法解析行动，默认结束"

    return thought, action_name, action_params


def _split_params(params_str: str) -> list:
    """按逗号分割参数字符串，但跳过引号内的逗号"""
    parts = []
    current = ""
    in_quote = False
    quote_char = None
    for ch in params_str:
        if ch in ('"', "'") and not in_quote:
            in_quote = True
            quote_char = ch
        elif ch == quote_char and in_quote:
            in_quote = False
            quote_char = None
        elif ch == "," and not in_quote:
            parts.append(current.strip())
            current = ""
            continue
        current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


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

    # 找到第一个 pending 任务
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]
    if not pending_tasks:
        logger.info("[ReAct入口] 没有待处理任务，跳过")
        exec_box.all_tasks_completed = True
        return {
            "execution": exec_box,
            "react_finished": True
        }

    task = pending_tasks[0]
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
        for t_id, code in exec_box.stage_outputs.items():
            previous_outputs += f"--- 子任务 {t_id} 的产出 ---\n{code[:3000]}\n\n"

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

    logger.info(f"[ReAct-Think] 第 {round_num}/{MAX_REACT_ROUNDS} 轮思考中...")

    response = llm.invoke([
        SystemMessage(content="你是一个严格遵循 ReAct 格式的编程助手。只输出 Thought: 和 Action:，不要加额外文字。"),
        HumanMessage(content=prompt)
    ]).content

    thought, action_name, action_params = _parse_react_response(response)

    logger.info(f"[ReAct-Think] 思考: {thought[:100]}")
    logger.info(f"[ReAct-Think] 行动: {action_name}({action_params})")

    return {
        "react_round": round_num,
        "_pending_thought": thought,
        "_pending_action_name": action_name,
        "_pending_action_params": action_params,
    }


def act_node(state: AgentState):
    """
    执行节点：执行 think 节点决定的工具调用，记录观察结果。
    """
    thought = state.get("_pending_thought", "")
    action_name = state.get("_pending_action_name", "FINISH")
    action_params = state.get("_pending_action_params", {})

    react_history = state.get("react_history", [])

    if action_name == "FINISH":
        # 任务完成，不需要执行工具
        step = ReActStep(
            thought=thought,
            action=ToolCall(tool_name="FINISH", tool_input={}, tool_output="任务完成"),
            observation="任务标记为完成"
        )
        react_history.append(step.model_dump())
        logger.info("[ReAct-Act] LLM 判定任务完成")
        return {
            "react_history": react_history,
            "react_finished": True,
            "_pending_thought": "",
            "_pending_action_name": "",
            "_pending_action_params": {},
        }

    # 执行工具
    logger.info(f"[ReAct-Act] 执行工具: {action_name}({action_params})")
    observation = execute_tool(action_name, **action_params)

    tool_call = ToolCall(
        tool_name=action_name,
        tool_input=action_params,
        tool_output=observation[:2000],
        timestamp=datetime.now().isoformat()
    )

    step = ReActStep(
        thought=thought,
        action=tool_call,
        observation=observation
    )
    react_history.append(step.model_dump())

    logger.info(f"[ReAct-Act] 观察结果: {observation[:150]}")

    return {
        "react_history": react_history,
        "_pending_thought": "",
        "_pending_action_name": "",
        "_pending_action_params": "",
    }


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
        # 任务完成——从历史中提取最终代码产出
        logger.info(f"[ReAct-Judge] 子任务 {task.task_id} ReAct 完成")
        final_code = _extract_final_code(react_history)
        if final_code:
            exec_box.stage_outputs[task.task_id] = final_code
        task.status = "testing"
        task.result = f"ReAct 循环完成（{round_num} 轮）"
        exec_box.current_task_index += 1
        return {
            "planning": plan_box,
            "execution": exec_box,
            "current_code": final_code,
            "react_finished": True,
            "react_blocked": False,
        }

    if round_num >= MAX_REACT_ROUNDS:
        # 达到上限——卡住，请求人工介入
        logger.warning(f"[ReAct-Judge] 子任务 {task.task_id} 达到 {MAX_REACT_ROUNDS} 轮上限，暂停等待人工介入")
        task.result = f"ReAct 循环达上限（{MAX_REACT_ROUNDS} 轮），等待人工介入"

        # 尝试从历史中提取已有的代码
        partial_code = _extract_final_code(react_history)
        if partial_code:
            exec_box.stage_outputs[task.task_id] = partial_code
            task.status = "testing"
        else:
            task.status = "pending"

        # 触发 LangGraph interrupt
        interrupt_msg = (
            f"子任务 {task.task_id}（{task.description}）在 {MAX_REACT_ROUNDS} 轮 "
            f"ReAct 循环后仍未完成。\n最后思考：{react_history[-1].get('thought', 'N/A') if react_history else 'N/A'}\n"
            f"请选择：继续执行 / 跳过此任务 / 修改需求"
        )
        user_decision = interrupt(interrupt_msg)

        # 用户选择处理
        if user_decision and "跳过" in str(user_decision):
            task.status = "failed"
            task.result = "用户跳过"
        elif user_decision and "修改" in str(user_decision):
            task.status = "pending"
            task.result = f"用户要求修改：{user_decision}"
        else:
            # 默认：继续执行（重置轮数）
            logger.info("[ReAct-Judge] 用户选择继续，重置轮数")
            return {
                "react_round": 0,
                "react_finished": False,
                "react_blocked": False,
            }

        return {
            "planning": plan_box,
            "execution": exec_box,
            "react_finished": True,
            "react_blocked": False,
        }

    # 继续循环——什么都不做，回到 think
    logger.info(f"[ReAct-Judge] 继续循环，当前 {round_num}/{MAX_REACT_ROUNDS} 轮")
    return {}


def _extract_final_code(react_history: list) -> str:
    """从 ReAct 历史中提取最终代码（最后一次 write_file 的内容）"""
    if not react_history:
        return ""
    # 从后往前找最后一次 write_file
    for step in reversed(react_history):
        if isinstance(step, dict):
            action = step.get("action", {})
            if isinstance(action, dict):
                if action.get("tool_name") == "write_file":
                    params = action.get("tool_input", {})
                    return params.get("content", "")
        elif hasattr(step, "action"):
            if step.action.tool_name == "write_file":
                return step.action.tool_input.get("content", "")
    return ""


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
    """judge 决定：继续循环 或 退出"""
    react_finished = state.get("react_finished", False)
    if react_finished:
        logger.info("[ReAct路由] 退出 ReAct 循环")
        return "end"
    logger.info("[ReAct路由] 继续下一轮")
    return "think"


# ==========================================================================
# 构建 ReAct 执行子图
# ==========================================================================

def build_react_worker_subgraph():
    """
    构建 ReAct 执行子图。

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

    return workflow.compile()
