"""
=============================================================================
复盘节点（Reviewer Node）—— 总结经验 → 写入向量库
=============================================================================

【定位】
    这是 8 阶段流水线的最后一站（复盘阶段）。
    每次流水线运行结束后，把本次执行的经验数字化存储到 Qdrant，
    供下一次运行的感知阶段检索参考，形成"越跑越聪明"的正反馈循环。

【存储策略】
    ✅ 正常完成（有 finished 任务）     → 存储（成功经验）
    ✅ 有 failed 任务 + 沙盒报错        → 存储（踩坑经验更宝贵）
    ❌ 执行结果为全空（可能是模型崩了） → 跳过（数据不可信）
    ❌ Embedding 模型不可用            → 跳过（无法生成向量）
    ❌ Qdrant 不可用                    → 降级跳过（不阻塞流水线）

【经验数据结构】（写入 Qdrant payload）
    {
        task_summary:      用户原始需求
        task_complexity:   simple / complex
        planning_strategy: {reasoning, task_plan}
        execution_result:  {success_rate, finished, failed, retry_total}
        pitfalls:          踩坑记录列表
        tools_used:        使用的工具列表
        tags:              自动提取的关键词标签
        timestamp:         执行时间
        session_id:        唯一运行 ID
    }

【使用方式】
    from core.nodes.reviewer import reviewer_node
    # 在主图中作为最后一个节点:
    # workflow.add_node("reviewer", reviewer_node)
    # workflow.add_edge("output", "reviewer")
    # workflow.add_edge("reviewer", END)
=============================================================================
"""

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from core.state import AgentState
from core.memory.embedding import create_embedder
from core.memory.store import ExperienceStore
from core.logger import logger

load_dotenv()

# 全局单例（避免每次进入节点都重新创建）
_embeddings = None
_store: ExperienceStore = None


def _get_store() -> ExperienceStore | None:
    """懒初始化向量存储（只连接一次）"""
    global _embeddings, _store

    if _store is not None:
        return _store

    # 检查是否配置了向量库
    if not os.getenv("EMBEDDING_MODEL_PATH") and not os.getenv("EMBEDDING_BASE_URL"):
        logger.info("[复盘] 未配置 Embedding 模型，跳过向量存储")
        return None

    try:
        _embeddings = create_embedder()
        _store = ExperienceStore(_embeddings)

        if not _store.connect():
            logger.warning("[复盘] Qdrant 连接失败，本次运行不会写入经验")
            return None

        return _store
    except Exception as e:
        logger.warning(f"[复盘] Embedding 模型不可用，跳过向量存储: {e}")
        return None


# ==========================================================================
# 数据收集
# ==========================================================================

def _collect_experience(state: AgentState) -> dict | None:
    """
    从 AgentState 中收集本次执行的结构化经验数据。

    返回 None 表示数据不可信（模型崩了），应跳过存储。
    """
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    messages = state.get("messages", [])

    if not plan_box or not exec_box:
        return None

    task_plan = plan_box.task_plan

    # ---- 提取用户需求 ----
    user_request = ""
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "human":
            user_request = msg.content if hasattr(msg, "content") else str(msg)
            break

    # ---- 统计执行结果 ----
    finished = sum(1 for t in task_plan if t.status == "finished")
    failed = sum(1 for t in task_plan if t.status == "failed")
    total = len(task_plan)
    success_rate = finished / total if total > 0 else 0.0

    # ---- 模型崩溃检测 ----
    # 条件：没有任何产出（finished=0 且 stage_outputs 为空）且有重试
    has_output = bool(exec_box.stage_outputs)
    if finished == 0 and not has_output:
        logger.info(
            "[复盘] 检测到可能是模型错误导致的空运行 "
            f"(finished=0, stage_outputs={len(exec_box.stage_outputs)})，跳过存储"
        )
        return None

    # ---- 拆解策略 ----
    reasoning = plan_box.thinking_chain if hasattr(plan_box, "thinking_chain") else []
    planning_strategy = {
        "reasoning": reasoning if isinstance(reasoning, list) else [reasoning],
        "task_plan": [
            {
                "id": t.task_id,
                "description": t.description,
                "objective": t.objective,
                "risk_level": t.risk_level,
                "risk_reason": t.risk_reason,
                "dependencies": t.dependencies,
                "final_status": t.status,
            }
            for t in task_plan
        ]
    }

    # ---- 执行结果 ----
    execution_result = {
        "success_rate": round(success_rate, 2),
        "finished": finished,
        "failed": failed,
        "total": total,
        "retry_total": exec_box.retry_count,
    }

    # ---- 踩坑记录 ----
    pitfalls = _extract_pitfalls(state, exec_box)

    # ---- 使用的工具 ----
    tools_used = _extract_tools_used(state)

    return {
        "task_summary": user_request if user_request else "未记录需求",
        "task_complexity": plan_box.task_complexity,
        "planning_strategy": planning_strategy,
        "execution_result": execution_result,
        "pitfalls": pitfalls,
        "tools_used": tools_used,
        "session_id": datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:6],
    }


def _extract_pitfalls(state: AgentState, exec_box) -> list:
    """
    从执行轨迹中提取踩坑记录。

    踩坑来源：
        1. 沙盒报错 → error_trace 不为空
        2. 子任务 failed → 从 task_plan 中获取失败原因
        3. ReAct 卡住 → react_blocked
    """
    pitfalls = []

    plan_box = state.get("planning")

    # 来源 1 & 2：子任务失败
    for t in (plan_box.task_plan if plan_box else []):
        if t.status == "failed":
            pitfall = {
                "task_id": t.task_id,
                "description": t.description,
                "error_snippet": "",
                "root_cause": t.result if t.result else "未知原因",
                "how_fixed": "",
            }

            # 如果有错误堆栈
            if exec_box.error_trace:
                # 提取关键行（前 3 行通常够定位问题）
                lines = exec_box.error_trace.strip().split("\n")
                pitfall["error_snippet"] = "\n".join(lines[:5])[:500]
                # 尝试提取异常类型
                for line in lines:
                    if "Error" in line or "error" in line:
                        pitfall["root_cause"] = line.strip()[:200]
                        break

            pitfalls.append(pitfall)

    # 来源 3：沙盒有报错但任务最终通过（修复经验也宝贵）
    if exec_box.error_trace and not pitfalls:
        # 所有任务都 finished，但曾经有过报错 → 记录修复经验
        finished_tasks = [t for t in (plan_box.task_plan if plan_box else []) if t.status == "finished"]
        if finished_tasks:
            lines = exec_box.error_trace.strip().split("\n")
            error_type = ""
            for line in lines:
                if "Error" in line or "error" in line:
                    error_type = line.strip()[:200]
                    break
            pitfalls.append({
                "task_id": "all",
                "description": "沙盒曾报错，已修复",
                "error_snippet": "\n".join(lines[:5])[:500],
                "root_cause": error_type or "沙盒验证失败",
                "how_fixed": f"经过 {exec_box.retry_count} 次重试修复",
            })

    # 来源 4：ReAct 卡住
    if state.get("react_blocked"):
        pitfalls.append({
            "task_id": "all",
            "description": "ReAct 循环卡住",
            "error_snippet": "",
            "root_cause": state.get("react_block_reason", "达到最大轮数"),
            "how_fixed": "人工介入或降级处理",
        })

    return pitfalls


def _extract_tools_used(state: AgentState) -> list:
    """从 ReAct 历史中提取使用的工具列表（去重）"""
    tools = set()
    react_history = state.get("react_history", [])
    for step in react_history:
        if isinstance(step, dict):
            tool_name = step.get("action", {}).get("tool_name", "")
        elif hasattr(step, "action"):
            tool_name = step.action.tool_name
        else:
            continue
        if tool_name and tool_name != "FINISH":
            tools.add(tool_name)
    return sorted(tools)


# ==========================================================================
# 主节点
# ==========================================================================

def reviewer_node(state: AgentState):
    """
    复盘节点：收集本次执行的经验数据，写入 Qdrant 向量库。

    这是流水线的最后一个节点，无论写入成功与否都不会阻断流水线。

    处理流程：
        1. 尝试连接 Qdrant（首次进入时初始化）
        2. 收集经验数据
        3. 调用 embedder 生成向量
        4. 写入 Qdrant
        5. 出错时降级处理（记录日志，继续返回）

    返回值：
        空 dict（不修改任何状态字段），或包含 review 信息的更新。
    """
    logger.info("[复盘] 开始复盘，收集执行经验...")

    # Step 1: 获取存储实例
    store = _get_store()
    if store is None:
        logger.info("[复盘] 向量存储不可用，跳过（降级正常）")
        return {"messages": []}  # 不修改状态

    # Step 2: 收集数据
    experience = _collect_experience(state)
    if experience is None:
        logger.info("[复盘] 数据不可信，跳过存储")
        return {"messages": []}

    # Step 3: 写入存储
    try:
        success = store.store_experience(
            task_summary=experience["task_summary"],
            task_complexity=experience["task_complexity"],
            planning_strategy=experience["planning_strategy"],
            execution_result=experience["execution_result"],
            pitfalls=experience["pitfalls"],
            tools_used=experience["tools_used"],
            session_id=experience["session_id"],
        )

        if success:
            logger.info(
                f"[复盘] ✅ 经验已归档 (session={experience['session_id']}, "
                f"成功率={experience['execution_result']['success_rate']:.0%}, "
                f"踩坑={len(experience['pitfalls'])} 条)"
            )
        else:
            logger.info("[复盘] 经验写入失败（降级正常）")

    except Exception as e:
        logger.warning(f"[复盘] 存储异常，跳过（非致命）: {e}")

    logger.info("[复盘] 复盘结束")

    # 复盘不修改任何核心状态，只旁观记录
    total_count = store.count() if store else 0
    return {
        "messages": [],
        "_review_summary": {
            "session_id": experience.get("session_id", ""),
            "success_rate": experience["execution_result"]["success_rate"],
            "pitfall_count": len(experience["pitfalls"]),
            "total_experiences_in_db": total_count,
        }
    }
