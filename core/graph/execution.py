"""
=============================================================================
执行子图（Execution Subgraph）—— 感知 + 规划 + 调度 + ReAct 执行循环
=============================================================================

【定位】
    这是整个流水线的"内环"，负责把用户需求转化为可验证的代码。
    它不关心代码跑不跑得通——那是外面主图（沙盒验证）的事。

【可重入设计】
    这个子图可以被主图反复调用：
    - 第一次进入：analyzer 先做需求分析 → scheduler 调度拆解 → ReAct worker 逐个执行
    - 再次进入（被主图触发重试）：跳过 analyzer 和 scheduler，ReAct worker 带着报错修复

    判断依据：PlanningContext.task_plan 是否已有数据。

【子图拓扑】

    START
      ↓
    router_start ──(task_plan 不为空)──→ react_worker
      ↓ (task_plan 为空)
    analyzer ──(需求模糊)──→ END
      ↓ (需求清晰)
    scheduler (拓扑排序 + 依赖校验 + 资源确认)
      ↓
    install_deps (预安装 Python 第三方库 + import 校验)
      ↓
    react_worker (ReAct 循环：think → act → judge，最多 7 轮)
      ↓ (还有 pending)──→ react_worker (循环处理下一个子任务)
      ↓ (全在 testing/finished)──→ END

【ReAct Worker vs 旧 Worker】
    旧 worker_node: 单次 LLM 调用生成代码，无工具，无自检
    ReAct worker:   think→act→observe 循环，4 个真实工具，最多 7 轮自纠错

【使用方式】
    from core.graph.execution import build_execution_subgraph

    subgraph = build_execution_subgraph()
    # 在主图中作为一个节点使用:
    # workflow.add_node("executor", subgraph)
=============================================================================
"""

from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.nodes.analyzer import analyzer_node
from core.nodes.scheduler import scheduler_node, install_deps_node
from core.nodes.react_worker import build_react_worker_subgraph
from core.nodes.worker import worker_node
from core.logger import logger

USE_REACT_WORKER = True


# ==========================================================================
# 子图入口路由：跳过已完成的规划阶段
# ==========================================================================

def route_entry(state: AgentState) -> str:
    """
    子图入口路由。

    如果 planning.task_plan 已经有数据（说明之前已经分析过），
    直接跳转到 worker，跳过 analyzer 和 scheduler。

    这个设计使子图可以被主图安全地多次调用（每次修复一个 bug 后重新进入）。
    """
    planning = state.get("planning")
    if planning and planning.task_plan:
        logger.info("[执行子图] 任务规划已存在，跳过分析直接进入编码")
        return "worker"
    logger.info("[执行子图] 首次进入，启动需求分析")
    return "analyzer"


# ==========================================================================
# analyzer 之后的路由
# ==========================================================================

def route_after_analyzer(state: AgentState) -> str:
    """
    analyzer 之后的分叉：
    - 需求模糊 → END（子图退出，等待用户补充）
    - 需求清晰 → scheduler（拓扑排序 + 依赖校验）
    """
    planning = state.get("planning")
    if planning is None or planning.need_clarification:
        if planning and planning.need_clarification:
            logger.info("[执行子图] 需求不清晰，退出子图等待用户补充")
        return "end"
    logger.info("[执行子图] 需求清晰，进入调度中枢")
    return "scheduler"


# ==========================================================================
# worker 之后的路由
# ==========================================================================

def route_after_worker(state: AgentState) -> str:
    """
    worker 之后的分叉——依赖感知调度：

    核心逻辑：
    - ReAct 阻塞 → 退出子图
    - 所有任务已处理 → 退出子图
    - 有 pending 任务，但下一个就绪任务的依赖尚未被沙盒验证（status=testing）
      → 退出子图，先让沙盒验证上游，避免基于未验证代码继续编码
    - 下一个就绪任务的所有依赖都是 finished（或没有依赖）
      → 继续循环编码（批量处理无依赖/已验证任务）

    简言之：有依赖链的任务逐个串行（编码→验证→下一个），
    无依赖关系的任务保持批量并行。
    """
    if state.get("react_blocked", False):
        logger.info("[执行子图] ReAct 阻塞，退出子图等待人工介入")
        return "end"

    plan_box = state.get("planning")
    pending = [t for t in plan_box.task_plan if t.status == "pending"]
    if not pending:
        logger.info("[执行子图] 全部子任务编码完毕，交给主图验证")
        return "end"

    # 找出下一个能满足依赖的就绪任务
    finished_ids = {t.task_id for t in plan_box.task_plan if t.status == "finished"}
    testing_ids = {t.task_id for t in plan_box.task_plan if t.status == "testing"}

    for task in pending:
        deps_ok = not task.dependencies or all(d in finished_ids for d in task.dependencies)
        if not deps_ok:
            continue  # 依赖未满足，跳过

        # 此任务可以编码——检查它的依赖中是否有 testing 的
        depends_on_testing = any(d in testing_ids for d in task.dependencies)
        if depends_on_testing:
            logger.info(
                f"[执行子图] 子任务 {task.task_id} 依赖 {task.dependencies}，"
                f"其中 {[d for d in task.dependencies if d in testing_ids]} 尚未验证，"
                f"退出子图等待沙盒先验证上游"
            )
            return "end"

        # 无依赖或依赖已全部验证 → 继续编码
        logger.info(f"[执行子图] 还有 {len(pending)} 个子任务待编码，继续...")
        return "worker"

    # 所有 pending 的依赖都未满足（等 testing 变 finished）
    logger.info("[执行子图] 所有待编码任务依赖未满足，退出等待沙盒验证")
    return "end"


# ==========================================================================
# scheduler 之后的路由：简单任务走快速通道
# ==========================================================================

def route_after_scheduler(state: AgentState) -> str:
    """
    scheduler 之后的分叉：
    - 简单任务 → simple_worker（单次 LLM 代码生成，跳过 ReAct 循环）
    - 复杂任务 → worker（ReAct 循环，带工具调用）
    """
    planning = state.get("planning")
    if planning and planning.task_complexity == "simple":
        logger.info("[执行子图] 简单任务，使用单次代码生成（跳过 ReAct 循环）")
        return "simple_worker"
    logger.info("[执行子图] 复杂任务，使用 ReAct Worker")
    return "worker"


# ==========================================================================
# 构建执行子图
# ==========================================================================

def build_execution_subgraph(checkpointer=None):
    """
    构建并编译执行子图。

    参数：
        checkpointer : SqliteSaver 实例，由上层传入，继续传给 ReAct worker 子图。

    返回值：
        已编译的 LangGraph 可运行对象，可作为节点嵌入主图。

    节点清单：
        - router_start : 入口路由（跳过 analyzer / 走 analyzer）
        - analyzer     : 感知+规划节点
        - scheduler    : 调度节点（拓扑排序 + 依赖校验 + 资源确认）
        - worker       : ReAct 执行子图（think→act→judge 循环，带工具调用）

    使用场景：
        作为主图编排的"内环"：生成代码 → 交给沙盒验证 → 失败则重入修复。
    """
    workflow = StateGraph(AgentState)

    # ---- 注册节点 ----
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("scheduler", scheduler_node)

    # 开启 ReAct Worker 子图（think→act→judge 循环，带工具调用）
    if USE_REACT_WORKER:
        react_subgraph = build_react_worker_subgraph(checkpointer=checkpointer)
        workflow.add_node("worker", react_subgraph)
        logger.info("[执行子图] 使用 ReAct Worker（think→act→observe 循环）")
    else:
        workflow.add_node("worker", worker_node)
        logger.info("[执行子图] 使用旧版 Worker（单次 LLM 代码生成）")

    # 简单任务快速通道（单次代码生成，无 ReAct 循环）
    workflow.add_node("simple_worker", worker_node)

    # ---- 入口路由 ----
    workflow.set_conditional_entry_point(
        route_entry,
        {
            "analyzer": "analyzer",
            "worker": "worker"
        }
    )

    # ---- analyzer 后分叉 ----
    workflow.add_conditional_edges(
        "analyzer",
        route_after_analyzer,
        {
            "scheduler": "scheduler",
            "end": END
        }
    )

    # ---- 环境准备：预安装依赖库 ----
    workflow.add_node("install_deps", install_deps_node)

    # ---- scheduler → install_deps ----
    workflow.add_edge("scheduler", "install_deps")

    # ---- install_deps → 按复杂度分叉进入 worker ----
    workflow.add_conditional_edges(
        "install_deps",
        route_after_scheduler,
        {
            "simple_worker": "simple_worker",
            "worker": "worker"
        }
    )

    # ---- worker 后分叉（循环核心） ----
    workflow.add_conditional_edges(
        "worker",
        route_after_worker,
        {
            "worker": "worker",
            "end": END
        }
    )

    # ---- simple_worker 后分叉（同 worker，支持重试时进 ReAct） ----
    workflow.add_conditional_edges(
        "simple_worker",
        route_after_worker,
        {
            "worker": "worker",
            "end": END
        }
    )

    return workflow.compile(checkpointer=checkpointer)
