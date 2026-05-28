"""
=============================================================================
主图谱（Master Graph）—— 整个流水线的"总经理"
=============================================================================

【定位】
    主图谱是所有子图的编排者。它不直接干活，而是调度三个子图：
    1. 执行子图（executor）—— 负责感知+规划+编码，产出代码
    2. 沙盒子图（sandbox） —— 负责验证代码是否能跑通
    3. 路由逻辑（router）  —— 根据测试结果决定下一步

【主图拓扑】

    START
      ↓
    executor (执行子图：分析需求 → 拆任务 → 逐个编码)
      ↓
    [路由 A] ── 没有 testing 任务 ──→ END
      ↓ (有待测试的代码)
    sandbox (沙盒子图：临时文件 → Python 执行 → 捕获报错)
      ↓
    [路由 B] ── 有 pending（代码挂了）────→ executor (重入修复)
              ── 还有 testing（继续验证）──→ sandbox
              ── 全部 finished ──────────→ END

【重试循环】

    当沙盒发现 bug 时：
    1. 主图把子任务状态设为 "pending" + 记录 error_trace
    2. 路由 B 检测到 pending → 再次进入 executor
    3. executor 子图的入口路由检测到 task_plan 已存在
       → 跳过 analyzer，直接进入 worker
    4. worker 读取 error_trace，带着报错修复代码
    5. 修复后设为 "testing" → 回到沙盒验证
    6. 循环直到测试通过

【使用方式】
    from core.graph.main import build_master_graph

    app = build_master_graph()
    result = app.invoke(
        {"messages": [HumanMessage(content="你的需求")]},
        {"recursion_limit": 100}
    )
=============================================================================
"""

from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.graph import build_execution_subgraph
from core.graph.sandbox import build_sandbox_subgraph
from core.logger import logger


# ==========================================================================
# 路由函数
# ==========================================================================

def route_after_executor(state: AgentState) -> str:
    """
    executor 之后的分叉：

    executor 刚跑完一轮，所有子任务的状态应该是 "testing" 或 "finished"。
    - 有 "testing" 任务 → 送入沙盒验证
    - 无 "testing" 任务（全部 finished）→ 大结局
    """
    plan_box = state.get("planning")
    for t in plan_box.task_plan:
        if t.status == "testing":
            logger.info("[主图路由] 有待测试的代码，送入沙盒验证")
            return "sandbox"
    logger.info("[主图路由] 所有子任务已完成，流水线结束")
    return "end"


def route_after_sandbox(state: AgentState) -> str:
    """
    sandbox 之后的分叉——主图的核心决策点：

    沙盒刚测完一个子任务，结果可能是通过或失败：
    1. 还有 sub_task.status == "pending"（刚验证失败）
       → executor 重入修复
    2. 还有 sub_task.status == "testing"（等验证但还没轮到）
       → 继续沙盒验证
    3. 全部 finished
       → 结束
    """
    plan_box = state.get("planning")

    # 优先级 1：有失败的任务需要修复
    pending = [t for t in plan_box.task_plan if t.status == "pending"]
    if pending:
        logger.warning(f"[主图路由] 检测到 {len(pending)} 个子任务未通过，打回重写")
        return "executor"

    # 优先级 2：还有等待验证的任务
    testing = [t for t in plan_box.task_plan if t.status == "testing"]
    if testing:
        logger.info(f"[主图路由] 还有 {len(testing)} 个子任务待验证，继续测试")
        return "sandbox"

    # 全部完成
    logger.info("[主图路由] 所有子任务均已通过测试，流水线结束")
    return "end"


# ==========================================================================
# 构建主图谱
# ==========================================================================

def build_master_graph():
    """
    构建并编译完整的主图谱。

    这个图谱编排了"编码 → 验证 → 修复"的完整循环，
    是项目对外暴露的唯一入口。

    节点清单：
        - executor : 执行子图（感知+规划+编码循环）
        - sandbox  : 沙盒子图（代码安全执行验证）

    路由逻辑：
        - executor → sandbox 或 END
        - sandbox  → executor（修复）或 sandbox（继续验证）或 END
    """
    # 子图实例（每个都是编译好的 StateGraph）
    exec_subgraph = build_execution_subgraph()
    sandbox_subgraph = build_sandbox_subgraph()

    # ---- 构建主图 ----
    workflow = StateGraph(AgentState)

    workflow.add_node("executor", exec_subgraph)
    workflow.add_node("sandbox", sandbox_subgraph)

    workflow.set_entry_point("executor")

    # executor → sandbox 或 END
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "sandbox": "sandbox",
            "end": END
        }
    )

    # sandbox → executor / sandbox / END（核心决策点）
    workflow.add_conditional_edges(
        "sandbox",
        route_after_sandbox,
        {
            "executor": "executor",
            "sandbox": "sandbox",
            "end": END
        }
    )

    return workflow.compile()
