"""
=============================================================================
主图谱（Master Graph）—— 整个流水线的"总经理"
=============================================================================

【定位】
    主图谱是所有子图的编排者。它不直接干活，而是调度：
    1. 执行子图（executor） —— 负责感知+规划+编码，产出代码
    2. 沙盒子图（sandbox）  —— 负责验证代码是否能跑通
    3. 整合节点（integrator）—— 汇总所有子任务结果，组装为结构化交付物
    4. 输出节点（output）    —— 写入磁盘 + 生成 README + 交付物清单

【主图拓扑】（8 阶段完整流水线 + 向量记忆闭环）

    START
      ↓
    executor (感知[RAG检索] → 规划 → 调度 → ReAct 编码)
      ↓
    [路由 A] ── 没有 testing 任务 ──→ integrator
      ↓ (有待测试的代码)
    sandbox (沙盒子图：临时文件 → Python 执行 → 捕获报错)
      ↓
    [路由 B] ── 有 pending（代码挂了）───────→ executor (重入修复)
              ── 还有 testing（继续验证）─────→ sandbox
              ── 全部 finished / retry ≥ 5 ──→ integrator
      ↓
    integrator (整合：组装文件 + 冲突检测 + LLM 一致性审核)
      ↓
    [路由 C] ── 需要重新生成 ──→ executor
              ── 整合完成 ────→ output
      ↓
    output (输出：写入磁盘 + 生成 README + 交付物清单)
      ↓
    reviewer (复盘：收集经验 → 写入 Qdrant 向量库 ──→ 下次感知阶段检索)
      ↓
    END

【重试循环】

    当沙盒发现 bug 时：
    1. 主图把子任务状态设为 "pending" + 记录 error_trace
    2. 路由 B 检测到 pending → 再次进入 executor
    3. executor 子图的入口路由检测到 task_plan 已存在
       → 跳过 analyzer，直接进入 worker
    4. worker 读取 error_trace，带着报错修复代码
    5. 修复后设为 "testing" → 回到沙盒验证
    6. 循环直到测试通过

【整合→重试】
    当 LLM 审核发现跨文件接口不匹配时，integrator 会把问题子任务打回 pending，
    路由 C 检测到后重新进入 executor 修复。

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
from core.nodes.integrator import integrator_node
from core.nodes.output import output_node
from core.nodes.reviewer import reviewer_node
from core.logger import logger


# ==========================================================================
# 路由函数
# ==========================================================================

def route_after_executor(state: AgentState) -> str:
    """
    executor 之后的分叉：

    - ReAct 阻塞（需人工介入）→ 直接退出主图，交 main.py 处理
    - 有 "testing" 任务 → 送入沙盒验证
    - 无 "testing" 任务（全部 finished）→ 大结局
    """
    if state.get("react_blocked", False):
        logger.info("[主图路由] ReAct 阻塞需人工介入，退出主图")
        return "end"
    plan_box = state.get("planning")
    for t in plan_box.task_plan:
        if t.status == "testing":
            logger.info("[主图路由] 有待测试的代码，送入沙盒验证")
            return "sandbox"
    logger.info("[主图路由] 所有子任务已完成，进入整合阶段")
    return "integrator"


def route_after_sandbox(state: AgentState) -> str:
    """
    sandbox 之后的分叉——主图的核心决策点：

    沙盒刚测完一个子任务，结果可能是通过或失败：
    1. 重试次数超限（≥3次）→ 强制终止，标记失败
    2. 还有 sub_task.status == "pending"（刚验证失败）
       → executor 重入修复
    3. 还有 sub_task.status == "testing"（等验证但还没轮到）
       → 继续沙盒验证
    4. 全部 finished
       → 结束
    """
    plan_box = state.get("planning")
    exec_box = state.get("execution")

    # 优先级 0：重试次数上限——防止任何形式的无限循环
    if exec_box and exec_box.retry_count >= 3:
        logger.error(
            f"[主图路由] ⛔ 重试次数已达上限（{exec_box.retry_count} 次），"
            f"强制终止未完成的任务"
        )
        for t in plan_box.task_plan:
            if t.status in ("pending", "testing"):
                t.status = "failed"
                t.result = (
                    f"重试超限（{exec_box.retry_count} 次）\n"
                    f"最后报错：{exec_box.error_trace[:500] if exec_box.error_trace else '无'}"
                )
        logger.info("[主图路由] 重试超限，转入整合（保留已完成的任务）")
        return "integrator"

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
    logger.info("[主图路由] 所有子任务均已通过测试，进入整合阶段")
    return "integrator"


# ==========================================================================
# 整合阶段之后的路由
# ==========================================================================

def route_after_integrator(state: AgentState) -> str:
    """
    integrator 之后的分叉：

    整合完成后检查：
    1. integration 未完成（LLM 要求重新生成某些文件）
       → 回到 executor 修复
    2. 整合完成 → output
    """
    integration = state.get("integration")
    plan_box = state.get("planning")

    if integration is None or integration.integration_done:
        logger.info("[主图路由] 整合完成，进入输出阶段")
        return "output"

    # 检查是否有被 integrator 打回 pending 的任务
    if plan_box:
        pending = [t for t in plan_box.task_plan if t.status == "pending"]
        if pending:
            logger.warning(
                f"[主图路由] 整合审核发现 {len(pending)} 个任务需要修复，打回 executor"
            )
            return "executor"

    # 兜底：整合未完成但没有 pending → 强制标记完成并进入输出
    logger.info("[主图路由] 整合未完成但无待修复任务，进入输出（兜底）")
    if integration:
        integration.integration_done = True
    return "output"


# ==========================================================================
# 构建主图谱
# ==========================================================================

def build_master_graph(checkpointer=None):
    """
    构建并编译完整的主图谱（8 阶段流水线）。

    这个图谱编排了"编码 → 验证 → 修复 → 整合 → 输出"的完整循环，
    是项目对外暴露的唯一入口。

    参数：
        checkpointer : SqliteSaver 实例，由调用方（main.py）创建并传入。
                       所有子图共享同一个实例，避免重复创建 SQLite 连接。

    节点清单：
        - executor   : 执行子图（感知[RAG] + 规划 + 编码循环）
        - sandbox    : 沙盒子图（代码安全执行验证）
        - integrator : 整合节点（组装文件 + LLM 一致性审核）
        - output     : 输出节点（写入磁盘 + README + 清单）
        - reviewer   : 复盘节点（收集经验 → 写入 Qdrant）

    路由逻辑：
        - executor   → sandbox 或 integrator
        - sandbox    → executor（修复）或 sandbox（继续验证）或 integrator
        - integrator → executor（重新生成）或 output
        - output     → reviewer（直接边，总是执行）
        - reviewer   → END
    """
    # 子图实例（每个都是编译好的 StateGraph）
    exec_subgraph = build_execution_subgraph(checkpointer=checkpointer)
    sandbox_subgraph = build_sandbox_subgraph()

    # ---- 构建主图 ----
    workflow = StateGraph(AgentState)

    workflow.add_node("executor", exec_subgraph)
    workflow.add_node("sandbox", sandbox_subgraph)
    workflow.add_node("integrator", integrator_node)
    workflow.add_node("output", output_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("executor")

    # executor → sandbox / integrator / end（阻塞时直接退出）
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "sandbox": "sandbox",
            "integrator": "integrator",
            "end": END
        }
    )

    # sandbox → executor / sandbox / integrator（核心决策点）
    workflow.add_conditional_edges(
        "sandbox",
        route_after_sandbox,
        {
            "executor": "executor",
            "sandbox": "sandbox",
            "integrator": "integrator"
        }
    )

    # integrator → executor（修复）或 output
    workflow.add_conditional_edges(
        "integrator",
        route_after_integrator,
        {
            "executor": "executor",
            "output": "output"
        }
    )

    # output → reviewer（总是执行，复盘内部处理降级）
    workflow.add_edge("output", "reviewer")

    # reviewer → END
    workflow.add_edge("reviewer", END)

    return workflow.compile(checkpointer=checkpointer)
