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

    - 需求不清晰（需用户补充）→ 退出主图，等待澄清
    - ReAct 阻塞（需人工介入）→ 直接退出主图，交 pipeline 处理
    - 有 "testing" 任务 → 送入沙盒验证
    - 无 "testing" 任务（全部 finished）→ 大结局
    """
    plan_box = state.get("planning")
    if plan_box and plan_box.need_clarification:
        logger.info("[主图路由] 需求不清晰，退出主图等待用户补充")
        return "end"
    if state.get("react_blocked", False):
        logger.info("[主图路由] ReAct 阻塞需人工介入，退出主图")
        return "end"
    for t in plan_box.task_plan:
        if t.status == "testing":
            logger.info("[主图路由] 有待测试的代码，送入沙盒验证")
            return "sandbox"
    logger.info("[主图路由] 所有子任务已完成，进入整合阶段")
    return "integrator"


def route_after_sandbox(state: AgentState) -> str:
    """
    sandbox 之后的分叉——主图的核心决策点：

    沙盒刚测完子任务，结果可能是通过或失败：
    0. react_blocked（沙盒失败直接人工介入）→ end
    1. 重试次数超限（≥max_retry_per_task/任务）→ 强制终止，标记失败
    2. testing → 优先清空验证队列，让沙盒连续测完
    3. pending（含 failed + not_started）→ 验证队列清空后统一打回 executor
    4. 全部 finished → 整合

    关键设计：testing 优先于 pending。
    如果 pending（失败待修复）优先，executor 可能因上游还在 testing 而空转一轮；
    testing 优先则先让沙盒把能过的都过了，解锁下游依赖，executor 有更多工作可做。
    """
    if state.get("react_blocked", False):
        logger.info("[主图路由] 沙盒验证失败，退出主图等待人工介入")
        return "end"

    plan_box = state.get("planning")
    exec_box = state.get("execution")

    # 优先级 0：按子任务独立重试上限——防止单个任务无限重试
    max_retry = exec_box.max_retry_per_task if exec_box else 3
    over_limit_tasks = [
        tid for tid, cnt in (exec_box.task_retry_count if exec_box else {}).items()
        if cnt >= max_retry
    ]
    if over_limit_tasks:
        logger.error(
            f"[主图路由] ⛔ 子任务 {over_limit_tasks} 重试次数已达上限（{max_retry} 次/任务），"
            f"全局总重试 {exec_box.retry_count} 次，强制终止这些任务"
        )
        for t in plan_box.task_plan:
            if t.task_id in over_limit_tasks and t.status in ("pending", "testing"):
                t.status = "failed"
                t.result = (
                    f"子任务 {t.task_id} 重试超限（{exec_box.task_retry_count[t.task_id]} 次）\n"
                    f"最后报错：{exec_box.error_trace[:500] if exec_box.error_trace else '无'}"
                )
        # 检查是否还有其他待处理任务
        remaining = [t for t in plan_box.task_plan if t.status in ("pending", "testing")]
        if not remaining:
            logger.info("[主图路由] 所有待处理任务均已处理，转入整合")
            return "integrator"
        logger.info(f"[主图路由] 仍有 {len(remaining)} 个任务待处理，继续执行")
        # 非超限任务继续进入下一优先级判断
        non_over_limit = [t for t in remaining if t.task_id not in over_limit_tasks]
        if non_over_limit:
            # 继续往下走 testing/pending 判断（不直接 return executor）
            pass
        else:
            # 只剩下超限 testing → 标记失败后结束
            for t in [t for t in remaining if t.task_id in over_limit_tasks]:
                t.status = "failed"
                t.result = (
                    f"子任务 {t.task_id} 重试超限（{exec_box.task_retry_count.get(t.task_id, 0)} 次）\n"
                    f"最后报错：{exec_box.error_trace[:500] if exec_box.error_trace else '无'}"
                )
            return "integrator"

    # 🌟 优先级 1：先清空验证队列！只要还有待验证的任务，就让沙盒继续咬牙测完
    testing = [t for t in plan_box.task_plan if t.status == "testing"]
    if testing:
        logger.info(f"[主图路由] 还有 {len(testing)} 个子任务待验证，保持沙盒队列连续测试...")
        return "sandbox"

    # 🌟 优先级 2：队列清空后，一揽子盘点有没有失败的任务，统一打回车间集中重修
    pending = [t for t in plan_box.task_plan if t.status == "pending"]
    if pending:
        logger.warning(f"[主图路由] 沙盒队列测完，检测到 {len(pending)} 个子任务未通过，集中打回重写")
        return "executor"

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
    workflow.add_node("output_writer", output_node)
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

    # sandbox → executor / sandbox / integrator / end（测试失败人工介入）
    workflow.add_conditional_edges(
        "sandbox",
        route_after_sandbox,
        {
            "executor": "executor",
            "sandbox": "sandbox",
            "integrator": "integrator",
            "end": END
        }
    )

    # integrator → executor（修复）或 output
    workflow.add_conditional_edges(
        "integrator",
        route_after_integrator,
        {
            "executor": "executor",
            "output": "output_writer"
        }
    )

    # output → reviewer（总是执行，复盘内部处理降级）
    workflow.add_edge("output_writer", "reviewer")

    # reviewer → END
    workflow.add_edge("reviewer", END)

    return workflow.compile(checkpointer=checkpointer)
