"""
=============================================================================
沙盒验证子图（Sandbox Subgraph）—— 代码安全执行与错误捕获
=============================================================================

【职责】
    把 Worker 生成的代码写进临时文件，用真实的 Python 解释器跑一遍，
    捕获 stdout/stderr，判断代码是否能正常运行。

【流程】
    1. 从 PlanningContext 中找第一个 status="testing" 的子任务
    2. 从 ExecutionContext.stage_outputs 中取出对应的代码
    3. 写入临时 .py 文件
    4. 用 subprocess 执行，捕获输出
    5. 根据返回码更新子任务状态：
       - returncode == 0 → 测试通过，设为 "finished"
       - returncode != 0 → 测试失败，设为 "pending" 并记录 error_trace
       - 执行超时 → 设为 "pending" 并记录超时错误

【与主图的协作】
    沙盒每次只测试一个子任务。主图的路由函数根据测试结果决定：
    - 通过 → 继续测试下一个 testing 任务，或全部完成结束
    - 失败 → 将工单打回给执行子图重新编码

【使用方式】
    from core.graph.sandbox import build_sandbox_subgraph

    sandbox = build_sandbox_subgraph()
    # workflow.add_node("sandbox", sandbox)
=============================================================================
"""

import sys
import os
import tempfile
import subprocess
from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.logger import logger


def sandbox_node(state: AgentState):
    """
    沙盒验证节点。

    从 PlanningContext 中找第一个 status="testing" 的任务，
    取出代码 → 写入临时文件 → Python 执行 → 分析结果 → 更新状态。

    返回值：
        更新后的 planning（任务状态变更）和 execution（错误记录）。
    """
    # 从状态中获取执行和规划上下文
    exec_box = state.get("execution")
    plan_box = state.get("planning")

    # 找第一个 waiting 测试的子任务
    target_task = None
    for t in plan_box.task_plan:
        if t.status == "testing":
            target_task = t
            break

    if target_task is None:
        # 没有要测试的任务，直接返回
        logger.debug("[沙盒验证] 没有待测试的子任务，跳过")
        return {"planning": plan_box, "execution": exec_box}

    task_id = target_task.task_id
    code = exec_box.stage_outputs.get(task_id, "")

    if not code:
        logger.error(f"[沙盒验证] 子任务 {task_id} 没有代码可测试")
        target_task.status = "pending"
        target_task.result = "无代码"
        exec_box.error_trace = "错误：子任务没有生成任何代码"
        return {"planning": plan_box, "execution": exec_box}

    logger.info(f"[沙盒验证] 正在测试子任务 {task_id}: {target_task.description}")

    # ==========================================
    # 写入临时文件
    # ==========================================
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_file = f.name

        logger.debug(f"已写入临时文件: {tmp_file}")

        # ==========================================
        # 用 Python 执行
        # ==========================================
        result = subprocess.run(
            [sys.executable, tmp_file],
            capture_output=True,
            text=True,
            timeout=30  # 30 秒超时，防止死循环
        )

        if result.returncode == 0:
            # 测试通过 🎉
            logger.info(f"子任务 {task_id} 测试通过")
            target_task.status = "finished"
            target_task.result = "沙盒测试通过"
            # 清除错误记录（之前可能有过报错，现在已经修好了）
            exec_box.error_trace = ""

            # 打印 stdout（截断防止刷屏）
            if result.stdout:
                stdout_preview = result.stdout[:500]
                logger.debug(f"stdout: {stdout_preview[:200]}")

        else:
            # 测试失败 ❌
            error_msg = result.stderr[:2000] if result.stderr else "无错误输出（可能是 exit code 非零）"
            logger.warning(f"子任务 {task_id} 测试失败")
            logger.debug(f"错误信息:\n{error_msg[:500]}")

            target_task.status = "pending"
            target_task.result = "沙盒测试未通过，等待修复"
            exec_box.error_trace = error_msg
            exec_box.retry_count += 1
            logger.info(f"已将子任务 {task_id} 打回重写（重试第 {exec_box.retry_count} 次）")

    except subprocess.TimeoutExpired:
        logger.warning(f"子任务 {task_id} 执行超时（>30秒）")
        target_task.status = "pending"
        target_task.result = "执行超时"
        exec_box.error_trace = "错误：代码执行超时（30秒），请检查是否有死循环"
        exec_box.retry_count += 1

    except Exception as e:
        logger.error(f"子任务 {task_id} 沙盒内部错误: {str(e)}")
        target_task.status = "pending"
        target_task.result = "沙盒异常"
        exec_box.error_trace = f"沙盒执行异常: {str(e)}"
        exec_box.retry_count += 1

    finally:
        # 清理临时文件
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
            logger.debug("已清理临时文件")

    return {"planning": plan_box, "execution": exec_box}


def build_sandbox_subgraph():
    """
    构建并编译沙盒验证子图。

    子图结构极简：一个 sandbox 节点，从开始直接走到结束。
    适合嵌入主图作为验证步骤。
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("sandbox", sandbox_node)

    workflow.set_entry_point("sandbox")
    workflow.add_edge("sandbox", END)

    return workflow.compile()
