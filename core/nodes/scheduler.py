"""
=============================================================================
调度节点 —— 拓扑排序 + 依赖管理 + 资源校验
=============================================================================

【定位】
    插入在 analyzer 和 worker 之间，负责将 LLM 产出的一堆子任务
    变为正确执行顺序的流水线。

【三个职责】
    1. 拓扑排序：根据 SubTask.dependencies 做 Kahn 算法排序
    2. 依赖就绪检查：每次选任务时检查其依赖是否全部完成
    3. 资源校验：对照 required_resources 和已注册工具，缺资源时告警

【使用方式】
    from core.nodes.scheduler import scheduler_node, get_next_ready_task

    # 作为节点一次调用
    state = scheduler_node(state)

    # 后续每次选任务时调用
    next_task = get_next_ready_task(state)
=============================================================================
"""

from collections import deque
from typing import Optional, List, Tuple

from core.state import AgentState, SubTask
from core.tools import TOOL_BY_NAME
from core.tools.docker_sandbox import docker_exec
from core.logger import logger


# ==========================================================================
# 拓扑排序（Kahn 算法）
# ==========================================================================

def topological_sort(tasks: List[SubTask]) -> Tuple[List[SubTask], bool]:
    """
    对子任务做 Kahn 拓扑排序。

    参数:
        tasks: 原始子任务列表

    返回:
        (sorted_tasks, has_cycle)
        - sorted_tasks: 排序后的子任务列表
        - has_cycle: True 表示检测到循环依赖，此时 sorted_tasks 为原序
    """
    if not tasks:
        return tasks, False

    # 构建 task_id → Task 映射
    task_map = {t.task_id: t for t in tasks}

    # 构建入度表和邻接表
    in_degree = {t.task_id: 0 for t in tasks}
    adj = {t.task_id: [] for t in tasks}

    for task in tasks:
        for dep_id in task.dependencies:
            if dep_id in task_map:
                adj[dep_id].append(task.task_id)
                in_degree[task.task_id] += 1
            else:
                logger.warning(
                    f"[调度] 子任务 {task.task_id} 依赖了不存在的任务 {dep_id}，已忽略"
                )

    # Kahn 算法
    queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
    sorted_ids = []

    while queue:
        tid = queue.popleft()
        sorted_ids.append(tid)
        for neighbor in adj[tid]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    has_cycle = len(sorted_ids) != len(tasks)

    if has_cycle:
        logger.warning(
            f"[调度] 检测到循环依赖！已排序 {len(sorted_ids)}/{len(tasks)} 个任务，"
            f"无法排序的任务 ID: {set(t.task_id for t in tasks) - set(sorted_ids)}"
        )
        return tasks, True

    # 按排序顺序重排
    sorted_tasks = [task_map[tid] for tid in sorted_ids]
    return sorted_tasks, False


# ==========================================================================
# 依赖就绪检查
# ==========================================================================

def get_next_ready_task(state: AgentState) -> Optional[SubTask]:
    """
    找到下一个「就绪」的子任务：
    条件1: status == "pending"
    条件2: 所有 dependencies 对应的任务都是 "finished"

    参数:
        state: 当前 AgentState

    返回:
        就绪的子任务，没有则返回 None
    """
    plan_box = state.get("planning")
    if not plan_box or not plan_box.task_plan:
        return None

    for task in plan_box.task_plan:
        if task.status != "pending":
            continue

        # 检查所有依赖是否已完成
        deps_ready = True
        for dep_id in task.dependencies:
            dep_task = _find_task_by_id(plan_box.task_plan, dep_id)
            if dep_task is None or dep_task.status != "finished":
                deps_ready = False
                break

        if deps_ready:
            return task

    return None


def _find_task_by_id(task_plan: List[SubTask], task_id: int) -> Optional[SubTask]:
    """在 task_plan 中按 task_id 查找"""
    for t in task_plan:
        if t.task_id == task_id:
            return t
    return None


# ==========================================================================
# 资源校验
# ==========================================================================

def check_resources(required: List[str]) -> List[str]:
    """
    检查所需资源/工具是否已注册。

    参数:
        required: 需要的工具名列表，如 ["write_file", "run_command"]

    返回:
        缺失的工具名列表（空列表 = 全部就绪）
    """
    if not required:
        return []

    available = set(TOOL_BY_NAME.keys())
    missing = [r for r in required if r not in available]

    if missing:
        logger.warning(f"[调度] ⚠️ 缺少以下工具/资源: {missing}，可用: {list(available)}")

    return missing


# ==========================================================================
# 调度节点（LangGraph 节点函数）
# ==========================================================================

def scheduler_node(state: AgentState):
    """
    【调度阶段】调度指挥官

    职责：
    1. 对 task_plan 做拓扑排序，确保依赖顺序正确
    2. 校验 required_resources 是否就绪
    3. 打印调度摘要
    """
    logger.info("[调度中枢] 正在对子任务进行拓扑排序与资源校验...")

    plan_box = state.get("planning")
    if not plan_box or not plan_box.task_plan:
        logger.info("[调度中枢] 无子任务需要调度，跳过")
        return {}

    tasks = plan_box.task_plan

    # 1. 拓扑排序
    sorted_tasks, has_cycle = topological_sort(tasks)

    if has_cycle:
        logger.warning("[调度中枢] 存在循环依赖，保持原始顺序继续执行")
    else:
        # 检查顺序是否变化
        old_order = [t.task_id for t in tasks]
        new_order = [t.task_id for t in sorted_tasks]
        if old_order != new_order:
            logger.info(f"[调度中枢] 拓扑重排: {old_order} → {new_order}")

    # 2. 资源校验
    missing = check_resources(plan_box.required_resources)

    # 3. 更新 task_plan
    plan_box.task_plan = sorted_tasks

    # 4. 打印调度摘要
    logger.info(f"[调度中枢] {'='*50}")
    logger.info(f"[调度中枢] 子任务总数: {len(sorted_tasks)}")
    logger.info(f"[调度中枢] 执行顺序: {' → '.join(f'[{t.task_id}]{t.description[:20]}' for t in sorted_tasks)}")
    logger.info(f"[调度中枢] 所需资源: {plan_box.required_resources or '无特殊需求'}")
    if missing:
        logger.warning(f"[调度中枢] ⚠️ 缺失资源: {missing}")
    else:
        logger.info("[调度中枢] ✅ 所有资源就绪")
    logger.info(f"[调度中枢] {'='*50}")

    return {"planning": plan_box}


# ==========================================================================
# 依赖库预安装节点
# ==========================================================================

def install_deps_node(state: AgentState):
    """
    【环境准备】在 Docker 容器中预安装任务所需的 Python 第三方库。

    在 ReAct 工作流启动前运行，避免 ReAct 轮次浪费在 pip install 上。
    安装完成后逐个 import 校验，通过/失败都会记录日志。

    状态更新：
        - execution.installed_libraries: 已成功安装并验证的库名列表
        - execution.missing_libraries: 安装或导入失败的库名列表
    """
    plan_box = state.get("planning")
    exec_box = state.get("execution")
    if exec_box is None:
        from core.state import ExecutionContext
        exec_box = ExecutionContext()

    libraries = plan_box.required_libraries if plan_box else []

    if not libraries:
        logger.info("[环境准备] 无第三方依赖库需要安装，跳过")
        return {}

    logger.info(f"[环境准备] 开始预安装 {len(libraries)} 个依赖库: {', '.join(libraries)}")

    installed = []
    missing = []

    for lib in libraries:
        lib = lib.strip()
        if not lib:
            continue

        # —— 步骤 1：pip install ——
        logger.info(f"[环境准备] pip install {lib} ...")
        result = docker_exec(f"pip install {lib}", timeout=120)

        if result["returncode"] != 0:
            err = result.get("stderr", "")[:150]
            logger.warning(f"[环境准备] ❌ {lib} 安装失败: {err}")
            missing.append(lib)
            continue

        # —— 步骤 2：import 校验 ——
        # 最简单的检查：能 import 就 OK，不追求打印版本号（避免 f-string 嵌套转义问题）
        import_name = lib.replace("-", "_")
        verify_cmd = f"python3 -c 'import {import_name}' 2>&1 || python -c 'import {import_name}' 2>&1"
        verify = docker_exec(verify_cmd, timeout=15)

        if verify["returncode"] == 0:
            logger.info(f"[环境准备] ✅ {lib} 已就绪 (import {import_name} OK)")
            installed.append(lib)
        else:
            err_detail = verify.get("stdout", "") or verify.get("stderr", "")
            logger.warning(
                f"[环境准备] ⚠️ {lib} 安装完成但 import {import_name} 失败: "
                f"{err_detail[:200]}"
            )
            installed.append(lib)  # 仍标记为已安装，ReAct 可以用自己的方式 import

    exec_box.installed_libraries = installed
    exec_box.missing_libraries = missing

    logger.info(
        f"[环境准备] 完成: 已就绪 {len(installed)} 个, "
        f"失败 {len(missing)} 个"
    )

    return {"execution": exec_box}
