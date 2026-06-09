"""
=============================================================================
沙盒验证子图（Sandbox Subgraph）—— 在 Docker 容器内执行代码验证
=============================================================================

【职责】
    把 Worker 生成的代码写入容器共享工作区，在容器内执行，验证代码是否能运行。

【流程】
    1. 从 PlanningContext 中找第一个 status="testing" 的子任务
    2. 从 ExecutionContext.stage_outputs 中取出文件产出
    3. 写入工作区的 task_<id>/ 子目录
    4. 在容器内执行主文件（CLI 直接跑，Web 服务后台启动+curl测试+kill）
    5. 根据返回码更新子任务状态

【与主图的协作】
    沙盒每次只测试一个子任务。主图的路由函数根据测试结果决定：
    - 通过 → 继续测试下一个 testing 任务，或全部完成结束
    - 失败 → 将工单打回给执行子图重新编码
=============================================================================
"""

import os
import time
from langgraph.graph import StateGraph, END
from core.state import AgentState
from core.tools.docker_sandbox import (
    docker_exec, docker_exec_background, is_web_service,
    map_to_host, WORKSPACE_HOST, WORKSPACE_CONTAINER,
)
from core.logger import logger


def sandbox_node(state: AgentState):
    """
    沙盒验证节点 —— 在 Docker 容器内执行代码。

    1. 找到第一个 status="testing" 的任务
    2. 解析 stage_outputs 中的文件产出
    3. 写入工作区 task_<id>/ 子目录
    4. 容器内执行 → CLI 直接跑，Web 后台+curl+kill
    5. 更新任务状态
    """
    exec_box = state.get("execution")
    plan_box = state.get("planning")

    # 找第一个等待测试的子任务
    target_task = None
    for t in plan_box.task_plan:
        if t.status == "testing":
            target_task = t
            break

    if target_task is None:
        logger.debug("[沙盒验证] 没有待测试的子任务，跳过")
        return {"planning": plan_box, "execution": exec_box}

    task_id = target_task.task_id
    task_output = exec_box.stage_outputs.get(task_id, "")

    # 解析新旧格式 → 统一的 files + main
    if isinstance(task_output, dict):
        files = task_output.get("files", {})
        main_file = task_output.get("main", "")
        if not main_file and len(files) == 1:
            main_file = list(files.keys())[0]
    elif isinstance(task_output, str) and task_output.strip():
        files = {"main.py": task_output}
        main_file = "main.py"
    else:
        files = {}
        main_file = ""

    if not files:
        logger.info(f"[沙盒验证] 子任务 {task_id} 无代码产出（非代码任务），直接标记完成")
        target_task.status = "finished"
        target_task.result = "无代码产出（非代码任务，已跳过沙盒验证）"
        exec_box.error_trace = ""
        return {"planning": plan_box, "execution": exec_box}

    file_count = len(files)
    web = is_web_service(files)
    web_tag = " [Web服务]" if web else ""
    logger.info(f"[沙盒验证] 正在测试子任务 {task_id}: {target_task.description}（{file_count} 个文件）{web_tag}")

    # ==========================================
    # 写入工作区的 task_<id>/ 子目录
    # ==========================================
    task_dir = f"task_{task_id}"
    container_task_dir = f"{WORKSPACE_CONTAINER}/{task_dir}"

    for rel_path, content in files.items():
        host_path = map_to_host(f"{container_task_dir}/{rel_path}")
        os.makedirs(os.path.dirname(host_path), exist_ok=True)
        with open(host_path, "w", encoding="utf-8") as f:
            f.write(content)

    logger.debug(f"[沙盒验证] 已写入: {WORKSPACE_HOST}/{task_dir}/")

    # ==========================================
    # 容器内执行
    # ==========================================
    try:
        if web:
            # ── Web 服务：后台启动 → 等待 → curl 测试 → kill ──
            _test_web_service(task_id, target_task, exec_box,
                              container_task_dir, main_file, files)
        else:
            # ── CLI 程序：直接执行 ──
            _test_cli(task_id, target_task, exec_box,
                      container_task_dir, main_file)

    except Exception as e:
        logger.error(f"[沙盒验证] 子任务 {task_id} 执行异常: {e}")
        target_task.status = "pending"
        target_task.result = "沙盒执行异常"
        exec_box.error_trace = str(e)
        exec_box.retry_count += 1
        exec_box.task_retry_count[task_id] = exec_box.task_retry_count.get(task_id, 0) + 1

    # 沙盒测试失败 → 直接触发人工介入，不再自动回 executor 重试
    result = {"planning": plan_box, "execution": exec_box}
    if target_task.status == "pending":
        result["react_blocked"] = True
        result["react_block_reason"] = (
            f"子任务 {task_id}（{target_task.description[:80]}）沙盒验证失败\n"
            f"报错：{exec_box.error_trace[:500]}"
        )
    return result


def _test_cli(task_id, target_task, exec_box, cwd: str, main_file: str):
    """测试 CLI 程序：直接执行，等待退出码"""
    result = docker_exec(f"python {main_file}", cwd=cwd, timeout=30)

    if result["returncode"] == 0:
        logger.info(f"[沙盒验证] 子任务 {task_id} 测试通过 ✅")
        target_task.status = "finished"
        target_task.result = "沙盒测试通过"
        exec_box.error_trace = ""
        if result["stdout"]:
            logger.debug(f"[沙盒验证] stdout: {result['stdout'][:200]}")
    else:
        error_msg = result["stderr"][:2000] if result["stderr"] else f"无错误输出（exit code={result['returncode']}）"
        logger.warning(f"[沙盒验证] 子任务 {task_id} 测试失败 ❌ —— {error_msg[:300]}")
        target_task.status = "pending"
        target_task.result = "沙盒测试未通过，等待修复"
        exec_box.error_trace = error_msg
        exec_box.retry_count += 1
        exec_box.task_retry_count[task_id] = exec_box.task_retry_count.get(task_id, 0) + 1


def _test_web_service(task_id, target_task, exec_box, cwd: str, main_file: str, files: dict):
    """测试 Web 服务：后台启动（stderr→日志）→ 轮询等待就绪 → curl 测试 → 杀进程"""
    port = _detect_port(files)
    test_url = f"http://localhost:{port}/"
    log_file = "/tmp/_flask_startup.log"

    # 1. 后台启动服务，stderr 重定向到日志文件（用于崩溃诊断）
    pid = docker_exec_background(f"python {main_file} 2>{log_file}", cwd=cwd)
    if pid is None:
        target_task.status = "pending"
        target_task.result = "Web 服务后台启动失败"
        exec_box.error_trace = "docker exec -d 启动失败（容器可能未运行）"
        exec_box.retry_count += 1
        exec_box.task_retry_count[task_id] = \
            exec_box.task_retry_count.get(task_id, 0) + 1
        return

    logger.debug(f"[沙盒验证] Web 服务 PID={pid}，轮询等待就绪（最多 15 秒）...")

    # 2. 轮询等待 + 进程存活检测（最多 15 次 × 1 秒）
    http_code = "000"
    for attempt in range(1, 16):
        time.sleep(1)

        # 检查进程是否还活着
        alive = docker_exec(
            f"kill -0 {pid} 2>/dev/null && echo alive || echo dead",
            cwd=cwd, timeout=3,
        )
        if "dead" in alive.get("stdout", ""):
            # 进程已死 → 读取启动日志获取真正的错误
            err_log = docker_exec(
                f"cat {log_file} 2>/dev/null || echo '(无启动日志)'",
                cwd=cwd, timeout=3,
            )
            crash_msg = err_log.get("stdout", "").strip() or "(无错误输出)"
            logger.warning(f"[沙盒验证] Web 服务 PID={pid} 在第 {attempt} 秒崩溃")
            target_task.status = "pending"
            target_task.result = f"Web 服务启动后崩溃（第 {attempt} 秒）"
            exec_box.error_trace = (
                f"Flask 启动失败（进程在第 {attempt} 秒退出）\n\n"
                f"启动错误日志:\n{crash_msg[:1500]}"
            )
            exec_box.retry_count += 1
            exec_box.task_retry_count[task_id] = \
                exec_box.task_retry_count.get(task_id, 0) + 1
            docker_exec(f"rm -f {log_file}", cwd=cwd, timeout=3)
            return

        # 进程活着 → 尝试 curl
        curl_result = docker_exec(
            f"curl -s -o /dev/null -w '%{{http_code}}' --max-time 3 {test_url}",
            cwd=cwd, timeout=5,
        )
        http_code = curl_result.get("stdout", "").strip()

        if http_code == "200":
            logger.info(
                f"[沙盒验证] 子任务 {task_id} Web 服务测试通过 ✅ "
                f"({test_url} → 200，第 {attempt} 秒就绪)"
            )
            target_task.status = "finished"
            target_task.result = f"Web 服务测试通过（端口 {port}，HTTP 200，{attempt} 秒就绪）"
            exec_box.error_trace = ""
            break

        logger.debug(f"[沙盒验证] 第 {attempt} 秒: HTTP {http_code}，继续等待...")

    # 3. 如果轮询结束仍未返回 200
    if target_task.status != "finished":
        # 尝试读取启动日志辅助诊断
        err_log = docker_exec(
            f"cat {log_file} 2>/dev/null || echo '(无)'", cwd=cwd, timeout=3,
        )
        startup_hint = err_log.get("stdout", "").strip()

        logger.warning(
            f"[沙盒验证] 子任务 {task_id} Web 服务测试失败 ❌ "
            f"(HTTP {http_code}, 等待 15 秒仍未就绪)"
        )
        target_task.status = "pending"
        target_task.result = f"Web 服务未在 15 秒内返回 200（最后 HTTP {http_code}）"
        exec_box.error_trace = (
            f"Web 服务测试: {test_url} → {http_code}（15 秒轮询超时）\n"
            + (f"启动日志:\n{startup_hint[:1000]}" if startup_hint else "(无启动错误输出)")
        )
        exec_box.retry_count += 1
        exec_box.task_retry_count[task_id] = \
            exec_box.task_retry_count.get(task_id, 0) + 1

    # 4. 清理
    docker_exec(f"kill {pid} 2>/dev/null; pkill -f 'python {main_file}' 2>/dev/null || true",
                cwd=cwd, timeout=5)
    docker_exec(f"rm -f {log_file}", cwd=cwd, timeout=3)


def _detect_port(files: dict) -> int:
    """从代码中检测 Web 服务端口号"""
    for content in files.values():
        # Flask: app.run(port=5000) 或 port=5000
        import re
        match = re.search(r'port\s*=\s*(\d{4,5})', content)
        if match:
            return int(match.group(1))
    return 5000  # 默认


def build_sandbox_subgraph():
    """构建并编译沙盒验证子图。"""
    workflow = StateGraph(AgentState)
    workflow.add_node("sandbox", sandbox_node)
    workflow.set_entry_point("sandbox")
    workflow.add_edge("sandbox", END)
    return workflow.compile()
