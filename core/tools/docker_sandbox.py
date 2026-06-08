"""
Docker 沙盒 —— 长活容器，agent 在容器内执行所有命令。

启动时创建一个 python:3.11 容器，挂载宿主机工作区到 /workspace。
容器保持运行，agent 的所有操作（写文件、执行命令、验证）都在容器内完成。

用法:
    from core.tools.docker_sandbox import init_container, docker_exec, map_to_host, WORKSPACE_HOST
"""

import os
import subprocess
from pathlib import Path
from core.logger import logger

CONTAINER_NAME = "agent-sandbox"
IMAGE = "python:3.11"

# 宿主机工作区路径 → 挂载到容器的 /workspace
# 使用正斜杠，因为 Docker on Windows 接受 "D:/path" 格式，且避免 bash 转义反斜杠
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Agent_upgrade/
WORKSPACE_HOST = str(_PROJECT_ROOT / "output" / "workspace").replace("\\", "/")
WORKSPACE_CONTAINER = "/workspace"

# Web 服务检测关键词
WEB_FRAMEWORKS = ["flask", "fastapi", "django", "aiohttp", "sanic", "tornado"]


def init_container() -> bool:
    """确保容器存在且运行。启动时调用一次。"""
    os.makedirs(WORKSPACE_HOST, exist_ok=True)

    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Status}}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    status = result.stdout.strip()

    if not status:
        logger.info(f"[Docker沙盒] 创建新容器 {CONTAINER_NAME}（镜像: {IMAGE}）...")
        result = subprocess.run(
            ["docker", "run", "-d", "--name", CONTAINER_NAME,
             "-v", f"{WORKSPACE_HOST}:{WORKSPACE_CONTAINER}",
             "-w", WORKSPACE_CONTAINER,
             IMAGE, "tail", "-f", "/dev/null"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode != 0:
            logger.error(f"[Docker沙盒] 容器创建失败: {result.stderr}")
            return False
        logger.info(f"[Docker沙盒] 容器已创建，工作区: {WORKSPACE_HOST} → {WORKSPACE_CONTAINER}")
        return True

    if "Up" in status:
        logger.info(f"[Docker沙盒] 容器 {CONTAINER_NAME} 已在运行")
        return True

    # 存在但停止了 → 重启
    logger.info(f"[Docker沙盒] 重启容器 {CONTAINER_NAME}...")
    subprocess.run(
        ["docker", "start", CONTAINER_NAME],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return True


def docker_exec(command: str, cwd: str = WORKSPACE_CONTAINER, timeout: int = 60) -> dict:
    """在容器中执行命令，返回 {"stdout": ..., "stderr": ..., "returncode": ...}

    使用 subprocess.run 参数列表模式（shell=False），命令字符串作为 sh -c 的
    文字参数直接传递给容器，宿主机 shell 不参与解析，杜绝注入风险。
    """
    try:
        result = subprocess.run(
            ["docker", "exec", "-w", cwd, CONTAINER_NAME, "sh", "-c", command],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"命令执行超时（{timeout}秒）", "returncode": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "returncode": -1}


def docker_exec_background(command: str, cwd: str = WORKSPACE_CONTAINER) -> int | None:
    """在容器后台启动进程，返回容器内 PID"""
    result = subprocess.run(
        ["docker", "exec", "-d", "-w", cwd, CONTAINER_NAME, "sh", "-c", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        return None
    # 获取容器内最后启动的进程 PID
    pid_result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "sh", "-c", f"pgrep -f '{command}' | tail -1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    try:
        return int(pid_result.stdout.strip()) if pid_result.stdout.strip() else None
    except ValueError:
        return None


def docker_kill(pid: int) -> bool:
    """杀掉容器内指定 PID 的进程"""
    result = subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "kill", str(pid)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return result.returncode == 0


def is_web_service(files: dict) -> bool:
    """检查代码是否包含 Web 框架"""
    for content in files.values():
        content_lower = content.lower()
        for fw in WEB_FRAMEWORKS:
            if f"import {fw}" in content_lower or f"from {fw}" in content_lower:
                return True
    return False


def map_to_host(path: str) -> str:
    """将容器路径或相对路径映射到宿主机路径"""
    if os.path.isabs(path):
        if path.startswith(WORKSPACE_CONTAINER):
            rel = path[len(WORKSPACE_CONTAINER):].lstrip("/")
            return os.path.join(WORKSPACE_HOST, rel)
        return os.path.join(WORKSPACE_HOST, os.path.basename(path))
    return os.path.join(WORKSPACE_HOST, path)
