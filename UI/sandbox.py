"""
=============================================================================
UI/sandbox.py —— 沙盒兼容层（Docker 优先，不可用时降级本地 subprocess）
=============================================================================

给 UI 模块提供统一的 docker_exec / map_to_host / is_web_service 等接口。
导入时自动探测 Docker，失败则注入本地实现到 core.tools.docker_sandbox。
=============================================================================
"""

import os
import sys
import signal
import subprocess
from typing import Optional

_LOCAL_WORKSPACE: Optional[str] = None
_DOCKER_READY = False


def _try_init_docker() -> bool:
    """尝试初始化 Docker 沙盒，失败返回 False"""
    try:
        from core.tools.docker_sandbox import init_container
        return init_container()
    except Exception:
        return False


_DOCKER_READY = _try_init_docker()

if _DOCKER_READY:
    from core.tools.docker_sandbox import (  # noqa: E402
        docker_exec, docker_exec_background, docker_kill,
        is_web_service, map_to_host,
        WORKSPACE_HOST, WORKSPACE_CONTAINER,
    )
    print("[UI] ✅ Docker 沙盒已就绪")
else:
    print("[UI] ⚠️ Docker 不可用，启用本地 subprocess 沙盒（HF Spaces 模式）")

    # 本地工作区
    _LOCAL_WORKSPACE = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "output", "workspace")
    )
    os.makedirs(_LOCAL_WORKSPACE, exist_ok=True)

    WORKSPACE_HOST = _LOCAL_WORKSPACE
    WORKSPACE_CONTAINER = _LOCAL_WORKSPACE

    def map_to_host(path: str) -> str:
        """本地模式：路径直接映射到本地工作区"""
        if os.path.isabs(path):
            if path.startswith("/workspace"):
                rel = path[len("/workspace"):].lstrip("/")
                return os.path.join(_LOCAL_WORKSPACE, rel)
            return os.path.join(_LOCAL_WORKSPACE, os.path.basename(path))
        return os.path.join(_LOCAL_WORKSPACE, path)

    def docker_exec(command: str, cwd: str = None, timeout: int = 60) -> dict:
        """本地 subprocess 执行（替代 Docker exec）"""
        try:
            work_dir = cwd if cwd else _LOCAL_WORKSPACE
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                encoding="utf-8",
                errors="replace",
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

    def docker_exec_background(command: str, cwd: str = None) -> Optional[int]:
        """本地后台启动进程"""
        try:
            work_dir = cwd if cwd else _LOCAL_WORKSPACE
            proc = subprocess.Popen(
                command, shell=True, cwd=work_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return proc.pid
        except Exception:
            return None

    def docker_kill(pid: int) -> bool:
        """本地杀进程"""
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False

    def is_web_service(files: dict) -> bool:
        """检查代码是否包含 Web 框架"""
        frameworks = ["flask", "fastapi", "django", "aiohttp", "sanic", "tornado"]
        for content in files.values():
            cl = content.lower()
            for fw in frameworks:
                if f"import {fw}" in cl or f"from {fw}" in cl:
                    return True
        return False

    # Monkey-patch docker_sandbox 模块，让沙盒子图透明使用本地执行
    try:
        import core.tools.docker_sandbox as _ds
        _ds.WORKSPACE_HOST = WORKSPACE_HOST
        _ds.WORKSPACE_CONTAINER = WORKSPACE_CONTAINER
        _ds.init_container = lambda: True
        _ds.docker_exec = docker_exec
        _ds.docker_exec_background = docker_exec_background
        _ds.docker_kill = docker_kill
        _ds.is_web_service = is_web_service
        _ds.map_to_host = map_to_host
        print("[UI] 已注入本地沙盒到 core.tools.docker_sandbox")
    except Exception:
        pass
