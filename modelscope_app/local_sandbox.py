"""
=============================================================================
local_sandbox.py —— 本地沙盒替代 Docker（魔塔社区 ModelScope 可部署）
=============================================================================

Docker 在魔塔不可用，启动时自动把 core.tools.docker_sandbox 的所有函数
替换为 subprocess 本地执行版。沙盒测代码、文件读写都在本地工作区完成。
替换时机在 import core.* 之前，对后续所有节点透明。

导入此模块即自动执行替换，并暴露 _LOCAL_WORKSPACE 供其他模块使用。
=============================================================================
"""

import os
import sys
import signal
import subprocess
import types
from pathlib import Path
from typing import Optional

# 确保 Windows 控制台能输出 emoji（魔塔 Linux 环境不受影响）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _setup_local_sandbox() -> str:
    """
    初始化本地沙盒工作区，返回工作区路径。
    把 core.tools.docker_sandbox 模块全部替换为本地 subprocess 版本，
    让后面的 sandbox / scheduler / filesystem / run_command 节点无感运行。
    """
    # 工作区：魔塔环境用 /data/workspace（持久化），本地用 output/workspace/
    project_root = Path(__file__).resolve().parent.parent  # modelscope_app/ 的父目录 = 项目根
    if os.path.isdir("/data"):
        workspace = "/data/workspace"
    else:
        workspace = os.path.join(project_root, "output", "workspace")
    os.makedirs(workspace, exist_ok=True)

    # ---- 本地版替代函数 ----

    def _local_exec(command: str, cwd: str = None, timeout: int = 60) -> dict:
        """本地 subprocess 执行（替代 docker exec）"""
        try:
            work_dir = cwd if cwd else workspace
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
            return {"stdout": "", "stderr": f"执行超时（{timeout}秒）", "returncode": -1}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1}

    def _local_exec_background(command: str, cwd: str = None) -> Optional[int]:
        """本地后台启动（替代 docker exec -d）"""
        try:
            work_dir = cwd if cwd else workspace
            proc = subprocess.Popen(
                command, shell=True, cwd=work_dir,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return proc.pid
        except Exception:
            return None

    def _local_kill(pid: int) -> bool:
        """本地杀进程"""
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False

    def _local_map_to_host(path: str) -> str:
        """容器路径 → 本地工作区路径"""
        if os.path.isabs(path):
            if path.startswith("/workspace"):
                rel = path[len("/workspace"):].lstrip("/")
                return os.path.join(workspace, rel)
            return os.path.join(workspace, os.path.basename(path))
        return os.path.join(workspace, path)

    def _is_web_service(files: dict) -> bool:
        """检查代码是否用了 Web 框架（纯逻辑，无 Docker 依赖）"""
        frameworks = ["flask", "fastapi", "django", "aiohttp", "sanic", "tornado"]
        for content in files.values():
            cl = content.lower()
            for fw in frameworks:
                if f"import {fw}" in cl or f"from {fw}" in cl:
                    return True
        return False

    # ---- 创建替代模块并注入 sys.modules ----
    fake_module = types.ModuleType("core.tools.docker_sandbox")
    fake_module.__file__ = __file__  # 避免某些框架按路径找文件
    fake_module.WORKSPACE_HOST = workspace
    fake_module.WORKSPACE_CONTAINER = workspace
    fake_module.CONTAINER_NAME = "local-sandbox"
    fake_module.init_container = lambda: True
    fake_module.docker_exec = _local_exec
    fake_module.docker_exec_background = _local_exec_background
    fake_module.docker_kill = _local_kill
    fake_module.is_web_service = _is_web_service
    fake_module.map_to_host = _local_map_to_host

    sys.modules["core.tools.docker_sandbox"] = fake_module

    # 如果已经 import 过（不太可能，但防御一下），原地替换
    existing = sys.modules.get("core.tools.docker_sandbox")
    if existing is not None:
        for attr in dir(fake_module):
            if not attr.startswith("__"):
                setattr(existing, attr, getattr(fake_module, attr))

    print(f"[本地沙盒] ✅ 已注入本地执行环境，工作区: {workspace}")
    return workspace


# 模块导入时自动执行沙盒替换
_LOCAL_WORKSPACE = _setup_local_sandbox()
