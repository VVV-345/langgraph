"""执行命令工具 —— 在 Docker 容器内执行"""

import shlex

from core.tools.docker_sandbox import docker_exec, WORKSPACE_CONTAINER


class RunCommandTool:
    name = "run_command"
    description = "在 Linux 终端中执行 shell 命令并返回输出。用于运行 Python 脚本、pip install 安装依赖、查看文件列表、测试 Web 服务等。"

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令（Linux 语法）"
            },
            "cwd": {
                "type": "string",
                "description": "命令执行的工作目录（可选，默认为 /workspace）"
            }
        },
        "required": ["command"]
    }

    # 命令名黑名单（精确匹配 basename，不允许绕过）
    DANGEROUS_COMMANDS = {
        "shutdown", "reboot", "halt", "poweroff", "init",
        "mkfs", "mkfs.ext2", "mkfs.ext3", "mkfs.ext4", "mkfs.xfs", "mkfs.btrfs",
        "mkswap", "fdisk", "parted",
        "dd",
    }

    # 参数模式黑名单（检查整个命令字符串）
    DANGEROUS_ARG_PATTERNS = [
        # fork 炸弹
        ":(){ :|:& };:",
        # 递归删除根目录（"-rf" 或 "-r -f" 后跟 "/" 或 "/*"）
        r"rm\s+.*(?:-r\b|--recursive).*/(?:\s|$|\*)",
        r"rm\s+.*-rf?\s+/(?:\s|$|\*)",
        # 覆盖块设备
        r">\s*/dev/sd[a-z]",
        r"dd\s+if=.*of=/dev/sd[a-z]",
        # 清除系统关键目录
        r"rm\s+-rf?\s+/(?:bin|boot|dev|etc|lib|lib64|proc|root|sbin|sys|usr|var)(?:\s|$|/\*)",
    ]

    def execute(self, command: str, cwd: str = None) -> str:
        # 1. Token-level 命令名校验
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if tokens:
            cmd_name = tokens[0].split("/")[-1]  # basename only
            if cmd_name in self.DANGEROUS_COMMANDS:
                return (
                    f"拒绝执行：命令 '{cmd_name}' 在禁止名单中。"
                    f"该命令可能对系统造成不可逆的破坏。"
                )
            # 也检查完整路径命令名（如 /usr/sbin/shutdown）
            full_basename = tokens[0].rstrip("/").rsplit("/", 1)[-1]
            if full_basename in self.DANGEROUS_COMMANDS:
                return (
                    f"拒绝执行：命令 '{full_basename}' 在禁止名单中。"
                    f"该命令可能对系统造成不可逆的破坏。"
                )

        # 2. 参数模式校验（正则匹配）
        import re
        normalized = " ".join(tokens)
        for pattern in self.DANGEROUS_ARG_PATTERNS:
            if re.search(pattern, normalized):
                return (
                    f"拒绝执行：命令包含危险操作模式（匹配: '{pattern}'）。"
                    f"请使用更安全的替代方案。"
                )

        work_dir = cwd or WORKSPACE_CONTAINER

        # 后台启动 Web 服务的命令需要特殊处理
        if command.strip().endswith("&"):
            # 去掉末尾的 &，用 docker exec -d 后台执行
            fg_cmd = command.strip().rstrip("&").strip()
            result = docker_exec(fg_cmd, cwd=work_dir, timeout=10)
            if result["returncode"] == 0 or not result["stderr"]:
                return f"后台启动成功: {fg_cmd}\n提示: 用 sleep 2 等待启动，然后用 curl 测试"
            return f"后台启动失败: {result['stderr'][:500]}"

        result = docker_exec(command, cwd=work_dir, timeout=60)

        output = ""
        if result["stdout"]:
            output += result["stdout"]
        if result["stderr"]:
            output += "\n[stderr]\n" + result["stderr"]
        if result["returncode"] != 0:
            output += f"\n[exit code: {result['returncode']}]"
        if len(output) > 4000:
            output = output[:4000] + "\n... (截断)"
        return output.strip() or "(无输出)"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
