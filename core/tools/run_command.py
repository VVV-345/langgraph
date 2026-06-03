"""执行命令工具 —— 在 Docker 容器内执行"""

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

    DANGEROUS_PATTERNS = ["rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:", "shutdown", "reboot"]

    def execute(self, command: str, cwd: str = None) -> str:
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in command:
                return f"拒绝执行：命令包含危险模式 '{pattern}'"

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
