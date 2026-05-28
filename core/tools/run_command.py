"""执行命令工具"""

import subprocess
import os


class RunCommandTool:
    name = "run_command"
    description = "在终端中执行 shell 命令并返回输出。用于运行 Python 脚本、安装依赖、查看文件列表等。注意：避免执行破坏性命令（rm -rf 等）。"

    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令"
            },
            "cwd": {
                "type": "string",
                "description": "命令执行的工作目录（可选，默认为当前目录）"
            }
        },
        "required": ["command"]
    }

    # 危险命令黑名单
    DANGEROUS_PATTERNS = ["rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:", "shutdown", "reboot"]

    def execute(self, command: str, cwd: str = None) -> str:
        # 安全检查
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in command:
                return f"拒绝执行：命令包含危险模式 '{pattern}'"

        work_dir = cwd or os.getcwd()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=work_dir
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            if len(output) > 4000:
                output = output[:4000] + f"\n... (截断)"
            return output.strip() or "(无输出)"
        except subprocess.TimeoutExpired:
            return "错误：命令执行超时（60秒）"
        except Exception as e:
            return f"命令执行失败：{str(e)}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
