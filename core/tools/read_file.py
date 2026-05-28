"""读取文件内容工具"""

import os


class ReadFileTool:
    name = "read_file"
    description = "读取指定路径的文件内容。用于了解项目结构、查看已有代码或读取配置文件。"

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要读取的文件路径（绝对路径或相对于工作目录的路径）"
            }
        },
        "required": ["path"]
    }

    def execute(self, path: str) -> str:
        if not os.path.exists(path):
            return f"错误：文件不存在 —— {path}"
        if os.path.isdir(path):
            return f"错误：路径是目录而非文件 —— {path}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 8000:
                content = content[:8000] + f"\n... (截断，共 {len(content)} 字符)"
            return content
        except UnicodeDecodeError:
            return f"错误：无法以 UTF-8 编码读取文件 —— {path}"
        except Exception as e:
            return f"读取文件失败：{str(e)}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
