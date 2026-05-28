"""写入文件工具"""

import os


class WriteFileTool:
    name = "write_file"
    description = "将内容写入指定路径的文件。如果文件已存在则覆盖，目录不存在则自动创建。"

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要写入的文件路径（绝对路径或相对于工作目录的路径）"
            },
            "content": {
                "type": "string",
                "description": "要写入文件的完整内容"
            }
        },
        "required": ["path", "content"]
    }

    def execute(self, path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            size = os.path.getsize(path)
            return f"写入成功：{path}（{size} 字节）"
        except PermissionError:
            return f"错误：没有写入权限 —— {path}"
        except Exception as e:
            return f"写入文件失败：{str(e)}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
