"""
=============================================================================
文件系统工具集 —— 7 个工具覆盖完整文件操作
=============================================================================

工具清单:
    list_directory  - 列出目录内容（文件 + 子目录）
    read_file       - 分页读取文件内容（带行号）
    write_file      - 创建或覆盖文件
    edit_file       - 外科手术式局部替换（改一行不用重写整个文件）
    search_content  - 在文件中搜索匹配行（grep）
    move_file       - 移动或重命名文件/目录
    delete_file     - 删除文件或空目录

所有路径相对于工作区（由 docker_sandbox.map_to_host 映射到宿主机路径）。
=============================================================================
"""

import os
import re
import shutil
from pathlib import Path
from core.tools.docker_sandbox import map_to_host, WORKSPACE_HOST

# ── 配置 ──────────────────────────────────────────────────────────
DEFAULT_READ_LINES = 700
MAX_READ_CHARS = 25000
MAX_SEARCH_RESULTS = 20
MAX_LIST_ITEMS = 50


# ===================================================================
# 工具 1: list_directory — 列出目录
# ===================================================================

class ListDirectoryTool:
    name = "list_directory"
    description = (
        "列出工作区目录的内容。用于查看项目结构、确认文件存在。"
        "默认递归深度 1（只列当前目录），最大深度 3。"
        "输出格式：📁 dir_name/ 或 📄 file_name (size)。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目录路径，相对于工作区（如 '.'、'templates'）。默认 '.'"
            },
            "depth": {
                "type": "integer",
                "description": "递归深度（1-3），默认 1"
            }
        },
        "required": []
    }

    def execute(self, path: str = ".", depth: int = 1) -> str:
        host_path = map_to_host(path)
        if not os.path.exists(host_path):
            return f"错误：路径不存在 —— {path}"
        if not os.path.isdir(host_path):
            return f"错误：不是目录 —— {path}，请用 read_file 读取文件"

        depth = max(1, min(depth, 3))
        lines = []
        count = 0

        def walk(dir_path: str, prefix: str, current_depth: int):
            nonlocal count
            if current_depth > depth or count >= MAX_LIST_ITEMS * 2:
                return
            try:
                entries = sorted(os.listdir(dir_path))
            except PermissionError:
                lines.append(f"{prefix}📁 <权限不足>")
                return

            for name in entries:
                if count >= MAX_LIST_ITEMS * 2:
                    break
                full = os.path.join(dir_path, name)
                rel = Path(full).relative_to(WORKSPACE_HOST).as_posix()
                try:
                    if os.path.isdir(full):
                        count += 1
                        lines.append(f"{prefix}📁 {rel}/")
                        if current_depth < depth:
                            walk(full, prefix + "  ", current_depth + 1)
                    else:
                        count += 1
                        size = os.path.getsize(full)
                        if size < 1024:
                            size_str = f"{size}B"
                        elif size < 1024 * 1024:
                            size_str = f"{size / 1024:.1f}KB"
                        else:
                            size_str = f"{size / 1024 / 1024:.1f}MB"
                        lines.append(f"{prefix}📄 {rel} ({size_str})")
                except OSError:
                    continue

        walk(host_path, "", 1)

        header = f"[目录: {path}, 深度={depth}, 显示 {count} 项]"
        if not lines:
            return f"{header}\n（空目录）"
        if count >= MAX_LIST_ITEMS * 2:
            lines.append(f"... 还有更多项目（已截断，试试缩小 path 范围）")
        return header + "\n" + "\n".join(lines)

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 工具 2: read_file — 分页读取
# ===================================================================

class ReadFileTool:
    name = "read_file"
    description = (
        "分页读取工作区文件内容（带行号）。默认读取前 700 行。"
        "输出截断 25000 字符，截断时提示剩余行数，用 start_line 继续读取。"
        "示例: read_file(path='app.py') → 前 700 行；"
        "read_file(path='app.py', start_line=701) → 继续读。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径，相对于工作区（如 'app.py'、'templates/index.html'）"
            },
            "start_line": {
                "type": "integer",
                "description": "从第几行开始读（默认 1）"
            },
            "end_line": {
                "type": "integer",
                "description": "读到第几行（含），不传则读 DEFAULT_READ_LINES 行"
            }
        },
        "required": ["path"]
    }

    def execute(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        host_path = map_to_host(path)
        if not os.path.exists(host_path):
            return (
                f"文件尚未创建: {path} —— 工作区还没有这个文件。"
                f"👉 用 write_file 创建它，不要换参数重试读取。"
            )
        if os.path.isdir(host_path):
            return f"错误：{path} 是目录，请用 list_directory"

        try:
            with open(host_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"错误：无法以 UTF-8 编码读取 —— {path}"
        except Exception as e:
            return f"读取文件失败：{str(e)}"

        total = len(lines)
        if total == 0:
            return f"[文件: {path}, 0 行]（空文件）"

        start = max(1, start_line) - 1
        if end_line <= 0:
            end_line = min(start + DEFAULT_READ_LINES, total)
        end = min(end_line - 1, total - 1)

        if start >= total:
            return f"错误：start_line={start_line} 超出文件总行数 {total}"

        selected = lines[start : end + 1]
        output_lines = []
        for i, line in enumerate(selected, start=start + 1):
            output_lines.append(f"{i:>6}|{line.rstrip()}")

        body = "\n".join(output_lines)

        if len(body) > MAX_READ_CHARS:
            body = body[:MAX_READ_CHARS]
            last_line_num = start + body[:MAX_READ_CHARS].count("\n")
            body += f"\n[... 截断，使用 start_line={last_line_num + 1} 继续]"

        header = f"[文件: {path}, L{start + 1}-{end + 1}/{total} 行]"
        footer = ""
        if end < total - 1:
            footer = f"\n[还有 {total - end - 1} 行未显示，start_line={end + 2} 继续读取]"

        return f"{header}\n{body}{footer}"

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 工具 3: write_file — 写入文件
# ===================================================================

class WriteFileTool:
    name = "write_file"
    description = (
        "将内容写入工作区文件。父目录不存在则自动创建。文件已存在则覆盖。"
        "大文件请分多次调用 write_file 创建不同的文件，不要在一个文件中堆砌所有代码。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "文件路径，相对于工作区（如 'app.py'、'templates/index.html'）"
            },
            "content": {
                "type": "string",
                "description": "要写入文件的完整源代码内容"
            }
        },
        "required": ["path", "content"]
    }

    def execute(self, path: str, content: str) -> str:
        try:
            host_path = map_to_host(path)
            os.makedirs(os.path.dirname(host_path) or WORKSPACE_HOST, exist_ok=True)
            with open(host_path, "w", encoding="utf-8") as f:
                f.write(content)
            size = os.path.getsize(host_path)
            lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"写入成功：{path}（{size} 字节, {lines} 行）"
        except PermissionError:
            return f"错误：没有写入权限 —— {path}"
        except Exception as e:
            return f"写入文件失败：{str(e)}"

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 工具 4: edit_file — 外科手术式局部替换
# ===================================================================

class EditFileTool:
    name = "edit_file"
    description = (
        "精确替换文件中的一段文本。old_string 必须精确匹配（包括缩进和换行）。"
        "优先使用此工具而非 write_file 重写整个文件，节省 token 并减少出错范围。"
        "old_string 要包含足够的上下文（前后各 1-2 行）以确保唯一匹配。"
        "示例: edit_file(path='app.py', old_string='port = 5000', new_string='port = 8080')"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要编辑的文件路径，相对于工作区"
            },
            "old_string": {
                "type": "string",
                "description": "要被替换的文本片段（必须精确匹配，含缩进）"
            },
            "new_string": {
                "type": "string",
                "description": "替换后的文本片段"
            }
        },
        "required": ["path", "old_string", "new_string"]
    }

    def execute(self, path: str, old_string: str, new_string: str) -> str:
        host_path = map_to_host(path)
        if not os.path.exists(host_path):
            return f"错误：文件不存在 —— {path}"

        try:
            with open(host_path, "r", encoding="utf-8") as f:
                original = f.read()
        except Exception as e:
            return f"读取文件失败：{str(e)}"

        count = original.count(old_string)
        if count == 0:
            return (
                f"错误：未找到匹配的 old_string。请用 read_file 确认文件内容，"
                f"确保 old_string 精确匹配（包括缩进、空格）。"
            )
        if count > 1:
            return (
                f"错误：old_string 匹配到 {count} 处，不唯一。请增加上下文（前后各 1-2 行）"
                f"使匹配唯一。"
            )

        new_content = original.replace(old_string, new_string, 1)
        try:
            with open(host_path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return f"写入文件失败：{str(e)}"

        # 计算变更摘要
        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        return f"替换成功：{path}（{old_lines} 行 → {new_lines} 行）"

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 工具 5: search_content — grep 搜索
# ===================================================================

class SearchContentTool:
    name = "search_content"
    description = (
        "在工作区文件中搜索匹配文本（类似 grep）。支持正则表达式。"
        "用于在代码中查找函数定义、import 语句、变量引用等，不需要逐个读取文件。"
        "示例: search_content(pattern='def get_user') → 查找函数定义；"
        "search_content(pattern='from flask', path='src/') → 搜索特定目录。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "搜索模式（纯文本或正则表达式）"
            },
            "path": {
                "type": "string",
                "description": "搜索范围（文件或目录），默认搜索整个工作区"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "是否区分大小写，默认 false（不区分）"
            },
            "file_pattern": {
                "type": "string",
                "description": "文件名过滤（如 '*.py'、'*.html'），默认 '*.py'"
            }
        },
        "required": ["pattern"]
    }

    def execute(
        self,
        pattern: str,
        path: str = ".",
        case_sensitive: bool = False,
        file_pattern: str = "*.py",
    ) -> str:
        host_path = map_to_host(path)
        if not os.path.exists(host_path):
            return f"错误：路径不存在 —— {path}"

        # 收集要搜索的文件
        if os.path.isfile(host_path):
            files = [host_path]
        else:
            files = _collect_files(host_path, file_pattern)

        if not files:
            return f"未找到匹配 {file_pattern} 的文件"

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"正则表达式错误：{str(e)}"

        results = []
        total_matches = 0
        for fpath in files:
            if total_matches >= MAX_SEARCH_RESULTS:
                break
            rel = Path(fpath).relative_to(WORKSPACE_HOST).as_posix()
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for lineno, line in enumerate(f, 1):
                        if total_matches >= MAX_SEARCH_RESULTS:
                            break
                        if regex.search(line):
                            total_matches += 1
                            results.append(f"  {rel}:{lineno}: {line.rstrip()[:200]}")
            except (UnicodeDecodeError, PermissionError):
                continue

        header = f"[搜索: '{pattern}' 在 {len(files)} 个文件中共 {total_matches} 处匹配]"
        if not results:
            return f"{header}\n（无匹配）"
        if total_matches >= MAX_SEARCH_RESULTS:
            results.append(f"... 已达 {MAX_SEARCH_RESULTS} 条上限，请缩小搜索范围")
        return header + "\n" + "\n".join(results)

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 工具 6: move_file — 移动/重命名
# ===================================================================

class MoveFileTool:
    name = "move_file"
    description = (
        "移动或重命名工作区中的文件/目录。相当于 mv 命令。"
        "父目录不存在则自动创建。"
        "示例: move_file(source='old_name.py', target='new_name.py') → 重命名；"
        "move_file(source='utils.py', target='lib/utils.py') → 移动到子目录。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "源路径，相对于工作区"
            },
            "target": {
                "type": "string",
                "description": "目标路径，相对于工作区"
            }
        },
        "required": ["source", "target"]
    }

    def execute(self, source: str, target: str) -> str:
        src_host = map_to_host(source)
        tgt_host = map_to_host(target)

        if not os.path.exists(src_host):
            return f"错误：源路径不存在 —— {source}"

        try:
            os.makedirs(os.path.dirname(tgt_host) or WORKSPACE_HOST, exist_ok=True)
            shutil.move(src_host, tgt_host)
            return f"移动成功：{source} → {target}"
        except PermissionError:
            return f"错误：没有权限移动 —— {source}"
        except Exception as e:
            return f"移动失败：{str(e)}"

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 工具 7: delete_file — 删除文件
# ===================================================================

class DeleteFileTool:
    name = "delete_file"
    description = (
        "删除工作区中的文件或空目录。"
        "⚠️ 不可恢复，请确认后再删除。不支持删除非空目录（需先清空）。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要删除的文件/目录路径，相对于工作区"
            }
        },
        "required": ["path"]
    }

    def execute(self, path: str) -> str:
        host_path = map_to_host(path)
        if not os.path.exists(host_path):
            return f"错误：路径不存在 —— {path}"

        try:
            if os.path.isdir(host_path):
                os.rmdir(host_path)  # 只删空目录
                return f"已删除目录：{path}"
            else:
                os.remove(host_path)
                return f"已删除文件：{path}"
        except OSError as e:
            if "directory not empty" in str(e).lower():
                return f"错误：目录非空 —— {path}，请先删除目录中的文件"
            return f"删除失败：{str(e)}"
        except Exception as e:
            return f"删除失败：{str(e)}"

    def to_openai_schema(self) -> dict:
        return _to_schema(self.name, self.description, self.parameters)


# ===================================================================
# 辅助函数
# ===================================================================

def _to_schema(name: str, description: str, parameters: dict) -> dict:
    """生成 OpenAI function calling 格式的 JSON Schema"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _collect_files(root: str, glob_pattern: str) -> list:
    """递归收集匹配 glob 模式的文件列表（忽略 __pycache__、.git 等）"""
    import fnmatch
    files = []
    skip_dirs = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".pytest_cache"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fname in filenames:
            if fnmatch.fnmatch(fname, glob_pattern):
                files.append(os.path.join(dirpath, fname))
    return files


# ===================================================================
# 工具清单（供注册使用）
# ===================================================================

FS_TOOLS = [
    ListDirectoryTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    SearchContentTool(),
    MoveFileTool(),
    DeleteFileTool(),
]

FS_TOOL_BY_NAME = {t.name: t for t in FS_TOOLS}
