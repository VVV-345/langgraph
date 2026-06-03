"""
工具注册表 —— 所有工具的统一入口。

工具分层：
    filesystem.py — 7 个文件系统工具（list/read/write/edit/search/move/delete）
    run_command.py — Shell 命令执行（Docker 沙盒内）
    web_search.py  — 网络搜索（DuckDuckGo）

每个工具遵循统一接口：name / description / parameters / execute() / to_openai_schema()
"""

from core.tools.filesystem import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    SearchContentTool,
    MoveFileTool,
    DeleteFileTool,
    FS_TOOLS,
    FS_TOOL_BY_NAME,
)
from core.tools.run_command import RunCommandTool
from core.tools.web_search import WebSearchTool
from core.tools.submit_task import SubmitTaskTool

# ── 工具清单（10 个工具）──────────────────────────────────────────
# 文件系统: list_directory, read_file, write_file, edit_file,
#           search_content, move_file, delete_file         (7 个)
# 命令执行: run_command                                     (1 个)
# 网络搜索: web_search                                      (1 个)
# 任务提交: submit_task                                     (1 个)

ALL_INFRA_TOOLS = [
    RunCommandTool(),
    WebSearchTool(),
    SubmitTaskTool(),
]

ALL_TOOLS = FS_TOOLS + ALL_INFRA_TOOLS

# 快速查找表
TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}


def get_tool_schemas() -> list[dict]:
    """返回所有工具的 OpenAI function-calling 格式 schema"""
    return [t.to_openai_schema() for t in ALL_TOOLS]


def execute_tool(name: str, **kwargs) -> str:
    """根据名称执行工具，返回结果字符串"""
    tool = TOOL_BY_NAME.get(name)
    if tool is None:
        return f"错误：未找到工具 '{name}'，可用工具：{list(TOOL_BY_NAME.keys())}"
    try:
        return tool.execute(**kwargs)
    except Exception as e:
        return f"工具执行异常：{str(e)}"
