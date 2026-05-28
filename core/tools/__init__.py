"""
工具注册表 —— 所有工具的统一入口。
每个工具遵循统一接口：name / description / parameters / execute()
"""

from core.tools.read_file import ReadFileTool
from core.tools.write_file import WriteFileTool
from core.tools.run_command import RunCommandTool
from core.tools.web_search import WebSearchTool

# 工具清单
ALL_TOOLS = [
    ReadFileTool(),
    WriteFileTool(),
    RunCommandTool(),
    WebSearchTool(),
]

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
