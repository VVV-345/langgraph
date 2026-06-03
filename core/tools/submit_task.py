"""
=============================================================================
submit_task 工具 —— 显式标记 ReAct 任务完成
=============================================================================

【为什么需要】
    隐式退出（模型不调工具 = 完成）对 LLM 不直观，导致反复调用
    list_directory / read_file 直到 7 轮上限。给模型一个明确的"完成按钮"。
"""


class SubmitTaskTool:
    """任务提交工具 —— 调用即结束当前 ReAct 循环。"""

    name = "submit_task"
    description = (
        "任务完成时调用此工具提交最终摘要。"
        "调用后 ReAct 循环立即结束，进入验证和整合阶段。"
    )

    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": (
                    "任务执行结果摘要，说明完成了什么、关键文件路径、运行结果等"
                ),
            }
        },
        "required": ["summary"],
    }

    def execute(self, summary: str) -> str:
        return f"[任务完成] {summary[:2000]}"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
