# core/state.py
from langgraph.graph import MessagesState
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

"""
用户输入任务
    ↓
【感知阶段】理解需求+上下文感知
    ↓
【规划阶段】任务拆解+风险评估+资源确认
    ↓
【调度阶段】子任务排序+依赖管理+资源分配
    ↓
【执行阶段】ReAct循环（思考→调用工具→观察结果）
    ↓
【验证阶段】结果校验+错误重试+用户确认
    ↓
【整合阶段】汇总所有子任务结果
    ↓
【输出阶段】生成最终交付物+反馈收集
    ↓
【复盘阶段】优化任务拆解策略（可选）

"""

# ==================================
# 子任务数据结构
# ==================================
class SubTask(BaseModel):
    task_id: int                  # 子任务唯一ID（1,2,3...）
    description: str              # 子任务描述
    objective: str                # 子任务目标
    dependencies: List[int] = []  # 依赖的子任务ID列表
    status: str = "pending"       # 执行状态：pending/doing/testing/finished/failed
    result: str = ""              # 子任务执行结果（代码/文本/文件路径等）


# ==================================
# 感知、规划、调度状态单
# ==================================

class PlanningContext(BaseModel):
    """
    【感知阶段】规划收纳盒
    由 Analyzer Node 填充，记录任务评估与拆解结果
    """
    # 任务复杂度评定结果："simple" / "complex" / "pending"
    task_complexity: str = "simple"
    # 拆解后的子任务清单一简单任务只有1个元素，复杂任务3~5个
    task_plan: List[SubTask] = Field(default_factory=list)
    # Planner 输出的原始思考链（用于调试与回溯）
    thinking_chain: str = ""
    # 是否需要向用户澄清需求（True=阻断执行，等待用户补充信息）
    need_clarification: bool = False
    # 需要向用户反问的具体问题
    clarification_question: str = ""


class ExecutionContext(BaseModel):
    """
    执行与调度状态
    """
    # ==================================
    # 执行进度字段（Worker Node 更新）
    # ==================================
    current_task_index: int = 0           # 当前正在执行第几个子任务（默认0，从第一个开始）
    stage_outputs: Dict[int, str] = Field(default_factory=dict) # 键=子任务ID，值=执行结果
    all_tasks_completed: bool = False     # 所有子任务是否完成
    
    error_trace: str = ""                 # 沙盒运行返回的报错
    retry_count: int = 0                  # 记录因为报错重写的次数


# ==================================
# ReAct 执行阶段数据结构
# ==================================

class ToolCall(BaseModel):
    """单次工具调用记录"""
    tool_name: str = ""           # 工具名
    tool_input: dict = {}         # 工具参数
    tool_output: str = ""         # 工具返回结果（截断后）
    timestamp: str = ""           # 调用时间


class ReActStep(BaseModel):
    """ReAct 循环中的一步：思考 → 行动 → 观察"""
    thought: str = ""             # LLM 的思考过程
    action: ToolCall = Field(default_factory=ToolCall)
    observation: str = ""         # 工具执行结果

    def to_prompt_str(self) -> str:
        """将这一步转成可喂给 LLM 的上下文文本"""
        lines = [f"Thought: {self.thought}"]
        if self.action.tool_name:
            lines.append(f"Action: {self.action.tool_name}({self.action.tool_input})")
        if self.observation:
            lines.append(f"Observation: {self.observation[:2000]}")
        return "\n".join(lines)


# ==================================
# 主图与子图的最终状态单
# ==================================

class AgentState(MessagesState):
    """
    主图状态单：供感知、调度、执行的大脑使用
    """
    planning: PlanningContext = Field(default_factory=PlanningContext)
    execution: ExecutionContext = Field(default_factory=ExecutionContext)

    current_code: str = ""                # Coder 生成的代码

    # ==================================
    # ReAct 循环状态
    # ==================================
    react_round: int = 0                  # 当前 ReAct 循环轮数
    react_history: list = Field(default_factory=list)  # List[ReActStep] 序列化形式
    react_blocked: bool = False           # 是否卡住需要人工介入
    react_block_reason: str = ""          # 卡住原因
    react_finished: bool = False          # 当前子任务 ReAct 是否完成


class SafeExecutionState(MessagesState):
    """
    独立的高危执行子图专用状态单：
    主图对它一无所知，只有进入高危权限校验时才临时创建。
    """
    # --- 高危执行子图专用状态 ---
    pending_action: str = "" # 准备执行的动作类型 (例如 "terminal_cmd" 或 "file_modify")
    action_payload: str = "" # 动作的具体内容 (例如 "rm -rf folder")
    auth_granted: bool = False # 用户是否授权通过 (True/False)
    backup_path: str = ""    # 备份文件的存放位置 (用于回撤)
    action_result: str = ""  # 终端执行完返回的结果或报错

