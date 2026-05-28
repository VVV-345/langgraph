# core/nodes/worker.py
import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from core.state import AgentState, ExecutionContext
from core.logger import logger

load_dotenv()

# 编码专用 LLM——不需要 JSON 输出，核心任务是写出高质量的纯代码
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),
    temperature=0.1,      # 低温保证代码逻辑严谨
    max_retries=8,
    timeout=60
)


def worker_node(state: AgentState):
    """
    【调度 + 执行阶段】打工人节点

    每次被图谱调用时只处理一个子任务：
    1. 从 PlanningContext 中捞出第一个 pending 子任务
    2. 翻阅前序已完成任务的代码作为上下文
    3. 如果 exec_box 中有 error_trace（上次沙盒报错），一并喂给 LLM 修复
    4. 调用 LLM 生成当前子任务的代码
    5. 将成果归档到 ExecutionContext，更新子任务状态为 "testing"

    图谱会通过条件边反复调用此节点，直到所有子任务完成。
    """
    # ==========================================
    # 从状态中取出两个核心盒子
    # ==========================================
    plan_box = state.get("planning")   # 规划收纳盒：记录子任务清单与状态
    exec_box = state.get("execution")  # 执行收纳盒：记录运行进度与成果

    # 兜底：首次进入 worker 时 execution 可能尚未初始化（analyzer 不会创建它）
    if exec_box is None:
        exec_box = ExecutionContext()

    # ==========================================
    # 1. 调度：找到下一个待办的子任务
    # ==========================================
    pending_tasks = [t for t in plan_box.task_plan if t.status == "pending"]

    if not pending_tasks:
        # 所有子任务已完成，插旗通知图谱可以收工了
        logger.info("[编码车间] 报告老板，白板上的任务全部干完了！")
        exec_box.all_tasks_completed = True
        return {"execution": exec_box}

    # 按顺序领取排在最前面的 pending 任务
    current_task = pending_tasks[0]
    task_id = current_task.task_id

    logger.info(f"[编码车间] 正在死磕子任务 {task_id}: {current_task.description}")

    # ==========================================
    # 2. 感知上下文：翻阅前序任务的历史成果 + 报错信息
    # ==========================================
    # 将已完成任务的代码拼接为参考上下文，防止多子任务之间出现断裂
    previous_outputs = ""
    if exec_box.stage_outputs:
        previous_outputs = "【已完成的前置任务成果参考——请确保新代码与以下代码兼容】\n"
        for t_id, code in exec_box.stage_outputs.items():
            previous_outputs += f"--- 子任务 {t_id} 代码 ---\n{code}\n\n"

    # 检查有没有上次沙盒返回的报错——这是"打工人觉醒"的关键
    error_feedback = ""
    if exec_box.error_trace:
        error_feedback = (
            "【⚠️ 上次运行报错——请务必修复以下问题】\n"
            f"{exec_box.error_trace}\n\n"
            "请仔细分析上面的错误，修改代码直到不会再出现同样的问题。\n"
        )
        logger.warning(f"[编码车间] 检测到上次报错，正在修复子任务 {task_id}...")

    # ==========================================
    # 3. 执行：局部思考 + 代码生成
    # ==========================================
    # 系统人设（告诉它规则）
    system_prompt = """你是一个顶级的 Python 程序员。
【输出要求】
1. 只输出纯 Python 代码，绝对不要包含任何解释性文字。
2. 不要使用 markdown 的 ```python 代码块标记包裹，直接输出代码本身！"""

    # 用户任务（告诉它具体干什么）
    user_prompt = f"""请严格按照以下当前子任务的需求编写代码：

【当前死磕的任务】
任务描述：{current_task.description}
具体目标：{current_task.objective}

{previous_outputs}
{error_feedback}"""

    # 拆分成 系统 + 人类 对话格式，通过所有大模型 API 校验
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]).content

    # 兜底清理：防止模型仍然带了 markdown 标记
    clean_code = response.replace("```python", "").replace("```", "").strip()

    logger.info(f"[编码车间] 子任务 {task_id} 代码编写完成，准备进入沙盒测试！")

    # ==========================================
    # 4. 归档：成果入库，状态设为 testing
    # ==========================================
    # 将当前子任务标记为 testing（等待沙盒验证，不直接 finished）
    for t in plan_box.task_plan:
        if t.task_id == task_id:
            t.status = "testing"
            t.result = "代码已生成，等待沙盒验证"

    # 将生成的代码存入档案馆（供后续子任务参考）
    exec_box.stage_outputs[task_id] = clean_code
    # 进度条向前推进一格
    exec_box.current_task_index += 1

    # 将更新后的盒子打包返回给图谱
    return {
        "planning": plan_box,
        "execution": exec_box,
        "current_code": clean_code
    }
