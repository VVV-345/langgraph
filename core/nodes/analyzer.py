# core/nodes/analyzer.py
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from core.state import AgentState, PlanningContext, SubTask
from langchain_openai import ChatOpenAI
from core.logger import logger

load_dotenv()  # 加载环境变量

"""
# 挂载本地大模型，强制开启 JSON 格式输出
llm = ChatOllama(
    model="qwen2.5:7b", 
    temperature=0.1, 
    format="json",
    timeout=60  # 增加超时时间，避免大任务拆解时卡住
)
"""

# 云端模型
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),  # 免费模型，速度极快
    temperature=0.1,
    # 模型的JSON输出参数（和Ollama的format="json"等价）
    model_kwargs={"response_format": {"type": "json_object"}},
    max_retries=8,
    timeout=60
)


def analyzer_node(state: AgentState):
    """
    【感知阶段】最高指挥官
    负责阅读用户完整对话历史，评估需求清晰度和任务难度，
    并把任务拆解写入 PlanningContext (规划收纳盒)
    """
    logger.info("[感知中枢] 正在评估任务复杂度与拆解需求...")
    
    
    # 保留最近10轮对话
    recent_messages = state["messages"][-20:]
    
    # Prompt必须极其严格，加入澄清逻辑和子任务数量限制
    prompt = """你是一个资深的软件架构师。请严格按照以下规则处理用户需求，只输出纯JSON，不要有任何解释性文字！

【第一步：评估需求清晰度】
如果用户的需求模糊、不完整、缺少必要信息（例如只说"帮我写个程序"），输出：
{
    "need_clarification": true,
    "clarification_question": "请具体说明你需要写什么程序，实现什么功能？"
}

【第二步：评估任务复杂度】
如果需求清晰，判断任务难度：
- simple (简单任务)：单文件脚本、基础算法、单一功能的函数（例如：写个计算器、重命名文件、处理单个CSV）。
- complex (复杂任务)：涉及多个文件、需要配置环境、完整的项目架构（例如：ROS2联合仿真系统、前后端架构）。

【第三步：生成任务计划】
- 简单任务：只生成1个子任务
- 复杂任务：拆分为3-5个子任务（最多5个，绝对不能超过）
- 每个子任务必须独立、可落地、有明确的目标
- dependencies字段填写依赖的子任务ID，没有依赖则为空数组

【输出格式】
如果需求清晰，输出：
{
    "need_clarification": false,
    "task_complexity": "simple/complex",
    "task_plan": [
        {"task_id": 1, "description": "子任务描述", "objective": "子任务目标", "dependencies": []},
        ...
    ]
}
"""
    
    # 调用模型，传入完整的最近对话历史
    response = llm.invoke([
        SystemMessage(content=prompt), 
        *recent_messages
    ]).content
    
    try:
        analysis = json.loads(response)
        
        # 需求澄清逻辑
        if analysis.get("need_clarification", False):
            question = analysis.get("clarification_question", "请补充更多任务细节")
            logger.info(f"[感知中枢] 需求不清晰，向用户提问：{question}")
            return {
                "planning": PlanningContext(
                    task_complexity="pending",
                    task_plan=[],
                    need_clarification=True,
                    clarification_question=question
                ),
                # 把模型的原始输出存入messages，方便调试和回溯
                "messages": [AIMessage(content=f"我需要你补充一些信息：{question}")]
            }
        
        # 解析任务计划
        complexity = analysis.get("task_complexity", "simple")
        raw_tasks = analysis.get("task_plan", [])
        
        # 🌟 简单任务强制合并
        if complexity == "simple" and len(raw_tasks) > 1:
            logger.warning("[护栏介入] 检测到大模型过度拆解简单任务，已强制合并为单步执行！")
            # 把大模型的目标拼接起来，防止信息丢失
            merged_objective = "；".join([t.get("objective", "") for t in raw_tasks])
            raw_tasks = [{
                "task_id": 1, 
                "description": "一次性编写并完成目标代码", 
                "objective": merged_objective, 
                "dependencies": []
            }]
            
        # 🌟 复杂任务限制上限（最多5个，避免死循环）
        elif len(raw_tasks) > 5:
            raw_tasks = raw_tasks[:5]
            logger.warning("[护栏介入] 子任务数量超过5个，已自动截断为前5个。")
        
        # 转换为SubTask对象，并校验依赖关系
        sub_tasks = []
        valid_task_ids = set()
        for task in raw_tasks:
            task_id = task.get("task_id", len(sub_tasks) + 1)
            valid_task_ids.add(task_id)
            
            # 校验依赖关系，移除不存在的依赖
            dependencies = task.get("dependencies", [])
            valid_dependencies = [d for d in dependencies if d in valid_task_ids]
            
            sub_tasks.append(
                SubTask(
                    task_id=task_id,
                    description=task.get("description", "未命名任务"),
                    objective=task.get("objective", ""),
                    dependencies=valid_dependencies
                )
            )
        
        # 打印结果
        if complexity == "simple":
            logger.info("[难度判定] 简单任务。已生成单步执行计划。")
        else:
            logger.info(f"[难度判定] 复杂系统任务！已拆解为 {len(sub_tasks)} 个子任务。")
            for t in sub_tasks:
                logger.info(f"  ➤ 步骤 {t.task_id}: {t.description}")
        
        # 更新规划上下文
        new_planning = PlanningContext(
            task_complexity=complexity,
            task_plan=sub_tasks,
            need_clarification=False,
            clarification_question=""
        )
        
        return {
            "planning": new_planning,
            "messages": [AIMessage(content=f"已为你生成{len(sub_tasks)}步执行计划")]
        }
        
    except json.JSONDecodeError:
        # 兜底防爆机制：解析失败自动降级为简单任务
        logger.warning("[解析失败] 模型未输出标准JSON，默认降级为单步简单任务。")
        fallback_task = SubTask(
            task_id=1, 
            description="执行用户指令", 
            objective=state["messages"][-1].content
        )
        return {
            "planning": PlanningContext(
                task_complexity="simple", 
                task_plan=[fallback_task],
                need_clarification=False,
                clarification_question=""
            ),
            "messages": [AIMessage(content="我将为你执行这个任务")]
        }
    
