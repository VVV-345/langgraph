# core/nodes/analyzer.py
import os
import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.state import AgentState, PlanningContext, SubTask
from langchain_openai import ChatOpenAI
from core.logger import logger
from core.memory.embedding import create_embedder
from core.memory.store import ExperienceStore

load_dotenv()  # 加载环境变量

# 云端模型
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),  # 免费模型，速度极快
    temperature=0.1,
    max_tokens=8192,
    # 模型的JSON输出参数
    model_kwargs={"response_format": {"type": "json_object"}},
    max_retries=8,
    timeout=60
)


# ==========================================================================
# RAG 历史经验检索（首次调用时连接 Qdrant）
# ==========================================================================

_rag_store: ExperienceStore = None


def _get_rag_store() -> ExperienceStore | None:
    """懒初始化 RAG 存储（复用 review 阶段的 store 单例逻辑）"""
    global _rag_store
    if _rag_store is not None:
        return _rag_store

    # 快速跳过：未配置任何 embedding 时不做检索，避免 create_embedder() 抛异常拖慢启动
    if not os.getenv("EMBEDDING_MODEL_PATH") and not os.getenv("EMBEDDING_BASE_URL") and not os.getenv("EMBEDDING_API_KEY"):
        logger.info("[感知-RAG] 未配置 Embedding 模型，跳过历史经验检索")
        _rag_store = None
        return None

    try:
        embeddings = create_embedder()
        _rag_store = ExperienceStore(embeddings)
        if _rag_store.connect():
            logger.info(f"[感知-RAG] 向量库已连接，共 {_rag_store.count()} 条历史经验")
            return _rag_store
        else:
            logger.info("[感知-RAG] Qdrant 不可用，本次分析不使用历史经验")
            _rag_store = None
            return None
    except Exception as e:
        logger.info(f"[感知-RAG] 初始化失败，不使用历史经验: {e}")
        _rag_store = None
        return None


def _build_rag_context(user_request: str) -> str:
    """
    检索历史相似经验，构建注入 prompt 的参考上下文。

    返回格式化的历史经验文本；如果检索不可用或无结果，返回空字符串。
    """
    store = _get_rag_store()
    if store is None:
        return ""

    try:
        docs = store.retrieve_similar(user_request, top_k=3)
    except Exception:
        return ""

    if not docs:
        logger.info("[感知-RAG] 无相似历史经验")
        return ""

    # 构建格式化的参考文本
    lines = [
        "【📚 历史参考经验——来自向量库的相似任务记录】",
        "以下是历史中与当前需求相似的任务及其执行结果，请参考其拆解策略和踩坑教训：",
        ""
    ]

    for i, doc in enumerate(docs):
        meta = doc.metadata
        sr = meta.get("execution_result", {}).get("success_rate", 0)
        status_icon = "✅" if sr >= 0.8 else "⚠️" if sr >= 0.5 else "❌"
        lines.append(f"--- {status_icon} 案例 {i + 1}（相似任务，成功率 {sr:.0%}）---")
        lines.append(f"需求：{meta.get('task_summary', 'N/A')[:200]}")
        lines.append(f"复杂度：{meta.get('task_complexity', '?')}")
        lines.append(f"子任务数：{meta.get('sub_task_count', '?')}")

        # 拆解策略
        plan = meta.get("planning_strategy", {})
        task_plan = plan.get("task_plan", [])
        if task_plan:
            lines.append("拆解方案：")
            for t in task_plan:
                risk = t.get("risk_level", "low")
                risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk, "⚪")
                lines.append(f"  {risk_icon} [{risk}] {t.get('description', '')}")

        # 踩坑
        pitfalls = meta.get("pitfalls", [])
        if pitfalls:
            lines.append(f"⚠️ 历史踩坑（{len(pitfalls)} 条），请避免：")
            for p in pitfalls[:3]:
                lines.append(f"  - {p.get('root_cause', 'N/A')[:150]}")
                if p.get("how_fixed"):
                    lines.append(f"    修复方式：{p.get('how_fixed', '')[:150]}")

        lines.append("")

    lines.append("【⚠️ 重要】参考以上经验，但不要照搬。根据当前需求的特点灵活调整。")
    lines.append("")

    context = "\n".join(lines)
    logger.info(f"[感知-RAG] 已检索 {len(docs)} 条历史经验，注入 prompt（{len(context)} 字符）")
    return context


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

【第三步：推理拆解理由】
用 reasoning 数组列出每个子任务的拆解理由，每个理由一句话，与 task_plan 一一对应。

【第四步：风险评估】
对每个子任务评估风险等级：
- low：纯计算/纯逻辑，不操作文件不联网
- medium：读写文件、执行普通命令
- high：涉及系统级操作、网络请求、数据删除、批量文件修改
并在 risk_reason 中简述原因。

【第五步：资源确认】
列出完成所有子任务需要哪些工具，从以下候选：
read_file / write_file / run_command / web_search

【第六步：识别 Python 依赖库】
分析所有子任务需要用到的 Python 第三方库（pip 包名），标注在 required_libraries 字段。
- 只列非标准库的第三方包（如 flask、requests、pandas、beautifulsoup4、Pillow）
- 标准库（os、json、sqlite3、pathlib、csv、datetime 等）不要列
- 数据库驱动注意区分：sqlite3 内置不列；mysqlclient、psycopg2、pymongo 需要列
- 如果任务纯粹只用标准库，返回空数组

【第七步：生成任务计划】
- 简单任务：只生成1个子任务
- 复杂任务：拆分为2-4个子任务（最多4个，绝对不能超过）
- 每个子任务必须独立、可落地、有明确的【代码编写】目标。
- 每个子任务可以产出多个文件（如同时写 models.py + init_db.py + app.py），用 write_file 分别写入各文件
- dependencies字段填写依赖的子任务ID，没有依赖则为空数组
- ⚠️【核心禁令】：严禁生成任何专门用于“测试”、“运行”、“验证”、“整合审查”的子任务！所有的代码运行测试将由后续的自动化【沙盒节点】完成。你的拆解只能包含实质性的代码编写和逻辑开发！

【输出格式】
如果需求清晰，输出：
{
    "need_clarification": false,
    "task_complexity": "simple/complex",
    "reasoning": ["拆解理由1", "拆解理由2"],
    "required_resources": ["write_file", "run_command"],
    "required_libraries": ["flask", "requests"],
    "task_plan": [
        {
            "task_id": 1,
            "description": "子任务描述",
            "objective": "子任务目标",
            "dependencies": [],
            "risk_level": "low/medium/high",
            "risk_reason": "风险评估原因"
        }
    ]
}
"""

    # 调用模型，传入完整的最近对话历史
    # ---- RAG 检索：从向量库获取相似历史经验 ----
    user_request = ""
    for msg in recent_messages:
        if hasattr(msg, "type") and msg.type == "human":
            user_request = msg.content if hasattr(msg, "content") else str(msg)
            break
    if not user_request and recent_messages:
        user_request = recent_messages[-1].content if hasattr(recent_messages[-1], "content") else str(recent_messages[-1])

    rag_context = _build_rag_context(user_request) if user_request else ""
    if rag_context:
        prompt = rag_context + "\n" + prompt

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
        
        # 提取思考链
        reasoning = analysis.get("reasoning", [])
        if not isinstance(reasoning, list):
            reasoning = [str(reasoning)]

        # 提取资源需求
        required_resources = analysis.get("required_resources", [])
        if not isinstance(required_resources, list):
            required_resources = []

        # 提取 Python 依赖库
        required_libraries = analysis.get("required_libraries", [])
        if not isinstance(required_libraries, list):
            required_libraries = []
        # 过滤明显不是 pip 包名的条目
        required_libraries = [
            lib for lib in required_libraries
            if isinstance(lib, str) and len(lib) > 0 and not lib.startswith(("#", "//"))
        ]

        # 转换为SubTask对象，并校验依赖关系
        sub_tasks = []
        valid_task_ids = set()
        for task in raw_tasks:
            task_id = task.get("task_id", len(sub_tasks) + 1)
            valid_task_ids.add(task_id)

            # 校验依赖关系，移除不存在的依赖
            dependencies = task.get("dependencies", [])
            valid_dependencies = [d for d in dependencies if d in valid_task_ids]

            # 校验风险等级
            risk_level = task.get("risk_level", "low")
            if risk_level not in ("low", "medium", "high"):
                risk_level = "low"

            sub_tasks.append(
                SubTask(
                    task_id=task_id,
                    description=task.get("description", "未命名任务"),
                    objective=task.get("objective", ""),
                    dependencies=valid_dependencies,
                    risk_level=risk_level,
                    risk_reason=task.get("risk_reason", "")
                )
            )
        
        # 打印结果
        if complexity == "simple":
            logger.info("[难度判定] 简单任务。已生成单步执行计划。")
        else:
            logger.info(f"[难度判定] 复杂系统任务！已拆解为 {len(sub_tasks)} 个子任务。")
            for t in sub_tasks:
                risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(t.risk_level, "⚪")
                logger.info(f"  ➤ 步骤 {t.task_id}: {t.description} {risk_icon}[{t.risk_level}]")
        if reasoning:
            logger.debug(f"[思考链] {reasoning}")
        if required_resources:
            logger.info(f"[资源需求] {', '.join(required_resources)}")
        if required_libraries:
            logger.info(f"[依赖库] {', '.join(required_libraries)}")
        
        # 更新规划上下文
        new_planning = PlanningContext(
            task_complexity=complexity,
            task_plan=sub_tasks,
            thinking_chain=reasoning,
            need_clarification=False,
            clarification_question="",
            required_resources=required_resources,
            required_libraries=required_libraries
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
            objective=state["messages"][-1].content,
            risk_level="low",
            risk_reason="兜底降级任务"
        )
        return {
            "planning": PlanningContext(
                task_complexity="simple",
                task_plan=[fallback_task],
                thinking_chain=["JSON解析失败，自动降级为简单单步任务"],
                need_clarification=False,
                clarification_question="",
                required_resources=[]
            ),
            "messages": [AIMessage(content="我将为你执行这个任务")]
        }
    
