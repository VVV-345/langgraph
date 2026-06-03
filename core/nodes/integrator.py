"""
=============================================================================
整合节点（Integrator Node）—— 汇总所有子任务结果 → 组装为结构化交付物
=============================================================================

【定位】
    这是 8 阶段流水线的第 7 站（整合阶段）。
    之前所有子任务的结果散落在 ExecutionContext.stage_outputs 里，
    这个节点的职责是把它们组装成一份结构化的交付物：
    - 从 ReAct 历史中提取 write_file 的 path 参数，建立「文件路径 → 代码」映射
    - 无明确路径的代码，由 LLM 推断文件名或默认命名
    - 多个子任务写同一文件时，智能合并（取最后一次写入，或 LLM 语义合并）
    - LLM 跨文件一致性审核（import 路径对齐、函数签名匹配）
    - 产出 IntegrationContext 填入 AgentState

【与上下阶段的协作】
    上游：sandbox 验证通过的全部 finished 子任务
    下游：output 节点负责将整合结果写入磁盘

【使用方式】
    from core.nodes.integrator import integrator_node
    # 在主图中:
    # workflow.add_node("integrator", integrator_node)
=============================================================================
"""

import os
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.state import AgentState, IntegratedFile, IntegrationContext
from core.logger import logger

load_dotenv()

# 整合专用 LLM（temperature 低，需要精确的结构化输出）
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),
    temperature=0.1,
    max_tokens=8192,
    model_kwargs={"response_format": {"type": "json_object"}},
    max_retries=5,
    timeout=90
)


# ==========================================================================
# 辅助函数
# ==========================================================================


def _group_code_by_file(
    stage_outputs: dict,  # Dict[int, dict] — {"files": {path: content}, "main": entry}
    react_history: list,
    task_plan: list,
) -> tuple:
    """
    将 stage_outputs 中的代码按文件路径分组。

    新格式：stage_outputs 的值已经是 {"files": {path: content}, "main": entry}，
    文件路径直接从数据中提取。同时兼容旧格式（单个代码字符串）。

    返回：(file_groups: dict, conflicts: list)
        file_groups = {"src/main.py": [(task_id, "code..."), ...], ...}
        conflicts = ["冲突描述1", ...]
    """
    file_groups: dict[str, list] = {}  # path → list of (task_id, content)
    conflicts: list[str] = []

    for task_id, output in stage_outputs.items():
        if isinstance(output, dict):
            files = output.get("files", {})
            main = output.get("main", "")
            if not files:
                continue
        elif isinstance(output, str):
            # 兼容旧格式：单个代码字符串
            if not output or not output.strip():
                continue
            # 尝试从 react_history 匹配路径（旧逻辑的简化版）
            matched_path = None
            for step in react_history:
                if isinstance(step, dict):
                    action = step.get("action", {})
                    if isinstance(action, dict) and action.get("tool_name") == "write_file":
                        hist_code = action.get("tool_input", {}).get("content", "")
                        if hist_code and hist_code[:500] == output[:500]:
                            matched_path = action.get("tool_input", {}).get("path", "")
                            break
                elif hasattr(step, "action") and step.action.tool_name == "write_file":
                    hist_code = step.action.tool_input.get("content", "")
                    if hist_code and hist_code[:500] == output[:500]:
                        matched_path = step.action.tool_input.get("path", "")
                        break
            if not matched_path:
                matched_path = f"module_{task_id}.py"
            files = {matched_path: output}
        else:
            continue

        for path, content in files.items():
            if not content or not content.strip():
                continue
            if path not in file_groups:
                file_groups[path] = []
            file_groups[path].append((task_id, content))

    # 冲突检测：同一路径有多个不同来源
    for path, entries in file_groups.items():
        if len(entries) > 1:
            task_ids = [e[0] for e in entries]
            conflicts.append(
                f"文件 [{path}] 被多个子任务写入：{task_ids}，"
                f"整合时取最后一次写入的内容"
            )

    return file_groups, conflicts


def _llm_merge_and_review(
    file_groups: dict,
    conflicts: list,
    user_request: str,
) -> dict:
    """
    调用 LLM 进行智能合并和跨文件一致性审核。

    返回 LLM JSON 解析后的 dict，包含 files、merge_summary、remaining_conflicts。
    """
    # 构建文件清单
    file_list = []
    for path, entries in file_groups.items():
        for task_id, code in entries:
            file_list.append({
                "path": path,
                "source_task": task_id,
                "content_preview": code[:8000],
                "content_length": len(code)
            })

    prompt = f"""你是一个资深软件架构师，正在进行代码整合审核。

【用户原始需求】
{user_request[:1000]}

【待整合的文件清单】
{json.dumps(file_list, ensure_ascii=False, indent=2)}

【已检测到的冲突】
{json.dumps(conflicts, ensure_ascii=False, indent=2) if conflicts else "无冲突"}

【你的任务】
1. 审核所有文件之间的接口是否对齐（import 路径是否正确、函数签名是否匹配）
2. 如果多个文件是同一项目的不同模块，确认它们能协同工作
3. 如果有冲突，判断是否可以自动合并；不能的标记出来
4. 生成 merge_summary（一段中文摘要，描述整合了哪些文件、各文件的作用）

【输出 JSON 格式】
{{
    "files": [
        {{
            "file_path": "src/main.py",
            "content": "完整代码（或从 content_preview 中已知的代码）",
            "source_tasks": [1]
        }}
    ],
    "merge_summary": "整合摘要（中文，100字以内）",
    "remaining_conflicts": ["无法自动解决的冲突（如有）"],
    "requires_regeneration": false
}}

如果发现跨文件接口严重不匹配、需要重新生成代码，将 requires_regeneration 设为 true，
并在 remaining_conflicts 中说明原因。"""

    response = llm.invoke([
        SystemMessage(content="你是一个严格的代码整合审核专家。只输出纯 JSON，不要任何解释文字。"),
        HumanMessage(content=prompt)
    ]).content

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        logger.warning("[整合] LLM 未输出标准 JSON，使用原始分组结果")
        # 兜底：直接把原始分组转成标准格式
        return {
            "files": [
                {
                    "file_path": path,
                    "content": entries[-1][1],  # 取最后一次写入
                    "source_tasks": [e[0] for e in entries]
                }
                for path, entries in file_groups.items()
            ],
            "merge_summary": f"共整合 {len(file_groups)} 个文件（兜底模式，未经 LLM 审核）",
            "remaining_conflicts": [],
            "requires_regeneration": False
        }


# ==========================================================================
# 主节点
# ==========================================================================

def integrator_node(state: AgentState):
    """
    智能整合节点：无冲突文件直接直通（绕过 LLM），有冲突文件交由 LLM 语义合并。

    核心设计：
        - 只有一个子任务写过的文件 → 100% 源码直通，不经过 LLM，零截断风险
        - 多个子任务写过的文件 → LLM 语义合并（此时才消耗 token）
        - LLM 跨文件一致性审核仅针对冲突文件

    返回值：
        更新后的 integration 上下文。
    """
    exec_box = state.get("execution")
    plan_box = state.get("planning")
    integration = state.get("integration", IntegrationContext())
    react_history = state.get("react_history", [])

    # 如果已经整合过，跳过
    if integration.integration_done:
        logger.info("[整合] 已整合过，跳过")
        return {}

    stage_outputs = exec_box.stage_outputs if exec_box else {}

    if not stage_outputs:
        logger.info("[整合] 无代码产出（纯查询/非代码任务），跳过整合")
        integration.integration_done = True
        integration.merge_summary = "无代码产出，无需整合"
        return {"integration": integration}

    logger.info(f"[整合] 开始分析 {len(stage_outputs)} 个代码产出...")

    # Step 1: 按文件路径分组
    task_plan = plan_box.task_plan if plan_box else []
    file_groups, conflicts = _group_code_by_file(stage_outputs, react_history, task_plan)

    # Step 2: 分离「无冲突文件」和「冲突文件」
    integrated_files = []
    files_need_llm_merge: dict[str, list] = {}

    for path, entries in file_groups.items():
        if len(entries) == 1:
            # ✅ 源码直通：只有一个子任务写过，直接取完整源码，零 token 消耗，零截断风险
            task_id, full_code = entries[0]
            integrated_files.append(IntegratedFile(
                file_path=path,
                content=full_code,
                source_tasks=[task_id]
            ))
            logger.info(f"[整合] ➔ [{path}] 无冲突，源码直通（{len(full_code)} 字符）")
        else:
            # ⚠️ 多任务冲突：暂存，交给 LLM 语义合并
            files_need_llm_merge[path] = entries
            logger.info(f"[整合] ⚠ [{path}] 被 {len(entries)} 个子任务修改，需 LLM 合并")

    # Step 3: 仅对冲突文件启用 LLM 审核
    if files_need_llm_merge:
        logger.info(f"[整合] 启动 LLM 语义合并（{len(files_need_llm_merge)} 个冲突文件）...")

        # 提取用户原始需求
        messages = state.get("messages", [])
        user_request = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                user_request = msg.content if hasattr(msg, "content") else str(msg)
                break
        if not user_request and messages:
            user_request = str(messages[0]) if not hasattr(messages[0], "content") else messages[0].content

        review = _llm_merge_and_review(files_need_llm_merge, conflicts, user_request)

        # 如果 LLM 要求重新生成
        if review.get("requires_regeneration"):
            logger.warning(f"[整合] ⚠️ LLM 合并发现严重接口冲突，打回重写")
            if plan_box and plan_box.task_plan:
                for t in plan_box.task_plan:
                    if t.status == "finished":
                        t.status = "pending"
                        t.result = f"整合审核未通过：{review.get('remaining_conflicts', [])}"
                        break
            integration.integration_done = False
            integration.conflicts = conflicts + review.get("remaining_conflicts", [])
            return {"integration": integration, "planning": plan_box}

        # 组装 LLM 合并后的文件
        for f in review.get("files", []):
            integrated_files.append(IntegratedFile(
                file_path=f.get("file_path", "unknown.py"),
                content=f.get("content", ""),
                source_tasks=f.get("source_tasks", [])
            ))

        all_conflicts = conflicts + review.get("remaining_conflicts", [])
        integration.conflicts = all_conflicts
        integration.merge_summary = review.get("merge_summary", f"冲突文件已合并（{len(files_need_llm_merge)} 个文件）")
    else:
        # 全部直通，皆大欢喜
        all_conflicts = conflicts
        integration.conflicts = all_conflicts
        integration.merge_summary = f"纯净直通：{len(integrated_files)} 个文件完整交付，无冲突"

    integration.files = integrated_files
    integration.integration_done = True

    logger.info(f"[整合] ✅ 完成：{len(integrated_files)} 个文件（{len(files_need_llm_merge)} 个经 LLM 合并），{len(all_conflicts)} 个冲突")
    if all_conflicts:
        for c in all_conflicts:
            logger.info(f"[整合]   ⚠ {c}")

    return {"integration": integration}
