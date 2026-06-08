"""
=============================================================================
输出节点（Output Node）—— 生成最终交付物 + 写入磁盘 + 反馈收集
=============================================================================

【定位】
    这是 8 阶段流水线的第 8 站（输出阶段）。
    整合阶段的产物（IntegrationContext）是结构化的文件列表，
    这个节点负责把它们真正写入磁盘，并生成配套文档。

【职责】
    1. 创建输出目录 output/<项目名_时间戳>/
    2. 逐个写入代码文件
    3. 调用 LLM 生成 README.md（项目说明、依赖、运行方式）
    4. 生成 deliverable_manifest.json（交付物清单）
    5. 更新 OutputContext 记录写入结果

【与上下阶段的协作】
    上游：integrator 节点产出的 IntegrationContext
    下游：用户反馈（可选，目前先实现单向输出）

【使用方式】
    from core.nodes.output import output_node
    # 在主图中:
    # workflow.add_node("output", output_node)
=============================================================================
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from core.state import AgentState, OutputContext
from core.logger import logger

load_dotenv()

# 输出专用 LLM
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("PROCESS_MODEL"),
    temperature=0.2,
    max_tokens=8192,
    model_kwargs={"response_format": {"type": "json_object"}},
    max_retries=5,
    timeout=90
)

# 输出根目录（相对于项目根目录）
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "output")


def _generate_readme(files: list, user_request: str, merge_summary: str) -> str:
    """
    调用 LLM 为交付物生成 README.md。

    参数：
        files: List[IntegratedFile] — 交付文件列表
        user_request: str — 用户原始需求
        merge_summary: str — 整合摘要

    返回：
        README.md 的完整 markdown 文本。
    """
    file_summaries = []
    for f in files:
        file_summaries.append({
            "path": f.file_path,
            "size": len(f.content),
            "preview": f.content[:500]
        })

    prompt = f"""你是一个技术文档撰写专家。请为以下项目生成一个 README.md 文档。

【用户原始需求】
{user_request[:1500]}

【整合摘要】
{merge_summary}

【交付文件列表】
{json.dumps(file_summaries, ensure_ascii=False, indent=2)}

【README 要求】
1. 包含项目名称（从需求中推断）
2. 功能简介（2-3 句话）
3. 文件结构说明
4. 环境依赖（从代码中推断用了哪些库）
5. 运行方式（如何启动/使用）
6. 使用示例（如有）

请输出 JSON：
{{
    "project_name": "项目名称",
    "readme": "完整的 README.md 内容（markdown 格式）"
}}"""

    response = llm.invoke([
        SystemMessage(content="你是一个专业的技术文档撰写专家。按 JSON 格式输出。"),
        HumanMessage(content=prompt)
    ]).content

    try:
        data = json.loads(response)
        readme = data.get("readme", "")
        project_name = data.get("project_name", "未命名项目")
        logger.info(f"[输出] 已生成 README，项目名：{project_name}")
        return readme, project_name
    except json.JSONDecodeError:
        logger.warning("[输出] README 生成 JSON 解析失败，使用兜底模板")
        return _fallback_readme(files, user_request), "未命名项目"


def _fallback_readme(files: list, user_request: str) -> str:
    """兜底 README 模板（无需 LLM）"""
    file_list = "\n".join(f"- `{f.file_path}`" for f in files)
    return f"""# 项目交付物

## 需求
{user_request[:500]}

## 文件结构
{file_list}

## 运行方式
```bash
# 根据项目类型选择对应命令
python main.py
```

## 生成信息
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 由 AI Agent 自动生成
"""


def _generate_manifest(
    output_dir: str,
    files: list,
    readme_path: str,
    merge_summary: str,
    project_name: str,
) -> str:
    """
    生成 deliverable_manifest.json 交付物清单。

    返回 manifest 文件路径。
    """
    manifest = {
        "project_name": project_name,
        "generated_at": datetime.now().isoformat(),
        "output_directory": output_dir,
        "merge_summary": merge_summary,
        "files": [
            {
                "path": f.file_path,
                "size_bytes": len(f.content),
                "source_tasks": f.source_tasks
            }
            for f in files
        ],
        "readme": "README.md"
    }

    manifest_path = os.path.join(output_dir, "deliverable_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    logger.info(f"[输出] 清单已写入：{manifest_path}")
    return manifest_path


# ==========================================================================
# 主节点
# ==========================================================================

def output_node(state: AgentState):
    """
    输出节点：将整合后的交付物写入磁盘。

    处理流程：
        1. 从 IntegrationContext 读取文件列表
        2. 创建输出目录
        3. 逐个写入代码文件
        4. 生成 README.md
        5. 生成 deliverable_manifest.json
        6. 更新 OutputContext

    返回值：
        更新后的 output 上下文。
    """
    integration = state.get("integration")
    output_ctx = state.get("output", OutputContext())
    plan_box = state.get("planning")

    # 如果已经输出过，跳过
    if output_ctx.output_done:
        logger.info("[输出] 已输出过，跳过")
        return {}

    if integration is None or not integration.files:
        logger.info("[输出] 无可交付文件，跳过输出")
        output_ctx.output_done = True
        output_ctx.output_dir = ""
        return {"output": output_ctx}

    files = integration.files

    # Step 1: 获取用户原始需求（用于命名和 README）
    messages = state.get("messages", [])
    user_request = ""
    for msg in messages:
        if hasattr(msg, "type") and msg.type == "human":
            user_request = msg.content if hasattr(msg, "content") else str(msg)
            break
    if not user_request and messages:
        user_request = str(messages[0]) if not hasattr(messages[0], "content") else messages[0].content

    # Step 2: 生成项目名和时间戳
    readme_content, project_name = _generate_readme(
        files, user_request, integration.merge_summary
    )

    # 安全文件名
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in project_name)
    safe_name = safe_name.strip()[:40].replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{safe_name}_{timestamp}" if safe_name else f"output_{timestamp}"
    output_dir = os.path.abspath(os.path.join(OUTPUT_ROOT, dir_name))

    # Step 3: 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[输出] 输出目录：{output_dir}")

    # Step 4: 写入代码文件（逐文件 try/except，单文件失败不影响其他文件）
    files_written = []
    failed_files = []
    for f in files:
        # 确保文件路径安全（防止路径穿越）
        safe_path = f.file_path.replace("\\", "/").lstrip("/")
        if ".." in safe_path:
            safe_path = os.path.basename(safe_path)
            logger.warning(f"[输出] 检测到路径穿越尝试，已截断：{f.file_path} → {safe_path}")

        try:
            full_path = os.path.join(output_dir, safe_path)
            # 创建子目录（如有）
            parent_dir = os.path.dirname(full_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            with open(full_path, "w", encoding="utf-8") as fh:
                fh.write(f.content)

            files_written.append(full_path)
            logger.info(f"[输出] ✅ 已写入：{safe_path}（{len(f.content)} 字符）")
        except OSError as e:
            failed_files.append(safe_path)
            logger.error(f"[输出] ❌ 写入失败 [{safe_path}]: {e}，跳过并继续")
        except Exception as e:
            failed_files.append(safe_path)
            logger.error(f"[输出] ❌ 写入异常 [{safe_path}]: {e}，跳过并继续")

    if failed_files:
        logger.warning(f"[输出] {len(failed_files)} 个文件写入失败: {', '.join(failed_files)}")

    # Step 5: 如果有失败任务，写入 FAILURE_REPORT.md
    if plan_box:
        failed_tasks = [t for t in plan_box.task_plan if t.status == "failed"]
        if failed_tasks:
            try:
                report_path = os.path.join(output_dir, "FAILURE_REPORT.md")
                report_lines = [
                    "# ❌ 失败任务报告",
                    "",
                    f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "| 子任务 | 描述 | 失败原因 |",
                    "|--------|------|----------|",
                ]
                for t in failed_tasks:
                    reason = (t.result or "未知错误").replace("\n", "<br>").replace("|", "\\|")
                    report_lines.append(f"| {t.task_id} | {t.description[:80]} | {reason[:500]} |")
                report_lines.append("")
                report_lines.append("---")
                report_lines.append(f"*共 {len(failed_tasks)} 个子任务失败，请根据以上报错修改需求或代码后重试。*")
                with open(report_path, "w", encoding="utf-8") as rf:
                    rf.write("\n".join(report_lines))
                files_written.append(report_path)
                logger.info(f"[输出] ⚠️ 已写入失败报告：FAILURE_REPORT.md（{len(failed_tasks)} 个失败任务）")
            except Exception as e:
                logger.error(f"[输出] 写入 FAILURE_REPORT.md 失败: {e}")

    # Step 6: 写入 README.md
    readme_path = os.path.join(output_dir, "README.md")
    try:
        with open(readme_path, "w", encoding="utf-8") as rh:
            rh.write(readme_content)
        files_written.append(readme_path)
        logger.info(f"[输出] ✅ README 已写入：README.md")
    except Exception as e:
        logger.error(f"[输出] 写入 README.md 失败: {e}")
        readme_path = ""

    # Step 7: 生成交付物清单
    try:
        manifest_path = _generate_manifest(
            output_dir, files, readme_path,
            integration.merge_summary, project_name
        )
        files_written.append(manifest_path)
    except Exception as e:
        logger.error(f"[输出] 生成交付物清单失败: {e}")
        manifest_path = ""

    # Step 8: 更新 latest 指针
    latest_pointer = os.path.join(OUTPUT_ROOT, "latest.txt")
    try:
        with open(latest_pointer, "w", encoding="utf-8") as lp:
            lp.write(dir_name)
    except Exception:
        pass  # latest pointer 不是关键路径

    # Step 9: 填充 OutputContext
    output_ctx.output_dir = output_dir
    output_ctx.manifest_path = manifest_path
    output_ctx.readme_path = readme_path
    output_ctx.files_written = files_written
    output_ctx.output_done = True

    logger.info(f"[输出] 🎉 交付完成！共 {len(files_written)} 个文件 → {output_dir}")

    return {"output": output_ctx}
