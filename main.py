#!/usr/bin/env python3
"""
=============================================================================
main.py —— 自主 AI 编码代理入口
=============================================================================

8 阶段流水线:
  感知(RAG检索) → 规划 → 调度 → 执行(ReAct) → 验证(沙盒) → 整合 → 输出 → 复盘(向量库)

使用方式:
  python main.py                              # 交互模式，输入任务
  python main.py -r "写一个批量重命名脚本"       # 直接传入任务
  python main.py -f task.txt                  # 从文件读取任务需求
  python main.py --resume                     # 断点续传——从上次中断处继续
  python main.py -r "..." -o ./my_output       # 指定输出目录
  python main.py -v                           # 显示详细日志
=============================================================================
"""

import argparse
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

from core.graph.main import build_master_graph
from core.state import PlanningContext, ExecutionContext, IntegrationContext, OutputContext
from core.logger import logger

# 断点续传状态文件
CHECKPOINT_STATE_FILE = "output/.pipeline_state.json"


# ==========================================================================
# CLI 参数解析
# ==========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="自主 AI 编码代理 —— 基于 LangGraph 的 8 阶段流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py
  python main.py -r "用 Python 写一个学生成绩管理系统"
  python main.py -r "写一个网页爬虫" -o ./output
  python main.py -f task.txt
  python main.py -v -r "写一个冒泡排序"
  python main.py --resume
        """,
    )
    parser.add_argument(
        "-r", "--request",
        type=str,
        default=None,
        help="任务需求描述（非交互模式，不传则进入交互输入）",
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="从文本文件读取任务需求",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="覆盖默认输出目录（默认: output/<项目名>_<时间戳>/）",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示 DEBUG 级别日志",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上一次中断处继续执行（断点续传）",
    )
    return parser.parse_args()


# ==========================================================================
# 任务获取
# ==========================================================================

DEFAULT_DEMO_TASK = (
    "用 Python 写一个学生成绩管理系统："
    "包含数据录入、成绩查询、统计分析三个模块"
)


def get_user_request(args: argparse.Namespace) -> str:
    """
    获取用户任务需求。
    优先级: -r 参数 > -f 文件 > 交互输入
    """
    # 方式 1: -r 命令行参数
    if args.request:
        return args.request.strip()

    # 方式 2: -f 从文件读取
    if args.file:
        path = Path(args.file)
        if not path.exists():
            logger.error(f"文件不存在: {args.file}")
            sys.exit(1)
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            logger.error(f"文件内容为空: {args.file}")
            sys.exit(1)
        logger.info(f"从文件读取任务需求: {args.file} ({len(content)} 字符)")
        return content

    # 方式 3: 交互输入
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       自主 AI 编码代理 —— 8 阶段流水线                        ║")
    print("║  感知 → 规划 → 调度 → 执行 → 验证 → 整合 → 输出 → 复盘       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    print("💡 提示: 也可以直接 python main.py -r \"你的任务\" 跳过交互")
    print()

    user_input = input("🙋 请输入你的任务需求: ").strip()
    if not user_input:
        user_input = DEFAULT_DEMO_TASK
        print(f"📝 使用默认演示任务:\n   {user_input}")
        print()
    return user_input


# ==========================================================================
# 输出展示
# ==========================================================================

def print_divider(title: str = "", width: int = 70) -> None:
    """打印分隔线，可选带标题"""
    c = "─"
    if title:
        side = max((width - len(title) - 2) // 2, 1)
        print(f"\n{c * side} {title} {c * side}")
    else:
        print(c * width)


def save_checkpoint(state: dict, thread_id: str) -> None:
    """保存管道状态快照到 JSON 文件，供断点续传使用。"""
    planning = state.get("planning")
    execution = state.get("execution")
    integration = state.get("integration")
    output_ctx = state.get("output")

    snapshot = {
        "thread_id": thread_id,
        "planning": planning.model_dump() if planning else {},
        "execution": execution.model_dump() if execution else {},
        "integration": integration.model_dump() if integration else {},
        "output": output_ctx.model_dump() if output_ctx else {},
        "messages": [
            {"role": "user" if m.type == "human" else "assistant", "content": str(m.content)}
            for m in state.get("messages", [])
        ],
    }

    os.makedirs("output", exist_ok=True)
    with open(CHECKPOINT_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"[断点] 状态已保存 → {CHECKPOINT_STATE_FILE}")


def load_checkpoint() -> tuple[dict, str]:
    """
    从 JSON 文件加载管道状态快照。

    失败任务会被重置为 pending（断点续传），
    已完成的任务保持 finished 状态不变。

    返回 (initial_state_dict, thread_id)
    """
    if not os.path.exists(CHECKPOINT_STATE_FILE):
        raise FileNotFoundError(f"断点文件不存在: {CHECKPOINT_STATE_FILE}")

    with open(CHECKPOINT_STATE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    thread_id = data.get("thread_id", "")

    # ── 重置失败任务为 pending，已完成任务保持不动 ──
    task_plan = data.get("planning", {}).get("task_plan", [])
    for t in task_plan:
        if t["status"] in ("failed",):
            t["status"] = "pending"
            t["result"] = f"[续传] 上次失败: {t.get('result', '')}"

    # ── 重建 Pydantic 对象 ──
    planning = PlanningContext(**data.get("planning", {}))
    execution = ExecutionContext(**data.get("execution", {}))
    # 重置重试计数和报错（新的一轮）
    execution.retry_count = 0
    execution.error_trace = ""
    execution.all_tasks_completed = False
    # 清理上一轮的 transient 状态（analyzer 每轮会重新生成）
    planning.need_clarification = False
    planning.clarification_question = ""

    # ── 重建消息 ──
    messages = []
    for m in data.get("messages", []):
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return {
        "messages": messages,
        "planning": planning,
        "execution": execution,
        "integration": IntegrationContext(),
        "output": OutputContext(),
    }, thread_id


def print_summary(final_state: dict) -> None:
    """打印精简的执行摘要"""
    planning = final_state.get("planning")
    execution = final_state.get("execution")
    output_ctx = final_state.get("output")
    review = final_state.get("_review_summary")

    print_divider("执行摘要", 60)

    # ── 任务概况 ──
    if planning:
        total = len(planning.task_plan)
        finished = sum(1 for t in planning.task_plan if t.status == "finished")
        failed = sum(1 for t in planning.task_plan if t.status == "failed")
        complexity = planning.task_complexity
        print(f"  复杂度: {complexity}  |  子任务: {total}  |  ✅ {finished} 通过  |  ❌ {failed} 失败")

        # ── 失败任务详情 ──
        if failed > 0:
            print()
            print("  ══════════════════════════════════════")
            print("  ❌ 失败任务详情：")
            print("  ──────────────────────────────────────")
            for t in planning.task_plan:
                if t.status == "failed":
                    reason = t.result or "(无错误信息)"
                    # 截断过长的报错，保留关键信息
                    if len(reason) > 600:
                        reason = reason[:600] + "\n  ... (截断，完整报错见输出目录下的 FAILURE_REPORT.md)"
                    # 缩进处理
                    indented = "\n  ".join(reason.split("\n"))
                    print(f"  📌 子任务 {t.task_id}：{t.description[:80]}")
                    print(f"     原因：{indented}")
                    print()

    # ── 重试 ──
    if execution:
        if execution.retry_count > 0:
            print(f"  重试次数: {execution.retry_count}")

    # ── 输出文件 ──
    if output_ctx and output_ctx.output_done:
        print(f"  输出目录: {output_ctx.output_dir}")
        files = output_ctx.files_written
        if files:
            if len(files) <= 8:
                for f in files:
                    print(f"    ✅ {f}")
            else:
                for f in files[:5]:
                    print(f"    ✅ {f}")
                print(f"    ... 共 {len(files)} 个文件")

    # ── 向量库 ──
    if review:
        sr = review.get("success_rate", 0)
        pit = review.get("pitfall_count", 0)
        total_exp = review.get("total_experiences_in_db", 0)
        print(f"  经验归档: 成功率 {sr:.0%}  |  踩坑 {pit} 条  |  向量库累计 {total_exp} 条")

    print_divider(width=60)
    print()


# ==========================================================================
# 主函数
# ==========================================================================

def main() -> None:
    args = parse_args()

    # 日志级别
    if args.verbose:
        import logging
        logger.setLevel(logging.DEBUG)

    # ── 初始化 Docker 沙盒容器（续传和全新都需要）──
    from core.tools.docker_sandbox import init_container, docker_exec
    if not init_container():
        logger.error("Docker 容器初始化失败，请确认 Docker Desktop 已启动")
        sys.exit(1)

    # ── 全新任务：清空工作区，避免旧文件干扰 LLM 判断 ──
    #    断点续传：保留工作区，LLM 需要看到上次的代码继续工作
    if not args.resume:
        docker_exec("rm -rf /workspace/* /workspace/.[!.]* 2>/dev/null; echo ok")
        logger.info("[沙盒] 新任务——已清空工作区")
    else:
        logger.info("[沙盒] 断点续传——保留工作区现场")

    # ── 构建图谱（共享一个 SqliteSaver 实例，避免重复创建连接）──
    logger.info("正在初始化 8 阶段流水线...")
    os.makedirs("output", exist_ok=True)
    db_conn = sqlite3.connect("output/checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(db_conn)
    app = build_master_graph(checkpointer=checkpointer)

    # ── 初始状态：--resume 从断点加载，否则走正常流程 ──
    if args.resume:
        try:
            initial_state, saved_thread_id = load_checkpoint()
        except FileNotFoundError:
            logger.error(f"断点文件不存在: {CHECKPOINT_STATE_FILE}")
            logger.error("请先正常执行一次管道，或检查 output 目录是否存在")
            sys.exit(1)

        config = {"configurable": {"thread_id": saved_thread_id}}
        print_divider("从断点恢复", 60)

        # 显示恢复摘要
        planning = initial_state.get("planning")
        if planning:
            finished = sum(1 for t in planning.task_plan if t.status == "finished")
            pending = sum(1 for t in planning.task_plan if t.status == "pending")
            logger.info(f"恢复状态：{finished} 个已完成，{pending} 个待重试")
        logger.info(f"使用 thread_id: {saved_thread_id}")

        if args.output_dir:
            initial_state["output_dir_override"] = args.output_dir
            logger.info(f"输出目录覆盖: {args.output_dir}")
    else:
        user_request = get_user_request(args)

        print_divider("流水线启动", 60)
        logger.info(f"用户需求: {user_request[:100]}{'...' if len(user_request) > 100 else ''}")

        initial_state = {"messages": [HumanMessage(content=user_request)]}
        if args.output_dir:
            initial_state["output_dir_override"] = args.output_dir
            logger.info(f"输出目录覆盖: {args.output_dir}")

        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # ── 执行流水线（支持 ReAct interrupt 恢复）──
    logger.info("流水线开始执行...")
    state = initial_state
    final_state = None
    clarify_count = 0  # 澄清轮次计数器，防止无限追问

    try:
        while True:
            try:
                state = app.invoke(state, {**config, "recursion_limit": 350})

                # ── 需求不清晰，向用户收集补充信息后重跑 ──
                planning = state.get("planning")
                if planning and planning.need_clarification:
                    clarify_count += 1

                    if clarify_count > 4:
                        # 已追问 4 轮，强制进入执行——在用户回答后追加明确指令
                        logger.warning(
                            f"[流水线] 已澄清 {clarify_count} 轮，本轮将强制要求模型停止追问"
                        )

                    print(f"\n🤔 {planning.clarification_question}")
                    answer = input("👉 请补充: ").strip()
                    if not answer:
                        logger.info("用户放弃补充信息，退出")
                        final_state = state
                        break

                    # 把用户补充信息追加到对话历史
                    # 超过 4 轮时在消息中强制要求不再追问
                    if clarify_count >= 4:
                        enriched = (
                            f"[系统指令] 已追问 {clarify_count} 轮，信息足够。"
                            f"请基于以下补充直接生成执行计划，绝对不要再提问。\n\n"
                            f"用户补充：{answer}"
                        )
                    else:
                        enriched = answer

                    state["messages"] = list(state.get("messages", [])) + [
                        HumanMessage(content=enriched)
                    ]
                    logger.info(f"已收集用户补充信息（第 {clarify_count} 轮），重新分析需求...")
                    try:
                        save_checkpoint(state, config["configurable"]["thread_id"])
                    except Exception:
                        pass
                    continue  # 重跑流水线

                # ── ReAct 阻塞（state flag 传播），人工介入 ──
                if state.get("react_blocked", False):
                    msg = state.get("react_block_reason", "ReAct 循环卡住")
                    plan_box = state.get("planning")

                    print_divider("⚠️  ReAct 循环卡住——需要人工决策", 60)
                    print(f"  {msg}")
                    print()
                    print("  选项:")
                    print("    [回车] 继续执行（重置轮数，再来 10 轮）")
                    print("    跳过      跳过此子任务")
                    print("    强制提交   代码已完整，直接结束并跳过沙盒验证")
                    print("    修改xxx    输入新的执行方向（如：修改 直接用 write_file 写代码，不要探索环境）")
                    print()

                    user_decision = input("👉 你的决策: ").strip()
                    logger.info(f"用户决策: {user_decision or '继续执行'}")

                    decision_str = user_decision.lower().strip() if user_decision else ""

                    # 找到阻塞的子任务（第一个 pending + result 含"等待人工介入"的）
                    stuck_task = None
                    if plan_box:
                        for t in plan_box.task_plan:
                            if t.status == "pending" and "等待人工介入" in (t.result or ""):
                                stuck_task = t
                                break

                    if decision_str == "skip" or "跳过" in decision_str:
                        if stuck_task:
                            stuck_task.status = "failed"
                            stuck_task.result = "用户跳过"
                            logger.info(f"[主循环] 跳过子任务 {stuck_task.task_id}")
                    elif "强制提交" in user_decision:
                        state["force_submit"] = True
                        if stuck_task:
                            logger.info(f"[主循环] 强制提交子任务 {stuck_task.task_id}，将跳过沙盒验证")
                    elif decision_str.startswith("edit") or "修改" in decision_str:
                        if stuck_task:
                            new_inst = decision_str[5:] if decision_str.startswith("edit ") else user_decision
                            stuck_task.description = f"{stuck_task.description}\n[用户补充指示] {new_inst}"
                            stuck_task.result = f"用户修改需求：{user_decision}"
                            logger.info(f"[主循环] 修改子任务 {stuck_task.task_id} 方向")
                    else:
                        # 继续执行：重置轮数，清阻塞
                        logger.info("[主循环] 用户选择继续执行，重置 ReAct 状态")

                    # 清理阻塞状态，重新进入 executor
                    state["react_blocked"] = False
                    state["react_block_reason"] = ""
                    state["react_round"] = 0
                    state["react_finished"] = False

                    try:
                        save_checkpoint(state, config["configurable"]["thread_id"])
                    except Exception:
                        pass
                    continue  # 重入流水线

                # ── 正常结束：保存断点 ──
                save_checkpoint(state, config["configurable"]["thread_id"])
                final_state = state
                break  # 正常结束
            except GraphInterrupt as e:
                # ── ReAct interrupt() 直接抛异常（兜底路径）──
                interrupt_values = e.args[0] if e.args else []
                msg = interrupt_values[0] if interrupt_values else "ReAct 循环卡住"

                print_divider("⚠️  ReAct 循环卡住——需要人工决策", 60)
                print(f"  {msg}")
                print()
                print("  选项:")
                print("    [回车] 继续执行（重置轮数，再来 10 轮）")
                print("    跳过      跳过此子任务")
                print("    强制提交   代码已完整，直接结束并跳过沙盒验证")
                print("    修改xxx    输入新的执行方向（如：修改 直接用 write_file 写代码，不要探索环境）")
                print()

                user_decision = input("👉 你的决策: ").strip()
                logger.info(f"用户决策: {user_decision or '继续执行'}")

                # Command(resume=...) 让 graph 从 interrupt() 处恢复
                if "强制提交" in user_decision:
                    state = Command(
                        resume=user_decision,
                        update={"force_submit": True, "react_blocked": False, "react_round": 0}
                    )
                else:
                    state = Command(resume=user_decision)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断 (Ctrl+C)")
        sys.exit(130)
    except Exception:
        # 异常退出前保存断点，方便恢复
        try:
            if state is not None and not isinstance(state, Command):
                save_checkpoint(state, config["configurable"]["thread_id"])
        except Exception:
            pass
        logger.exception("流水线执行异常")
        sys.exit(1)

    if final_state is None:
        logger.error("流水线未能产生最终状态")
        sys.exit(1)

    # ── 结果输出 ──
    print_summary(final_state)

    # ── 交付物清单 ──
    integration = final_state.get("integration")
    if integration and integration.files:
        print("📦 最终交付物:")
        for f in integration.files:
            size_kb = len(f.content) / 1024
            src = f"← 子任务 {f.source_tasks}" if f.source_tasks else ""
            print(f"   📄 {f.file_path}  ({size_kb:.1f} KB)  {src}")
        print()

    # ── 冲突报告 ──
    if integration and integration.conflicts:
        print(f"⚠️ 整合冲突 ({len(integration.conflicts)} 项):")
        for c in integration.conflicts:
            print(f"   - {c}")
        print()

    # ── 退出码 ──
    planning = final_state.get("planning")
    if planning:
        total = len(planning.task_plan)
        failed = sum(1 for t in planning.task_plan if t.status == "failed")
        if total > 0 and failed == total:
            logger.error(f"所有 {total} 个子任务均失败")
            sys.exit(1)

    logger.info("流水线完成")
    sys.exit(0)


if __name__ == "__main__":
    main()
