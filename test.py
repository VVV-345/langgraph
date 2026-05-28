# test.py —— 完整流水线集成测试入口
# 使用核心图谱主图，体验"编码 → 沙盒验证 → 修复"完整流程

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from core.graph.main import build_master_graph

load_dotenv()


def main():
    print("🚀 启动 Agent 完整流水线集成测试...")
    print("=" * 60)

    # 从 core.graph.main 获取已编译的主图谱
    app = build_master_graph()

    # 测试场景提示
    print("\n💡 建议测试以下三种场景：")
    print('  1. 模糊需求: "帮我写个程序"')
    print('  2. 简单任务: "帮我写一个批量重命名当前目录文件的 Python 脚本"')
    print('  3. 复杂任务: "用 Python 写一个 Web 爬虫系统：包含网页抓取、数据清洗、存储到 SQLite"')

    user_input = input("\n🙋‍♂️ 请输入你的任务需求: ").strip()
    if not user_input:
        user_input = "用 Python 写一个学生成绩管理系统：包含数据录入、成绩查询、统计分析三个模块"
        print(f"检测到输入为空，自动使用默认测试任务：\n  → {user_input}")

    # 构建初始状态——只传入用户消息，其余字段使用默认值
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    # 启动主图谱流水线
    # recursion_limit=100 以容纳：1 次分析 + N 次编码 + N 次沙盒验证 + 重试循环
    print("\n" + "=" * 60)
    print("⚙️  系统开始流转...")
    print("=" * 60 + "\n")

    final_state = app.invoke(initial_state, {"recursion_limit": 100})

    print("\n" + "=" * 60)
    print("⚙️  系统流转结束")
    print("=" * 60 + "\n")

    # ==========================================
    # 结果汇总输出
    # ==========================================
    planning_box = final_state.get("planning")
    exec_box = final_state.get("execution")

    # 对话历史回放
    print("📝 【对话历史】")
    messages = final_state.get("messages", [])
    for i, msg in enumerate(messages):
        role = "👤 用户" if hasattr(msg, "type") and msg.type == "human" else "🤖 Agent"
        content = msg.content if hasattr(msg, "content") else str(msg)
        if len(content) > 200:
            content = content[:200] + "...(截断)"
        print(f"  [{i}] {role}: {content}")
    print()

    # 规划收纳盒
    if planning_box is None:
        print("❌ 错误：planning 盒子为空！")
        return

    print("📋 【规划收纳盒 (PlanningContext)】")
    print(f" ├─ 需求是否需要澄清？  : {planning_box.need_clarification}")
    if planning_box.need_clarification:
        print(f" ├─ 项目经理的反问      : {planning_box.clarification_question}")
        print(" └─ ⚠️ 流程已阻断，请根据上方问题补充需求后重新运行。")
        return

    print(f" ├─ 任务复杂度判定      : {planning_box.task_complexity}")
    print(f" ├─ 子任务总数          : {len(planning_box.task_plan)}")
    if planning_box.thinking_chain:
        print(f" ├─ 思考链              : {planning_box.thinking_chain[:100]}...")

    print(" └─ 📊 子任务清单:")
    for task in planning_box.task_plan:
        status_map = {
            "pending": "⏳ 待处理",
            "doing": "🔄 进行中",
            "testing": "🧪 沙盒验证中",
            "finished": "✅ 已完成",
            "failed": "❌ 失败"
        }
        status_display = status_map.get(task.status, f"❓ {task.status}")
        print(f"      [子任务 {task.task_id}] {status_display}")
        print(f"        - 描述: {task.description}")
        print(f"        - 目标: {task.objective}")
        print(f"        - 依赖: {task.dependencies if task.dependencies else '无'}")
        if task.status == "pending" and task.result:
            print(f"        - 状态说明: {task.result}")
    print()

    # 执行收纳盒
    if exec_box is None:
        print("❌ 错误：execution 盒子为空！")
        return

    print("📦 【执行收纳盒 (ExecutionContext)】")
    print(f" ├─ 当前子任务进度      : 第 {exec_box.current_task_index}/{len(planning_box.task_plan)} 个")
    print(f" ├─ 是否全部完成？      : {'是 ✅' if exec_box.all_tasks_completed else '否'}")
    print(f" ├─ 错误重试次数        : {exec_box.retry_count}")
    if exec_box.error_trace:
        print(f" ├─ 最近错误堆栈        : {exec_box.error_trace[:300]}")
    print(f" └─ 📂 已归档成果 ({len(exec_box.stage_outputs)} 个子任务):")

    for task_id, code in exec_box.stage_outputs.items():
        code_preview = code[:300].replace("\n", "\n        ")
        print(f"      [子任务 {task_id}] 代码预览:")
        print(f"        {code_preview}")
        if len(code) > 300:
            print(f"        ... (共 {len(code)} 字符，已截断预览)")
        print()

    # 最终交付物预览
    print("📦 【最终交付物 (current_code)】")
    final_code = final_state.get("current_code", "")
    if final_code:
        print(f"  最新生成的代码（共 {len(final_code)} 字符）:")
        print("  " + final_code[:500].replace("\n", "\n  "))
        if len(final_code) > 500:
            print(f"  ... (截断，完整代码见上方归档区)")
    else:
        print("  (无最终代码输出)")

    print("\n" + "=" * 60)
    print("✅ 集成测试完成。")
    print("=" * 60)


if __name__ == "__main__":
    main()
