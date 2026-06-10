"""
=============================================================================
ui_layout.py —— Gradio 界面布局 & CSS
=============================================================================

build_ui() 接收回调函数引用，返回组装好的 Gradio Blocks 实例。
=============================================================================
"""

import gradio as gr


# ==========================================================================
# CSS
# ==========================================================================

CSS = """
.gradio-container { max-width: 1400px !important; margin: 0 auto !important; }
#agent-chatbot { border-radius: 8px !important; border: 1px solid #e0e0e0 !important; }
#agent-chatbot .message-row { font-size: 14px !important; }
#left-panel { background: #fafbfc; color: #1a1a1a; border-right: 1px solid #e0e0e0; padding: 8px; }
#left-panel textarea, #left-panel input, #left-panel label, #left-panel p, #left-panel div, #left-panel h3, #left-panel pre, #left-panel code, #left-panel b, #left-panel span { color: #1a1a1a !important; }
#left-panel textarea[disabled], #left-panel textarea[readonly] { color: #1a1a1a !important; -webkit-text-fill-color: #1a1a1a !important; }
#control-panel { background: #fafbfc; color: #1a1a1a; border-left: 1px solid #e0e0e0; padding: 8px; }
#control-panel textarea, #control-panel input, #control-panel label, #control-panel p, #control-panel div, #control-panel h3, #control-panel pre, #control-panel code, #control-panel b, #control-panel span, #control-panel table, #control-panel th, #control-panel td, #control-panel li { color: #1a1a1a !important; }
#control-panel textarea[disabled], #control-panel textarea[readonly] { color: #1a1a1a !important; -webkit-text-fill-color: #1a1a1a !important; }
#intervene-panel * { color: #1a1a1a !important; }
#intervene-panel {
    border: 2px solid #ff4d4f !important;
    box-shadow: 0 0 12px rgba(255,77,79,0.3) !important;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 4px rgba(255,77,79,0.15); }
    50% { box-shadow: 0 0 16px rgba(255,77,79,0.45); }
    100% { box-shadow: 0 0 4px rgba(255,77,79,0.15); }
}
pre, code { font-family: 'Fira Code','Consolas',monospace !important; font-size: 12px !important; }
#start-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important; color: white !important; font-weight: bold !important;
}
#start-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(102,126,234,0.4) !important; }
#status-text textarea { font-weight: bold !important; font-size: 13px !important; }
footer { display: none !important; }
"""


# ==========================================================================
# 布局构建
# ==========================================================================

def build_ui(
    on_start,
    on_clarify,
    on_intervene,
    on_action_change,
    list_workspace,
    list_sessions,
    preview_file,
    clear_workspace,
    on_embed_mode_change,
    on_embed_load,
    auth_user="admin",
    auth_pass="123456",
):
    """构建 Gradio Blocks 界面，接收回调函数引用"""

    with gr.Blocks(
        title="AI 编码代理控制台",
        css=CSS,
        theme=gr.themes.Soft(),
    ) as demo:
        agent_state = gr.State({})
        thread_id_state = gr.State("")
        is_interrupt_state = gr.State(False)
        login_state = gr.State(False)

        # ==================================================================
        # 登录界面
        # ==================================================================
        with gr.Column(visible=True) as login_screen:
            gr.Markdown("# 🔐 AI 编码代理控制台")
            gr.Markdown("请输入账号密码以继续")
            login_user = gr.Textbox(
                label="用户名",
                placeholder="请输入用户名",
                value="admin",
            )
            login_pass = gr.Textbox(
                label="密码",
                type="password",
                placeholder="请输入密码",
            )
            login_btn = gr.Button("🔓 登录", variant="primary")
            login_msg = gr.Markdown("")

        # ==================================================================
        # 主界面（登录后显示）
        # ==================================================================
        with gr.Column(visible=False) as main_screen:
            gr.Markdown(
                "# 🤖 AI 编码代理控制台\n"
                "8 阶段流水线：感知 → 规划 → 调度 → 执行 → 验证 → 整合 → 输出 → 复盘"
            )

            with gr.Row(equal_height=True):
                # 左栏：任务进度 + 工作区文件
                with gr.Column(scale=1, elem_id="left-panel"):
                    gr.Markdown("### 📋 任务进度")
                    task_html = gr.HTML(
                        value="<p style='color:#888'>暂无任务</p>",
                        elem_id="task-progress",
                    )
                    gr.Markdown("---\n### 📂 工作区文件")
                    file_selector = gr.Dropdown(
                        choices=["(工作区为空)"],
                        value=None,
                        label="选择文件预览",
                        interactive=True,
                        filterable=True,
                        show_label=False,
                    )
                    file_preview = gr.Code(
                        value="# 选择一个文件即可预览",
                        language="python",
                        lines=18,
                        label="文件内容",
                    )
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 刷新", size="sm")
                        clear_workspace_btn = gr.Button(
                            "🗑️ 清空", size="sm", variant="stop"
                        )

                # 中栏：对话日志
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        value=[{
                            "role": "assistant",
                            "content": "👋 欢迎使用 AI 编码代理！输入需求后点击 ▶️ 启动任务。"
                        }],
                        label="执行日志",
                        height=520,
                        elem_id="agent-chatbot"
                    )
                    status_text = gr.Textbox(
                        value="🟢 就绪",
                        label="状态",
                        interactive=False,
                        elem_id="status-text",
                    )
                    progress_bar = gr.HTML(
                        value='<div style="margin:6px 0;font-size:12px;color:#555">⏳ 等待启动...</div>'
                              '<div style="width:100%;background:#e8e8e8;border-radius:6px;height:14px">'
                              '<div style="width:0%;background:#1677ff;height:14px;border-radius:6px"></div>'
                              '</div>',
                        elem_id="progress-bar",
                    )
                    result_md = gr.Markdown(
                        "### 🤖 AI 编码代理\n"
                        "输入需求后点击 **▶️ 启动任务** 按钮开始"
                    )

                # 右栏：控制台
                with gr.Column(scale=1, elem_id="control-panel"):
                    gr.Markdown("### ⚙️ 控制台")
                    user_input = gr.Textbox(
                        label="📝 任务需求",
                        lines=3,
                        placeholder="例如：用 Python 写一个数独游戏\n或：写一个 Flask 学生成绩管理系统",
                    )
                    with gr.Row():
                        resume_dropdown = gr.Dropdown(
                            label="🆔 断点续传",
                            choices=[],
                            value=None,
                            allow_custom_value=True,
                            scale=5,
                            filterable=True,
                        )
                        refresh_sessions_btn = gr.Button(
                            "🔄", size="sm", variant="secondary", scale=1,
                            min_width=40,
                        )
                    show_completed_cb = gr.Checkbox(
                        label="显示已完成任务",
                        value=False,
                    )
                    start_btn = gr.Button(
                        "▶️ 启动任务",
                        variant="primary",
                        size="lg",
                        elem_id="start-btn",
                    )

                    gr.Markdown("---")

                    # ── 向量模型配置 ──
                    with gr.Accordion("🧠 向量模型（经验记忆）", open=False):
                        embed_status = gr.HTML(
                            value="<div style='font-size:12px;color:#888'>"
                            "⚪ <b>向量模型:</b> 未加载</div>"
                        )
                        embed_mode = gr.Radio(
                            choices=[("API 接口", "api"), ("本地模型", "local")],
                            label="模型来源",
                            value="api",
                        )
                        embed_path = gr.Textbox(
                            label="本地模型路径",
                            placeholder="如: /data/models/bge-m3",
                            visible=False,
                        )
                        embed_url = gr.Textbox(
                            label="API 地址",
                            placeholder="如: https://api.openai.com/v1",
                            visible=True,
                        )
                        embed_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            placeholder="sk-xxx",
                            visible=True,
                        )
                        embed_model_name = gr.Textbox(
                            label="模型名",
                            placeholder="text-embedding-3-small",
                            value="text-embedding-3-small",
                            visible=True,
                        )
                        embed_load_btn = gr.Button(
                            "🔄 加载模型", variant="secondary", size="sm"
                        )

                    # 需求澄清面板
                    with gr.Column(visible=False) as clarify_panel:
                        gr.Markdown("### 🤔 需求澄清")
                        clarify_question_md = gr.Markdown("")
                        clarify_input = gr.Textbox(
                            label="请补充信息",
                            lines=2,
                            placeholder="输入补充说明...",
                        )
                        clarify_btn = gr.Button("📤 提交补充", variant="secondary")

                    # 人工介入面板
                    with gr.Column(visible=False, elem_id="intervene-panel") as intervene_panel:
                        gr.Markdown("### ⚠️ 人工决策")
                        intervene_content = gr.HTML(value="")

                        gr.Markdown("""
                        | 操作 | 效果 |
                        |------|------|
                        | 🔄 **继续执行** | 保留报错信息，让 AI 带着错误上下文重新修复 |
                        | ✅ **强制提交** | 跳过沙盒验证，直接把当前代码标记为完成 |
                        | ⏭️ **跳过任务** | 标记当前子任务失败，继续执行下一个 |
                        | 📝 **修改需求** | 输入补充指示或修改后的需求，重新编码 |
                        """)
                        gr.Markdown("""
                        > 💡 **提示：** 若报错为容器/环境问题（如 `sandbox timeout`、`docker not available`、
                        > `connection refused` 等），并非模型能力不足，选择 **✅ 强制提交** 即可跳过沙盒，
                        > 代码本身已正确生成。
                        """)
                        intervene_action = gr.Radio(
                            choices=["继续执行", "强制提交", "跳过任务", "修改需求"],
                            label="选择操作",
                            value="继续执行",
                        )
                        customize_input = gr.Textbox(
                            label="自定义修改内容（仅「修改需求」时填写）",
                            lines=3,
                            visible=False,
                            placeholder="输入修改后的需求或补充指示...",
                        )
                        intervene_btn = gr.Button(
                            "🛠️ 执行决策",
                            variant="secondary",
                            size="lg",
                        )

            # ── 统一输出 ──
            OUTPUTS = [
                chatbot, agent_state, status_text,
                task_html, file_selector,
                clarify_panel, intervene_panel, intervene_content,
                progress_bar,
            ]

            # ── 事件绑定 ──
            start_btn.click(
                fn=on_start,
                inputs=[user_input, resume_dropdown],
                outputs=OUTPUTS,
            )
            clarify_btn.click(
                fn=on_clarify,
                inputs=[clarify_input, agent_state, thread_id_state],
                outputs=OUTPUTS,
            )
            intervene_action.change(
                fn=on_action_change,
                inputs=[intervene_action],
                outputs=[customize_input],
            )
            intervene_btn.click(
                fn=on_intervene,
                inputs=[intervene_action, customize_input,
                        agent_state, thread_id_state, is_interrupt_state],
                outputs=OUTPUTS,
            )
            refresh_btn.click(
                fn=lambda: list_workspace(),
                inputs=[],
                outputs=[file_selector],
            )
            file_selector.change(
                fn=preview_file,
                inputs=[file_selector],
                outputs=[file_preview],
            )
            clear_workspace_btn.click(
                fn=lambda: (clear_workspace(), "# 工作区已清空"),
                inputs=[],
                outputs=[file_selector, file_preview],
            )
            refresh_sessions_btn.click(
                fn=lambda sc: list_sessions(show_completed=sc),
                inputs=[show_completed_cb],
                outputs=[resume_dropdown],
            )
            show_completed_cb.change(
                fn=lambda sc: list_sessions(show_completed=sc),
                inputs=[show_completed_cb],
                outputs=[resume_dropdown],
            )
            # 向量模型 UI 联动
            embed_mode.change(
                fn=on_embed_mode_change,
                inputs=[embed_mode],
                outputs=[embed_path, embed_url, embed_key, embed_model_name],
            )
            embed_load_btn.click(
                fn=on_embed_load,
                inputs=[embed_mode, embed_path, embed_url, embed_key, embed_model_name],
                outputs=[embed_status],
            )
            # 页面加载时自动填充
            demo.load(
                fn=lambda: list_sessions(show_completed=False),
                inputs=[],
                outputs=[resume_dropdown],
            )
            demo.load(
                fn=lambda: list_workspace(),
                inputs=[],
                outputs=[file_selector],
            )

        # ==================================================================
        # 登录验证
        # ==================================================================
        def do_login(user, pwd):
            if user == auth_user and pwd == auth_pass:
                return (
                    gr.update(visible=False),  # login_screen
                    gr.update(visible=True),   # main_screen
                    "",                        # login_msg
                    True,                      # login_state
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "❌ 用户名或密码错误",
                    False,
                )

        login_btn.click(
            fn=do_login,
            inputs=[login_user, login_pass],
            outputs=[login_screen, main_screen, login_msg, login_state],
        )
        login_pass.submit(
            fn=do_login,
            inputs=[login_user, login_pass],
            outputs=[login_screen, main_screen, login_msg, login_state],
        )

    return demo
