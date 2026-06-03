"""
core.graph 包 —— 图编排层

模块结构：
    execution.py — 执行子图（感知 → 规划 → 调度 → ReAct 执行）
    sandbox.py   — 沙盒验证子图（代码安全执行与错误捕获）
    main.py      — 主图（编排 executor + sandbox 的编码→验证→修复循环）

使用方式：
    from core.graph.main import build_master_graph
    from core.graph.execution import build_execution_subgraph
    from core.graph.sandbox import build_sandbox_subgraph
"""

from core.graph.execution import build_execution_subgraph

__all__ = ["build_execution_subgraph"]
