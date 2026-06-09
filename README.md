# 🤖 AI 编码代理控制台

> 基于 **LangGraph** 的 8 阶段自主编码流水线，支持 ReAct 思考循环、沙盒验证、向量记忆闭环。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain.com/langgraph)
[![Gradio](https://img.shields.io/badge/Gradio-5.0+-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 目录

- [架构全景](#架构全景)
- [模块详解](#模块详解)
  - [主图谱 `core/graph/main.py`](#主图谱-coregraphmainpy)
  - [执行子图 `core/graph/execution.py`](#执行子图-coregraphexecutionpy)
  - [沙盒子图 `core/graph/sandbox.py`](#沙盒子图-coregraphsandboxpy)
  - [ReAct 工作者 `core/nodes/react_worker.py`](#react-工作者-corenodesreact_workerpy)
  - [感知分析器 `core/nodes/analyzer.py`](#感知分析器-corenodesanalyzerpy)
  - [调度器 `core/nodes/scheduler.py`](#调度器-corenodesschedulerpy)
  - [整合器 `core/nodes/integrator.py`](#整合器-corenodesintegratorpy)
  - [输出器 `core/nodes/output.py`](#输出器-corenodesoutputpy)
  - [复盘器 `core/nodes/reviewer.py`](#复盘器-corenodesreviewerpy)
  - [工具集 `core/tools/`](#工具集-coretools)
  - [向量记忆 `core/memory/`](#向量记忆-corememory)
- [执行流程演示](#执行流程演示)
- [项目结构](#项目结构)
- [安装与配置](#安装与配置)
- [使用方式](#使用方式)
- [部署到魔塔](#部署到魔塔)
- [人工干预说明](#人工干预说明)

---

## 架构全景

```
                     ┌─────────────────────────────────────┐
                     │           主图谱 (Main Graph)        │
                     │         "总经理" — 编排所有阶段       │
                     └─────────────────────────────────────┘
                                        │
        ┌───────────┬───────────┬───────┴───────┬───────────┬───────────┐
        ▼           ▼           ▼               ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐    ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ ①感知   │ │ ②规划   │ │ ③调度   │    │ ④执行   │ │ ⑤验证   │ │ ⑥整合   │
   │ 需求分析│ │ 任务拆解 │ │ 拓扑排序│    │ ReAct编码│ │ 沙盒测试│ │ 组装交付│
   └─────────┘ └─────────┘ └─────────┘    └─────────┘ └─────────┘ └─────────┘
        │           │           │               │           │           │
        └───────────┴───────────┴───────────────┴───────────┴───────────┘
                                        │
                                    ┌───┴───┐
                                    │ ⑦输出 │
                                    │写磁盘 │
                                    └───┬───┘
                                        │
                                    ┌───┴───┐
                                    │ ⑧复盘 │
                                    │向量归档│
                                    └───────┘
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **主图谱** | 顶层编排器，调度 5 个节点（executor / sandbox / integrator / output / reviewer） |
| **子图** | 主图节点内部的完整 LangGraph，有自己的路由和循环 |
| **条件路由** | 根据状态动态决定下一步走向（如：测试失败 → 回到编码修复） |
| **ReAct 循环** | Think → Act → Observe 循环，LLM 调用工具，观察结果，决定下一步 |
| **AgentState** | 贯穿全图的共享状态对象，所有节点读写同一个 state |

---

## 模块详解

### 主图谱 `core/graph/main.py`

**角色：** 流水线的总指挥，不亲自写代码，只做三件事——调度子图、判断走向、兜底止损。

**图谱拓扑：**

```
START → executor → [路由A] → sandbox → [路由B] → integrator → [路由C] → output → reviewer → END
         ↑                      ↑                      │                      │
         └──────────────────────┘                      └──────────────────────┘
              (重试循环)                                   (修复循环)
```

**节点清单：**

| 节点 | 实现 | 职责 |
|------|------|------|
| `executor` | `execution.py` 子图 | 感知 + 规划 + 调度 + ReAct 编码 |
| `sandbox` | `sandbox.py` 子图 | 代码执行验证（CLI / Web 服务） |
| `integrator` | `integrator.py` 节点 | 组装交付物 + LLM 一致性审核 |
| `output_writer` | `output.py` 节点 | 写入磁盘 + README + 交付清单 |
| `reviewer` | `reviewer.py` 节点 | 收集经验 → 写入 Qdrant 向量库 |

**三个核心路由：**

| 路由函数 | 触发时机 | 判断逻辑 |
|----------|----------|----------|
| `route_after_executor` | executor 完成后 | 有 testing → sandbox / 全部 finished → integrator / react_blocked → END |
| `route_after_sandbox` | sandbox 完成后 | 有 testing → 继续 sandbox / 有 pending → 回 executor / 超限 → integrator |
| `route_after_integrator` | integrator 完成后 | 有 pending 打回 → executor / 完成 → output |

---

### 执行子图 `core/graph/execution.py`

**角色：** 流水线的"内环"，把用户需求转化为可验证的代码。可被主图反复重入（修复模式）。

**图谱拓扑：**

```
START → [入口路由] ──(task_plan 为空)──→ analyzer
      └──(task_plan 已存在)────────────→ worker (跳过分析)
                                          ↓
analyzer → [路由] ──(需求模糊)──→ END
         └──(需求清晰)──→ scheduler → install_deps → worker
                                                       ↓
                                              [路由] ──(还有 pending)──→ worker (循环)
                                                     └──(全部处理完)──→ END
```

**节点清单：**

| 节点 | 职责 |
|------|------|
| `router_start` | 入口判断：task_plan 是否已有数据（首次 vs 重入） |
| `analyzer` | LLM 需求分析 → 复杂度评定 → 子任务拆解 → RAG 检索相似经验 |
| `scheduler` | 拓扑排序（Kahn 算法）+ 依赖校验 + 资源确认 |
| `install_deps` | 用 pip 预安装 required_libraries + import 校验 |
| `worker` | **ReAct 子图**：think → act → judge 循环，每任务最多 7 轮 |

**重入机制：**
```
第 1 次进入：analyzer → scheduler → install_deps → worker（全新编码）
第 2 次进入（沙盒失败后）：直接跳到 worker（带着 error_trace 修复）
第 N 次进入：同上，直到所有任务 finished 或超过重试上限
```

---

### 沙盒子图 `core/graph/sandbox.py`

**角色：** 在沙盒内执行代码，验证是否能跑通。只测一个子任务，主图负责循环调度。

**图谱拓扑（单节点子图）：**

```
START → sandbox_node → END

sandbox_node 内部：
  1. 找第一个 status="testing" 的子任务
  2. 从 stage_outputs 提取代码文件
  3. 写入工作区 task_<id>/ 子目录
  4. CLI 程序：直接 python 执行，检测返回码
  5. Web 服务：后台启动 → 轮询 curl 测试（15 秒）→ 杀进程
  6. 更新子任务状态：finished / pending（失败回写）
```

**测试策略：**

| 类型 | 检测方式 | 超时 |
|------|----------|------|
| CLI 程序 | `python main.py` → 检查 exit code | 30 秒 |
| Web 服务 | 后台启动 → `curl` 轮询 HTTP 200 | 15 秒 × 1 秒间隔 |
| Web 崩溃 | `kill -0` 检测进程存活 → 读启动日志 | 实时 |

**两种沙盒模式：**

| 模式 | 沙盒实现 | 适用场景 |
|------|----------|----------|
| Docker 模式 | `python:3.11` 容器 | 本地开发，完全隔离 |
| 本地模式 | `subprocess` 直接执行 | 魔塔部署，Docker 不可用 |

---

### ReAct 工作者 `core/nodes/react_worker.py`

**角色：** 整个流水线最核心的执行单元。每个子任务进入后，LLM 在 Think → Act → Observe 循环中调用工具、读写文件、运行命令，直到完成或卡住。

**子图拓扑：**

```
START → start_react → think → act → judge ──(continue)──→ think (循环)
                                  ↓
                             (done/blocked) → END
```

**节点清单：**

| 节点 | 职责 |
|------|------|
| `start_react` | 选下一个就绪任务 + 重置轮数 + 判断「全新编码」还是「修复模式」 |
| `think` | LLM 思考：分析当前状态，决定调用哪个工具 or 提交任务 |
| `act` | 执行工具调用，收集 Observation |
| `judge` | 裁决：任务完成 / 继续循环 / 卡住（7 轮仍无进展） |

**循环控制：**

| 条件 | 结果 |
|------|------|
| 调用了 `submit_task` | → 标记 finished / testing，退出循环 |
| 当前轮数 < 7 且有进展 | → 继续 think |
| 当前轮数 ≥ 7 且无进展 | → react_blocked = True，等待人工介入 |
| 连续 3 轮无工具调用 | → 提醒 LLM 采取行动 |
| 沙盒失败重入 | → 自动进入「修复模式」，提示带上 error_trace |

**10 个可用工具：**

| 工具 | 分类 | 功能 |
|------|------|------|
| `list_directory` | 文件系统 | 列出目录内容 |
| `read_file` | 文件系统 | 分页读取文件（带行号） |
| `write_file` | 文件系统 | 创建或覆盖文件 |
| `edit_file` | 文件系统 | 外科手术式局部替换 |
| `search_content` | 文件系统 | grep 搜索文件内容 |
| `move_file` | 文件系统 | 移动/重命名文件 |
| `delete_file` | 文件系统 | 删除文件或空目录 |
| `run_command` | 命令执行 | 在沙盒内执行任意 shell 命令 |
| `web_search` | 信息获取 | DuckDuckGo 搜索 API 文档/报错 |
| `submit_task` | 任务控制 | 提交任务完成（带摘要） |

---

### 感知分析器 `core/nodes/analyzer.py`

**职责：** 流水线第一站。接收用户原始需求，做三件事：

1. **RAG 检索**：从 Qdrant 向量库中检索相似历史经验，作为上下文参考
2. **复杂度评定**：判断任务属于 simple / complex
3. **子任务拆解**：调用 LLM 将需求拆解为 SubTask 列表（含依赖、风险等级、所需库）
4. **需求澄清**：如果需求模糊，设置 `need_clarification=True`，阻断流水线等待用户补充

**输出：** 填充 `AgentState.planning`（`PlanningContext` 对象）

---

### 调度器 `core/nodes/scheduler.py`

**职责：** 插入在 analyzer 和 worker 之间，确保子任务按正确顺序执行。

1. **拓扑排序**：对 task_plan 做 Kahn 算法排序，保证依赖关系正确
2. **依赖校验**：检查每个任务的依赖是否存在（不存在则标记为 failed）
3. **资源校验**：对照 required_resources 和已注册工具，缺资源时告警

---

### 整合器 `core/nodes/integrator.py`

**职责：** 所有子任务通过沙盒验证后，将散落在 `stage_outputs` 中的代码文件组装为结构化交付物。

1. **代码提取**：从 ReAct 历史中提取 `write_file` 的 path 参数，建立「文件路径 → 代码」映射
2. **冲突检测**：多子任务写同一文件时，智能合并（取最后写入 / LLM 语义合并）
3. **跨文件审核**：LLM 检查 import 路径对齐、函数签名匹配、接口一致性
4. **打回机制**：发现问题 → 将相关任务设回 pending → 主图路由回 executor 修复

**输出：** 填充 `AgentState.integration`（`IntegrationContext` 对象）

---

### 输出器 `core/nodes/output.py`

**职责：** 将整合后的交付物真正写入磁盘。

1. 创建 `output/<项目名_时间戳>/` 目录
2. 逐个写入代码文件
3. 调用 LLM 生成 `README.md`（含依赖说明、运行方式）
4. 生成 `deliverable_manifest.json`（交付物清单）

**输出：** 填充 `AgentState.output`（`OutputContext` 对象）

---

### 复盘器 `core/nodes/reviewer.py`

**职责：** 流水线最后一站，形成"越跑越聪明"的正反馈闭环。

| 条件 | 行为 |
|------|------|
| ✅ 有 finished 任务 | 存储成功经验 |
| ❌ 有 failed 任务 + 沙盒报错 | 存储踩坑经验（更宝贵） |
| ⚠️ 执行结果全空 | 跳过（数据不可信） |
| ⚠️ Embedding / Qdrant 不可用 | 降级跳过（不阻塞流水线） |

**经验数据结构：** 任务描述、复杂度、拆解方案、执行结果、踩坑记录、工具使用、关键词标签、时间戳。

---

### 工具集 `core/tools/`

| 模块 | 工具数 | 说明 |
|------|--------|------|
| `filesystem.py` | 7 | list_directory / read_file / write_file / edit_file / search_content / move_file / delete_file |
| `run_command.py` | 1 | 沙盒内 shell 命令执行（`shlex` 参数防注入） |
| `web_search.py` | 1 | DuckDuckGo 网页搜索（Lite HTML 解析 + Instant Answer 兜底） |
| `submit_task.py` | 1 | 任务提交（ReAct 结束信号） |
| `docker_sandbox.py` | 基础 | Docker 容器管理（init / exec / exec_background / kill / is_web_service / map_to_host） |

---

### 向量记忆 `core/memory/`

| 模块 | 职责 |
|------|------|
| `embedding.py` | Embedding 工厂：自动检测本地模型 vs API 端点，返回 LangChain Embeddings 实例 |
| `store.py` | Qdrant 向量存储：写入经验 + MMR 语义检索（k=5, fetch_k=20, λ=0.6） |

**检索时机：** 每次 analyzer 做需求分析时，自动从 Qdrant 检索相似历史经验作为上下文。

---

## 执行流程演示

### 完整流水线（以"写一个 Flask 学生成绩管理系统"为例）

```
用户输入："写一个 Flask 学生成绩管理系统"
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ① analyzer（感知分析）                                          │
│   • RAG 检索：找到 3 条相似经验（"Flask 博客系统"、"图书管理"） │
│   • 复杂度评定：complex（涉及数据库、路由、模板）               │
│   • 拆解为 4 个子任务：                                         │
│     T1: 数据库模型 + 初始化 (独立, low)                         │
│     T2: Flask 路由 + 视图函数 (依赖 T1, medium)                 │
│     T3: HTML 模板渲染 (依赖 T2, low)                             │
│     T4: 表单验证 + 错误处理 (依赖 T2, medium)                   │
│   • required_libraries: ["flask", "sqlalchemy"]                 │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ② scheduler（调度排序）                                         │
│   • 拓扑排序：T1 → T2 → T3, T4                                 │
│   • 依赖校验：通过                                               │
│   • 资源校验：10/10 工具可用                                    │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ③ install_deps（环境准备）                                      │
│   • pip install flask sqlalchemy                                │
│   • import 校验：通过                                            │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ④ ReAct worker — 第 1 轮（处理 T1）                            │
│   think: "先创建数据库模型 models.py"                           │
│   act: write_file("models.py", "class Student(db.Model)...")    │
│   observe: "文件写入成功"                                        │
│   think: "创建初始化脚本 init_db.py"                             │
│   act: write_file("init_db.py", "from app import db...")        │
│   observe: "文件写入成功"                                        │
│   think: "任务完成"                                              │
│   act: submit_task(summary="已创建数据库模型和初始化脚本")      │
│   judge: → T1 标记为 testing                                    │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑤ sandbox（验证 T1）                                            │
│   • 检测类型：CLI 程序                                          │
│   • 写入 work/task_1/models.py, init_db.py                      │
│   • docker exec python init_db.py                               │
│   • exit code = 0 ✅                                             │
│   • T1 标记为 finished                                           │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑥ ReAct worker — 第 2 轮（处理 T2）                            │
│   think: "创建 app.py，导入 T1 的 models"                       │
│   act: write_file("app.py", "from models import *...")          │
│   ...（类似流程）                                               │
│   act: submit_task(...)                                         │
│   → T2 标记为 testing                                           │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑦ sandbox（验证 T2 — Web 服务）                                │
│   • 检测类型：Web 服务（检测到 from flask import...）           │
│   • docker exec -d python app.py 2>/tmp/log                     │
│   • 第 1 秒: kill -0 {pid} → alive, curl → "000"              │
│   • 第 2 秒: kill -0 {pid} → alive, curl → "000"              │
│   • 第 3 秒: kill -0 {pid} → alive, curl → "200" ✅            │
│   • kill {pid} 清理                                             │
│   • T2 标记为 finished                                           │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
  ...（T3、T4 类似：编码 → 验证 → 通过）...
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑧ integrator（整合）                                            │
│   • 收集所有 stage_outputs 的文件                               │
│   • 冲突检测：无冲突                                             │
│   • LLM 审核：import 路径一致，接口匹配                          │
│   • integration.integration_done = True                          │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑨ output（输出）                                                │
│   • 目录：output/学生成绩管理系统_20260609_182157/              │
│   • 写入：models.py, init_db.py, app.py, templates/index.html   │
│   • 生成：README.md, deliverable_manifest.json                   │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ ⑩ reviewer（复盘）                                              │
│   • 计算 success_rate: 4/4 = 100%                               │
│   • 提取标签: ["Flask", "Web", "数据库", "API"]                 │
│   • 向量化写入 Qdrant → 下次类似任务可检索                      │
└─────────────────────────────────────────────────────────────────┘
  │
  ▼
执行完成 ✅
```

### 沙盒失败修复流程

```
T2 写入 app.py → sandbox 测试
  │
  ▼
curl → HTTP 500 (崩溃)
  │
  ▼
sandbox: T2.status = "pending"
sandbox: error_trace = "ImportError: No module named 'models'"
  │
  ▼
主图路由: 有 pending → 回 executor
  │
  ▼
ReAct worker — T2 修复模式:
  think: "报错是 ImportError，说明 models.py 没有被正确导入。
         让我检查 app.py 的 import 语句。"
  act: read_file("app.py")
  observe: "from models import * 的 import 语句正确。models.py 在 task_1/ 目录。"
  think: "需要把 models.py 复制到当前 task_2/ 目录，或者修改 import 路径。"
  act: edit_file("app.py", old="from models import *", new="import sys\nsys.path.append('..')\nfrom task_1.models import *")
  observe: "文件已修改"
  think: "修复完成"
  act: submit_task(...)
  → T2 重新标记为 testing
  │
  ▼
sandbox 重新验证 → curl → HTTP 200 ✅
```

---

## 项目结构

```
Agent_upgrade/
├── core/                        # 🔧 核心引擎（与 UI 无关）
│   ├── state.py                 #   AgentState — 全图共享状态定义
│   ├── logger.py                #   统一日志
│   ├── graph/                   #   📊 图谱层（编排逻辑）
│   │   ├── main.py              #     主图谱：8 阶段流水线 + 5 节点 + 路由
│   │   ├── execution.py         #     执行子图：感知→规划→调度→ReAct 编码
│   │   └── sandbox.py           #     沙盒子图：CLI/Web 服务测试
│   ├── nodes/                   #   🧩 节点层（具体实现）
│   │   ├── analyzer.py          #     感知分析：需求理解 + 任务拆解 + RAG
│   │   ├── scheduler.py         #     调度排序：拓扑排序 + 依赖校验
│   │   ├── react_worker.py      #     ★ ReAct 子图：Think→Act→Observe 循环
│   │   ├── integrator.py        #     整合组装：文件合并 + 一致性审核
│   │   ├── output.py            #     输出写入：磁盘 + README + 清单
│   │   ├── reviewer.py          #     复盘归档：经验向量化 → Qdrant
│   │   └── worker.py            #     旧版 Worker（已弃用，保留备用）
│   ├── tools/                   #   🔨 工具层（LLM 可调用）
│   │   ├── __init__.py          #     工具注册表 + 统一入口
│   │   ├── filesystem.py        #     7 个文件系统工具
│   │   ├── run_command.py       #     Shell 命令执行
│   │   ├── web_search.py        #     DuckDuckGo 网页搜索
│   │   ├── submit_task.py       #     任务提交信号
│   │   └── docker_sandbox.py    #     Docker 沙盒基础设施
│   └── memory/                  #   🧠 记忆层（向量存储）
│       ├── embedding.py         #     Embedding 工厂（本地/API 自动检测）
│       └── store.py             #     Qdrant 向量存储（MMR 检索）
│
├── UI/                          # 🖥️ 本地 UI（Docker 沙盒模式）
│   ├── app.py                   #   入口：组装图谱 + 启动 Gradio
│   ├── layout.py                #   布局 + CSS + 事件绑定
│   ├── pipeline.py              #   流水线执行生成器
│   ├── handlers.py              #   事件处理（启动/澄清/干预）
│   ├── helpers.py               #   格式化渲染 + 会话列表
│   └── sandbox.py               #   Docker 沙盒探测 + 初始化
│
├── modelscope_app/              # ☁️ 魔塔 UI（本地 subprocess 沙盒）
│   ├── app.py                   #   入口：沙盒注入 → 构建图谱 → 启动
│   ├── ui_layout.py             #   布局 + CSS + 事件绑定
│   ├── pipeline.py              #   流水线执行生成器
│   ├── event_handlers.py        #   事件处理（启动/澄清/干预）
│   ├── ui_helpers.py            #   格式化渲染 + 会话列表
│   └── local_sandbox.py         #   ★ 沙盒注入模块（替换 Docker 为本地 subprocess）
│
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量示例
├── .gitignore                   # Git 忽略规则
└── CLAUDE.md                    # 开发行为准则（Karpathy 指南）
```

---

## 安装与配置

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd Agent_upgrade
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 API Key 和模型配置
```

**必需配置：**

```env
BASE_URL = "https://api.deepseek.com/v1/"
LLM_API_KEY = "sk-your-api-key-here"
PROCESS_MODEL = "deepseek-v4-flash"
```

**可选配置：**

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CODING_MODEL` | 编码专用模型 | 复用 PROCESS_MODEL |
| `EMBEDDING_MODEL_PATH` | 本地嵌入模型路径 | 无（走 API） |
| `EMBEDDING_BASE_URL` | API 嵌入端点 | 复用 BASE_URL |
| `QDRANT_PATH` | Qdrant 数据目录 | 自动检测 |
| `QDRANT_COLLECTION` | Collection 名称 | `task_experiences` |

### 4. （可选）Docker 沙盒

```bash
# 确保 Docker 已安装并运行
docker ps
```

---

## 使用方式

### 本地 UI（Docker 沙盒）

```bash
cd Agent_upgrade
python -m UI.app
# 浏览器打开 http://localhost:7860
```

### 魔塔 UI（本地 subprocess）

```bash
cd Agent_upgrade
python -m modelscope_app.app
# 浏览器打开 http://localhost:7860
```

### 界面说明

| 区域 | 功能 |
|------|------|
| **左栏** | 任务进度（T1/T2/...状态）+ 工作区文件树 |
| **中栏** | 执行日志（Chatbot）+ 状态栏 + 进度条 + 执行摘要 |
| **右栏** | 任务输入框 + 断点续传下拉框 + 启动按钮 + 介入面板（中断时出现） |

---

## 部署到魔塔

1. 将仓库上传到魔塔 Space
2. Space SDK 选 **Gradio**
3. 入口文件指向 `modelscope_app/app.py`
4. 在 Space 设置 → Environment Variables 中配置：

```env
BASE_URL = "https://your-api.com/v1/"
LLM_API_KEY = "sk-your-key"
PROCESS_MODEL = "deepseek-v4-flash"
EMBEDDING_BASE_URL = "https://your-api.com/v1/"
EMBEDDING_API_KEY = "sk-your-key"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
```

5. `QDRANT_PATH` 会自动使用 `/data/qdrant_data`（魔塔持久存储）

---

## 人工干预说明

流水线在以下情况会暂停，等待用户决策：

| 触发条件 | 场景 | 可选操作 |
|----------|------|----------|
| 需求模糊 | analyzer 无法理解需求 | 补充信息 → 重新分析 |
| ReAct 卡住 | think→act 循环超过 7 轮无进展 | 继续执行 / 修改需求 |
| 沙盒失败 | 代码测试不通过 | 继续执行 / 强制提交 / 跳过任务 / 修改需求 |
| 重试超限 | 单任务重试超过 3 次 | 强制提交 / 跳过任务 |

**四种干预操作：**

| 操作 | 效果 |
|------|------|
| **继续执行** | 保留错误上下文，让 LLM 带着报错重新尝试修复 |
| **强制提交** | 跳过沙盒验证，直接标记任务完成 |
| **跳过任务** | 标记任务失败，继续下一个 |
| **修改需求** | 输入补充指示，注入到对话中重新执行 |

---

## License

MIT

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
