# PreFit AI（启跃智伴）技术设计参考

> 本文档整合 PreFit AI 的核心架构与设计决策，作为公开技术参考。详细内部设计文档不随仓库分发。

## 1. 系统定位

PreFit AI 是基于 LangGraph 的状态驱动、多 Agent 协同、传感器数据增强型运动前智能体。核心架构：

> **LangGraph 负责决策流程编排，LangChain 负责工具与知识增强，LangSmith 负责生产级质量闭环。**

当前版本聚焦**运动前阶段（0–10 分钟）**：健康评估 → 风险预警 → 可行性分析 → 身体电量 → 目标问询 → 计划生成 → 方案编排 → 设备检查 → 热身指导。

## 2. 技术栈三层关系

| 组件 | 定位 | 解决的问题 | 类比 |
|---|---|---|---|
| **LangChain** | LLM 应用开发的组件库 | Prompt 模板、Tool 封装、Retriever、Output Parser、Memory 等标准化积木 | 乐高积木盒 |
| **LangGraph** | 有状态、多步骤 Agent 的编排框架 | 条件分支、循环、并行、人机交互打断、跨会话状态持久化 | 流程图引擎 / 状态机 |
| **LangSmith** | 可观测性 + 评测平台 | 全链路 trace、延迟/成本监控、离线评估集、线上质量监控 | APM + 测试平台 |

一句话：**LangChain 给零件，LangGraph 决定零件怎么按状态流转，LangSmith 让你看见并验证流转是否正确、够快、够准。**

## 3. 为什么必须用 LangGraph

运动智能体的需求特征几乎都是线性 Chain 处理不好、但 LangGraph 的核心能力：

- **条件分支**：血氧 < 90% 强制禁止高强度、高风险走就医分支 → `StateGraph` 条件边路由。
- **人工打断**：问询支持打断、打断后语音不可继续播报 → `interrupt()` 原语在节点内暂停等待用户输入，恢复时清空播报状态。
- **跨会话持久化**：体能基线需 7 天数据校准、用户长期偏好 → `checkpointer`（Postgres/SQLite Saver）落盘，每次开 APP 从上次 checkpoint 恢复。
- **流式输出**：方案分析按流式呈现 → LangGraph 原生 `stream_mode="messages"` 逐 token / 逐节点透传。
- **多路数据聚合再决策**：可行性评估综合生理数据 + 历史档案 → 典型"并行节点 → 汇聚节点"图结构。

## 4. 系统架构

```
                         用户
                          |
              -------------------------
              |                       |
          智能耳机                 APP
       (Sensor + Voice)        (UI + Control)
                          |
                    Agent Gateway
                          |
        =====================================
                    LangGraph Runtime
        +-----------------------------------+
        |          Sport Supervisor Agent   |  总控
        +-----------------------------------+
              |          |          |          |
           Health     Safety     Coach     Device
           Agent      Agent      Agent     Agent
        =====================================
                    LangChain Layer
             Tools / Retriever / Memory
        -------------------------------------
                    Data Layer
         PostgreSQL    InfluxDB    Vector DB
         用户画像     生理时序     运动知识库
        -------------------------------------
                    LangSmith
          Trace / Evaluation / Monitoring
```

## 5. State 设计

所有节点共享状态，贯穿全图：

```python
from typing import TypedDict, Optional, Literal, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]     # 对话历史（含打断记录）
    user_id: str
    is_first_time: bool

    # 运动前评估
    vitals: dict                                  # 静息心率/血氧/HRV/压力值
    risk_level: Literal["low", "mid", "high"]
    risk_flags: list[str]                          # 异常项标记
    baseline_score: Optional[float]                # 身体电量
    baseline_ready: bool                            # 是否满 7 天数据

    # 用户画像/意图
    goal: Optional[str]                             # 减脂/有氧/耐力
    intensity_pref: Optional[str]
    interrupted: bool

    # 运动前计划
    weather: dict
    location: dict
    plan: Optional[dict]
    plan_analysis_stream: Optional[str]
```

## 6. 主流程与节点

```
START
  ↓
memory_load（Mem0 记忆快照）
  ↓ ← 是否上传体检报告
  ├─ 上传 → report_analyze（解析 PDF + 更新画像）┐
  └─ 未上传 ──────────────────────────────────┤
                                             ↓
health_collect (F001) → risk_assess (F002)
  ↓ ← 风险等级
  ├─ high/forbidden → safety_stop → memory_write → END
  └─ low/mid
        ↓
feasibility (F003) → recovery (F004) → avatar (F005)
        ↓ ← 新老用户分流
        ├─ 新用户 → interview (F006, interrupt) ┐
        └─ 老用户 ────────────────────────────┤
                                             ↓
plan_generate (F007) → plan_analyze (F008) → plan_compose (F009) → plan_edit (F010)
        ↓
device_check (F011) → warmup (F012)
        ↓
memory_write（记忆沉淀，后台不阻塞）→ END
```

## 7. 关键设计决策

### 7.1 风险判定用纯代码，不靠 LLM

风险初筛有硬性 SLA（误报率 ≤ 3%、漏报率 ≤ 1%、响应 ≤ 1s），纯 LLM 判断达不到稳定性。因此：

- **规则引擎**（纯 Python）：HR > 100 → high，SpO2 < 90% → forbidden 等确定性判定。
- **医学 RAG + LLM**：仅负责灰色地带的话术生成与建议，不参与安全判定主链路。
- **身体电量**：确定性算法引擎计算基线/分数，LLM 仅负责解读与个性化建议。

### 7.2 interrupt 实现问询打断

```python
from langgraph.types import interrupt, Command

def goal_inquiry_node(state: AgentState):
    user_input = interrupt({"question": "你今天想做哪种类型的运动？"})
    if not is_valid_answer(user_input):
        return Command(goto="goal_inquiry")   # 模糊指令重新追问
    return {"goal": parse_goal(user_input)}
```

State 中 `interrupted: bool` 标志"打断即清空当前播报状态"，前端只需上报打断事件。

### 7.3 流式方案分析

`analyze_plan` 节点用 LangChain `.astream_events()` 实现真流式，配合 LangGraph `stream_mode="messages"` 把 token 一路透传到前端。

### 7.4 checkpointer 落盘

- 默认 `InMemorySaver`（开发可跑通，支持 interrupt resume，重启丢失）。
- 配置 Postgres 后自动切换 `PostgresSaver`（落盘，支撑 7 天基线校准与长期记忆）。
- `interrupt` 必须配合 checkpointer + `thread_id` 才能恢复。

## 8. 三层 Memory 设计

| 层级 | 存储 | 内容 |
|---|---|---|
| Session Memory | Redis | 当前运动的对话与状态 |
| User Long Memory | PostgreSQL + Mem0 | 用户长期画像、偏好、常用地点、目标 |
| Vector Memory | pgvector | 运动医学知识、动作库、训练方案 |

## 9. LangSmith 在项目中的作用

需求表里几乎每行都带硬性 SLA（响应时间、误报率、识别率、匹配度），LangSmith 不是可选项：

1. **全链路 Trace 排查延迟**：定位瓶颈在检索还是生成。
2. **离线评估集**：风险识别误报率/漏报率、语音识别率、方案匹配度的回归测试。
3. **在线监控**：P95 延迟、异常节点占比、high-risk 分支触发率。
4. **Prompt A/B 测试**：对比方案采纳率。
5. **反馈回流**：bad case 标注进数据集持续迭代。

只需开启环境变量即自动接入，无需侵入业务代码。

## 10. 落地优先级

1. 先搭 **状态机骨架** 与 **高风险分支**（安全底线，规则用纯代码）。
2. 医学/运动知识库用 LangChain Retriever + 向量库跑通 RAG，再接入结构化输出。
3. 从第一天起打开 LangSmith tracing 积累种子数据。
4. 7 天基线、长期记忆用 checkpointer 落 Postgres，而非内存态。
5. 打断交互细节先在 LangGraph 层定义清楚语义，前端只上报事件。

## 11. 推荐技术栈

| 模块 | 技术 |
|---|---|
| Agent Framework | LangGraph |
| LLM Framework | LangChain |
| LLM | OpenAI / DeepSeek / Qwen 等 OpenAI 协议兼容 |
| Vector DB | pgvector |
| Database | PostgreSQL |
| Cache | Redis |
| 长期记忆 | Mem0 |
| 报告解析 | pypdf |
| API | FastAPI |
| Streaming | WebSocket / SSE |
| Monitoring | LangSmith |
| Deployment | K8s |
