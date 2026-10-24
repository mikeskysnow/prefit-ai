"""医学知识库 RAG 工具层：pgvector 向量检索 + 知识库 seeding。

对应文档《风险规则方案.md》Layer 3：
- 复用已有 PostgreSQL（pgvector 扩展），不引入 ChromaDB。
- 检索运动医学指南、PAR-Q+ 标准、禁忌症文献，为 LLM 灰色地带研判提供循证依据。
- 作为辅助层，不在安全判定（红线规则）的关键路径上。

核心 API：
- get_medical_retriever() → LangChain Retriever
- retrieve_medical_knowledge(query) → str（拼接的上下文文本）
- seed_medical_kb() → 初始化知识库（幂等，已存在则跳过）
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from sport_agent.config import get_settings
from sport_agent.utils import get_logger

logger = get_logger(__name__)

_COLLECTION = "sport_medical_kb"
_vectorstore: Optional[Any] = None
_seeded = False

# ===== 医学知识库内容（seed 用）=====
MEDICAL_KB_DOCS: list[dict[str, str]] = [
    {
        "category": "运动禁忌",
        "title": "ACSM 绝对禁忌症",
        "content": (
            "ACSM（美国运动医学会）运动测试绝对禁忌症："
            "近期急性心肌梗死（2天内）、不稳定型心绞痛、未控制的心律失常、"
            "急性肺栓塞或梗死、急性心肌炎或心包炎、主动脉夹层、"
            "急性心力衰竭、未控制的高血压（收缩压>200或舒张压>110 mmHg）。"
            "存在以上任一情况应禁止运动，立即就医。"
        ),
    },
    {
        "category": "运动禁忌",
        "title": "血氧安全阈值",
        "content": (
            "血氧饱和度（SpO2）运动安全标准："
            "正常值 95-100%；90-94% 为偏低，应限制运动强度；"
            "低于 90% 为低氧血症，禁止中高强度运动，建议医学检查。"
            "运动中 SpO2 下降超过 4 个百分点应停止运动。"
        ),
    },
    {
        "category": "运动禁忌",
        "title": "静息心率异常",
        "content": (
            "静息心率（RHR）评估标准："
            "成人正常范围 60-100 次/分。"
            "静息心率>100次/分（心动过速）可能提示甲状腺功能亢进、贫血、感染或心脏疾病，"
            "不建议进行剧烈运动，应进一步检查。"
            "静息心率<50次/分（非运动员）可能提示传导异常。"
        ),
    },
    {
        "category": "运动禁忌",
        "title": "高血压运动限制",
        "content": (
            "高血压患者运动指南（ACSM）："
            "血压>180/110 mmHg 为运动禁忌，应先药物控制。"
            "血压 160-179/100-109 mmHg 仅可进行低强度有氧运动（散步、太极）。"
            "血压控制良好（<140/90）可进行中等强度运动，避免 Valsalva 动作（屏气发力）。"
            "推荐有氧运动为主，避免大重量抗阻训练。"
        ),
    },
    {
        "category": "运动禁忌",
        "title": "糖尿病运动注意事项",
        "content": (
            "糖尿病患者运动指南："
            "血糖>16.7 mmol/L 且血酮阳性时禁止运动；"
            "血糖<5.6 mmol/L 运动前应补充碳水化合物。"
            "运动中携带糖果防低血糖；避免在胰岛素作用高峰期运动。"
            "推荐中低强度有氧运动，每周150分钟以上。"
            "伴有视网膜病变者避免高冲击和头部倒立运动。"
        ),
    },
    {
        "category": "指标阈值",
        "title": "心率变异性（HRV）解读",
        "content": (
            "心率变异性（HRV，rMSSD）评估恢复状态："
            "成年男性正常范围 25-70 ms，女性 20-60 ms（随年龄下降）。"
            "HRV 较个人基线下降>20% 提示恢复不足或过度训练。"
            "HRV 持续偏低（<25ms）可能提示慢性压力、睡眠不足或自主神经功能异常。"
            "HRV 高于基线提示恢复良好，可进行高强度训练。"
        ),
    },
    {
        "category": "指标阈值",
        "title": "年龄别最大心率与靶心率区间",
        "content": (
            "最大心率估算公式：HRmax = 208 - 0.7×年龄（田中公式，推荐使用）。"
            "Zone1（恢复）：50-60% HRmax；Zone2（有氧）：60-70% HRmax；"
            "Zone3（耐力）：70-80% HRmax；Zone4（阈值）：80-90% HRmax；Zone5（最大）：90-100% HRmax。"
            "初学者从 Zone1-2 开始，逐步进阶。"
            "55岁以上人群靶心率建议降低 5-10%。"
        ),
    },
    {
        "category": "指标阈值",
        "title": "压力与恢复评估",
        "content": (
            "压力指数（Stress Score）评估："
            "0-30 为低压力，恢复良好；31-60 为中等压力，可正常训练；"
            "61-80 为高压力，建议降低训练强度；>80 为极高压力，推荐恢复性活动。"
            "持续高压力状态（>70持续3天以上）提示过度训练或心理负荷过大。"
            "压力指数结合 HRV 下降可更准确判断恢复不足。"
        ),
    },
    {
        "category": "特殊人群",
        "title": "中老年人运动处方",
        "content": (
            "50岁以上人群运动注意事项："
            "运动前建议完成 PAR-Q+ 问卷；有慢性病者需医生评估。"
            "推荐有氧运动（快走、游泳、骑行）为主，每周150分钟中等强度。"
            "每周2-3次力量训练，使用自重或轻阻力。"
            "平衡训练（太极、单脚站立）预防跌倒。"
            "避免突然高强度运动，热身和拉伸时间延长至10-15分钟。"
        ),
    },
    {
        "category": "特殊人群",
        "title": "心血管疾病患者运动",
        "content": (
            "心脏病/冠心病患者运动指南："
            "必须在医生评估后制定运动处方。"
            "仅限低中强度有氧运动（心率不超过靶心率下限）。"
            "避免等长收缩运动（如平板支撑、重负荷力量训练）。"
            "运动中如出现胸闷、气短、头晕、心悸应立即停止。"
            "建议在有监护条件下运动，随身携带急救药物。"
        ),
    },
    {
        "category": "就医指征",
        "title": "运动前就医指征",
        "content": (
            "出现以下情况应建议就医，不宜自行运动："
            "1. 静息心率突然升高>20次/分且持续2天以上；"
            "2. 血氧持续<95%；"
            "3. 运动中出现胸痛、晕厥、严重气短；"
            "4. 静息血压>160/100 mmHg；"
            "5. 心律不齐（心跳不规则感）；"
            "6. 近期有感冒发烧等感染症状（建议症状消失后休息7天再运动）。"
        ),
    },
    {
        "category": "药物影响",
        "title": "常见药物对运动心率的影响",
        "content": (
            "β受体阻滞剂（如美托洛尔）：降低静息和运动心率，靶心率公式不适用，"
            "应使用 RPE（主观疲劳感）评估强度。"
            "利尿剂：可能引起电解质紊乱，运动中注意补钾补水。"
            "胰岛素/口服降糖药：运动可增强降糖效果，注意低血糖风险。"
            "钙通道阻滞剂：可能影响心率，运动强度应保守。"
            "服药人群运动前应咨询医生，运动中密切监测身体反应。"
        ),
    },
    {
        "category": "灰色地带",
        "title": "PAR-Q+ 筛查标准",
        "content": (
            "PAR-Q+（体力活动适应能力问卷）核心筛查项："
            "1. 医生是否曾告知你只能在医生监督下进行体力活动？"
            "2. 你在体力活动时是否曾感到胸痛？"
            "3. 过去一个月内你是否曾在没有体力活动时感到胸痛？"
            "4. 你是否曾因头晕失去平衡或晕倒？"
            "5. 你是否有 bone/joint 问题可能因运动加重？"
            "6. 医生是否正在为你的血压或心脏问题开药？"
            "7. 你是否知道自己有任何不适合运动的原因？"
            "任一回答为'是'应进一步医学评估。"
        ),
    },
    {
        "category": "灰色地带",
        "title": "综合风险研判原则",
        "content": (
            "灰色地带综合研判原则："
            "当单项指标未触发红线但多个指标组合异常时，需综合评估："
            "1. 年龄>55 + 静息心率>90 → 建议中低强度，避免剧烈运动；"
            "2. HRV低于基线20% + 压力>60 → 恢复不足，建议恢复性运动；"
            "3. 血氧92-94% + 有呼吸系统疾病史 → 限制运动强度，建议检查；"
            "4. 近期感冒/发烧恢复期（<7天）→ 仅低强度，逐步恢复。"
            "研判结果应标注置信度，低置信度时建议保守处理。"
        ),
    },
]


def get_embeddings() -> Any:
    """获取 Embedding 模型实例（使用 embedding 专用配置，未配置时复用 LLM 配置）。"""
    from langchain_openai import OpenAIEmbeddings

    s = get_settings()
    return OpenAIEmbeddings(
        api_key=s.effective_embedding_api_key,
        base_url=s.effective_embedding_base_url,
        model=s.embedding_model,
    )


def get_vectorstore() -> Any:
    """获取 PGVector 向量库单例（复用已有 Postgres 连接）。

    首次调用时自动建表 + seeding 医学知识库。
    若 pgvector 不可用则返回 None，调用方需处理降级。
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    s = get_settings()
    if not s.postgres_url:
        logger.warning("rag.skip_no_postgres")
        return None

    try:
        from langchain_postgres import PGVector

        _vectorstore = PGVector(
            embeddings=get_embeddings(),
            connection=s.postgres_url,
            collection_name=_COLLECTION,
        )
        logger.info("rag.vectorstore.ready", collection=_COLLECTION)
        # 首次使用时自动 seeding
        _auto_seed(_vectorstore)
        return _vectorstore
    except Exception as e:  # noqa: BLE001
        logger.warning("rag.vectorstore_failed", error=str(e))
        return None


def _auto_seed(vs: Any) -> None:
    """幂等 seeding：检查知识库是否已有数据，无则写入。"""
    global _seeded
    if _seeded:
        return
    try:
        # 用一条简单查询探测是否已有数据
        probe = vs.similarity_search("运动禁忌", k=1)
        if probe:
            _seeded = True
            return
        seed_medical_kb(vs)
        _seeded = True
    except Exception as e:  # noqa: BLE001
        logger.warning("rag.auto_seed_failed", error=str(e))


def seed_medical_kb(vs: Any | None = None) -> int:
    """将医学知识库文档写入 pgvector（幂等）。

    Returns:
        写入的文档数量
    """
    vs = vs or get_vectorstore()
    if vs is None:
        logger.warning("rag.seed_skip_no_vectorstore")
        return 0

    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=d["content"],
            metadata={"category": d["category"], "title": d["title"]},
        )
        for d in MEDICAL_KB_DOCS
    ]
    try:
        vs.add_documents(docs)
        logger.info("rag.seeded", count=len(docs))
        return len(docs)
    except Exception as e:  # noqa: BLE001
        logger.warning("rag.seed_failed", error=str(e))
        return 0


def get_medical_retriever() -> Any:
    """获取医学知识库检索器（MMR 搜索，k=3，减少冗余）。"""
    vs = get_vectorstore()
    if vs is None:
        return None
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},
    )


def _retrieve_uncached(query: str) -> str:
    retriever = get_medical_retriever()
    if retriever is None:
        return ""

    try:
        docs = retriever.invoke(query)
        if not docs:
            return ""
        context = "\n---\n".join(
            f"[{d.metadata.get('title', '')}] {d.page_content}" for d in docs
        )
        logger.info("rag.retrieved", query=query[:50], docs=len(docs))
        return context
    except Exception as e:  # noqa: BLE001
        logger.warning("rag.retrieve_failed", error=str(e))
        return ""


# 检索需要先把 query 送去 embedding（实测约 0.5s/次，网络抖动时更久），
# 而 query 由生理指标档位拼成，同类状态高度重复，故按 query 缓存整段上下文。
# 知识库是启动时 seeding 的静态内容，进程生命周期内缓存不会失效。
_retrieve_cached = lru_cache(maxsize=256)(_retrieve_uncached)


def retrieve_medical_knowledge(query: str) -> str:
    """检索医学知识库，返回拼接的上下文文本。

    用于 LLM 灰色地带研判的 context（Layer 3）。
    若 RAG 不可用则返回空字符串，调用方降级到纯 LLM 推理。
    """
    context = _retrieve_cached(query)
    if not context:
        # 空结果不缓存：可能是 RAG 暂时不可用，下次应重试
        _retrieve_cached.cache_clear()
    return context
