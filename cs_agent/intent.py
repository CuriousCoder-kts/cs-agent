"""cs_agent.intent — 意图识别路由（v0.1 核心）

Agent 的"分诊台"：先弄清用户要什么，再决定走工具（查订单/退款）、
走 RAG（商品咨询）还是直接回复（闲聊）。路由错了，后面全白干。

迁移自 Day 1 的 02_structured_output.py，升级三点：
1. 实体抽取（order_id）——v0.5 工具调用要用的参数，现在就开始收集；
2. 低置信度转人工（confidence < 阈值）——v0.5 的转人工逻辑种子；
3. 解析失败降级（degraded=True + 转人工）——兜底不是崩溃。
"""

from dataclasses import dataclass, asdict

from .llm import LLMClient, parse_json_safely

# 意图枚举（路由表的关键：模型只能从这里选）
INTENTS = ["查询订单", "申请退款", "咨询购买", "投诉建议", "闲聊", "其他"]
EMOTIONS = ["平静", "不满", "愤怒"]

INTENT_SYSTEM_PROMPT = """你是一名电商客服意图识别引擎。分析用户输入，只输出一个 JSON 对象，不要任何多余文字。

可选意图（intent 字段只能取以下值之一）：
- 查询订单：询问订单状态、物流、发货进度
- 申请退款：要求退款、退货、取消订单
- 咨询购买：询问商品信息、价格、优惠、库存
- 投诉建议：表达不满、投诉服务、提出建议
- 闲聊：打招呼、与购物无关的话题
- 其他：无法归类的诉求

输出 JSON 格式：
{
  "intent": "上述意图之一",
  "confidence": 0到1之间的小数,
  "emotion": "平静 | 不满 | 愤怒",
  "need_human": true或false（用户强烈不满或诉求超出客服范围时为 true）,
  "order_id": "用户提到的订单号（如 A1001），没有则为 null"
}"""


@dataclass
class IntentResult:
    intent: str
    confidence: float
    emotion: str
    need_human: bool
    order_id: str | None
    low_confidence: bool = False   # 置信度低于阈值（v0.5 转人工依据）
    degraded: bool = False         # 模型输出解析失败，走了降级

    @property
    def should_handoff(self) -> bool:
        """是否转人工：用户情绪失控 / 诉求超范围 / 置信度不足 / 解析失败。"""
        return self.need_human or self.low_confidence or self.degraded

    @classmethod
    def from_dict(cls, data: dict, threshold: float) -> "IntentResult":
        """从模型输出的 JSON 构造结果，所有字段都做防御性校验。"""
        intent = data.get("intent", "其他")
        if intent not in INTENTS:
            intent = "其他"
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))  # 钳位到 [0,1]
        emotion = data.get("emotion", "平静")
        if emotion not in EMOTIONS:
            emotion = "平静"
        need_human = bool(data.get("need_human", False))
        order_id = data.get("order_id") or None
        return cls(
            intent=intent,
            confidence=confidence,
            emotion=emotion,
            need_human=need_human,
            order_id=order_id,
            low_confidence=confidence < threshold,
        )

    @classmethod
    def fallback(cls) -> "IntentResult":
        """模型输出解析失败时的降级结果：意图未知 → 必须转人工。"""
        return cls(
            intent="其他",
            confidence=0.0,
            emotion="平静",
            need_human=True,
            order_id=None,
            degraded=True,
        )

    def to_dict(self) -> dict:
        return asdict(self)


class IntentRouter:
    """意图识别器：一句话 → 结构化意图（JSON）+ 路由信号。"""

    def __init__(self, llm: LLMClient, threshold: float | None = None):
        self.llm = llm
        self.threshold = (
            threshold if threshold is not None
            else llm.config.intent_confidence_threshold
        )

    def classify(self, text: str) -> IntentResult:
        """识别意图。解析失败自动降级（fallback），绝不抛异常给上层。"""
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        msg = self.llm.chat(messages, json_mode=True)
        data = parse_json_safely(msg.get("content", "") or "")
        if data is None:
            return IntentResult.fallback()
        return IntentResult.from_dict(data, self.threshold)
