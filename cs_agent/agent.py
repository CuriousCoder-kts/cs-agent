"""cs_agent.agent — Agent 内核（v0.5 核心）

把 Day 2 手写的 guardrailed_agent 移植进主项目，并升级三处：
    1. 工具从"硬编码清单"换成注册表（tools.get_tools_schema()）
    2. 增加对话记忆（messages 跨轮存活）
    3. 接入 RAG：闲聊/咨询类问题先检索知识库再答

保留 Day 2 的三道护栏（它们已在 day02/step3 验证过）：
    ① max_steps 防无限循环    ② seen_calls 去重    ③ 异常全量回传

公开接口：
    AgentSession(llm, kb).chat(user_text) -> AgentReply
"""

import json
from dataclasses import dataclass, field

from .llm import LLMClient
from .rag import KnowledgeBase
from .tools import execute_tool, get_tools_schema, tool_names

SYSTEM_PROMPT = """你是一名专业、友善的电商客服智能体。你的工作流程：

1. 先判断用户诉求类型：
   - 订单/物流/退款/库存 → 调用对应工具获取真实数据，严禁凭空编造订单信息；
   - 商品咨询/售后政策 → 参考下方"知识库资料"作答，并自然说明依据；
   - 打招呼/闲聊 → 直接友好回应。

2. 涉及金额（退款、赔付）必须调用工具拿到实付金额，严禁用单价心算。

3. 如果用户情绪激动或明确要求人工，调用 create_ticket 创建工单并安抚用户。

4. 信息不足时（如没给订单号），先礼貌追问，不要猜测。

5. 回答简洁、口语化，像真人客服，不要暴露你是程序或提及工具名称。

【知识库资料】
{retrieved_context}
"""


@dataclass
class AgentReply:
    text: str
    steps: int = 0                       # 模型调用轮数
    tools_used: list[str] = field(default_factory=list)
    handoff: bool = False                # 是否已转人工
    trace: list[str] = field(default_factory=list)   # 每个工具调用的可读记录


class AgentSession:
    """一次会话：维护对话记忆，驱动 FC 循环。"""

    def __init__(self, llm: LLMClient, kb: KnowledgeBase, max_steps: int = 6):
        self.llm = llm
        self.kb = kb
        self.max_steps = max_steps
        self.messages: list[dict] = []   # 对话记忆：跨轮存活
        self._seen_calls: set[str] = set()

    def _build_system_prompt(self, user_text: str) -> str:
        """每轮按用户问题动态检索知识库，注入 system prompt。"""
        context = self.kb.as_context(user_text, top_k=3) or "（本轮无需知识库资料）"
        return SYSTEM_PROMPT.format(retrieved_context=context)

    def chat(self, user_text: str) -> AgentReply:
        # 每轮重建 system（含最新检索），历史 user/assistant/tool 消息保留
        system_msg = {"role": "system", "content": self._build_system_prompt(user_text)}
        self.messages.append({"role": "user", "content": user_text})

        reply = AgentReply(text="")
        tools = get_tools_schema()

        for step in range(1, self.max_steps + 1):
            reply.steps = step
            # 护栏③：模型调用本身也可能失败，包装成人话
            try:
                msg = self.llm.chat([system_msg, *self.messages], tools=tools)
            except SystemExit:
                raise
            except Exception as e:  # noqa: BLE001
                reply.text = f"抱歉，系统暂时繁忙，请稍后再试。（{type(e).__name__}）"
                return reply

            tool_calls = msg.get("tool_calls")
            # 记忆：assistant 消息（含 tool_calls）必须回存，否则协议违约
            self.messages.append(msg)

            if not tool_calls:
                reply.text = msg.get("content", "") or "（无回复）"
                return reply

            for call in tool_calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}") or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                # 护栏②：同参数重复调用 → 回 warning，促模型换思路
                signature = f"{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                if signature in self._seen_calls:
                    result = json.dumps(
                        {"warning": f"已用相同参数调用过 {name}，请勿重复，换个思路或直接作答"},
                        ensure_ascii=False,
                    )
                else:
                    self._seen_calls.add(signature)
                    result = execute_tool(name, args)   # 护栏③在 tools 内部
                    reply.tools_used.append(name)
                    reply.trace.append(f"{name}({raw_args})")

                if name == "create_ticket":
                    reply.handoff = True

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                })

        # 护栏①：步数用尽
        reply.text = "抱歉，这个问题我处理得有点久了，已为您转接人工客服。"
        reply.handoff = True
        return reply
