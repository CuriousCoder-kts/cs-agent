"""cs_agent.cli — 命令行入口（v0.5：多轮对话版）

用法：
    单次模式：python -m cs_agent --text "我的订单 A1002 发货了吗"
    交互模式：python -m cs_agent              （多轮记忆，输入 q 退出）
    --intent-only：只跑意图识别（v0.1 行为，便于对照）

v0.5 变化：默认走 AgentSession（多轮 + 工具 + RAG），
意图路由作为"分诊台"在第一轮打印，展示"先分类、后处理"的架构。
"""

import argparse

from .agent import AgentSession
from .config import Config
from .intent import IntentRouter
from .llm import LLMClient
from .rag import KnowledgeBase
from .tools import tool_names


def _print_intent(router: IntentRouter, text: str) -> None:
    r = router.classify(text)
    flag = "  ⚠️ 建议人工" if r.should_handoff else ""
    print(f"  [分诊] 意图={r.intent} 置信度={r.confidence:.2f} "
          f"情绪={r.emotion} 订单号={r.order_id}{flag}")


def _print_reply(reply) -> None:
    print(f"\n客服 > {reply.text}")
    meta = [f"模型调用 {reply.steps} 轮"]
    if reply.tools_used:
        meta.append("工具：" + "、".join(reply.tools_used))
    if reply.handoff:
        meta.append("已转人工")
    print(f"       （{' ｜ '.join(meta)}）")
    for t in reply.trace:
        print(f"       ↳ {t}")


def main() -> None:
    parser = argparse.ArgumentParser(description="cs-agent v0.5 · 多轮对话客服")
    parser.add_argument("--text", type=str, default=None, help="单次模式：问一句就退出")
    parser.add_argument("--intent-only", action="store_true", help="只跑意图识别（v0.1 行为）")
    parser.add_argument("--no-intent", action="store_true", help="不打印分诊信息")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖置信度阈值")
    args = parser.parse_args()

    config = Config.load()
    llm = LLMClient(config)
    router = IntentRouter(llm, threshold=args.threshold)

    if args.intent_only:
        if not args.text:
            print("--intent-only 需要配合 --text 使用")
            return
        _print_intent(router, args.text)
        return

    kb = KnowledgeBase.load(config.knowledge_dir)
    session = AgentSession(llm, kb)
    print(f"cs-agent v0.5 · 知识块 {len(kb.chunks)} 个 / 工具 {len(tool_names())} 个")
    print(f"  工具：{'、'.join(tool_names())}\n")

    if args.text:
        if not args.no_intent:
            _print_intent(router, args.text)
        _print_reply(session.chat(args.text))
        return

    print("交互模式（多轮记忆已开启，输入 q / quit / 退出 结束）")
    print("试试：订单 A1002 显示签收了但我没收到，我要退款！\n")
    while True:
        try:
            text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            continue
        if text.lower() in {"q", "quit", "退出"}:
            print("再见。")
            break
        if not args.no_intent:
            _print_intent(router, text)
        _print_reply(session.chat(text))


if __name__ == "__main__":
    main()
