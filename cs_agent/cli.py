"""cs_agent.cli — 命令行入口（v0.1 验收界面）

用法：
    单次模式：python -m cs_agent --text "我的订单 A1002 发货了吗"
    交互模式：python -m cs_agent
"""

import argparse
import json

from .config import Config
from .intent import IntentRouter
from .llm import LLMClient


def _print_result(result) -> None:
    """v0.1 验收输出：意图 JSON + 路由信号。"""
    print("\n>>> 意图识别结果：")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if result.should_handoff:
        print("\n⚠️  [路由信号] 建议转人工"
              + ("（置信度不足）" if result.low_confidence else "")
              + ("（输出解析失败，已降级）" if result.degraded else "")
              + ("（用户情绪/诉求超范围）" if result.need_human else ""))
    else:
        print(f"\n✅ [路由信号] 意图明确（{result.intent}），可进入自动处理流程。")


def main() -> None:
    parser = argparse.ArgumentParser(description="cs-agent v0.1 · 意图识别路由")
    parser.add_argument("--text", type=str, default=None,
                        help="单次模式：直接识别这句话的意图")
    parser.add_argument("--threshold", type=float, default=None,
                        help="覆盖置信度阈值（默认取 .env 配置）")
    args = parser.parse_args()

    llm = LLMClient(Config.load())
    router = IntentRouter(llm, threshold=args.threshold)

    if args.text:
        _print_result(router.classify(args.text))
        return

    # 交互模式
    print("cs-agent v0.1 · 意图识别路由（输入 q / quit / 退出 结束）")
    print("提示：试试包含订单号的句子，比如「订单 A1003 怎么还没到，我要退款！」\n")
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
        _print_result(router.classify(text))


if __name__ == "__main__":
    main()
