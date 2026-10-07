"""cs-agent 评测器（v0.6 · 第一把尺子）

用法（在仓库根目录）：
    ../agent-learning/.venv/Scripts/python.exe eval/run_eval.py --tag baseline_v05

评测维度（对齐外部评审 2026-10-07 建议）：
    1. 意图准确率   —— IntentRouter 分类是否命中预期
    2. 工具选择     —— 应调的调了吗（子集匹配）、不该调的忍住了吗（空集匹配）
    3. 转人工判定   —— 该转的转了吗、不该转的误转了吗
    4. 检索命中     —— 期望的知识小节是否进入 TopK
    5. 系统表现     —— 每例延迟、模型调用轮数

原则：失败案例全部留档进报告——它们是之后每次改动的回归集。
数字不求漂亮，求可复现。
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cs_agent.agent import AgentSession          # noqa: E402
from cs_agent.config import Config               # noqa: E402
from cs_agent.intent import IntentRouter         # noqa: E402
from cs_agent.llm import LLMClient               # noqa: E402
from cs_agent.rag import KnowledgeBase           # noqa: E402

CATEGORIES = ["正常查询", "信息缺失", "歧义表达", "越权请求", "无关问题", "需人工接手"]


def load_cases(path: Path) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            cases.append(json.loads(line))
    return cases


def run_case(case: dict, llm: LLMClient, router: IntentRouter, kb: KnowledgeBase) -> dict:
    result = {
        "id": case["id"], "category": case["category"], "query": case["query"],
        "note": case.get("note", ""),
        "intent_ok": None, "tools_ok": None, "handoff_ok": None, "retrieval_ok": None,
        "got_intent": None, "got_tools": [], "got_handoff": None,
        "got_steps": 0, "latency_s": 0.0, "error": None, "reply_snip": "",
    }
    try:
        t0 = time.perf_counter()
        # ① 意图（独立调用，不影响 Agent 会话）
        if case["expected_intent"] is not None:
            ir = router.classify(case["query"])
            result["got_intent"] = ir.intent
            result["intent_ok"] = ir.intent == case["expected_intent"]

        # ② Agent 全流程
        session = AgentSession(llm, kb)
        reply = session.chat(case["query"])
        result["latency_s"] = round(time.perf_counter() - t0, 2)
        result["got_steps"] = reply.steps
        result["got_tools"] = reply.tools_used
        result["got_handoff"] = reply.handoff
        result["reply_snip"] = (reply.text or "")[:120].replace("\n", " ")

        # 工具判定：null=不判；[]=必须零调用；列表=必须包含（子集匹配）
        if case["expected_tools"] is not None:
            exp, got = set(case["expected_tools"]), set(reply.tools_used)
            result["tools_ok"] = exp.issubset(got) if exp else (got == set())

        # 转人工判定
        result["handoff_ok"] = reply.handoff == case["expected_handoff"]

        # 检索命中判定
        if case["expected_chunk"]:
            titles = [c.title for c in kb.retrieve(case["query"], top_k=3)]
            result["retrieval_ok"] = case["expected_chunk"] in titles
    except SystemExit as e:
        result["error"] = f"配置/鉴权中断: {e}"
    except Exception as e:  # noqa: BLE001
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n / d * 100:.0f}%)" if d else "—"


def build_report(results: list[dict], tag: str, model: str) -> str:
    lines = [f"# 评测报告 · {tag}", "",
             f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}",
             f"- 模型：{model}",
             f"- 用例数：{len(results)}",
             "- 判定口径：意图=分类命中（null 不判）；工具=null 不判 / [] 必须零调用 / 列表须包含；"
             "转人工=与预期一致；检索=期望小节进 Top3", ""]

    # 总览
    checked = lambda k: [r for r in results if r[k] is not None]  # noqa: E731
    lines += ["## 总览", "", "| 维度 | 通过率 |", "|---|---|"]
    for key, name in [("intent_ok", "意图准确"), ("tools_ok", "工具选择"), ("handoff_ok", "转人工判定"),
                      ("retrieval_ok", "检索命中(Top3)")]:
        sub = checked(key)
        ok = sum(1 for r in sub if r[key])
        lines.append(f"| {name} | {fmt_pct(ok, len(sub))} |")
    lat = [r["latency_s"] for r in results if r["latency_s"] and not r["error"]]
    if lat:
        lines.append(f"| 平均延迟 | {statistics.mean(lat):.1f}s（p95 {sorted(lat)[int(len(lat)*0.95)-1]:.1f}s） |")
    errs = [r for r in results if r["error"]]
    if errs:
        lines.append(f"| 运行错误 | {len(errs)} 例 |")
    lines.append("")

    # 分场景
    lines += ["## 分场景", "", "| 场景 | 例数 | 意图 | 工具 | 转人工 | 检索 |", "|---|---|---|---|---|---|"]
    for cat in CATEGORIES:
        sub = [r for r in results if r["category"] == cat]
        if not sub:
            continue
        row = [cat, str(len(sub))]
        for key in ("intent_ok", "tools_ok", "handoff_ok", "retrieval_ok"):
            s2 = [r for r in sub if r[key] is not None]
            row.append(fmt_pct(sum(1 for r in s2 if r[key]), len(s2)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # 失败案例（回归集种子）
    lines += ["## 失败案例（回归集种子，改动后必须重跑）", ""]
    fails = [r for r in results if any(r[k] is False for k in ("intent_ok", "tools_ok", "handoff_ok", "retrieval_ok")) or r["error"]]
    if not fails:
        lines.append("（无失败案例——警惕：评测集可能太软，下一步加难度）")
    for r in fails:
        bad = [k for k in ("intent_ok", "tools_ok", "handoff_ok", "retrieval_ok") if r[k] is False]
        lines.append(f"- **{r['id']}**（{r['category']}）「{r['query']}」")
        lines.append(f"  - 未过：{', '.join(bad) if bad else r['error']}")
        exp_intent = r.get("got_intent")
        lines.append(f"  - 意图 got={exp_intent}｜工具 got={r['got_tools']}｜转人工 got={r['got_handoff']}")
        lines.append(f"  - 回复摘录：{r['reply_snip']}…")
        if r["note"]:
            lines.append(f"  - 备注：{r['note']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="cs-agent 评测器")
    parser.add_argument("--cases", default=str(REPO_ROOT / "eval" / "cases.jsonl"))
    parser.add_argument("--tag", default="baseline_v05", help="报告标签")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    args = parser.parse_args()

    config = Config.load()
    llm = LLMClient(config)
    router = IntentRouter(llm)
    kb = KnowledgeBase.load(config.knowledge_dir)
    cases = load_cases(Path(args.cases))
    if args.limit:
        cases = cases[: args.limit]

    print(f"评测开始：{len(cases)} 例，模型 {config.model}，知识块 {len(kb.chunks)}")
    results = []
    for i, case in enumerate(cases, 1):
        r = run_case(case, llm, router, kb)
        marks = "".join(
            "✓" if r[k] else ("✗" if r[k] is False else "－")
            for k in ("intent_ok", "tools_ok", "handoff_ok", "retrieval_ok")
        )
        print(f"[{i:>2}/{len(cases)}] {r['id']} {r['category']:<5} {marks} "
              f"{r['latency_s']}s" + (f"  ERR:{r['error']}" if r["error"] else ""))
        results.append(r)

    report = build_report(results, args.tag, config.model)
    out_dir = REPO_ROOT / "docs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.tag}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n报告已写入 {out_path}")
    # 控制台简版
    for key, name in [("intent_ok", "意图"), ("tools_ok", "工具"), ("handoff_ok", "转人工"), ("retrieval_ok", "检索")]:
        sub = [r for r in results if r[key] is not None]
        print(f"  {name}: {fmt_pct(sum(1 for r in sub if r[key]), len(sub))}")


if __name__ == "__main__":
    main()
