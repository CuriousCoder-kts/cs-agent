"""cs-agent 骨架自检（不消耗 API 额度）

运行：python tests/test_skeleton.py
v0.1 覆盖：解析兜底 / 意图构造 / 路由信号 / 配置守卫 / Prompt 完整性
v0.5 新增：工具注册表 / 工具执行异常安全 / RAG 切分与检索 / paid_total 语义
"""

import sys
import pathlib
import unittest

# 让 tests/ 目录下的脚本无论从哪里运行都能找到仓库根目录
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cs_agent.config import Config
from cs_agent.intent import INTENTS, INTENT_SYSTEM_PROMPT, IntentResult
from cs_agent.llm import parse_json_safely
from cs_agent.rag import KnowledgeBase, Chunk, _tokenize
from cs_agent.tools import execute_tool, get_tools_schema, tool_names


class TestParseJsonSafely(unittest.TestCase):
    """结构化输出的解析兜底——Day 1 铁律在这里落地。"""

    def test_bare_json(self):
        self.assertEqual(parse_json_safely('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(parse_json_safely('```json\n{"a": 1}\n```'), {"a": 1})

    def test_fenced_without_label(self):
        self.assertEqual(parse_json_safely('```\n{"a": 1}\n```'), {"a": 1})

    def test_chatter_prefix(self):
        # 模型偶尔"客气"：好的，结果如下：{...}
        self.assertEqual(
            parse_json_safely('好的，结果如下：{"intent": "查询订单"}'),
            {"intent": "查询订单"},
        )

    def test_garbage_returns_none(self):
        self.assertIsNone(parse_json_safely("这不是JSON"))
        self.assertIsNone(parse_json_safely(""))


class TestIntentResult(unittest.TestCase):
    def test_normal_construction(self):
        r = IntentResult.from_dict(
            {"intent": "查询订单", "confidence": 0.95, "emotion": "平静",
             "need_human": False, "order_id": "A1002"},
            threshold=0.6,
        )
        self.assertEqual(r.intent, "查询订单")
        self.assertFalse(r.should_handoff)

    def test_invalid_intent_falls_back(self):
        r = IntentResult.from_dict({"intent": "外星指令", "confidence": 0.9}, threshold=0.6)
        self.assertEqual(r.intent, "其他")

    def test_low_confidence_handoff(self):
        r = IntentResult.from_dict({"intent": "闲聊", "confidence": 0.3}, threshold=0.6)
        self.assertTrue(r.low_confidence)
        self.assertTrue(r.should_handoff)

    def test_confidence_clamped(self):
        r = IntentResult.from_dict({"intent": "闲聊", "confidence": 7}, threshold=0.6)
        self.assertEqual(r.confidence, 1.0)
        r2 = IntentResult.from_dict({"intent": "闲聊", "confidence": "垃圾值"}, threshold=0.6)
        self.assertEqual(r2.confidence, 0.0)

    def test_parse_failure_degrades_to_human(self):
        r = IntentResult.fallback()
        self.assertTrue(r.degraded)
        self.assertTrue(r.should_handoff)

    def test_order_id_none_when_missing(self):
        r = IntentResult.from_dict({"intent": "闲聊", "confidence": 0.9}, threshold=0.6)
        self.assertIsNone(r.order_id)


class TestPromptIntegrity(unittest.TestCase):
    """Prompt 是路由的'策略代码'——枚举与格式说明必须齐备。"""

    def test_all_intents_in_prompt(self):
        for intent in INTENTS:
            self.assertIn(intent, INTENT_SYSTEM_PROMPT)

    def test_prompt_has_json_contract(self):
        self.assertIn('"intent"', INTENT_SYSTEM_PROMPT)
        self.assertIn('"confidence"', INTENT_SYSTEM_PROMPT)
        self.assertIn('"order_id"', INTENT_SYSTEM_PROMPT)


class TestConfigGuard(unittest.TestCase):
    def test_require_raises_on_placeholder(self):
        cfg = Config(base_url="", api_key="your_api_key_here", model="")
        with self.assertRaises(SystemExit):
            cfg.require()

    def test_require_passes_on_valid(self):
        cfg = Config(base_url="https://x", api_key="sk-real", model="m")
        cfg.require()  # 不应抛异常


class TestToolRegistry(unittest.TestCase):
    """v0.5 工具注册表——不变量断言，不硬编码数量（Day 2 教训）。"""

    def test_expected_tools_registered(self):
        for expected in ["get_order_status", "get_order_price", "check_stock", "create_ticket"]:
            self.assertIn(expected, tool_names())

    def test_schema_impl_consistent(self):
        """schema 里的每个工具都必须有 name/description/parameters（一致性不变量）。"""
        for schema in get_tools_schema():
            fn = schema["function"]
            self.assertTrue(fn["name"])
            self.assertTrue(fn["description"].strip(), f"{fn['name']} 缺 description")
            self.assertIn("parameters", fn)

    def test_unknown_tool_returns_error_not_raise(self):
        """护栏③：未知工具不抛异常，返回可读 error。"""
        import json
        out = json.loads(execute_tool("no_such_tool", {}))
        self.assertIn("error", out)

    def test_bad_args_return_readable_error(self):
        """护栏③：参数错误不能抛异常，要变成可读 Observation。"""
        import json
        out = json.loads(execute_tool("get_order_status", {"wrong_param": 1}))
        self.assertIn("error", out)

    def test_order_price_is_authoritative(self):
        """paid_total 语义：退款依据，不是单价×数量（A1002 = 307.2 不是 384）。"""
        import json
        out = json.loads(execute_tool("get_order_price", {"order_id": "A1002"}))
        self.assertEqual(out["paid_total"], 307.2)
        self.assertNotEqual(out["paid_total"], out["unit_price"] * out["quantity"])

    def test_missing_order_returns_found_false(self):
        import json
        out = json.loads(execute_tool("get_order_status", {"order_id": "ZZZZ999"}))
        self.assertFalse(out["found"])


class TestTokenize(unittest.TestCase):
    def test_chinese_bigram(self):
        toks = _tokenize("退款政策")
        self.assertIn("退款", toks)
        self.assertIn("政策", toks)

    def test_alnum(self):
        toks = _tokenize("SKU-1001 价格")
        self.assertIn("sku-1001", toks)


class TestKnowledgeBase(unittest.TestCase):
    def test_retrieve_finds_refund_policy(self):
        kb = KnowledgeBase([
            Chunk("p.md", "退款政策", "支持 7 天无理由退货，按实付金额退回"),
            Chunk("p.md", "配送时效", "现货 48 小时内发货"),
        ])
        hits = kb.retrieve("我要退款，怎么退", top_k=1)
        self.assertTrue(hits)
        self.assertEqual(hits[0].title, "退款政策")

    def test_empty_kb_returns_empty(self):
        kb = KnowledgeBase([])
        self.assertEqual(kb.retrieve("任意问题"), [])
        self.assertEqual(kb.as_context("任意问题"), "")

    def test_cite_format(self):
        c = Chunk("policies.md", "退款政策", "正文")
        self.assertEqual(c.cite(), "[policies.md · 退款政策]")


if __name__ == "__main__":
    unittest.main(verbosity=2)
