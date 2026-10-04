"""cs-agent v0.1 骨架自检（不消耗 API 额度）

运行：python tests/test_skeleton.py
覆盖：JSON 解析兜底 / 意图结果构造 / 路由信号逻辑 / 配置自检 / Prompt 完整性。
"""

import sys
import pathlib
import unittest

# 让 tests/ 目录下的脚本无论从哪里运行都能找到仓库根目录
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cs_agent.config import Config
from cs_agent.intent import INTENTS, INTENT_SYSTEM_PROMPT, IntentResult
from cs_agent.llm import parse_json_safely


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
