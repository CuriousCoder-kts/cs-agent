"""cs_agent.llm — LLM 客户端（OpenAI 兼容接口）

迁移自 agent-learning/day02 的 llm_client.py（Sir 手写版）。
原则不变：HTTP 细节、配置自检、错误翻译只写这一遍。

公开接口：
    LLMClient(config).chat(messages, tools=None, json_mode=False) -> dict（完整 message）
    parse_json_safely(text) -> dict | None（结构化输出的解析兜底）
"""

import json

import requests

from .config import Config


def parse_json_safely(text: str) -> dict | None:
    """安全解析 JSON：处理模型偶尔包裹 ```json 代码块 / 前后废话的情况。

    解析失败返回 None（而不是抛异常）——调用方据此走降级策略（转人工）。
    这是 Day 1 的铁律：模型一定会有不听话的时候，兜底必须存在。
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 兜底：截取第一个 { 到最后一个 } 之间再试一次（对付"好的，结果如下：{...}"）
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


class LLMClient:
    """OpenAI 兼容接口的极简客户端。"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        json_mode: bool = False,
    ) -> dict:
        """单次模型调用，返回完整 message 对象（不丢 tool_calls）。

        - tools=None：不给模型工具菜单；
        - json_mode=True：尝试启用 response_format json_object（不支持时静默降级）。
        """
        self.config.require()
        url = f"{self.config.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0.0,  # 决策类任务要稳定，不要创造性
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = requests.post(url, headers=headers, json=payload, timeout=60)

        # 高频错误的人话翻译（生产里对应日志 + 分级告警）
        if resp.status_code == 401:
            raise SystemExit(
                "\n[鉴权失败 401] Key 无效。检查 .env 中 LLM_API_KEY 是否为真实 Key、"
                "复制是否完整、与 BASE_URL 是否同一平台。\n"
            )
        if resp.status_code == 404:
            raise SystemExit(
                f"\n[404] 模型名 {self.config.model!r} 可能不被该平台支持，核对 .env 的 LLM_MODEL。\n"
            )
        if resp.status_code == 429:
            raise SystemExit("\n[限流 429] 请求过频或额度用尽，稍后重试。\n")
        resp.raise_for_status()

        return resp.json()["choices"][0]["message"]
