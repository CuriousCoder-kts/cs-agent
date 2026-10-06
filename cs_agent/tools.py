"""cs_agent.tools — 工具注册表（v0.5 核心）

设计目标：把 day02 手写的"硬编码工具清单"升级为**装饰器注册表**。
一个 @tool 装饰器同时登记三件事，从此加工具只改一处：

    1. Python 实现（模型点菜后由我们执行）
    2. JSON schema（交给模型的菜单）
    3. 策略性 description（写进 prompt，影响模型何时选它）

为什么用装饰器？回顾 Day 2 的教训：工具枚举、prompt 说明、执行函数三处
不同步就是 bug 之源。注册表让"加一个工具"变成只写一个函数 + 一个装饰器。

公开接口：
    @tool(name, description, parameters)   —— 注册一个工具
    TOOLS_SCHEMA                            —— 给模型看的工具菜单（list[dict]）
    execute_tool(name, args) -> str         —— 执行工具，异常安全（永远返回字符串）
"""

import json
from typing import Any, Callable

# 内部注册表：name -> {"fn": callable, "schema": dict}
_REGISTRY: dict[str, dict[str, Any]] = {}


def tool(name: str, description: str, parameters: dict):
    """装饰器：把一个函数注册为可被模型调用的工具。

    parameters 用 JSON Schema 描述参数。description 是"策略注入点"——
    Day 2 讲过：工具描述可承载策略，写清"什么时候用、什么时候不用"，
    模型的选择准确率显著提升。
    """
    def decorator(fn: Callable) -> Callable:
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        if name in _REGISTRY:
            raise ValueError(f"工具名重复：{name}（工具名必须全局唯一）")
        _REGISTRY[name] = {"fn": fn, "schema": schema}
        return fn  # 原样返回，函数本身仍可直接调用（便于测试）
    return decorator


# 冒烟数据（v0.5 用内存假数据；v1 换真实数据库 / API）
_FAKE_ORDERS = {
    "A1001": {
        "status": "已发货",
        "carrier": "顺丰",
        "tracking_no": "SF1234567890",
        "unit_price": 299.0,
        "quantity": 1,
        "paid_total": 289.0,   # 用了新人券减 10
        "item": "牛油果绿连衣裙",
    },
    "A1002": {
        "status": "已签收",
        "carrier": "中通",
        "tracking_no": "ZT9876543210",
        "unit_price": 128.0,
        "quantity": 3,
        "paid_total": 307.2,   # 128×3=384，打八折后实付 307.2
        "item": "纯棉基础款T恤 ×3",
    },
    "A1003": {
        "status": "运输中",
        "carrier": "顺丰",
        "tracking_no": "SF2233445566",
        "unit_price": 459.0,
        "quantity": 1,
        "paid_total": 459.0,
        "item": "轻量运动跑鞋",
    },
}

_FAKE_STOCK = {
    "SKU-1001": {"name": "牛油果绿连衣裙", "price": 299.0, "stock": 128},
    "SKU-1002": {"name": "轻量运动跑鞋", "price": 459.0, "stock": 30},
    "SKU-1003": {"name": "无线降噪耳机", "price": 899.0, "stock": 200},
    "SKU-1004": {"name": "保温杯", "price": 129.0, "stock": 500},
}


@tool(
    name="get_order_status",
    description=(
        "查询订单状态与物流信息。需要用户提供订单号（如 A1001）。"
        "如果用户没给订单号，不要猜，先在回复里向用户追问订单号。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "订单号，形如 A1001",
            }
        },
        "required": ["order_id"],
    },
)
def get_order_status(order_id: str) -> dict:
    order = _FAKE_ORDERS.get(order_id.upper())
    if order is None:
        # 注意：这里不抛异常，而是返回结构化"未找到"——让模型能读懂并自救
        return {"found": False, "message": f"未查询到订单 {order_id}，请确认订单号是否正确"}
    return {"found": True, "order_id": order_id.upper(), **order}


@tool(
    name="get_order_price",
    description=(
        "查询订单的实付金额（paid_total）。退款、赔付类问题必须先调它拿到实付金额，"
        "严禁用单价乘以数量心算——订单可能有折扣，实付才是退款依据。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "订单号，形如 A1002"}
        },
        "required": ["order_id"],
    },
)
def get_order_price(order_id: str) -> dict:
    order = _FAKE_ORDERS.get(order_id.upper())
    if order is None:
        return {"found": False, "message": f"未查询到订单 {order_id}"}
    return {
        "found": True,
        "order_id": order_id.upper(),
        "unit_price": order["unit_price"],
        "quantity": order["quantity"],
        "paid_total": order["paid_total"],   # 唯一权威的退款依据
        "note": "退款金额以此 paid_total 为准，不要用单价×数量",
    }


@tool(
    name="check_stock",
    description="查询商品库存与价格。用户问某商品有没有货、多少钱时调用。",
    parameters={
        "type": "object",
        "properties": {
            "sku": {"type": "string", "description": "商品编号，形如 SKU-1001"},
        },
        "required": ["sku"],
    },
)
def check_stock(sku: str) -> dict:
    item = _FAKE_STOCK.get(sku.upper())
    if item is None:
        return {"found": False, "message": f"未找到商品 {sku}"}
    return {"found": True, "sku": sku.upper(), **item}


@tool(
    name="create_ticket",
    description=(
        "创建人工工单。当用户情绪激动、要求人工、或诉求超出自动处理范围时调用。"
        "summary 要概括用户问题，便于人工客服快速接手。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "问题摘要，一句话"},
            "order_id": {"type": "string", "description": "关联订单号，可为空"},
            "priority": {
                "type": "string",
                "enum": ["low", "normal", "high"],
                "description": "优先级，投诉类用 high",
            },
        },
        "required": ["summary"],
    },
)
def create_ticket(summary: str, order_id: str | None = None, priority: str = "normal") -> dict:
    # v0.5 只模拟；v1 落库并推送工单系统
    return {
        "created": True,
        "ticket_id": "T" + str(abs(hash(summary)) % 100000).zfill(5),
        "summary": summary,
        "order_id": order_id,
        "priority": priority,
        "message": "工单已创建，人工客服将在工作时间内跟进",
    }


def get_tools_schema() -> list[dict]:
    """返回给模型看的工具菜单（每次调用都新建副本，防止外部误改注册表）。"""
    return [entry["schema"] for entry in _REGISTRY.values()]


def tool_names() -> list[str]:
    return list(_REGISTRY.keys())


def execute_tool(name: str, arguments: dict) -> str:
    """执行工具，永远返回字符串（异常也包装成可读文本）。

    这是 Day 2 "错误也是信息"的落地：工具崩溃不该杀死循环，
    而应变成一条 Observation 让模型自己决定下一步。
    """
    entry = _REGISTRY.get(name)
    if entry is None:
        return json.dumps({"error": f"未知工具 {name}"}, ensure_ascii=False)
    try:
        result = entry["fn"](**arguments)
        return json.dumps(result, ensure_ascii=False)
    except TypeError as e:
        # 参数不匹配（模型给错参数名/少给参数）——可读报错，模型能自救
        return json.dumps({"error": f"参数错误：{e}"}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001 —— 护栏：任何异常都变 Observation
        return json.dumps({"error": f"工具执行失败：{type(e).__name__}: {e}"}, ensure_ascii=False)
