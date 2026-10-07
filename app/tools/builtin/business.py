"""ch08 内置业务只读工具:query_order(归属由引擎把门,内部防御性复核)/ query_product。
query_logistics 已下线,物流查询由 MCP Server 接管(spec §4.1)。"""
import json
import random
from dataclasses import asdict
from datetime import datetime, timedelta

from pydantic import BaseModel, Field

from app.services import orders as _orders
from app.tools.builtin import builtin_tool


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


class QueryOrderArgs(BaseModel):
    order_id: str = Field(min_length=1, max_length=64, description="订单号")


@builtin_tool(name="query_order",
              description="查询订单信息。参数 order_id 为订单号。返回订单状态、金额、商品名、下单时间。",
              args_model=QueryOrderArgs, ownership="order", budget_priority=100)
def query_order_fn(args: dict, ctx) -> str:
    o = _orders.get_order(ctx.user_id, args["order_id"])
    if o is None:
        return _json({"error": "订单不存在或不属于当前用户"})
    ctx.order_snapshots.append(asdict(o))
    return _json({"order_id": o.order_id, "status": o.status, "amount": o.amount,
                  "product": o.product, "created_at": o.created_at,
                  "delivered_at": o.delivered_at})


class QueryProductArgs(BaseModel):
    product_name: str = Field(min_length=1, max_length=128, description="商品名称关键词")


@builtin_tool(name="query_product",
              description="查询商品信息。参数 product_name 为商品名称关键词。返回价格、库存、是否在售。",
              args_model=QueryProductArgs, budget_priority=100)
def query_product_fn(args: dict, ctx) -> str:
    return _json({
        "product_name": args["product_name"],
        "price": round(random.uniform(9.9, 1999.0), 2),
        "stock": random.randint(0, 500),
        "on_sale": random.random() > 0.1,
    })
