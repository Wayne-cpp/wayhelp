import json
import random
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import StringConstraints

from app.models import Conversation, Faq, Ticket

OrderId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
ProductName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Keyword = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
Description = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]

_ORDER_STATUS = ["待付款", "待发货", "已发货", "已完成"]
_COMPANIES = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
_TRACE_TEXT = ["已揽收", "到达转运中心", "派件中", "已签收"]


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


@tool
def query_order(order_id: OrderId) -> str:
    """查询订单信息。参数 order_id 为订单号。返回订单状态、金额、商品名、下单时间。"""
    return _json({
        "order_id": order_id,
        "status": random.choice(_ORDER_STATUS),
        "amount": round(random.uniform(9.9, 999.0), 2),
        "product": random.choice(["保温杯", "蓝牙耳机", "机械键盘", "帆布包"]),
        "created_at": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(timespec="seconds"),
    })


@tool
def query_product(product_name: ProductName) -> str:
    """查询商品信息。参数 product_name 为商品名称关键词。返回价格、库存、是否在售。"""
    return _json({
        "product_name": product_name,
        "price": round(random.uniform(9.9, 1999.0), 2),
        "stock": random.randint(0, 500),
        "on_sale": random.random() > 0.1,
    })


@tool
def query_logistics(order_id: OrderId) -> str:
    """查询订单物流。参数 order_id 为订单号。返回物流公司、运单号与物流节点列表。"""
    n = random.randint(2, 4)
    base = datetime.now() - timedelta(hours=n * 8)
    traces = [
        {"time": (base + timedelta(hours=i * 8)).isoformat(timespec="seconds"),
         "description": _TRACE_TEXT[i - 1] if i - 1 < len(_TRACE_TEXT) else "运输中"}
        for i in range(1, n + 1)
    ]
    return _json({
        "order_id": order_id,
        "company": random.choice(_COMPANIES),
        "tracking_no": "SF" + "".join(random.choices("0123456789", k=10)),
        "traces": traces,
    })


MOCK_TOOLS: list[BaseTool] = [query_order, query_product, query_logistics]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def build_tools(session_factory, conversation_id: int) -> list[BaseTool]:
    """每轮请求构造绑定该会话的工具实例;conversation_id 经闭包注入,不对模型暴露。"""

    @tool
    def query_faq(keyword: Keyword) -> str:
        """查询常见问题库。参数 keyword 为关键词,对问题与答案做模糊匹配,返回最相关的前 5 条。"""
        pattern = f"%{_escape_like(keyword)}%"
        with session_factory() as s:
            rows = (
                s.query(Faq)
                .filter((Faq.question.like(pattern, escape="\\"))
                        | (Faq.answer.like(pattern, escape="\\")))
                .order_by(Faq.id)
                .limit(5)
                .all()
            )
        if not rows:
            return _json({"results": [], "note": "未找到与关键词相关的常见问题"})
        return _json({"results": [
            {"question": r.question, "answer": r.answer, "category": r.category} for r in rows
        ]})

    @tool
    def create_ticket(
        description: Description,
        ticket_type: Literal["售后", "投诉", "咨询"],
    ) -> str:
        """创建人工工单并转接人工客服。参数 description 为问题描述,ticket_type 只能是 售后/投诉/咨询。"""
        ticket_no = "T" + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + secrets.token_hex(6)
        with session_factory() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None:
                raise ValueError("conversation not found")
            s.add(Ticket(ticket_no=ticket_no, conversation_id=conversation_id,
                         description=description, ticket_type=ticket_type))
            conv.status = "已转人工"
            s.commit()  # 工单与会话状态同事务,同成同败
        return _json({"ticket_no": ticket_no, "status": "待处理"})

    return [query_order, query_product, query_logistics, query_faq, create_ticket]
