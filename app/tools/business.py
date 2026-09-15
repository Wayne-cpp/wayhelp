import json
import random
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Literal

from langchain_core.tools import BaseTool, tool
from pydantic import StringConstraints

from app.knowledge.retriever import (
    NOTE_REBUILDING,
    NOTE_REBUILD_REQUIRED,
    NOTE_UNCONFIGURED,
    RetrievalResult,
    assemble_evidence,
)
from app.models import Conversation, Ticket

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

SCOPE_ENUM = Literal["faq", "policy", "product_spec", "after_sales_manual",
                     "qa_mined", "manual"]


@dataclass
class RetrievalTrace:
    """每轮 query_faq 调用的显式 interface(spec §7.1);初始 not_called,完成后只写一次。"""
    status: str = "not_called"   # not_called | ok | low_confidence | tool_error
    result: RetrievalResult | None = None
    evidence: list[dict] | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class TurnToolset:
    tools: list
    retrieval_trace: RetrievalTrace

    @property
    def tools_by_name(self) -> dict:
        return {t.name: t for t in self.tools}

    # T7→T9 桥接:ChatService 在 T9 前仍把 factory 结果当 list 消费
    # (ToolRegistry 迭代 / `if tools` 真值判断),补齐列表协议保持全链不红
    def __iter__(self):
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def __getitem__(self, index):
        return self.tools[index]


def build_tools(session_factory, conversation_id: int, retriever=None,
                settings=None) -> TurnToolset:
    """每轮请求构造绑定该会话的工具实例;conversation_id 经闭包注入,不对模型暴露。"""
    trace = RetrievalTrace()

    @tool
    def query_faq(keyword: Keyword, scope: SCOPE_ENUM | None = None) -> str:
        """查询知识库。参数 keyword 为用户问题或关键词;scope 可选,限定知识范围
        (faq 常见问答 / policy 退货退款政策 / product_spec 商品规格 / after_sales_manual
        售后手册 / qa_mined 历史挖掘 / manual 手工录入)。返回检索策略、低置信标记与证据列表。"""
        if retriever is None:
            trace.status = "tool_error"
            trace.error_code = "kb_unconfigured"
            return _json({"evidence": [], "note": "知识检索未配置",
                          "low_confidence": False, "effective_strategy": None})
        result = retriever.search(keyword, scope=scope)
        if result.note in (NOTE_REBUILDING, NOTE_REBUILD_REQUIRED, NOTE_UNCONFIGURED):
            trace.status = "tool_error"
            trace.error_code = ("kb_rebuilding" if result.note == NOTE_REBUILDING
                                else "kb_rebuild_required" if result.note == NOTE_REBUILD_REQUIRED
                                else "kb_unconfigured")
            return _json({"evidence": [], "note": result.note,
                          "low_confidence": False,
                          "effective_strategy": result.effective_strategy})
        evidence = [e.to_dict() for e in assemble_evidence(
            result.hits, max_items=settings.rerank_top_n if settings else 10,
            budget_chars=(settings.max_tool_result_chars if settings else 4000),
            overhead_chars=200)] if result.hits else []
        trace.status = "low_confidence" if result.low_confidence else "ok"
        trace.result = result
        trace.evidence = evidence
        return _json({"effective_strategy": result.effective_strategy,
                      "low_confidence": result.low_confidence,
                      "note": result.note, "evidence": evidence})

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

    return TurnToolset([query_order, query_product, query_logistics, query_faq,
                        create_ticket], trace)
