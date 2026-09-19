"""ch06 确定性 mock 订单服务(spec §11):固定模板按 user_id 实例化,
订单号带用户命名空间({NS}-{seq}),不同用户不重合;不存在与非本人均返回 None。
演示时钟可注入(测试冻结);覆盖 7 天内可退/超期/定制/在途四种政策场景。"""

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Callable


@dataclass(frozen=True)
class OrderInfo:
    order_id: str
    product: str
    amount: float
    status: str            # 待付款/待发货/已发货/已完成
    created_at: str        # ISO 秒级
    delivered_at: str | None
    returnable_note: str   # 无理由退货受限说明;质量问题仍以政策为准
    queried_at: str        # 查询时刻(快照语义)


# (序号, 商品, 金额, 状态, 下单距今天数, 签收距今天数|None, 类目说明)
_TEMPLATES = (
    ("1001", "保温杯", 89.0, "已完成", 3, 2, "普通商品,在 7 天无理由退货期内"),
    ("1002", "蓝牙耳机", 199.0, "已完成", 20, 18, "普通商品,已超 7 天无理由退货期"),
    ("1003", "定制帆布包", 59.0, "已完成", 5, 4, "定制商品,不支持 7 天无理由退货"),
    ("1004", "机械键盘", 349.0, "已发货", 1, None, "普通商品,运输在途"),
)

_clock: Callable[[], datetime] = datetime.now


def set_clock(fn: Callable[[], datetime]) -> None:
    global _clock
    _clock = fn


def reset_clock() -> None:
    global _clock
    _clock = datetime.now


def _ns(user_id: str) -> str:
    return user_id.replace("-", "")[-4:].upper()


_ORDER_ID_RE = re.compile(r"\b([0-9A-Za-z]{4}-\d{4})\b")


def find_order_ids(text: str) -> list[str]:
    """从文本提取订单号候选(大写归一、保序去重),不做归属判断。"""
    out: list[str] = []
    for m in _ORDER_ID_RE.finditer(text or ""):
        oid = m.group(1).upper()
        if oid not in out:
            out.append(oid)
    return out


def _instantiate(user_id: str) -> list[OrderInfo]:
    now = _clock()
    ns = _ns(user_id)
    out: list[OrderInfo] = []
    for seq, product, amount, status, d_created, d_delivered, note in _TEMPLATES:
        created = now - timedelta(days=d_created)
        delivered = (now - timedelta(days=d_delivered)) if d_delivered is not None else None
        out.append(OrderInfo(
            order_id=f"{ns}-{seq}", product=product, amount=amount, status=status,
            created_at=created.isoformat(timespec="seconds"),
            delivered_at=delivered.isoformat(timespec="seconds") if delivered else None,
            returnable_note=note, queried_at=now.isoformat(timespec="seconds")))
    return out


def list_orders(user_id: str) -> list[OrderInfo]:
    return _instantiate(user_id)


def get_order(user_id: str, order_id: str) -> OrderInfo | None:
    oid = (order_id or "").strip().upper()
    for o in _instantiate(user_id):
        if o.order_id == oid:
            return o
    return None


def order_brief(o: OrderInfo) -> dict:
    """order_selector 帧载荷(不含 returnable_note/queried_at)。"""
    return {"order_id": o.order_id, "product": o.product, "amount": o.amount,
            "status": o.status, "created_at": o.created_at}


def order_summary(o: OrderInfo | dict) -> str:
    """prompt/上下文用单行摘要(接受 OrderInfo 或其 asdict)。"""
    d = asdict(o) if isinstance(o, OrderInfo) else o
    return (f"订单 {d['order_id']}:商品 {d['product']},金额 {d['amount']},"
            f"状态 {d['status']},下单 {d['created_at']},"
            f"签收 {d.get('delivered_at') or '未签收'},{d['returnable_note']}")
