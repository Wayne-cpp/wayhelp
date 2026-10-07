"""ch08 两个业务 MCP Server 的 mock 数据(spec §4.1):照 ch02 做法随机生成,
按 order_id 种子稳定(同一单多次查一致);不接真实系统、不建表、不认用户。"""
import hashlib
from datetime import datetime, timedelta

_COMPANIES = ["顺丰速运", "中通快递", "圆通速递", "京东物流"]
_TRACE_POOL = ["已揽收", "到达转运中心", "运输中", "派件中", "已签收"]


def _seed(order_id: str) -> int:
    return int(hashlib.sha256(order_id.encode("utf-8")).hexdigest()[:8], 16)


def logistics_trace(order_id: str) -> dict:
    s = _seed(order_id)
    company = _COMPANIES[s % len(_COMPANIES)]
    tracking_no = "SF" + f"{s % 10**10:010d}"
    n = 2 + s % 3                      # 2~4 条轨迹
    base = datetime.now() - timedelta(hours=8 * n)
    traces = [{"time": (base + timedelta(hours=8 * (i + 1))).isoformat(timespec="seconds"),
               "description": _TRACE_POOL[min(i, len(_TRACE_POOL) - 1)]}
              for i in range(n)]
    return {"order_id": order_id, "company": company,
            "tracking_no": tracking_no, "traces": traces}


def warranty_status(order_id: str) -> dict:
    s = _seed(order_id)
    in_warranty = (s % 10) < 6         # 60% 在保
    expire = (datetime.now() + timedelta(days=180 if in_warranty else -30)).date().isoformat()
    return {"order_id": order_id, "in_warranty": in_warranty,
            "warranty_expire": expire,
            "note": "整机保修期内,非人为损坏可免费维修" if in_warranty
                    else "已过保修期,维修需自费"}


def return_progress(order_id: str) -> dict:
    s = _seed(order_id)
    stages = ["退货申请已提交", "商家已同意", "退货物流已揽收", "商家已收货", "退款已到账"]
    n = 1 + s % len(stages)            # 进度停在某一节点
    base = datetime.now() - timedelta(days=n)
    nodes = [{"stage": stages[i],
              "time": (base + timedelta(days=i)).isoformat(timespec="seconds"),
              "done": True} for i in range(n)]
    return {"order_id": order_id, "return_no": "R" + f"{s % 10**8:08d}",
            "current": stages[n - 1], "nodes": nodes}
