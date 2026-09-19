# tests/test_orders.py
import pytest

from app.services import orders
from app.services.orders import find_order_ids, get_order, list_orders
from tests.conftest import TEST_USER_ID

OTHER_USER = "22222222-2222-2222-2222-222222222222"


@pytest.fixture(autouse=True)
def _frozen_clock():
    from datetime import datetime
    orders.set_clock(lambda: datetime(2026, 9, 19, 12, 0, 0))
    yield
    orders.reset_clock()


def test_list_orders_deterministic_and_namespaced():
    a = list_orders(TEST_USER_ID)
    assert [o.order_id for o in a] == ["1111-1001", "1111-1002", "1111-1003", "1111-1004"]
    assert a == list_orders(TEST_USER_ID)  # 确定性:同一用户两次相同
    assert all(o.queried_at == "2026-09-19T12:00:00" for o in a)


def test_users_isolated():
    a = list_orders(TEST_USER_ID)
    b = list_orders(OTHER_USER)
    assert {o.order_id for o in a}.isdisjoint({o.order_id for o in b})
    assert get_order(OTHER_USER, "1111-1001") is None  # 交叉访问不得命中


def test_get_order_normalizes_and_validates():
    o = get_order(TEST_USER_ID, " 1111-1001 ")
    assert o is not None and o.product == "保温杯" and o.status == "已完成"
    assert get_order(TEST_USER_ID, "1111-9999") is None
    assert get_order(TEST_USER_ID, "1001") is None  # 缺命名空间不猜


def test_scenarios_cover_policy_cases():
    by_id = {o.order_id: o for o in list_orders(TEST_USER_ID)}
    assert by_id["1111-1001"].delivered_at is not None          # 7 天内可退
    assert "超" in by_id["1111-1002"].returnable_note            # 超期
    assert "定制" in by_id["1111-1003"].returnable_note          # 特殊类目
    assert by_id["1111-1004"].delivered_at is None               # 在途未签收


def test_find_order_ids():
    assert find_order_ids("订单 1111-1001 和 1111-1002 能退吗") == ["1111-1001", "1111-1002"]
    assert find_order_ids("小写 1111-1001 重复 1111-1001") == ["1111-1001"]
    assert find_order_ids("没有订单号") == []
