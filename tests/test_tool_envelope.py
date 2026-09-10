import json

import pytest

from app.tool_envelope import truncate_content, unwrap, wrap


def test_roundtrip_success():
    text = wrap("结果", True, None, 4000)
    assert json.loads(text)["v"] == 1
    assert unwrap(text) == ("结果", True)


def test_roundtrip_error():
    text = wrap("超时", False, "timeout", 4000)
    payload = json.loads(text)
    assert payload["ok"] is False and payload["error_code"] == "timeout"
    assert unwrap(text) == ("超时", False)


def test_wrap_respects_max_chars():
    text = wrap("x" * 10000, True, None, 300)
    assert len(text) <= 300
    payload = json.loads(text)
    assert payload["truncated"] is True
    content, ok = unwrap(text)
    assert ok is True and len(content) < 10000


def test_truncate_content_matches_wrap():
    content = "订单详情abc中文混合" * 1000  # 万级字符,中英混合
    truncated = truncate_content(content, True, None, 300)
    assert truncated != content  # 必须真的截断(原实现恒不截断)
    payload = json.loads(wrap(truncated, True, None, 300))
    assert "truncated" not in payload  # 一次到位,wrap 不再二次截断
    assert payload["content"] == truncated  # 落库与回灌模型所见逐字一致


def test_truncate_content_error_envelope_matches_wrap():
    message = "超时请重试timeout" * 500
    truncated = truncate_content(message, False, "timeout", 300)
    assert truncated != message
    payload = json.loads(wrap(truncated, False, "timeout", 300))
    assert "truncated" not in payload
    assert payload["message"] == truncated


def test_truncate_content_short_untouched():
    assert truncate_content("余额查询成功", True, None, 300) == "余额查询成功"
    assert truncate_content("工具暂时不可用(已重试)", False, "tool_unavailable", 300) == \
        "工具暂时不可用(已重试)"


def test_unwrap_corrupted_raises():
    with pytest.raises(ValueError):
        unwrap("not-json")
    with pytest.raises(ValueError):
        unwrap('{"v":2,"ok":true,"content":"x"}')  # 版本不符
    with pytest.raises(ValueError):
        unwrap('{"v":1,"content":"x"}')  # 缺 ok
