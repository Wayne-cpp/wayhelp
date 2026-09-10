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
    truncated = truncate_content("x" * 10000, True, None, 300)
    assert wrap(truncated, True, None, 300) == wrap("x" * 10000, True, None, 300)


def test_unwrap_corrupted_raises():
    with pytest.raises(ValueError):
        unwrap("not-json")
    with pytest.raises(ValueError):
        unwrap('{"v":2,"ok":true,"content":"x"}')  # 版本不符
    with pytest.raises(ValueError):
        unwrap('{"v":1,"content":"x"}')  # 缺 ok
