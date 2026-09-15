import json


def wrap(content: str, ok: bool, error_code: str | None, max_chars: int,
         metadata: dict | None = None) -> str:
    """工具结果持久化 envelope;metadata 非 None → v2,否则保持 v1 逐字节兼容。"""
    truncated = False
    while True:
        payload: dict = {"v": 2 if metadata is not None else 1, "ok": ok}
        if ok:
            payload["content"] = content
        else:
            payload["error_code"] = error_code or "tool_error"
            payload["message"] = content
        if metadata is not None:
            payload["metadata"] = metadata
        if truncated:
            payload["truncated"] = True
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) <= max_chars:
            return text
        # 需要截断:保守估算每字符 1 字节(英文)到 3 字节(中文),逐次减半收敛
        # v2 超长时只截 content 并标 truncated,metadata 原样保留(预算由证据组装阶段保证)
        truncated = True
        overflow = len(text) - max_chars
        cut = max(1, int(len(content) - max(overflow, len(content) // 10)))
        content = content[:cut]


def truncate_content(content: str, ok: bool, error_code: str | None, max_chars: int) -> str:
    """返回使 wrap(...) 不超长的 content(供回灌模型与落库一致使用)。"""
    # wrap 内部循环保证返回值必 <= max_chars,不能用它判断是否超长;
    # 精确式:空 content 的 envelope(含两个引号)+ json 序列化后的 content 长度,与 wrap 输出逐字节一致
    base = len(wrap("", ok, error_code, max_chars)) - 2

    def fits(candidate: str) -> bool:
        return base + len(json.dumps(candidate, ensure_ascii=False)) <= max_chars

    if fits(content):
        return content
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(content[:mid]):
            lo = mid
        else:
            hi = mid - 1
    return content[:lo]


def unwrap(text: str) -> tuple[str, bool]:
    """v1/v2 均返回 (content, ok);其余版本/非法格式抛 ValueError。"""
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("corrupted tool envelope") from exc
    if (not isinstance(payload, dict) or payload.get("v") not in (1, 2)
            or "ok" not in payload):
        raise ValueError("corrupted tool envelope")
    ok = bool(payload["ok"])
    body = payload.get("content") if ok else payload.get("message")
    if not isinstance(body, str):
        raise ValueError("corrupted tool envelope")
    return body, ok


def unwrap_metadata(text: str) -> dict | None:
    """v2 envelope 的 metadata dict;v1 或无 metadata 返回 None。"""
    payload = json.loads(text)
    if not isinstance(payload, dict) or payload.get("v") != 2:
        return None
    md = payload.get("metadata")
    return md if isinstance(md, dict) else None
