"""确定性清洗:训练与推理共用同一实现(spec §4.1)。

处理顺序冻结:脱敏 → 繁简(OpenCC t2s)→ 全半角 → 空白折叠。
在线推理只走本模块,不调用 LLM 校对。改动任一步 = 换 PREPROCESS_VERSION。
"""

from __future__ import annotations

import re
import unicodedata

from opencc import OpenCC

PREPROCESS_VERSION = "topic-preprocess-v1"

_CC = OpenCC("t2s")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 邮箱各段限定 ASCII:Python re 的 \w 含汉字,原 [\w.+-] 会把「发票发到a@b.com了吗」
# 整句吞进占位符;spec §4.1 脱敏是 token 级替换,只换地址、保留其余文本。
_EMAIL_RE = re.compile(r"[A-Za-z0-9_.+-]+@[A-Za-z0-9_-]+\.[A-Za-z0-9_.]+")
_ORDER_RE = re.compile(r"\b[A-Za-z]\d{4,}\b|\b\d{8,}\b")
_WS_RE = re.compile(r"\s+")


def _to_halfwidth(text: str) -> str:
    return "".join(
        chr(ord(ch) - 0xFEE0) if 0xFF01 <= ord(ch) <= 0xFF5E
        else (" " if ch == "　" else ch)
        for ch in text)


def clean_text(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ORDER_RE.sub("[ORDER_ID]", text)
    text = _CC.convert(text)
    text = unicodedata.normalize("NFC", text)
    text = _to_halfwidth(text)
    return _WS_RE.sub(" ", text).strip()


def preprocess_config() -> dict:
    return {"preprocess_version": PREPROCESS_VERSION,
            "steps": ["mask_email", "mask_phone", "mask_order_id",
                      "opencc_t2s", "nfc", "halfwidth", "space_collapse"]}
