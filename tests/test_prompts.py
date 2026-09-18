from app.prompts.intent import INTENT_PROMPT
from app.prompts.service import (
    AGENT_BUDGET_ANSWER, CHITCHAT_REPLY, COMPLAINT_REPLY, FALLBACK_ANSWER,
    KB_UNAVAILABLE_ANSWER, REFUSAL_ANSWER, SERVICE_SYSTEM_PROMPT,
)


def test_refusal_constant_embedded_verbatim():
    assert REFUSAL_ANSWER in SERVICE_SYSTEM_PROMPT
    assert SERVICE_SYSTEM_PROMPT.count(REFUSAL_ANSWER) == 1


def test_prompt_contracts_present():
    for needle in ("[ref_no]", "不承诺", "suggest_options", "不得声称"):
        assert needle in SERVICE_SYSTEM_PROMPT


def test_fixed_answers():
    assert COMPLAINT_REPLY.startswith("非常抱歉")
    assert "客服小蜜" in CHITCHAT_REPLY
    assert KB_UNAVAILABLE_ANSWER == "知识库暂时不可用,请稍后重试。"
    assert AGENT_BUDGET_ANSWER == "本轮处理已达到上限,请缩小问题范围后重试。"
    assert "请求人工帮助并选择相应按钮" in FALLBACK_ANSWER


def test_intent_prompt_shape():
    assert "{query}" in INTENT_PROMPT
    for intent in ("物流", "订单", "商品咨询", "退款退货", "售后", "投诉", "闲聊"):
        assert intent in INTENT_PROMPT
    assert "needs_knowledge" in INTENT_PROMPT
    # 渲染必须用 .replace(字面花括号不能被 .format 吞掉)
    rendered = INTENT_PROMPT.replace("{query}", "测试问题")
    assert "测试问题" in rendered and "{query}" not in rendered
