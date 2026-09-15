from app.prompts.service import REFUSAL_ANSWER, SERVICE_SYSTEM_PROMPT


def test_refusal_constant_embedded_verbatim():
    assert REFUSAL_ANSWER in SERVICE_SYSTEM_PROMPT
    assert SERVICE_SYSTEM_PROMPT.count(REFUSAL_ANSWER) == 1


def test_prompt_contracts_present():
    for needle in ("[ref_no]", "low_confidence", "不承诺", "query_faq 每轮最多调用一次"):
        assert needle in SERVICE_SYSTEM_PROMPT
