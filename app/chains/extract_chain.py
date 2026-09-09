import logging
from typing import Any

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate

from app.errors import MessageTooLongError, UpstreamError
from app.schemas import AfterSaleExtraction

logger = logging.getLogger(__name__)

EXTRACT_INSTRUCTION = """你是售后工单信息提取器。从用户的售后描述中提取四个字段,输出严格的 JSON 对象,不要输出任何其他内容。

字段定义:
- order_id: 订单号字符串;用户没提则为 null;原样保留前导零。出现多个订单号时,取用户明确指定的主订单,未指定则取首次出现者。
- intent: 诉求类型,只能是 ["退货退款","换货","仅退款","维修","催发货","投诉","其他"] 之一。出现多个诉求时,以最后明确表达的处理诉求为主;只有抱怨且未提出处理诉求时归"投诉";无法判断归"其他"。
- expectation: 期望方案,只能是 ["全额退款","部分退款","换货","补发","维修","赔偿","其他"] 之一或 null。用户未表达补偿或处理结果期望时为 null,不得推测。
- summary: 一句话问题摘要,简短、忠于原文,不得编造原文没有的信息。

只输出包含以上四个键的 JSON 对象,键必须全部存在,不得增加其他键。"""

# few-shot 样例与 evals/extract_cases.jsonl 不重叠
_FEW_SHOT: list[tuple[str, str]] = [
    (
        "我买的锅收到时把手断了,订单 20260711001,想换个新的",
        '{{"order_id":"20260711001","intent":"换货","expectation":"换货","summary":"锅具把手断裂,要求换货"}}',
    ),
    (
        "什么时候发货啊,都等两天了",
        '{{"order_id":null,"intent":"催发货","expectation":null,"summary":"催促商家尽快发货"}}',
    ),
]


def build_extract_prompt() -> ChatPromptTemplate:
    messages: list = [("system", EXTRACT_INSTRUCTION)]
    for text, output in _FEW_SHOT:
        messages.append(("human", text))
        messages.append(("assistant", output))
    messages.append(("human", "{text}"))
    return ChatPromptTemplate.from_messages(messages)


async def run_extraction(
    model: Any, method: str, text: str, max_input_tokens: int
) -> AfterSaleExtraction:
    prompt = build_extract_prompt()
    messages = prompt.invoke({"text": text}).to_messages()
    if count_tokens_approximately(messages) > max_input_tokens:
        raise MessageTooLongError("extraction input exceeds input token budget")
    structured = model.with_structured_output(
        AfterSaleExtraction, method=method, include_raw=True
    )
    try:
        result = await structured.ainvoke(messages)
    except MessageTooLongError:
        raise
    except Exception as exc:
        logger.warning("extract upstream error: %s", type(exc).__name__)
        raise UpstreamError("extraction upstream call failed") from exc
    if result.get("parsing_error") is not None or result.get("parsed") is None:
        parsing_error = result.get("parsing_error")
        logger.warning(
            "extract parsing failed: %s",
            type(parsing_error).__name__ if parsing_error is not None else "missing_parsed",
        )
        raise UpstreamError("structured output parsing failed")
    return result["parsed"]
