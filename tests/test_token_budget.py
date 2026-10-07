from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.token_budget import (
    ContextBudget, build_summary_projection, compute_budget, estimate_message,
    estimate_messages, estimate_tokens, measure_sys_tokens, truncate_to_tokens,
)


def _settings(**over):
    from app.config import Settings
    base = dict(openai_base_url="http://x", openai_api_key="k", model_name="m",
                database_url="mysql+pymysql://u:p@h/d",
                model_context_window=18000, max_output_tokens=2000,
                max_user_input_tokens=2000, max_agent_steps=3,
                tool_result_max_tokens=1200, rerank_top_k=5)
    base.update(over)
    return Settings(**base)


def test_estimate_tokens_cjk_and_ascii():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好") == 2          # CJK 1字=1token
    assert estimate_tokens("abcd") == 1          # 其余 ceil(/4)
    assert estimate_tokens("你好abcd") == 3
    assert estimate_tokens("。") == 1            # 中文标点 U+3000–U+303F
    assert estimate_tokens("!") == 1


def test_estimate_message_overhead_and_tool_calls():
    assert estimate_message(HumanMessage(content="你好")) == 2 + 4
    # langchain-core 1.6.2 content 无 None 分支,model_copy 绕过校验取真 None,钉 None 按空串语义
    none_content = AIMessage(content="").model_copy(update={"content": None})
    assert estimate_message(none_content) == 4                     # None 按空串
    m = AIMessage(content="", tool_calls=[{"name": "query_order", "args": {"order_id": "A1"}, "id": "c1", "type": "tool_call"}])
    assert estimate_message(m) > 4
    assert estimate_messages([HumanMessage(content="你好"), ToolMessage(content="x", tool_call_id="c1")]) == 2 + 4 + 1 + 4


def test_truncate_to_tokens():
    text = "汉" * 100
    out = truncate_to_tokens(text, 30)
    assert estimate_tokens(out) <= 30 and len(out) == 30
    assert truncate_to_tokens("短", 30) == "短"


def test_budget_acceptance_numbers():
    b = compute_budget(_settings(), sys_tokens=1883)
    assert (b.peak, b.fixed, b.avail) == (3750, 4583, 5667)
    assert (b.total, b.layer1, b.layer2) == (5667, 3966, 1701)
    assert b.sufficient


def test_budget_min_arm_and_clamp():
    b = compute_budget(_settings(model_context_window=65536), sys_tokens=1883)
    assert b.total == 20 * 300                       # 想留住的轮数 × 稳态 取小
    tiny = compute_budget(_settings(model_context_window=5000), sys_tokens=1883)
    assert tiny.avail >= 0 and not tiny.sufficient   # 一轮装不下 → 自检报警语义


def test_build_summary_projection():
    segs = ["第一段订单A1001要退款", "第二段", "第三段", "第四段"]
    assert build_summary_projection(segs, 10_000) == "\n\n".join(segs)  # 全额
    # 四段全文约 20 token,40 不触发截短路径;16 使每段等分配截短生效(原计划预算估错,断言语义不变)
    out = build_summary_projection(segs, 16)       # 超预算:每段等分配截短,段数不丢
    assert out.count("……") >= 1 and estimate_tokens(out) <= 16
    assert len(out.split("\n\n")) == 4             # 方案 A:无「最近 3 段」硬上限
    assert build_summary_projection(segs, 0) == ""


def test_measure_sys_tokens():
    assert measure_sys_tokens("系统", []) == 2 + 4
