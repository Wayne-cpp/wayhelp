from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.services.context_layers import (
    build_layered_view, evaluate_degrade, needs_reconcile, reconcile_checkpoint_ids,
    render_history_text, tool_omitted_text,
)
from app.sessions import ContextMeta, PersistedMessageRecord


def _mk(mid, m):
    return m.model_copy(update={"additional_kwargs": {**m.additional_kwargs, "db_id": mid}})


def _settings():
    from app.config import Settings
    return Settings(openai_base_url="http://x", openai_api_key="k", model_name="m",
                    database_url="mysql+pymysql://u:p@h/d")


def _history():
    out, mid = [], 0
    for i in range(3):
        mid += 1
        out.append(_mk(mid, HumanMessage(content=f"问题{i}")))
        mid += 1
        out.append(_mk(mid, AIMessage(content=f"回答{i}" + "长" * 200)))
    return out  # ids 1..6


def test_three_layer_split_and_render():
    msgs = _history()
    meta = ContextMeta("早期摘要:订单A1001", summary_upto=2, layer1_from=4)
    v = build_layered_view(msgs, meta, _settings())
    assert v.summary == "早期摘要:订单A1001"
    assert [m.content for m in v.layer2_rendered] == ["问题1", "回答1" + "长" * 47 + "…"]
    assert [m.content for m in v.layer1] == ["问题2", "回答2" + "长" * 200]
    assert v.l1_tokens > 0 and v.l2_tokens > 0 and v.summary_tokens > 0


def test_null_anchors_all_layer1_and_none_content_safe():
    # langchain-core 1.6.2 content 无 None 分支,model_copy 绕过校验取真 None(Task 3 先例)
    none_ai = AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c",
                                                 "type": "tool_call"}]).model_copy(
        update={"content": None})
    msgs = _history() + [_mk(7, none_ai)]
    v = build_layered_view(msgs, ContextMeta(None, None, None), _settings())
    assert len(v.layer1) == len(msgs) and v.layer2_rendered == [] and v.summary is None
    assert v.l1_tokens > 0  # content=None 不炸


def test_tool_message_inherits_group_id():
    msgs = _history()
    msgs.insert(3, ToolMessage(content="x" * 500, tool_call_id="c1", name="query_order"))
    meta = ContextMeta(None, None, 2)   # 组 id=3(assistant)>2 → 层1;若按 0 会错进层2
    v = build_layered_view(msgs, meta, _settings())
    assert any(isinstance(m, ToolMessage) for m in v.layer1)


def test_layer2_tool_omitted():
    msgs = _history()
    tm = ToolMessage(content="x" * 500, tool_call_id="c1", name="query_order")
    msgs.insert(3, _mk(3, tm))  # 该 tool 自带 id 落层2?否——tool 不落库;这里验证截短形态即可
    v = build_layered_view(msgs, ContextMeta(None, None, 4), _settings())
    texts = [m.content for m in v.layer2_rendered if isinstance(m, ToolMessage)]
    assert texts == [tool_omitted_text("query_order")]


def test_needs_reconcile_and_passthrough():
    assert not needs_reconcile(_history())
    assert needs_reconcile([HumanMessage(content="没盖章")])


def _records(tool_rows: bool):
    recs, mid = [], 0
    for i in range(2):
        mid += 1
        recs.append(PersistedMessageRecord(str(mid), "user", f"问题{i}", None, None))
        mid += 1
        recs.append(PersistedMessageRecord(str(mid), "assistant", f"回答{i}", None, None))
        if tool_rows and i == 0:
            mid += 1
            recs.append(PersistedMessageRecord(str(mid), "tool", "env", None, "c1"))
    return recs


def test_reconcile_legacy_tool_rows_and_new_db():
    msgs = [HumanMessage(content="问题0"), AIMessage(content="回答0"),
            ToolMessage(content="r", tool_call_id="c1", name="t"),
            HumanMessage(content="问题1"), AIMessage(content="回答1")]
    out = reconcile_checkpoint_ids(msgs, _records(tool_rows=True))
    assert out is not None and out[0].additional_kwargs["db_id"] == 1
    assert out[3].additional_kwargs["db_id"] == 4
    out2 = reconcile_checkpoint_ids(msgs, _records(tool_rows=False))
    assert out2 is not None  # 新库无 tool 行:过滤后对齐


def test_reconcile_duplicate_content_fails():
    msgs = [HumanMessage(content="一样"), HumanMessage(content="一样")]
    recs = [PersistedMessageRecord("1", "user", "一样", None, None),
            PersistedMessageRecord("2", "user", "一样", None, None)]
    assert reconcile_checkpoint_ids(msgs, recs) is None  # 歧义,不猜 ID


def test_reconcile_trailing_uncommitted_fails():
    msgs = [HumanMessage(content="问题0"), AIMessage(content="回答0"),
            HumanMessage(content="没落库的问")]
    assert reconcile_checkpoint_ids(msgs, _records(tool_rows=False)) is None


def test_render_history_text():
    msgs = _history()
    meta = ContextMeta("摘要:退款诉求", 2, 4)
    text = render_history_text(build_layered_view(msgs, meta, _settings()))
    assert text.startswith("早期对话摘要:\n摘要:退款诉求")
    assert "用户:问题1" in text and "助手:回答1" in text and "用户:问题2" in text


def test_evaluate_degrade_moves_boundary():
    # 原计划预算估错:末轮=7+207=214,60/210 均触发 kept=[] 或孤立 assistant 弹除;
    # 214 恰好装下末轮,断言语义(boundary<6、切在 user 边界)不变(Task 3 先例)
    msgs = _history()  # 3 轮,assistant 各 200+ 字
    boundary = evaluate_degrade(msgs, None, budget_l1=214)
    assert boundary is not None and boundary < 6     # 只保留末尾装进 214 token 的部分
    kept = [m for m in msgs if (m.additional_kwargs.get("db_id") or 0) > boundary]
    assert kept and kept[0].additional_kwargs["db_id"] % 2 == 1  # 切在 user 边界(奇数 id)
    assert evaluate_degrade(msgs, None, budget_l1=10**6) is None


def test_evaluate_degrade_single_turn_over_budget():
    msgs = _history()[-2:]  # 一轮就 200+ 字
    boundary = evaluate_degrade(msgs, None, budget_l1=10)
    assert boundary == 6  # 层1 空:边界推到最末,层2/摘要下轮接住
