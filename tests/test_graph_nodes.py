# tests/test_graph_nodes.py(本任务先放分类与解析用例,后续任务继续追加)
# 注:classify 测试经编译图驱动——langgraph 1.2.11 裸调节点时 get_stream_writer()
# 抛 RuntimeError(计划 Task 9 注记预授权的兜底路线,裁决采纳)。
import json
import logging

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.graph.nodes import GraphDeps, build_front_nodes, parse_intent_output
from app.graph.state import ROUTE_TABLE, ChatGraphState, new_turn_state
from tests.conftest import make_settings


class _ClassifyModel:
    """只支持 ainvoke 的桩:返回固定文本。"""

    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=self._text)


def _nodes(model_text):
    deps = GraphDeps(model=_ClassifyModel(model_text), settings=make_settings(),
                     retriever=None, store=None)
    return build_front_nodes(deps)


def _graph(model):
    """前段两节点串成最小图:START → resolve_reference → classify_intent → END。"""
    nodes = build_front_nodes(GraphDeps(model=model, settings=make_settings(),
                                        retriever=None, store=None))
    g = StateGraph(ChatGraphState)
    g.add_node("resolve_reference", nodes["resolve_reference"])
    g.add_node("classify_intent", nodes["classify_intent"])
    g.add_edge(START, "resolve_reference")
    g.add_edge("resolve_reference", "classify_intent")
    g.add_edge("classify_intent", END)
    return g.compile()


@pytest.mark.parametrize("text,expected", [
    ('{"intent":"物流","needs_knowledge":false}', ("物流", False)),
    ('{"intent":"售后","needs_knowledge":true}', ("售后", True)),
    ('前缀文本{"intent":"投诉","needs_knowledge":false}后缀', ("投诉", False)),
    ('{"intent":"售后","needs_knowledge":"false"}', None),   # 字符串 false 非法
    ('{"intent":"售后"}', None),                              # 缺字段
    ('{"intent":"退票","needs_knowledge":true}', None),       # 非法枚举
    ('不是 JSON', None),
    ('{"intent":"闲聊","needs_knowledge":false,"x":1}', ("闲聊", False)),  # 容忍多余字段
])
def test_parse_intent_output(text, expected):
    assert parse_intent_output(text) == expected


async def test_classify_sets_route_from_table():
    g = _graph(_ClassifyModel('{"intent":"退款退货","needs_knowledge":false}'))
    out = await g.ainvoke(new_turn_state("我想退货"))
    assert out["intent"] == "退款退货" and out["needs_knowledge"] is False
    assert out["route"] == "knowledge"  # 路由表保守覆盖


async def test_classify_parse_failure_falls_back_to_knowledge(caplog):
    g = _graph(_ClassifyModel("模型输出了一坨废话"))
    with caplog.at_level(logging.WARNING, logger="wayhelp.graph"):
        out = await g.ainvoke(new_turn_state("查订单"))
    assert (out["intent"], out["needs_knowledge"]) == ("售后", True)
    assert out["route"] == "knowledge"  # 解析失败不得退回 business
    assert "解析失败" in caplog.text or "parse" in caplog.text


async def test_classify_model_exception_aborts_with_upstream_error():
    class _Boom:
        async def ainvoke(self, messages):
            raise ConnectionError("upstream down")

    from app.graph.errors import TurnAbortError
    with pytest.raises(TurnAbortError):
        await _graph(_Boom()).ainvoke(new_turn_state("q"))


async def test_resolve_reference_passthrough():
    nodes = _nodes("{}")
    out = await nodes["resolve_reference"](new_turn_state("原样 透传"))
    assert out["resolved_query"] == "原样 透传"
