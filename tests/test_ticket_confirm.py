"""ch08 工单预览(spec §5):挂起轮帧序 session → ticket_preview → [DONE];
确认/取消全链在 Task 11 resume 分派落地后补全。"""
from app.main import create_app
from tests.conftest import ScriptedChatModel, TEST_USER_ID, make_runtime, make_settings
from tests.test_ch05_acceptance import _client, _turn, _types

_SCRIPTS = [
    ['{"intent":"订单","confidence":0.9}'],          # classify(首轮 understand 跳过)
    [("tool", [{"index": 0, "name": "create_ticket", "id": "w1",
                "args": '{"description":"商品有质量问题,要求换货","ticket_type":"售后"}'}])],
    ["已为您提交售后工单,工单号见上方卡片。"],         # Task 11 接通后 main_agent 收尾
]


async def test_write_call_suspends_with_ticket_preview_frame():
    app = create_app(settings=make_settings(),
                     model=ScriptedChatModel(scripts=list(_SCRIPTS)),
                     runtime=make_runtime(tools=[]))
    async with await _client(app) as client:
        frames, sid = await _turn(client, "帮我建个工单")
    assert _types(frames) == ["session", "ticket_preview", "[DONE]"]
    card = frames[1]
    assert card["interrupt_id"]
    assert card["ticket_type"] == "售后"
    assert "质量问题" in card["description"]
