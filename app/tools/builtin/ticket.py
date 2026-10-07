"""ch08 create_ticket(写):tickets 与幂等行同一事务;不改 conv.status(spec §3/§5)。"""
import json
from typing import Literal

from pydantic import BaseModel, Field

from app.models import Conversation, ToolWriteIdempotency
from app.tools.builtin import builtin_tool
from app.tools.business import write_ticket
from app.tools.catalog import WriteMeta


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


class CreateTicketArgs(BaseModel):
    description: str = Field(min_length=1, max_length=2000, description="问题描述")
    ticket_type: Literal["售后", "投诉", "咨询"] = Field(description="工单类型")


@builtin_tool(name="create_ticket",
              description="创建人工工单。仅在顾客明确要求建工单且工单类型、问题描述已核对齐全时调用;调用后需用户在前端预览确认才真正生效。",
              args_model=CreateTicketArgs, permission="write", budget_priority=50)
def create_ticket_fn(args: dict, ctx, write: WriteMeta) -> str:
    if ctx.conversation_id is None or ctx.session_factory is None:
        raise ValueError("conversation context required")
    with ctx.session_factory() as s:
        conv = s.get(Conversation, ctx.conversation_id)
        if conv is None:
            raise ValueError("conversation not found")
        ticket_no = write_ticket(s, ctx.conversation_id,
                                 args["description"], args["ticket_type"])
        s.add(ToolWriteIdempotency(idempotency_key=write.key,
                                   arguments_sha256=write.args_sha256,
                                   ticket_no=ticket_no))
        s.commit()  # 工单与幂等映射同事务;不改 conv.status
    return _json({"ticket_no": ticket_no, "status": "待处理"})
