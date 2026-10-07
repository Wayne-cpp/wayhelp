"""ch08 工具目录(spec §1):进程内唯一工具表;内置启动登记,MCP 每轮发现现合成。
权限/归属分类只认本地策略,Server 自报 annotations 一律不看。"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

Permission = Literal["read", "write", "deny"]
Ownership = Literal["none", "order"]
Source = Literal["builtin", "mcp"]


@dataclass(frozen=True)
class WriteMeta:
    """execute_confirmed 传给写函数的幂等元数据(spec §3)。"""
    key: str
    args_sha256: str


@dataclass
class TurnContext:
    """每轮显式上下文:替代闭包绑定;conversation_id 只取已确认的 MySQL 会话 id。"""
    user_id: str
    conversation_id: int | None
    resolved_query: str
    retriever: Any
    settings: Any
    session_factory: Any = None
    faq_trace: Any = None            # FaqRetrievalTrace(business 分支装配时挂)
    order_snapshots: list = field(default_factory=list)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_schema: dict                 # JSON Schema dict(模型可见 + 校验依据)
    args_model: type[BaseModel]       # 校验/StructuredTool 载体
    source: Source
    permission: Permission
    ownership: Ownership = "none"
    mcp_server: str | None = None
    budget_priority: int = 0          # 超预算剔除次序;数值越大越优先保留
    fn: Callable | None = None        # builtin:(args, ctx) -> str;写函数多收 write: WriteMeta
    formatter: Callable[[str], str] | None = None  # MCP 结果挑字段 → JSON str


class ToolCatalog:
    def __init__(self):
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())


# ── MCP 本地能力策略(spec §1.1/§4.2)──

KNOWN_READONLY_SERVERS = frozenset({"logistics", "after_sales"})


def classify_mcp_tool(server: str, schema: dict) -> tuple[str, str] | None:
    """返回 (permission, ownership);None = 不进工具面。只放行两种形态:
    零参数 → (read, none);仅必填字符串 order_id → (read, order)。"""
    if server not in KNOWN_READONLY_SERVERS:
        return None
    props = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    if not props and not required:
        return ("read", "none")
    if set(props) == {"order_id"} and required == {"order_id"} \
            and (props["order_id"] or {}).get("type") == "string":
        return ("read", "order")
    return None


def mcp_args_model(shape: str) -> type[BaseModel]:
    if shape == "order":
        class _OrderArgs(BaseModel):
            order_id: str = Field(min_length=1, max_length=64, description="订单号")
        return _OrderArgs

    class _EmptyArgs(BaseModel):
        pass
    return _EmptyArgs


def canonical_args_sha(args: dict) -> str:
    raw = json.dumps(args or {}, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def idempotency_key_for(conversation_id: int, tool_call_id: str) -> str:
    return hashlib.sha256(f"{conversation_id}:{tool_call_id}".encode("utf-8")).hexdigest()
