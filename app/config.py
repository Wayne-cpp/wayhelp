from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_base_url: str = Field(min_length=1)
    openai_api_key: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    structured_output_method: Literal["json_schema", "json_mode"] = "json_schema"
    max_input_tokens: int = Field(default=2000, gt=0)  # extract/mining 专用,聊天链路不用
    max_output_tokens: int = Field(default=1000, gt=0)
    max_message_chars: int = Field(default=8000, gt=0)
    max_sessions: int = Field(default=1000, gt=0)
    max_messages_per_session: int = Field(default=100, gt=0)
    database_url: str = Field(min_length=1)
    test_admin_database_url: str = Field(
        default="mysql+pymysql://root:root-password@127.0.0.1:3306/mysql?charset=utf8mb4"
    )
    test_database_url: str = Field(
        default="mysql+pymysql://root:root-password@127.0.0.1:3306/wayhelp_test?charset=utf8mb4"
    )
    tool_timeout_seconds: float = Field(default=5, gt=0)
    tool_max_retries: int = Field(default=2, ge=0)
    max_tool_calls_per_turn: int = Field(default=5, gt=0)
    max_tool_result_chars: int = Field(default=4000, ge=256)  # 证据组装字符预算(assemble_evidence budget_chars)
    embedding_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: Literal[1024] = 1024
    milvus_uri: str = "./data/milvus_lite.db"
    knowledge_top_k: int = Field(default=5, gt=0)
    knowledge_min_score: float = Field(default=0.6491, ge=-1, le=1)  # 2026-09-16 四策略评估 0.10 档重冻(evals/results/20260916T021207Z_compare.json;旧值 0.623 系 ch03 旧语料校准)
    mining_batch_size: int = Field(default=10, gt=0)
    max_chunk_chars: int = Field(default=500, gt=0)
    chunk_overlap_chars: int = Field(default=80, ge=0)
    rerank_base_url: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_timeout_seconds: float = Field(default=5, gt=0)
    rerank_max_retries: int = Field(default=2, ge=0)   # 内部追加重试;2 = 最多共 3 次请求
    retrieval_candidate_k: int = Field(default=50, gt=0)
    rerank_top_k: int = Field(default=10, gt=0)
    rerank_confidence_signal: Literal["top1", "margin12", "product", "ratio5"] = "top1"  # 闸门置信信号;胜者由评估校准选拔后冻结
    rerank_min_score: float = Field(default=0.0553)  # 2026-09-16 校准冻结(0.10 档,par=0.971;同上报告);阈值语义随 confidence_signal
    bm25_min_score: float = Field(default=0.0)     # 校准确认无可用闸门(分数退化),保持 0.0
    hybrid_min_score: float = Field(default=0.0)   # 校准确认 RRF 同分不可闸,保持 0.0;rerank 降级路径靠自评软闸门
    knowledge_strategy: Literal["dense", "bm25", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    query_rewrite_enabled: bool = True
    knowledge_tool_timeout_seconds: float = Field(default=20, gt=0)
    max_agent_steps: int = Field(default=8, gt=0)      # main_agent 内模型调用次数上限
    max_agent_tokens: int = Field(default=20000, gt=0)  # 本轮累计 token 预算(usage 或预留估算)
    checkpoint_db_path: str = "./data/checkpoints.db"
    # ch07 会话上下文预算(spec §5)
    model_context_window: int = Field(default=65536, gt=0)
    max_user_input_tokens: int = Field(default=2000, gt=0)
    tool_result_max_tokens: int = Field(default=1200, gt=0)
    summary_projection_tokens: int = Field(default=650, gt=0)
    history_target_turns: int = Field(default=20, gt=0)
    steady_tokens_per_turn: int = Field(default=300, gt=0)
    summary_max_chars: int = Field(default=200, gt=0)
    safety_margin_tokens: int = Field(default=550, ge=0)
    intent_model_name: str | None = None       # reserved,未接线(双模型 cascade 明确不在本章范围);None = 复用主模型(D5)
    refund_expand_enabled: bool = True         # order_specific 扩写开关(§6.4)
    # ch08 工具系统(spec §6.1)
    mcp_logistics_url: str = ""          # 例 http://127.0.0.1:8101/mcp;空 = 不接入
    mcp_after_sales_url: str = ""        # 例 http://127.0.0.1:8102/mcp;空 = 不接入
    mcp_discovery_timeout_seconds: float = Field(default=2, gt=0)  # 每轮按 Server 发现超时,超时即跳过
    tool_write_timeout_seconds: float = Field(default=10, gt=0)    # 写路径超时(超时≠没执行,绝不重试)
    tool_timeout_overrides: dict[str, float] = Field(default_factory=dict)  # JSON 按名覆盖等待期限(运维 knob/故障注入)
    audit_result_max_chars: int = Field(default=2000, gt=0)        # 审计 result_summary 截断
    # ch09 可观测性(spec §5.1):未配齐密钥 = 完全不挂回调,系统行为不变
    langfuse_enabled: bool = False
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    low_conf_snapshot_top_n: int = Field(default=3, gt=0)  # 落池召回快照条数(审核页展示)
    # 正式置信闸冻结参数(spec §5.2):默认 uncalibrated = 退化为旧 top1 信号行为
    # (既有 rerank_confidence_signal/rerank_min_score 口径继续生效);
    # 校准(uv run python evals/run_retrieval_compare.py --calibrate-evidence,Task 16 实跑)
    # 后把 evals/calibration/evidence_confidence.json 的值同步到此处默认值并注明日期。
    # 注:evidence_min_confidence 即 spec §7 表的 RERANK_EVIDENCE_MIN_CONFIDENCE(同一 knob,
    # 命名以计划为准);evidence_confidence_version 即 EVIDENCE_CONFIDENCE_VERSION。
    evidence_weight_top1: float = Field(default=1.0)   # 待校准回填,占位保守值
    evidence_weight_count: float = Field(default=0.0)  # 待校准回填,占位保守值
    evidence_weight_margin: float = Field(default=0.0)  # 待校准回填,占位保守值
    evidence_min_effective_score: float = Field(default=0.0)  # 待校准回填,占位保守值
    evidence_min_confidence: float = Field(default=0.0553)    # 待校准回填(暂沿用 rerank_min_score 冻结值)
    evidence_confidence_version: str = "uncalibrated"

    @field_validator("tool_timeout_overrides", mode="before")
    @classmethod
    def _parse_overrides(cls, v):
        if isinstance(v, str):
            import json as _json
            raw = _json.loads(v or "{}")
            if not isinstance(raw, dict):
                raise ValueError("TOOL_TIMEOUT_OVERRIDES 必须是 JSON 对象")
            return {str(k): float(x) for k, x in raw.items()}
        return v

    def has_rerank_key(self) -> bool:
        return bool(self.rerank_api_key.strip() or self.embedding_api_key.strip())

    def rerank_key(self) -> str:
        return (self.rerank_api_key.strip() or self.embedding_api_key.strip())

    @field_validator("embedding_dim", mode="before")
    @classmethod
    def _coerce_dim(cls, v):
        # env/.env 读进来是字符串,Literal[1024] 不做 str→int 强转,先归一
        if isinstance(v, str) and v.isdigit():
            return int(v)
        return v

    @model_validator(mode="after")
    def _overlap_less_than_chunk(self):
        if self.chunk_overlap_chars >= self.max_chunk_chars:
            raise ValueError("chunk_overlap_chars 必须小于 max_chunk_chars")
        return self

    def has_embedding_key(self) -> bool:
        return bool(self.embedding_api_key.strip())

    def has_langfuse_key(self) -> bool:
        return bool(self.langfuse_public_key.strip()
                    and self.langfuse_secret_key.strip())
