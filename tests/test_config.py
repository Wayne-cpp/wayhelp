import pytest
from pydantic import ValidationError

from app.config import Settings
from tests.conftest import make_settings

REQUIRED_ENV = {
    "OPENAI_BASE_URL": "http://localhost:11434/v1",
    "OPENAI_API_KEY": "test-key",
    "MODEL_NAME": "qwen2.5",
    "DATABASE_URL": "mysql+pymysql://u:p@127.0.0.1:9/wayhelp",
}


def _set_required(monkeypatch):
    for k, v in REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)


def test_loads_with_required_env(monkeypatch):
    _set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.openai_base_url == "http://localhost:11434/v1"
    assert s.openai_api_key == "test-key"
    assert s.model_name == "qwen2.5"
    assert s.structured_output_method == "json_schema"
    assert s.max_input_tokens == 2000
    assert s.max_output_tokens == 1000
    assert s.max_message_chars == 8000
    assert s.max_sessions == 1000
    assert s.max_messages_per_session == 100


def test_missing_required_fails(monkeypatch):
    for k in REQUIRED_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_structured_method_fails(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("STRUCTURED_OUTPUT_METHOD", "function_calling")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_non_positive_value_fails(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("MAX_SESSIONS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_kwargs_override_env(monkeypatch):
    _set_required(monkeypatch)
    s = Settings(_env_file=None, max_sessions=5, max_messages_per_session=10)
    assert s.max_sessions == 5
    assert s.max_messages_per_session == 10


def test_odd_max_messages_allowed():
    s = make_settings(max_messages_per_session=7)
    assert s.max_messages_per_session == 7


def test_tool_settings_defaults():
    s = make_settings()
    assert s.tool_timeout_seconds == 5
    assert s.tool_max_retries == 2
    assert s.max_tool_calls_per_turn == 5
    assert s.max_tool_result_chars == 4000


def test_tool_max_retries_zero_allowed():
    assert make_settings(tool_max_retries=0).tool_max_retries == 0


def test_bad_tool_settings_rejected():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        make_settings(tool_timeout_seconds=0)
    with pytest.raises(pydantic.ValidationError):
        make_settings(tool_max_retries=-1)
    with pytest.raises(pydantic.ValidationError):
        make_settings(max_tool_result_chars=100)  # < 256
    with pytest.raises(pydantic.ValidationError):
        make_settings(database_url="")
