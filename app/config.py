from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_base_url: str = Field(min_length=1)
    openai_api_key: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    structured_output_method: Literal["json_schema", "json_mode"] = "json_schema"
    max_input_tokens: int = Field(default=2000, gt=0)
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
    max_tool_result_chars: int = Field(default=4000, ge=256)
