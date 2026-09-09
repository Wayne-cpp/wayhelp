from typing import Literal

from pydantic import Field, field_validator
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

    @field_validator("max_messages_per_session")
    @classmethod
    def _must_be_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError("MAX_MESSAGES_PER_SESSION must be even")
        return v
