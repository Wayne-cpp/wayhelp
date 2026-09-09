from typing import Any

from fastapi import Request

from app.config import Settings
from app.services.chat_service import ChatService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_model(request: Request) -> Any:
    return request.app.state.model


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service
