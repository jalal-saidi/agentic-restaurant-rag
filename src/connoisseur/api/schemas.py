"""Versioned HTTP request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

BackendName = Literal["langgraph", "agno"]


class HistoryMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("content cannot be blank")
        return normalized


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    backend: BackendName = "langgraph"
    session_id: str | None = Field(default=None, min_length=1, max_length=200)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=100)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("message cannot be blank")
        return normalized

    @field_validator("session_id")
    @classmethod
    def normalize_session_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("session_id cannot be blank")
        return normalized


class ChatResponse(BaseModel):
    answer: str
    backend: BackendName
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackendInfo(BaseModel):
    name: BackendName
    available: bool
    reason: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    backends: list[BackendInfo]
