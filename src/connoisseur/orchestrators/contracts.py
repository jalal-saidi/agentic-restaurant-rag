"""Framework-neutral contracts shared by the API and both backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A normalized chat message."""

    role: MessageRole
    content: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ChatMessage:
        role = str(value.get("role", "")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported chat role: {role or '(empty)'}")
        content = str(value.get("content", "")).strip()
        if not content:
            raise ValueError("Chat message content cannot be empty")
        return cls(role=role, content=content)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ChatInput:
    """One request sent to an orchestration backend."""

    message: str
    session_id: str
    history: tuple[ChatMessage, ...] = ()

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError("message cannot be empty")
        if not self.session_id.strip():
            raise ValueError("session_id cannot be empty")


@dataclass(frozen=True, slots=True)
class OrchestrationResult:
    """Backend output in a stable shape consumed by the HTTP API."""

    answer: str
    backend: str
    session_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Whether a backend can accept requests in the current environment."""

    name: str
    available: bool
    reason: str | None = None


@runtime_checkable
class LanguageModel(Protocol):
    """Minimal async LLM interface used by deterministic graph nodes."""

    async def complete(self, *, system: str, prompt: str) -> str:
        """Return one assistant response."""

    async def aclose(self) -> None:
        """Release any model-side resources."""


@runtime_checkable
class RetrievalGateway(Protocol):
    """Retrieval operations exposed through an MCP connection."""

    async def retrieve(
        self,
        *,
        query: str,
        profile: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Retrieve restaurant and recipe evidence for a request."""

    async def health(self) -> Mapping[str, Any]:
        """Return MCP server health information."""

    async def aclose(self) -> None:
        """Release any MCP-side resources."""


@runtime_checkable
class ToolProvider(Protocol):
    """Dynamically load model-callable tools from an MCP server."""

    async def get_tools(self) -> Sequence[Any]:
        """Return the currently exposed tool definitions."""

    async def aclose(self) -> None:
        """Release any discovery-client resources."""


@runtime_checkable
class Orchestrator(Protocol):
    """Common interface implemented by LangGraph and Agno."""

    name: str

    async def run(self, request: ChatInput) -> OrchestrationResult:
        """Execute a complete multi-agent request."""

    async def aclose(self) -> None:
        """Release backend resources."""


def normalize_history(
    history: Sequence[ChatMessage | Mapping[str, Any]] | None,
) -> tuple[ChatMessage, ...]:
    """Convert API/client history values into immutable messages."""

    if not history:
        return ()
    return tuple(
        item if isinstance(item, ChatMessage) else ChatMessage.from_mapping(item)
        for item in history
    )
