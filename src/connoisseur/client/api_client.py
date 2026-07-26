"""Async, framework-neutral client for the FastAPI service."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

HttpClientFactory = Callable[..., Any]


class ApiClientError(RuntimeError):
    """The remote API could not fulfill a request."""


@dataclass(frozen=True, slots=True)
class RemoteChatResult:
    answer: str
    backend: str
    session_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def normalize_ui_history(
    history: Sequence[Any] | None,
) -> list[dict[str, str]]:
    """Accept Gradio's message format and its older pair format."""

    normalized: list[dict[str, str]] = []
    for item in history or ():
        if isinstance(item, Mapping):
            role = str(item.get("role", "")).strip().lower()
            content = str(item.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                normalized.append({"role": role, "content": content})
            continue
        role = str(getattr(item, "role", "")).strip().lower()
        content = str(getattr(item, "content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
            continue
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
            if user is not None and str(user).strip():
                normalized.append({"role": "user", "content": str(user).strip()})
            if assistant is not None and str(assistant).strip():
                normalized.append(
                    {"role": "assistant", "content": str(assistant).strip()}
                )
    return normalized


class ConnoisseurApiClient:
    """Call the independently deployed API process."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 300.0,
        max_history_messages: int = 100,
        client_factory: HttpClientFactory | None = None,
    ) -> None:
        if not 0 <= max_history_messages <= 100:
            raise ValueError("max_history_messages must be between 0 and 100")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_history_messages = max_history_messages
        self._client_factory = client_factory

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        try:
            import httpx
        except ImportError as exc:
            raise ApiClientError(
                "The 'httpx' package is required by the remote client"
            ) from exc
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
        )

    @staticmethod
    def _error_detail(response: Any) -> str:
        try:
            body = response.json()
        except Exception:
            return f"API returned HTTP {response.status_code}"
        if isinstance(body, Mapping):
            detail = body.get("detail")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        return f"API returned HTTP {response.status_code}"

    async def chat(
        self,
        message: str,
        *,
        backend: str,
        session_id: str | None = None,
        history: Sequence[Any] | None = None,
    ) -> RemoteChatResult:
        normalized_history = normalize_ui_history(history)
        if self.max_history_messages == 0:
            normalized_history = []
        else:
            normalized_history = normalized_history[-self.max_history_messages :]
        payload: dict[str, Any] = {
            "message": message,
            "backend": backend,
            "history": normalized_history,
        }
        if session_id:
            payload["session_id"] = session_id
        try:
            async with self._make_client() as client:
                response = await client.post("/v1/chat", json=payload)
        except ApiClientError:
            raise
        except Exception as exc:
            raise ApiClientError(
                f"Could not connect to the API: {type(exc).__name__}"
            ) from exc
        if response.status_code >= 400:
            raise ApiClientError(self._error_detail(response))
        try:
            data = response.json()
            return RemoteChatResult(
                answer=str(data["answer"]),
                backend=str(data["backend"]),
                session_id=str(data["session_id"]),
                metadata=(
                    data.get("metadata", {})
                    if isinstance(data.get("metadata", {}), Mapping)
                    else {}
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiClientError("API returned an invalid chat response") from exc

    async def backends(self) -> list[dict[str, Any]]:
        try:
            async with self._make_client() as client:
                response = await client.get("/v1/backends")
        except Exception as exc:
            raise ApiClientError(
                f"Could not connect to the API: {type(exc).__name__}"
            ) from exc
        if response.status_code >= 400:
            raise ApiClientError(self._error_detail(response))
        data = response.json()
        if not isinstance(data, list):
            raise ApiClientError("API returned an invalid backend list")
        return [dict(item) for item in data if isinstance(item, Mapping)]
