"""MCP tool discovery and OpenAI tool-binding for the LangGraph backend."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

from .contracts import ToolProvider
from .errors import (
    BackendUnavailableError,
    ModelConfigurationError,
    RetrievalError,
)
from .settings import AppSettings

MCPClientFactory = Callable[[dict[str, dict[str, Any]]], Any]


class LangChainMCPToolProvider(ToolProvider):
    """Discover LangChain-compatible tools from the FastMCP HTTP endpoint."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        client_factory: MCPClientFactory | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory
        self._client: Any | None = None
        self._tools: tuple[Any, ...] | None = None
        self._lock = asyncio.Lock()

    def _make_client(self) -> Any:
        connections: Any = {
            "connoisseur": {
                "transport": "http",
                "url": self.settings.mcp_server_url,
            }
        }
        if self._client_factory is not None:
            return self._client_factory(connections)
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'langchain-mcp-adapters' package is required for "
                "LangGraph MCP tool discovery"
            ) from exc
        return MultiServerMCPClient(connections)

    async def get_tools(self) -> Sequence[Any]:
        """List and cache the MCP server's current tool definitions."""

        if self._tools is not None:
            return self._tools
        async with self._lock:
            if self._tools is not None:
                return self._tools
            try:
                self._client = self._make_client()
                discovered = tuple(await self._client.get_tools())
            except BackendUnavailableError:
                raise
            except Exception as exc:
                raise RetrievalError(
                    f"MCP tool discovery failed: {type(exc).__name__}"
                ) from exc
            if not discovered:
                raise RetrievalError("MCP tool discovery returned no tools")
            self._tools = discovered
            return self._tools

    async def aclose(self) -> None:
        """Release a client resource when a future adapter version exposes one."""

        if self._client is None:
            return
        close = getattr(self._client, "aclose", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result


def bind_openai_mcp_tools(
    settings: AppSettings,
    tools: Sequence[Any],
    *,
    chat_model_cls: Any | None = None,
) -> Any:
    """Create the tool-selecting ChatOpenAI runnable used by LangGraph."""

    configuration_error = settings.model_configuration_error()
    if configuration_error:
        raise ModelConfigurationError(configuration_error)
    if chat_model_cls is None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'langchain-openai' package is required for LangGraph "
                "MCP tool selection"
            ) from exc
        chat_model_cls = ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.openai_client_api_key,
        "timeout": settings.request_timeout_seconds,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    if settings.llm_reasoning_effort is not None:
        kwargs["reasoning_effort"] = settings.llm_reasoning_effort
    if settings.llm_max_tokens_parameter == "max_completion_tokens":
        kwargs["max_completion_tokens"] = settings.llm_max_tokens
    else:
        # ChatOpenAI aliases ``max_tokens`` to ``max_completion_tokens``.
        # ``extra_body`` preserves the legacy field for compatible local APIs.
        kwargs["extra_body"] = {"max_tokens": settings.llm_max_tokens}

    model = chat_model_cls(**kwargs)
    return model.bind_tools(list(tools))
