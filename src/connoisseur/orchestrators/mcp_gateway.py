"""Deterministic MCP retrieval adapter shared by graph nodes."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import RetrievalGateway
from .errors import BackendUnavailableError, RetrievalError
from .settings import AppSettings
from .utils import to_jsonable

ClientFactory = Callable[[str], Any]


def _first_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (list, tuple)):
        for item in value:
            result = _first_string(item)
            if result:
                return result
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _tool_result_data(result: Any) -> Any:
    data = getattr(result, "data", None)
    if data is not None:
        return to_jsonable(data)
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return to_jsonable(structured)
    content = getattr(result, "content", None) or []
    text_parts = [
        str(item.text)
        for item in content
        if getattr(item, "text", None) is not None
    ]
    text = "\n".join(text_parts).strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"text": text}


class FastMCPRetrievalGateway(RetrievalGateway):
    """Call the deployed retrieval server over streamable HTTP."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.settings = settings
        self._client_factory = client_factory

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory(self.settings.mcp_server_url)
        try:
            from fastmcp import Client
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'fastmcp' package is required to call the retrieval server"
            ) from exc
        return Client(
            self.settings.mcp_server_url,
            timeout=self.settings.request_timeout_seconds,
        )

    @staticmethod
    def _restaurant_arguments(
        query: str,
        profile: Mapping[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"query": query, "limit": limit}
        optional: dict[str, Any] = {
            "cuisine": _first_string(profile.get("cuisines")),
            "neighborhood": _first_string(profile.get("neighborhoods")),
            "price_range": _first_string(profile.get("price_range")),
        }
        min_rating = profile.get("min_rating")
        if isinstance(min_rating, (int, float)):
            optional["min_rating"] = float(min_rating)
        arguments.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return arguments

    @staticmethod
    def _recipe_arguments(
        query: str,
        profile: Mapping[str, Any],
        limit: int,
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"query": query, "limit": limit}
        cuisine = _first_string(profile.get("cuisines"))
        ingredients = _string_list(profile.get("preferred_ingredients"))
        if cuisine:
            arguments["cuisine"] = cuisine
        if ingredients:
            arguments["ingredients"] = ingredients
        return arguments

    async def retrieve(
        self,
        *,
        query: str,
        profile: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        client = self._make_client()
        tool_calls = (
            (
                "search_restaurants",
                self._restaurant_arguments(
                    query,
                    profile,
                    self.settings.retrieval_limit,
                ),
            ),
            (
                "search_recipes",
                self._recipe_arguments(
                    query,
                    profile,
                    self.settings.retrieval_limit,
                ),
            ),
        )
        evidence: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        try:
            async with client:
                for tool_name, arguments in tool_calls:
                    try:
                        result = await client.call_tool(tool_name, arguments)
                        evidence[
                            "restaurants" if tool_name == "search_restaurants" else "recipes"
                        ] = _tool_result_data(result)
                    except Exception as exc:
                        errors.append(
                            {
                                "tool": tool_name,
                                "error": type(exc).__name__,
                            }
                        )
        except Exception as exc:
            raise RetrievalError(
                f"MCP retrieval connection failed: {type(exc).__name__}"
            ) from exc

        if len(errors) == len(tool_calls):
            raise RetrievalError("All MCP retrieval tool calls failed")
        evidence.setdefault("restaurants", {})
        evidence.setdefault("recipes", {})
        evidence["tool_calls"] = [name for name, _ in tool_calls]
        if errors:
            evidence["errors"] = errors
        return evidence

    async def health(self) -> Mapping[str, Any]:
        client = self._make_client()
        try:
            async with client:
                result = await client.call_tool("health_check", {})
            data = _tool_result_data(result)
            return data if isinstance(data, Mapping) else {"result": data}
        except Exception as exc:
            return {
                "status": "unavailable",
                "error": type(exc).__name__,
            }

    async def aclose(self) -> None:
        # Connections are scoped to each call by FastMCP's async context manager.
        return None
