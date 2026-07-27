from __future__ import annotations

import unittest
from typing import Any

from connoisseur.orchestrators.settings import AppSettings
from connoisseur.orchestrators.tool_discovery import (
    LangChainMCPToolProvider,
    bind_openai_mcp_tools,
)


class FakeMCPClient:
    def __init__(self, tools: list[Any]) -> None:
        self.tools = tools
        self.calls = 0

    async def get_tools(self) -> list[Any]:
        self.calls += 1
        return self.tools


class LangChainMCPToolProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovers_http_tools_once_and_caches_them(self) -> None:
        tool = object()
        client = FakeMCPClient([tool])
        captured: dict[str, Any] = {}

        def factory(connections: dict[str, dict[str, Any]]) -> FakeMCPClient:
            captured.update(connections)
            return client

        provider = LangChainMCPToolProvider(
            AppSettings(mcp_server_url="http://mcp.test:8001/mcp"),
            client_factory=factory,
        )

        first = await provider.get_tools()
        second = await provider.get_tools()

        self.assertEqual(first, (tool,))
        self.assertIs(first, second)
        self.assertEqual(client.calls, 1)
        self.assertEqual(
            captured,
            {
                "connoisseur": {
                    "transport": "http",
                    "url": "http://mcp.test:8001/mcp",
                }
            },
        )


class FakeChatModel:
    latest: FakeChatModel | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.tools: list[Any] = []
        FakeChatModel.latest = self

    def bind_tools(self, tools: list[Any]) -> FakeChatModel:
        self.tools = tools
        return self


class BindOpenAIToolsTests(unittest.TestCase):
    def test_forwards_reasoning_and_current_openai_token_parameter(self) -> None:
        tool = object()
        settings = AppSettings(
            llm_model="gpt-test",
            openai_api_key="test-only",
            llm_reasoning_effort="none",
            llm_max_tokens=900,
        )

        bound = bind_openai_mcp_tools(
            settings,
            [tool],
            chat_model_cls=FakeChatModel,
        )

        self.assertIs(bound, FakeChatModel.latest)
        assert FakeChatModel.latest is not None
        self.assertEqual(FakeChatModel.latest.tools, [tool])
        self.assertEqual(FakeChatModel.latest.kwargs["reasoning_effort"], "none")
        self.assertEqual(
            FakeChatModel.latest.kwargs["max_completion_tokens"],
            900,
        )

    def test_preserves_legacy_max_tokens_for_compatible_endpoints(self) -> None:
        settings = AppSettings(
            llm_model="local",
            openai_base_url="http://model.test/v1",
            llm_max_tokens_parameter="max_tokens",
            llm_max_tokens=700,
        )

        bind_openai_mcp_tools(
            settings,
            [object()],
            chat_model_cls=FakeChatModel,
        )

        assert FakeChatModel.latest is not None
        self.assertEqual(
            FakeChatModel.latest.kwargs["extra_body"],
            {"max_tokens": 700},
        )
