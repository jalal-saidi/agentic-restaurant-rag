from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from connoisseur.orchestrators.mcp_gateway import FastMCPRetrievalGateway
from connoisseur.orchestrators.settings import AppSettings


class FakeClient:
    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> FakeClient:
        self.entered = True
        return self

    async def __aexit__(self, *_: Any) -> None:
        self.exited = True

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        response = self.responses[name]
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(data=response)


class FastMCPRetrievalGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_calls_restaurant_and_recipe_tools_with_profile_filters(self) -> None:
        client = FakeClient(
            {
                "search_restaurants": {
                    "status": "ok",
                    "count": 1,
                    "results": [{"name": "Sakura Garden"}],
                },
                "search_recipes": {
                    "status": "ok",
                    "count": 1,
                    "results": [{"name": "Miso Soup"}],
                },
            }
        )
        settings = AppSettings(
            llm_model="model",
            openai_api_key="test-only",
            retrieval_limit=4,
        )
        gateway = FastMCPRetrievalGateway(
            settings,
            client_factory=lambda _: client,
        )

        result = await gateway.retrieve(
            query="quiet Japanese date",
            profile={
                "cuisines": ["Japanese"],
                "neighborhoods": ["Little Tokyo"],
                "price_range": "$$$",
                "min_rating": 4.5,
                "preferred_ingredients": ["miso"],
            },
        )

        self.assertTrue(client.entered)
        self.assertTrue(client.exited)
        self.assertEqual(
            [name for name, _ in client.calls],
            ["search_restaurants", "search_recipes"],
        )
        restaurant_args = client.calls[0][1]
        self.assertEqual(restaurant_args["cuisine"], "Japanese")
        self.assertEqual(restaurant_args["neighborhood"], "Little Tokyo")
        self.assertEqual(restaurant_args["min_rating"], 4.5)
        self.assertEqual(client.calls[1][1]["ingredients"], ["miso"])
        self.assertEqual(result["restaurants"]["count"], 1)
        self.assertEqual(
            result["tool_calls"],
            ["search_restaurants", "search_recipes"],
        )

    async def test_preserves_partial_results_when_one_tool_fails(self) -> None:
        client = FakeClient(
            {
                "search_restaurants": {"count": 1, "results": [{"name": "A"}]},
                "search_recipes": RuntimeError("recipe index unavailable"),
            }
        )
        gateway = FastMCPRetrievalGateway(
            AppSettings(llm_model="model", openai_api_key="test-only"),
            client_factory=lambda _: client,
        )

        result = await gateway.retrieve(query="dinner", profile={})

        self.assertEqual(result["restaurants"]["count"], 1)
        self.assertEqual(result["recipes"], {})
        self.assertEqual(result["errors"][0]["tool"], "search_recipes")
