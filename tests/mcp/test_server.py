from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import patch

import connoisseur.mcp.server as server_module
from connoisseur.mcp.server import ConnoisseurTools


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def search_restaurants(self, query: str, **kwargs: object) -> dict:
        self.calls.append(("search_restaurants", (query, kwargs)))
        return {"status": "ok", "results": [{"name": "Example"}]}

    def restaurant_details(self, **kwargs: object) -> dict:
        self.calls.append(("restaurant_details", kwargs))
        return {"status": "found"}

    def search_recipes(self, query: str, **kwargs: object) -> dict:
        self.calls.append(("search_recipes", (query, kwargs)))
        return {"status": "ok", "results": [{"name": "Recipe"}]}

    def restaurant_reviews(self, **kwargs: object) -> dict:
        self.calls.append(("restaurant_reviews", kwargs))
        return {"status": "found", "reviews": []}

    def health(self) -> dict:
        self.calls.append(("health", None))
        return {"status": "ok"}

    def corpus_stats(self) -> dict:
        self.calls.append(("corpus_stats", None))
        return {"restaurants": 1}


class FakeFastMCP:
    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, object] = {}
        self.resources: dict[str, object] = {}
        self.routes: dict[str, object] = {}
        self.run_kwargs: dict | None = None

    def tool(self, function):
        self.tools[function.__name__] = function
        return function

    def resource(self, uri: str):
        def register(function):
            self.resources[uri] = function
            return function

        return register

    def custom_route(self, path: str, methods: list[str]):
        def register(function):
            self.routes[path] = (methods, function)
            return function

        return register

    def run(self, **kwargs):
        self.run_kwargs = kwargs


class MCPServerTests(unittest.TestCase):
    def test_tool_facade_delegates_structured_arguments(self) -> None:
        service = FakeService()
        tools = ConnoisseurTools(service)  # type: ignore[arg-type]

        result = tools.search_recipes(
            "tomato",
            limit=3,
            cuisine="Italian",
            max_total_minutes=30,
            ingredients=["basil"],
        )

        self.assertEqual(result["results"][0]["name"], "Recipe")
        name, payload = service.calls[0]
        self.assertEqual(name, "search_recipes")
        self.assertEqual(payload[1]["ingredients"], ["basil"])

    def test_factory_registers_all_tools_without_loading_models(self) -> None:
        service = FakeService()
        with patch.object(server_module, "FastMCP", FakeFastMCP):
            server = server_module.create_mcp(service)  # type: ignore[arg-type]

        self.assertEqual(
            set(server.tools),
            {
                "search_restaurants",
                "get_restaurant_details",
                "search_recipes",
                "get_restaurant_reviews",
                "health_check",
                "corpus_stats",
            },
        )
        self.assertIn("connoisseur://health", server.resources)
        self.assertIn("/healthz", server.routes)
        result = server.tools["health_check"]()
        self.assertEqual(result, {"status": "ok"})

    def test_http_health_route_returns_degraded_as_ready(self) -> None:
        service = FakeService()
        service.health = lambda: {
            "status": "degraded",
            "active_backend": "lexical",
        }

        class FakeJSONResponse:
            def __init__(self, content, status_code):
                self.content = content
                self.status_code = status_code

        responses = types.ModuleType("starlette.responses")
        responses.JSONResponse = FakeJSONResponse
        starlette = types.ModuleType("starlette")
        starlette.responses = responses
        with (
            patch.object(server_module, "FastMCP", FakeFastMCP),
            patch.dict(
                sys.modules,
                {
                    "starlette": starlette,
                    "starlette.responses": responses,
                },
            ),
        ):
            server = server_module.create_mcp(service)  # type: ignore[arg-type]
            handler = server.routes["/healthz"][1]
            response = asyncio.run(handler(object()))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content["status"], "degraded")

    def test_main_runs_streamable_http_transport(self) -> None:
        with (
            patch.object(server_module, "FastMCP", FakeFastMCP),
            patch.object(server_module, "mcp", None),
            patch.dict(
                "os.environ",
                {
                    "MCP_TRANSPORT": "http",
                    "MCP_HOST": "127.0.0.1",
                    "MCP_PORT": "9001",
                },
                clear=False,
            ),
        ):
            created = FakeFastMCP("captured")
            with patch.object(
                server_module, "create_mcp", return_value=created
            ):
                server_module.main()

        self.assertEqual(
            created.run_kwargs,
            {
                "transport": "http",
                "host": "127.0.0.1",
                "port": 9001,
                "path": "/mcp",
            },
        )


if __name__ == "__main__":
    unittest.main()
