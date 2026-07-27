from __future__ import annotations

import importlib.util
import unittest
from typing import Any

from connoisseur.client.gradio_app import create_demo
from connoisseur.mcp.server import create_mcp
from connoisseur.orchestrators.agno_backend import AgnoRuntime
from connoisseur.orchestrators.contracts import ChatInput
from connoisseur.orchestrators.langgraph_backend import LangGraphOrchestrator
from connoisseur.orchestrators.settings import AppSettings

HAS_FRAMEWORKS = all(
    importlib.util.find_spec(name) is not None
    for name in (
        "agno",
        "fastmcp",
        "gradio",
        "langchain_mcp_adapters",
        "langchain_openai",
        "langgraph",
        "openai",
    )
)


class NoopLLM:
    async def complete(self, *, system: str, prompt: str) -> str:
        del system, prompt
        return "unused"

    async def aclose(self) -> None:
        return None


class NoopToolModel:
    async def ainvoke(self, messages: list[Any]) -> Any:
        del messages
        return object()


@unittest.skipUnless(HAS_FRAMEWORKS, "Pinned framework dependencies are required")
class FrameworkConstructionTests(unittest.TestCase):
    def test_constructs_real_langgraph_agno_mcp_and_gradio_objects(self) -> None:
        from langchain_core.tools import tool

        @tool
        async def discovery_search(query: str) -> dict[str, str]:
            """Return test evidence for a discovery query."""

            return {"query": query}

        settings = AppSettings(
            llm_model="constructor-smoke-model",
            openai_api_key="test-only",
            mcp_server_url="http://mcp.test/mcp",
        )

        langgraph = LangGraphOrchestrator(
            settings=settings,
            llm=NoopLLM(),
            tools=[discovery_search],
            tool_model=NoopToolModel(),
        )
        agno = AgnoRuntime(settings)
        agno_model = agno._make_model()  # noqa: SLF001
        mcp_tools = agno.components.mcp_tools_cls(
            transport="streamable-http",
            url=settings.mcp_server_url,
        )
        team = agno._make_team(  # noqa: SLF001
            agno_model,
            agno._make_members(agno_model, mcp_tools),  # noqa: SLF001
        )
        mcp_server = create_mcp(service=object())  # type: ignore[arg-type]
        demo = create_demo(api_client=object())  # type: ignore[arg-type]

        self.assertEqual(type(langgraph.graph).__name__, "CompiledStateGraph")
        self.assertEqual(type(team).__name__, "Team")
        self.assertEqual(type(mcp_server).__name__, "FastMCP")
        self.assertEqual(type(demo).__name__, "Blocks")


@unittest.skipUnless(HAS_FRAMEWORKS, "Pinned framework dependencies are required")
class LangGraphToolLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_selects_and_tool_node_executes_discovered_tool(self) -> None:
        from langchain_core.messages import AIMessage
        from langchain_core.tools import tool

        calls: list[str] = []

        @tool
        async def scenario_search(query: str) -> dict[str, str]:
            """Search evidence for the current dining scenario."""

            calls.append(query)
            return {"match": "Sakura Garden"}

        class ToolSelectingModel:
            def __init__(self) -> None:
                self.invocations = 0

            async def ainvoke(self, messages: list[Any]) -> AIMessage:
                del messages
                self.invocations += 1
                if self.invocations == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "scenario_search",
                                "args": {"query": "quiet Japanese date"},
                                "id": "call-1",
                                "type": "tool_call",
                            }
                        ],
                    )
                return AIMessage(content="RETRIEVAL_COMPLETE")

        orchestrator = LangGraphOrchestrator(
            settings=AppSettings(
                llm_model="constructor-smoke-model",
                openai_api_key="test-only",
            ),
            llm=NoopLLM(),
            tools=[scenario_search],
            tool_model=ToolSelectingModel(),
        )

        result = await orchestrator.run(
            ChatInput(
                message="Suggest a Japanese dining experience for a quiet date.",
                session_id="tool-loop-test",
            )
        )

        self.assertEqual(calls, ["quiet Japanese date"])
        self.assertEqual(result.metadata["mcp_tools"], ["scenario_search"])
        self.assertIn("scenario_search", result.metadata["discovered_tools"])
        self.assertEqual(result.metadata["tool_rounds"], 2)
