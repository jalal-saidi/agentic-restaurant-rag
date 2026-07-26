from __future__ import annotations

import importlib.util
import unittest
from collections.abc import Mapping
from typing import Any

from connoisseur.client.gradio_app import create_demo
from connoisseur.mcp.server import create_mcp
from connoisseur.orchestrators.agno_backend import AgnoRuntime
from connoisseur.orchestrators.langgraph_backend import LangGraphOrchestrator
from connoisseur.orchestrators.settings import AppSettings

HAS_FRAMEWORKS = all(
    importlib.util.find_spec(name) is not None
    for name in ("agno", "fastmcp", "gradio", "langgraph", "openai")
)


class NoopLLM:
    async def complete(self, *, system: str, prompt: str) -> str:
        del system, prompt
        return "unused"

    async def aclose(self) -> None:
        return None


class NoopRetrieval:
    async def retrieve(
        self,
        *,
        query: str,
        profile: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del query, profile
        return {}

    async def health(self) -> Mapping[str, Any]:
        return {"status": "ok"}

    async def aclose(self) -> None:
        return None


@unittest.skipUnless(HAS_FRAMEWORKS, "Pinned framework dependencies are required")
class FrameworkConstructionTests(unittest.TestCase):
    def test_constructs_real_langgraph_agno_mcp_and_gradio_objects(self) -> None:
        settings = AppSettings(
            llm_model="constructor-smoke-model",
            openai_api_key="test-only",
            mcp_server_url="http://mcp.test/mcp",
        )

        langgraph = LangGraphOrchestrator(
            settings=settings,
            llm=NoopLLM(),
            retrieval=NoopRetrieval(),
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
