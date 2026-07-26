from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from connoisseur.orchestrators.agno_backend import (
    AgnoComponents,
    AgnoRuntime,
)
from connoisseur.orchestrators.contracts import ChatInput
from connoisseur.orchestrators.settings import AppSettings


class FakeModel:
    instances: list[FakeModel] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


class FakeAgent:
    instances: list[FakeAgent] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.instances.append(self)


class FakeMCPTools:
    instances: list[FakeMCPTools] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.connected = False
        self.closed = False
        self.instances.append(self)

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.closed = True


class FakeTeam:
    instances: list[FakeTeam] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.run_input: str | None = None
        self.run_kwargs: dict[str, Any] | None = None
        self.instances.append(self)

    async def arun(self, value: str, **kwargs: Any) -> Any:
        self.run_input = value
        self.run_kwargs = kwargs
        return SimpleNamespace(
            content="Sakura Garden matches the requested quiet atmosphere.",
            run_id="run-1",
            metrics={"input_tokens": 10},
            member_responses=[
                SimpleNamespace(agent_name="Culinary Retriever"),
                SimpleNamespace(agent_name="Style Matcher"),
            ],
        )


class AgnoRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        FakeModel.instances.clear()
        FakeAgent.instances.clear()
        FakeMCPTools.instances.clear()
        FakeTeam.instances.clear()

    async def test_builds_coordinate_team_with_mcp_retrieval_member(self) -> None:
        components = AgnoComponents(
            agent_cls=FakeAgent,
            team_cls=FakeTeam,
            model_cls=FakeModel,
            mcp_tools_cls=FakeMCPTools,
            coordinate_mode="coordinate",
        )
        runtime = AgnoRuntime(
            AppSettings(
                llm_model="model",
                openai_base_url="http://model.test/v1",
                mcp_server_url="http://mcp:8001/mcp",
            ),
            components=components,
        )

        result = await runtime.run(
            ChatInput(
                message="Find a quiet Japanese restaurant",
                session_id="session-1",
            )
        )

        self.assertIn("Sakura Garden", result.answer)
        self.assertEqual(len(FakeAgent.instances), 5)
        retriever = next(
            item
            for item in FakeAgent.instances
            if item.kwargs["name"] == "Culinary Retriever"
        )
        tools = retriever.kwargs["tools"]
        self.assertEqual(len(tools), 1)
        self.assertIs(tools[0], FakeMCPTools.instances[0])
        self.assertEqual(retriever.kwargs["id"], "culinary-retriever")
        self.assertTrue(
            any(
                "limit=5" in instruction
                for instruction in retriever.kwargs["instructions"]
            )
        )
        self.assertTrue(FakeMCPTools.instances[0].connected)
        self.assertTrue(FakeMCPTools.instances[0].closed)
        self.assertEqual(
            FakeMCPTools.instances[0].kwargs["timeout_seconds"],
            90,
        )
        team = FakeTeam.instances[0]
        self.assertEqual(team.kwargs["id"], "connoisseur-coordination-team")
        self.assertEqual(team.kwargs["mode"], "coordinate")
        self.assertEqual(len(team.kwargs["members"]), 5)
        self.assertEqual(team.run_kwargs["session_id"], "session-1")  # type: ignore[index]
        self.assertEqual(
            FakeModel.instances[0].kwargs["base_url"],
            "http://model.test/v1",
        )
        self.assertEqual(
            result.metadata["delegated_members"],
            ["Culinary Retriever", "Style Matcher"],
        )
        self.assertEqual(
            FakeModel.instances[0].kwargs["api_key"],
            "not-required",
        )
        self.assertEqual(
            FakeModel.instances[0].kwargs["max_completion_tokens"],
            1_200,
        )
        self.assertNotIn("temperature", FakeModel.instances[0].kwargs)
