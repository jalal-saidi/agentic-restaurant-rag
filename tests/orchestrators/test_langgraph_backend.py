from __future__ import annotations

import unittest
from collections.abc import Mapping
from typing import Any

from connoisseur.orchestrators.contracts import ChatInput, ChatMessage
from connoisseur.orchestrators.langgraph_backend import LangGraphOrchestrator
from connoisseur.orchestrators.settings import AppSettings


class FakeLLM:
    async def complete(self, *, system: str, prompt: str) -> str:
        return "unused"

    async def aclose(self) -> None:
        return None


class CapturingLLM(FakeLLM):
    def __init__(self) -> None:
        self.prompt = ""

    async def complete(self, *, system: str, prompt: str) -> str:
        del system
        self.prompt = prompt
        return "Synthesized answer"


class FakeRetrieval:
    async def retrieve(
        self,
        *,
        query: str,
        profile: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return {}

    async def health(self) -> Mapping[str, Any]:
        return {"status": "ok"}

    async def aclose(self) -> None:
        return None


class FakeCompiledGraph:
    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self.output = output or {}
        self.input: dict[str, Any] | None = None
        self.config: dict[str, Any] | None = None

    async def ainvoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.input = state
        self.config = config
        return self.output


class RecordingStateGraph:
    latest: RecordingStateGraph | None = None

    def __init__(self, state_schema: Any) -> None:
        self.state_schema = state_schema
        self.nodes: dict[str, Any] = {}
        self.edges: list[tuple[Any, Any]] = []
        self.compiled = FakeCompiledGraph()
        RecordingStateGraph.latest = self

    def add_node(self, name: str, function: Any) -> None:
        self.nodes[name] = function

    def add_edge(self, source: Any, target: Any) -> None:
        self.edges.append((source, target))

    def compile(self, **_: Any) -> FakeCompiledGraph:
        return self.compiled


class LangGraphTopologyTests(unittest.TestCase):
    def test_builds_explicit_parallel_fan_out_and_fan_in(self) -> None:
        LangGraphOrchestrator(
            settings=AppSettings(llm_model="model", openai_api_key="test-only"),
            llm=FakeLLM(),
            retrieval=FakeRetrieval(),
            state_graph_cls=RecordingStateGraph,
            start_node="START",
            end_node="END",
        )
        graph = RecordingStateGraph.latest
        assert graph is not None

        self.assertEqual(
            set(graph.nodes),
            {"profile", "retrieve", "trend", "style", "nutrition", "synthesis"},
        )
        self.assertIn(("retrieve", "trend"), graph.edges)
        self.assertIn(("retrieve", "style"), graph.edges)
        self.assertIn(("retrieve", "nutrition"), graph.edges)
        self.assertIn(("trend", "synthesis"), graph.edges)
        self.assertIn(("style", "synthesis"), graph.edges)
        self.assertIn(("nutrition", "synthesis"), graph.edges)


class LangGraphRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_shared_contract_and_thread_configuration(self) -> None:
        graph = FakeCompiledGraph(
            {
                "answer": "Sakura Garden is the strongest match.",
                "evidence": {
                    "tool_calls": ["search_restaurants", "search_recipes"]
                },
            }
        )
        orchestrator = LangGraphOrchestrator(
            settings=AppSettings(llm_model="model", openai_api_key="test-only"),
            llm=FakeLLM(),
            retrieval=FakeRetrieval(),
            graph=graph,
        )

        result = await orchestrator.run(
            ChatInput(
                message="Find a quiet Japanese date spot",
                session_id="session-1",
            )
        )

        self.assertEqual(result.backend, "langgraph")
        self.assertEqual(result.session_id, "session-1")
        self.assertIn("Sakura Garden", result.answer)
        self.assertEqual(
            graph.config["configurable"]["thread_id"],  # type: ignore[index]
            "session-1",
        )
        self.assertEqual(graph.config["max_concurrency"], 3)  # type: ignore[index]

    async def test_zero_history_limit_sends_no_history_to_graph(self) -> None:
        graph = FakeCompiledGraph({"answer": "Done.", "evidence": {}})
        orchestrator = LangGraphOrchestrator(
            settings=AppSettings(
                llm_model="model",
                openai_api_key="test-only",
                max_history_messages=0,
            ),
            llm=FakeLLM(),
            retrieval=FakeRetrieval(),
            graph=graph,
        )

        await orchestrator.run(
            ChatInput(
                message="Find dinner",
                session_id="session-1",
                history=(
                    ChatMessage(role="user", content="Earlier question"),
                    ChatMessage(role="assistant", content="Earlier answer"),
                ),
            )
        )

        self.assertEqual(graph.input["history"], [])  # type: ignore[index]

    async def test_synthesis_preserves_every_parallel_report_when_evidence_is_large(
        self,
    ) -> None:
        llm = CapturingLLM()
        orchestrator = LangGraphOrchestrator(
            settings=AppSettings(
                llm_model="model",
                openai_api_key="test-only",
                max_context_chars=1_000,
            ),
            llm=llm,
            retrieval=FakeRetrieval(),
            graph=FakeCompiledGraph(),
        )

        await orchestrator._synthesis_node(  # noqa: SLF001
            {
                "message": "Find dinner",
                "profile": {"cuisine": "Japanese"},
                "evidence": {"large": "x" * 10_000},
                "trend_analysis": "TREND_SENTINEL",
                "style_analysis": "STYLE_SENTINEL",
                "nutrition_analysis": "NUTRITION_SENTINEL",
            }
        )

        self.assertIn("TREND_SENTINEL", llm.prompt)
        self.assertIn("STYLE_SENTINEL", llm.prompt)
        self.assertIn("NUTRITION_SENTINEL", llm.prompt)
