"""Explicit LangGraph implementation of the culinary multi-agent workflow."""

from __future__ import annotations

import time
from typing import Any, TypedDict

from .contracts import (
    BackendStatus,
    ChatInput,
    LanguageModel,
    OrchestrationResult,
    RetrievalGateway,
)
from .errors import (
    BackendUnavailableError,
    OrchestrationError,
)
from .prompts import (
    NUTRITION_SYSTEM,
    PROFILE_SYSTEM,
    STYLE_SYSTEM,
    SYNTHESIS_SYSTEM,
    TREND_SYSTEM,
)
from .settings import AppSettings
from .utils import (
    json_for_prompt,
    module_available,
    parse_json_object,
)


class CulinaryGraphState(TypedDict, total=False):
    """State channels for the explicit fan-out/fan-in graph."""

    message: str
    session_id: str
    history: list[dict[str, str]]
    profile: dict[str, Any]
    evidence: dict[str, Any]
    trend_analysis: str
    style_analysis: str
    nutrition_analysis: str
    answer: str


class LangGraphOrchestrator:
    """Profile -> MCP retrieval -> parallel specialists -> synthesis."""

    name = "langgraph"
    topology = (
        ("__start__", "profile"),
        ("profile", "retrieve"),
        ("retrieve", "trend"),
        ("retrieve", "style"),
        ("retrieve", "nutrition"),
        ("trend", "synthesis"),
        ("style", "synthesis"),
        ("nutrition", "synthesis"),
        ("synthesis", "__end__"),
    )

    def __init__(
        self,
        *,
        settings: AppSettings,
        llm: LanguageModel,
        retrieval: RetrievalGateway,
        graph: Any | None = None,
        checkpointer: Any | None = None,
        state_graph_cls: Any | None = None,
        start_node: Any | None = None,
        end_node: Any | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.retrieval = retrieval
        self.graph = graph or self._build_graph(
            checkpointer=checkpointer,
            state_graph_cls=state_graph_cls,
            start_node=start_node,
            end_node=end_node,
        )

    @classmethod
    def status(cls, settings: AppSettings) -> BackendStatus:
        reason = settings.model_configuration_error()
        if reason:
            return BackendStatus(cls.name, False, reason)
        missing = [
            package
            for package in ("langgraph", "openai", "fastmcp")
            if not module_available(package)
        ]
        if missing:
            return BackendStatus(
                cls.name,
                False,
                f"Missing package(s): {', '.join(missing)}",
            )
        return BackendStatus(cls.name, True)

    def _build_graph(
        self,
        *,
        checkpointer: Any | None,
        state_graph_cls: Any | None,
        start_node: Any | None,
        end_node: Any | None,
    ) -> Any:
        if state_graph_cls is None:
            try:
                from langgraph.graph import END, START, StateGraph
            except ImportError as exc:
                raise BackendUnavailableError(
                    "The 'langgraph' package is required for this backend"
                ) from exc
            state_graph_cls = StateGraph
            start_node = START
            end_node = END
        elif start_node is None or end_node is None:
            raise ValueError(
                "start_node and end_node are required with a custom StateGraph"
            )

        builder = state_graph_cls(CulinaryGraphState)
        builder.add_node("profile", self._profile_node)
        builder.add_node("retrieve", self._retrieval_node)
        builder.add_node("trend", self._trend_node)
        builder.add_node("style", self._style_node)
        builder.add_node("nutrition", self._nutrition_node)
        builder.add_node("synthesis", self._synthesis_node)

        builder.add_edge(start_node, "profile")
        builder.add_edge("profile", "retrieve")
        # Native LangGraph fan-out: these nodes share one superstep.
        builder.add_edge("retrieve", "trend")
        builder.add_edge("retrieve", "style")
        builder.add_edge("retrieve", "nutrition")
        # Native fan-in: synthesis starts after all three branches complete.
        builder.add_edge("trend", "synthesis")
        builder.add_edge("style", "synthesis")
        builder.add_edge("nutrition", "synthesis")
        builder.add_edge("synthesis", end_node)

        if checkpointer is None:
            return builder.compile()
        return builder.compile(checkpointer=checkpointer)

    async def _profile_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, Any]:
        history = "\n".join(
            f"{item['role']}: {item['content']}" for item in state.get("history", [])
        ) or "(no prior conversation)"
        prompt = (
            f"Conversation:\n{history}\n\n"
            f"Current request:\n{state['message']}\n\n"
            "Return only the requested JSON object."
        )
        raw = await self.llm.complete(system=PROFILE_SYSTEM, prompt=prompt)
        try:
            profile = parse_json_object(raw)
        except (ValueError, TypeError):
            profile = {
                "intent": state["message"],
                "cuisines": [],
                "neighborhoods": [],
                "vibes": [],
                "dietary_needs": [],
                "disliked_ingredients": [],
                "preferred_ingredients": [],
                "price_range": None,
                "min_rating": None,
                "occasion": None,
                "recipe_request": False,
                "notes": "Profile model returned unstructured output.",
            }
        return {"profile": profile}

    async def _retrieval_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, Any]:
        evidence = await self.retrieval.retrieve(
            query=state["message"],
            profile=state["profile"],
        )
        return {"evidence": dict(evidence)}

    def _specialist_prompt(self, state: CulinaryGraphState) -> str:
        profile = json_for_prompt(
            state["profile"],
            max_chars=max(1_000, self.settings.max_context_chars // 4),
        )
        evidence = json_for_prompt(
            state["evidence"],
            max_chars=self.settings.max_context_chars,
        )
        return (
            f"User request:\n{state['message']}\n\n"
            f"Extracted profile:\n{profile}\n\n"
            f"Retrieved evidence:\n{evidence}"
        )

    async def _trend_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, str]:
        answer = await self.llm.complete(
            system=TREND_SYSTEM,
            prompt=self._specialist_prompt(state),
        )
        return {"trend_analysis": answer}

    async def _style_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, str]:
        answer = await self.llm.complete(
            system=STYLE_SYSTEM,
            prompt=self._specialist_prompt(state),
        )
        return {"style_analysis": answer}

    async def _nutrition_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, str]:
        answer = await self.llm.complete(
            system=NUTRITION_SYSTEM,
            prompt=self._specialist_prompt(state),
        )
        return {"nutrition_analysis": answer}

    async def _synthesis_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, str]:
        max_context = self.settings.max_context_chars
        profile_budget = max(160, max_context // 6)
        evidence_budget = max(320, max_context // 2)
        report_budget = max(
            120,
            (max_context - profile_budget - evidence_budget) // 3,
        )
        prompt = (
            f"User request:\n{state['message']}\n\n"
            "Extracted profile:\n"
            f"{json_for_prompt(state['profile'], max_chars=profile_budget)}\n\n"
            "Retrieved evidence:\n"
            f"{json_for_prompt(state['evidence'], max_chars=evidence_budget)}\n\n"
            "Trend specialist report:\n"
            f"{json_for_prompt(state['trend_analysis'], max_chars=report_budget)}\n\n"
            "Style specialist report:\n"
            f"{json_for_prompt(state['style_analysis'], max_chars=report_budget)}\n\n"
            "Nutrition specialist report:\n"
            f"{json_for_prompt(state['nutrition_analysis'], max_chars=report_budget)}"
        )
        answer = await self.llm.complete(
            system=SYNTHESIS_SYSTEM,
            prompt=prompt,
        )
        return {"answer": answer}

    async def run(self, request: ChatInput) -> OrchestrationResult:
        started = time.perf_counter()
        history = (
            request.history[-self.settings.max_history_messages :]
            if self.settings.max_history_messages
            else ()
        )
        state: CulinaryGraphState = {
            "message": request.message.strip(),
            "session_id": request.session_id,
            "history": [
                {"role": message.role, "content": message.content}
                for message in history
            ],
        }
        try:
            output = await self.graph.ainvoke(
                state,
                config={
                    "configurable": {"thread_id": request.session_id},
                    "max_concurrency": 3,
                },
            )
        except OrchestrationError:
            raise
        except Exception as exc:
            raise OrchestrationError(
                f"LangGraph execution failed: {type(exc).__name__}"
            ) from exc
        answer = str(output.get("answer", "")).strip()
        if not answer:
            raise OrchestrationError("LangGraph completed without an answer")
        return OrchestrationResult(
            answer=answer,
            backend=self.name,
            session_id=request.session_id,
            metadata={
                "orchestration": "StateGraph",
                "route": [
                    "profile",
                    "retrieve",
                    ["trend", "style", "nutrition"],
                    "synthesis",
                ],
                "mcp_tools": output.get("evidence", {}).get("tool_calls", []),
                "elapsed_ms": round((time.perf_counter() - started) * 1_000, 2),
            },
        )

    async def aclose(self) -> None:
        await self.llm.aclose()
        await self.retrieval.aclose()
