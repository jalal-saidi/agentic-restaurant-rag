"""Explicit LangGraph implementation of the culinary multi-agent workflow."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Annotated, Any, TypedDict

from .contracts import (
    BackendStatus,
    ChatInput,
    LanguageModel,
    OrchestrationResult,
    ToolProvider,
)
from .errors import (
    BackendUnavailableError,
    ModelInvocationError,
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
from .tool_discovery import (
    LangChainMCPToolProvider,
    bind_openai_mcp_tools,
)
from .utils import (
    json_for_prompt,
    module_available,
    parse_json_object,
    to_jsonable,
)


def _append_messages(left: list[Any], right: list[Any]) -> list[Any]:
    """Reducer for the model/tool conversation inside one graph execution."""

    return [*left, *right]


class CulinaryGraphState(TypedDict, total=False):
    """State channels for the explicit fan-out/fan-in graph."""

    message: str
    session_id: str
    history: list[dict[str, str]]
    profile: dict[str, Any]
    messages: Annotated[list[Any], _append_messages]
    tool_rounds: int
    evidence: dict[str, Any]
    trend_analysis: str
    style_analysis: str
    nutrition_analysis: str
    answer: str


class LangGraphOrchestrator:
    """Profile -> model-selected MCP tools -> specialists -> synthesis."""

    name = "langgraph"
    topology = (
        ("__start__", "profile"),
        ("profile", "select_tools"),
        ("select_tools", "tools"),
        ("select_tools", "collect_evidence"),
        ("tools", "select_tools"),
        ("tools", "collect_evidence"),
        ("collect_evidence", "trend"),
        ("collect_evidence", "style"),
        ("collect_evidence", "nutrition"),
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
        tool_provider: ToolProvider | None = None,
        tools: Sequence[Any] | None = None,
        tool_model: Any | None = None,
        graph: Any | None = None,
        checkpointer: Any | None = None,
        state_graph_cls: Any | None = None,
        start_node: Any | None = None,
        end_node: Any | None = None,
        tool_node_cls: Any | None = None,
        tools_condition_fn: Callable[[Any], str] | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tool_provider = tool_provider or LangChainMCPToolProvider(settings)
        self._tools = tuple(tools) if tools is not None else None
        self._tool_model = tool_model
        self._tool_names = (
            [str(getattr(tool, "name", type(tool).__name__)) for tool in self._tools]
            if self._tools is not None
            else []
        )
        self._checkpointer = checkpointer
        self._state_graph_cls = state_graph_cls
        self._start_node = start_node
        self._end_node = end_node
        self._tool_node_cls = tool_node_cls
        self._tools_condition_fn = tools_condition_fn
        self._graph_lock = asyncio.Lock()
        self.graph = graph
        if self.graph is None and self._tools is not None:
            if self._tool_model is None:
                self._tool_model = bind_openai_mcp_tools(
                    self.settings,
                    self._tools,
                )
            self.graph = self._build_graph()

    @classmethod
    def status(cls, settings: AppSettings) -> BackendStatus:
        reason = settings.model_configuration_error()
        if reason:
            return BackendStatus(cls.name, False, reason)
        missing = [
            package
            for package in (
                "langgraph",
                "langchain_mcp_adapters",
                "langchain_openai",
                "openai",
            )
            if not module_available(package)
        ]
        if missing:
            return BackendStatus(
                cls.name,
                False,
                f"Missing package(s): {', '.join(missing)}",
            )
        return BackendStatus(cls.name, True)

    async def _ensure_graph(self) -> Any:
        """Discover MCP tools and compile the graph once, on first use."""

        if self.graph is not None:
            return self.graph
        async with self._graph_lock:
            if self.graph is not None:
                return self.graph
            self._tools = tuple(await self.tool_provider.get_tools())
            if not self._tools:
                raise OrchestrationError("LangGraph has no discovered MCP tools")
            self._tool_names = [
                str(getattr(tool, "name", type(tool).__name__))
                for tool in self._tools
            ]
            if self._tool_model is None:
                self._tool_model = bind_openai_mcp_tools(
                    self.settings,
                    self._tools,
                )
            self.graph = self._build_graph()
            return self.graph

    def _build_graph(self) -> Any:
        state_graph_cls = self._state_graph_cls
        start_node = self._start_node
        end_node = self._end_node
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
        if self._tools is None or self._tool_model is None:
            raise RuntimeError("MCP tools and a bound tool model are required")

        tool_node_cls = self._tool_node_cls
        tools_condition_fn = self._tools_condition_fn
        if tool_node_cls is None or tools_condition_fn is None:
            try:
                from langgraph.prebuilt import ToolNode, tools_condition
            except ImportError as exc:
                raise BackendUnavailableError(
                    "LangGraph's prebuilt tool nodes are required"
                ) from exc
            tool_node_cls = tool_node_cls or ToolNode
            tools_condition_fn = tools_condition_fn or tools_condition
        self._tools_condition_fn = tools_condition_fn

        builder = state_graph_cls(CulinaryGraphState)
        builder.add_node("profile", self._profile_node)
        builder.add_node("select_tools", self._tool_selection_node)
        builder.add_node(
            "tools",
            tool_node_cls(
                self._tools,
                handle_tool_errors=True,
                messages_key="messages",
            ),
        )
        builder.add_node("collect_evidence", self._collect_evidence_node)
        builder.add_node("trend", self._trend_node)
        builder.add_node("style", self._style_node)
        builder.add_node("nutrition", self._nutrition_node)
        builder.add_node("synthesis", self._synthesis_node)

        builder.add_edge(start_node, "profile")
        builder.add_edge("profile", "select_tools")
        builder.add_conditional_edges(
            "select_tools",
            self._route_after_tool_selection,
            {
                "tools": "tools",
                "collect_evidence": "collect_evidence",
            },
        )
        builder.add_conditional_edges(
            "tools",
            self._route_after_tools,
            {
                "select_tools": "select_tools",
                "collect_evidence": "collect_evidence",
            },
        )
        # Native LangGraph fan-out: these nodes share one superstep.
        builder.add_edge("collect_evidence", "trend")
        builder.add_edge("collect_evidence", "style")
        builder.add_edge("collect_evidence", "nutrition")
        # Native fan-in: synthesis starts after all three branches complete.
        builder.add_edge("trend", "synthesis")
        builder.add_edge("style", "synthesis")
        builder.add_edge("nutrition", "synthesis")
        builder.add_edge("synthesis", end_node)

        if self._checkpointer is None:
            return builder.compile()
        return builder.compile(checkpointer=self._checkpointer)

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

    def _tool_selection_prompt(self, state: CulinaryGraphState) -> str:
        history = json_for_prompt(
            state.get("history", []),
            max_chars=max(1_000, self.settings.max_context_chars // 6),
        )
        profile = json_for_prompt(
            state["profile"],
            max_chars=max(1_000, self.settings.max_context_chars // 3),
        )
        return (
            "Select evidence from the MCP tools discovered for this request.\n\n"
            f"Current request:\n{state['message']}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Extracted profile:\n{profile}\n\n"
            f"Default result limit: {self.settings.retrieval_limit}"
        )

    async def _tool_selection_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, Any]:
        messages = state.get("messages", [])
        invocation_messages: list[Any]
        if messages:
            invocation_messages = messages
        else:
            invocation_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the retrieval planner in a culinary RAG system. "
                        "Inspect the available MCP tool names, descriptions, and "
                        "argument schemas at runtime. Call the smallest useful set "
                        "of evidence-producing tools for the user's request. You "
                        "must call at least one tool before stopping. You may call "
                        "independent tools in parallel and use returned identifiers "
                        "for follow-up detail or review calls. Do not assume fixed "
                        "tool names, do not repeat an identical call, and do not "
                        "use health or administrative tools unless needed to "
                        "diagnose missing evidence. When the evidence is sufficient, "
                        "respond with RETRIEVAL_COMPLETE without another tool call. "
                        "Do not answer the user's culinary question yourself."
                    ),
                },
                {
                    "role": "user",
                    "content": self._tool_selection_prompt(state),
                },
            ]
        if self._tool_model is None:
            raise RuntimeError("LangGraph tool model is not configured")
        try:
            response = await self._tool_model.ainvoke(invocation_messages)
        except OrchestrationError:
            raise
        except Exception as exc:
            raise ModelInvocationError(
                f"LangGraph tool selection failed: {type(exc).__name__}"
            ) from exc

        has_previous_results = any(
            getattr(message, "type", None) == "tool" for message in messages
        )
        if not getattr(response, "tool_calls", None) and not has_previous_results:
            raise OrchestrationError(
                "LangGraph tool selector stopped before calling an MCP tool"
            )
        additions = [response] if messages else [*invocation_messages, response]
        return {
            "messages": additions,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
        }

    def _route_after_tool_selection(
        self,
        state: CulinaryGraphState,
    ) -> str:
        if self._tools_condition_fn is None:
            raise RuntimeError("LangGraph tools condition is not configured")
        route = self._tools_condition_fn(state)
        return "tools" if route == "tools" else "collect_evidence"

    def _route_after_tools(self, state: CulinaryGraphState) -> str:
        if (
            state.get("tool_rounds", 0)
            >= self.settings.langgraph_max_tool_rounds
        ):
            return "collect_evidence"
        return "select_tools"

    async def _collect_evidence_node(
        self,
        state: CulinaryGraphState,
    ) -> dict[str, Any]:
        calls_by_id: dict[str, str] = {}
        called_tools: list[str] = []
        for message in state.get("messages", []):
            for call in getattr(message, "tool_calls", None) or []:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name", "")).strip()
                call_id = str(call.get("id", "")).strip()
                if name:
                    called_tools.append(name)
                if name and call_id:
                    calls_by_id[call_id] = name

        results: list[dict[str, Any]] = []
        for message in state.get("messages", []):
            if getattr(message, "type", None) != "tool":
                continue
            call_id = str(getattr(message, "tool_call_id", "") or "")
            name = str(
                getattr(message, "name", None)
                or calls_by_id.get(call_id)
                or "unknown"
            )
            result: dict[str, Any] = {
                "tool": name,
                "status": str(getattr(message, "status", "success")),
                "content": to_jsonable(getattr(message, "content", "")),
            }
            artifact = getattr(message, "artifact", None)
            if artifact is not None:
                result["artifact"] = to_jsonable(artifact)
            results.append(result)

        if not results:
            raise OrchestrationError(
                "LangGraph completed tool selection without MCP evidence"
            )
        return {
            "evidence": {
                "discovered_tools": list(self._tool_names),
                "tool_calls": called_tools,
                "tool_results": results,
            }
        }

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
            "messages": [],
            "tool_rounds": 0,
        }
        try:
            graph = await self._ensure_graph()
            output = await graph.ainvoke(
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
                    "select_tools",
                    "tools",
                    "collect_evidence",
                    ["trend", "style", "nutrition"],
                    "synthesis",
                ],
                "mcp_tools": output.get("evidence", {}).get("tool_calls", []),
                "discovered_tools": output.get("evidence", {}).get(
                    "discovered_tools",
                    self._tool_names,
                ),
                "tool_rounds": output.get("tool_rounds", 0),
                "elapsed_ms": round((time.perf_counter() - started) * 1_000, 2),
            },
        )

    async def aclose(self) -> None:
        await asyncio.gather(
            self.llm.aclose(),
            self.tool_provider.aclose(),
            return_exceptions=True,
        )
