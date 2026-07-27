"""Agno coordinate-team implementation with MCP-connected retrieval."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import BackendStatus, ChatInput, OrchestrationResult
from .errors import (
    BackendUnavailableError,
    OrchestrationError,
    RetrievalError,
)
from .prompts import AGNO_EXPECTED_OUTPUT, AGNO_TEAM_INSTRUCTIONS
from .settings import AppSettings
from .utils import (
    content_to_text,
    format_history,
    module_available,
    to_jsonable,
)


@dataclass(frozen=True, slots=True)
class AgnoComponents:
    """Version-sensitive Agno classes, isolated for testing and upgrades."""

    agent_cls: Any
    team_cls: Any
    model_cls: Any
    mcp_tools_cls: Any
    coordinate_mode: Any


@dataclass(frozen=True, slots=True)
class AgnoRunResult:
    answer: str
    metadata: Mapping[str, Any]


def load_agno_components() -> AgnoComponents:
    """Load Agno lazily so the LangGraph backend remains independently usable."""

    try:
        from agno.agent import Agent
        from agno.models.openai import OpenAIChat
        from agno.team import Team
        from agno.team.mode import TeamMode
        from agno.tools.mcp import MCPTools
    except ImportError as exc:
        raise BackendUnavailableError(
            "Agno with its MCP and OpenAI extras is required for this backend"
        ) from exc
    return AgnoComponents(
        agent_cls=Agent,
        team_cls=Team,
        model_cls=OpenAIChat,
        mcp_tools_cls=MCPTools,
        coordinate_mode=TeamMode.coordinate,
    )


class AgnoRuntime:
    """Build and execute one coordinate team with explicit MCP lifecycle."""

    member_names = (
        "Profile Strategist",
        "Culinary Retriever",
        "Trend Analyst",
        "Style Matcher",
        "Nutrition Advisor",
    )

    def __init__(
        self,
        settings: AppSettings,
        *,
        components: AgnoComponents | None = None,
    ) -> None:
        self.settings = settings
        self.components = components or load_agno_components()

    def _make_model(self) -> Any:
        kwargs: dict[str, Any] = {
            "id": self.settings.llm_model,
            "api_key": self.settings.openai_client_api_key,
            self.settings.llm_max_tokens_parameter: self.settings.llm_max_tokens,
            "timeout": self.settings.request_timeout_seconds,
        }
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature
        if self.settings.llm_reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.settings.llm_reasoning_effort
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        return self.components.model_cls(**kwargs)

    def _make_members(self, model: Any, mcp_tools: Any) -> list[Any]:
        Agent = self.components.agent_cls
        return [
            Agent(
                id="profile-strategist",
                name="Profile Strategist",
                role="Extracts the user's explicit dining and recipe constraints",
                model=model,
                instructions=[
                    "Read the full request and conversation.",
                    "List only explicit cuisine, location, vibe, budget, occasion, ingredient, allergy, and dietary constraints.",
                    "Do not invent missing preferences.",
                ],
            ),
            Agent(
                id="culinary-retriever",
                name="Culinary Retriever",
                role="Retrieves restaurant, review, and recipe evidence",
                model=model,
                tools=[mcp_tools],
                instructions=[
                    "You must call search_restaurants for every recommendation request.",
                    "Also call search_recipes when recipes or dishes are relevant.",
                    (
                        "Pass limit="
                        f"{self.settings.retrieval_limit} to each search tool."
                    ),
                    "Use get_restaurant_details or get_restaurant_reviews when the request asks about a named restaurant.",
                    "Return the relevant records and their identifiers; do not add facts from memory.",
                ],
            ),
            Agent(
                id="trend-analyst",
                name="Trend Analyst",
                role="Analyzes culinary patterns without overstating trend evidence",
                model=model,
                instructions=[
                    "Analyze only evidence supplied by the retrieval specialist.",
                    "Distinguish corpus patterns from claims about current popularity.",
                    "Return short, evidence-grounded notes.",
                ],
            ),
            Agent(
                id="style-matcher",
                name="Style Matcher",
                role="Matches atmosphere, cuisine, price, and occasion",
                model=model,
                instructions=[
                    "Compare explicit profile constraints with retrieved records.",
                    "Name the strongest matches and explain each fit briefly.",
                    "Never recommend an item absent from the retrieved records.",
                ],
            ),
            Agent(
                id="nutrition-advisor",
                name="Nutrition Advisor",
                role="Checks explicit dietary and ingredient constraints",
                model=model,
                instructions=[
                    "Do not infer nutrition facts that are absent from retrieved records.",
                    "Flag possible conflicts with explicit restrictions.",
                    "For severe allergies, advise verifying ingredients and cross-contact directly; do not give medical advice.",
                ],
            ),
        ]

    def _make_team(self, model: Any, members: list[Any]) -> Any:
        return self.components.team_cls(
            id="connoisseur-coordination-team",
            name="Connoisseur Coordination Team",
            mode=self.components.coordinate_mode,
            model=model,
            members=members,
            instructions=AGNO_TEAM_INSTRUCTIONS,
            expected_output=AGNO_EXPECTED_OUTPUT,
            add_member_tools_to_context=True,
            share_member_interactions=True,
            store_member_responses=True,
            markdown=True,
            telemetry=False,
        )

    @staticmethod
    def _delegated_member_names(response: Any) -> list[str]:
        """Read the members Agno actually invoked, when run details expose them."""

        names: list[str] = []
        for member_response in getattr(response, "member_responses", None) or ():
            name = getattr(member_response, "agent_name", None) or getattr(
                member_response,
                "team_name",
                None,
            )
            normalized = str(name).strip() if name else ""
            if normalized and normalized not in names:
                names.append(normalized)
        return names

    async def run(self, request: ChatInput) -> AgnoRunResult:
        model = self._make_model()
        mcp_tools = self.components.mcp_tools_cls(
            transport="streamable-http",
            url=self.settings.mcp_server_url,
            timeout_seconds=max(1, int(self.settings.request_timeout_seconds)),
        )
        connected = False
        try:
            connect_result = mcp_tools.connect()
            if inspect.isawaitable(connect_result):
                await connect_result
            connected = True

            members = self._make_members(model, mcp_tools)
            team = self._make_team(model, members)
            history = format_history(
                request.history,
                max_messages=self.settings.max_history_messages,
            )
            team_input = (
                f"Conversation history:\n{history}\n\n"
                f"Current user request:\n{request.message}\n\n"
                "Follow the mandatory delegation and evidence-grounding instructions."
            )
            response = await team.arun(
                team_input,
                session_id=request.session_id,
                stream=False,
            )
            answer = content_to_text(getattr(response, "content", response))
            if not answer:
                raise OrchestrationError("The Agno team completed without an answer")
            metadata: dict[str, Any] = {
                "orchestration": "Agno Team",
                "team_mode": "coordinate",
                "configured_members": list(self.member_names),
                "mcp_transport": "streamable-http",
                "retrieval_limit": self.settings.retrieval_limit,
            }
            delegated_members = self._delegated_member_names(response)
            if delegated_members:
                metadata["delegated_members"] = delegated_members
            run_id = getattr(response, "run_id", None)
            if run_id:
                metadata["run_id"] = str(run_id)
            metrics = getattr(response, "metrics", None)
            if metrics is not None:
                normalized_metrics = to_jsonable(metrics)
                if isinstance(normalized_metrics, Mapping):
                    metadata["metrics"] = normalized_metrics
            return AgnoRunResult(answer=answer, metadata=metadata)
        except OrchestrationError:
            raise
        except Exception as exc:
            if not connected:
                raise RetrievalError(
                    f"Agno could not connect to MCP: {type(exc).__name__}"
                ) from exc
            raise OrchestrationError(
                f"Agno team execution failed: {type(exc).__name__}"
            ) from exc
        finally:
            close = getattr(mcp_tools, "close", None)
            if close is not None:
                try:
                    close_result = close()
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception:
                    # Never replace the request result/error with cleanup failure.
                    pass


RuntimeFactory = Callable[[AppSettings], Any]


class AgnoOrchestrator:
    """Adapter exposing Agno Team through the shared backend contract."""

    name = "agno"

    def __init__(
        self,
        *,
        settings: AppSettings,
        runtime_factory: RuntimeFactory | None = None,
    ) -> None:
        self.settings = settings
        self._runtime_factory = runtime_factory or AgnoRuntime

    @classmethod
    def status(cls, settings: AppSettings) -> BackendStatus:
        reason = settings.model_configuration_error()
        if reason:
            return BackendStatus(cls.name, False, reason)
        if not module_available("agno"):
            return BackendStatus(cls.name, False, "Missing package: agno")
        return BackendStatus(cls.name, True)

    async def run(self, request: ChatInput) -> OrchestrationResult:
        configuration_error = self.settings.model_configuration_error()
        if configuration_error:
            raise BackendUnavailableError(configuration_error)
        started = time.perf_counter()
        runtime = self._runtime_factory(self.settings)
        result = await runtime.run(request)
        metadata = dict(result.metadata)
        metadata["elapsed_ms"] = round(
            (time.perf_counter() - started) * 1_000,
            2,
        )
        return OrchestrationResult(
            answer=result.answer,
            backend=self.name,
            session_id=request.session_id,
            metadata=metadata,
        )

    async def aclose(self) -> None:
        # AgnoRuntime scopes its MCP connection to each request.
        return None
