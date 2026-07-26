"""Lazy backend selection for the HTTP service."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from .agno_backend import AgnoOrchestrator
from .contracts import BackendStatus, Orchestrator
from .errors import BackendUnavailableError
from .langgraph_backend import LangGraphOrchestrator
from .llm import OpenAICompatibleLLM
from .mcp_gateway import FastMCPRetrievalGateway
from .settings import AppSettings

BackendFactory = Callable[[], Orchestrator]


class BackendRegistry:
    """Probe and lazily create independently installable backends."""

    supported = ("langgraph", "agno")

    def __init__(
        self,
        settings: AppSettings | None = None,
        *,
        factories: dict[str, BackendFactory] | None = None,
        status_providers: dict[str, Callable[[], BackendStatus]] | None = None,
    ) -> None:
        self.settings = settings or AppSettings.from_env()
        self._instances: dict[str, Orchestrator] = {}
        self._lock = asyncio.Lock()
        self._factories = factories or {
            "langgraph": self._make_langgraph,
            "agno": self._make_agno,
        }
        self._status_providers = status_providers or {
            "langgraph": lambda: LangGraphOrchestrator.status(self.settings),
            "agno": lambda: AgnoOrchestrator.status(self.settings),
        }

    def _make_langgraph(self) -> Orchestrator:
        return LangGraphOrchestrator(
            settings=self.settings,
            llm=OpenAICompatibleLLM(self.settings),
            retrieval=FastMCPRetrievalGateway(self.settings),
        )

    def _make_agno(self) -> Orchestrator:
        return AgnoOrchestrator(settings=self.settings)

    def statuses(self) -> list[BackendStatus]:
        return [
            self._status_providers[name]()
            for name in self.supported
            if name in self._status_providers
        ]

    async def readiness(self) -> dict[str, Any]:
        """Check configuration/framework readiness plus the shared MCP service."""

        statuses = self.statuses()
        retrieval = await FastMCPRetrievalGateway(self.settings).health()
        retrieval_status = str(retrieval.get("status", "")).lower()
        ready = any(item.available for item in statuses) and retrieval_status in {
            "ok",
            "degraded",
        }
        return {
            "status": "ready" if ready else "not_ready",
            "backends": [
                {
                    "name": item.name,
                    "available": item.available,
                    "reason": item.reason,
                }
                for item in statuses
            ],
            "retrieval": dict(retrieval),
        }

    async def get(self, name: str) -> Orchestrator:
        normalized = name.strip().lower()
        if normalized not in self._factories:
            raise BackendUnavailableError(f"Unknown backend: {name}")
        status_provider = self._status_providers.get(normalized)
        status = status_provider() if status_provider else None
        if status is not None and not status.available:
            raise BackendUnavailableError(
                status.reason or f"{normalized} backend is unavailable"
            )
        if normalized in self._instances:
            return self._instances[normalized]
        async with self._lock:
            if normalized not in self._instances:
                self._instances[normalized] = self._factories[normalized]()
        return self._instances[normalized]

    async def aclose(self) -> None:
        instances = list(self._instances.values())
        self._instances.clear()
        results = await asyncio.gather(
            *(instance.aclose() for instance in instances),
            return_exceptions=True,
        )
        # Cleanup is best-effort during application shutdown.
        _ = results
