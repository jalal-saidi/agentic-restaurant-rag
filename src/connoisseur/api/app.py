"""HTTP boundary for selecting and invoking an orchestration backend."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from connoisseur.orchestrators.contracts import (
    ChatInput,
    ChatMessage,
)
from connoisseur.orchestrators.errors import (
    BackendUnavailableError,
    OrchestrationError,
)
from connoisseur.orchestrators.registry import BackendRegistry
from connoisseur.orchestrators.utils import to_jsonable

from .schemas import (
    BackendInfo,
    ChatRequest,
    ChatResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)


def create_app(registry: BackendRegistry | None = None) -> FastAPI:
    """Create an app with an injectable registry for tests and deployment."""

    selected_registry = registry or BackendRegistry()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await selected_registry.aclose()

    application = FastAPI(
        title="Connoisseur Companion API",
        version="1.0.0",
        description=(
            "A backend-selectable culinary assistant powered by MCP retrieval "
            "and either LangGraph or Agno multi-agent orchestration."
        ),
        lifespan=lifespan,
    )
    application.state.backend_registry = selected_registry

    @application.get(
        "/healthz",
        response_model=HealthResponse,
        tags=["operations"],
    )
    async def healthz(request: Request) -> HealthResponse:
        backend_registry: BackendRegistry = request.app.state.backend_registry
        backends = [
            BackendInfo(
                name=item.name,  # type: ignore[arg-type]
                available=item.available,
                reason=item.reason,
            )
            for item in backend_registry.statuses()
        ]
        return HealthResponse(
            status="ok",
            service="connoisseur-api",
            backends=backends,
        )

    @application.get("/readyz", tags=["operations"])
    async def readyz(request: Request) -> JSONResponse:
        """Return 200 only when a backend is configured and MCP is usable."""

        backend_registry: BackendRegistry = request.app.state.backend_registry
        readiness = await backend_registry.readiness()
        status_code = (
            status.HTTP_200_OK
            if readiness.get("status") == "ready"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )
        return JSONResponse(content=to_jsonable(readiness), status_code=status_code)

    @application.get(
        "/v1/backends",
        response_model=list[BackendInfo],
        tags=["chat"],
    )
    async def list_backends(request: Request) -> list[BackendInfo]:
        backend_registry: BackendRegistry = request.app.state.backend_registry
        return [
            BackendInfo(
                name=item.name,  # type: ignore[arg-type]
                available=item.available,
                reason=item.reason,
            )
            for item in backend_registry.statuses()
        ]

    @application.post(
        "/v1/chat",
        response_model=ChatResponse,
        tags=["chat"],
    )
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        session_id = payload.session_id or str(uuid4())
        backend_registry: BackendRegistry = request.app.state.backend_registry
        try:
            backend = await backend_registry.get(payload.backend)
            result = await backend.run(
                ChatInput(
                    message=payload.message,
                    session_id=session_id,
                    history=tuple(
                        ChatMessage(role=item.role, content=item.content)
                        for item in payload.history
                    ),
                )
            )
        except BackendUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except OrchestrationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.exception(
                "Unexpected orchestration error for backend %s",
                payload.backend,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="The orchestration request failed unexpectedly",
            ) from exc

        metadata = to_jsonable(result.metadata)
        return ChatResponse(
            answer=result.answer,
            backend=result.backend,  # type: ignore[arg-type]
            session_id=result.session_id,
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    return application


app = create_app()
