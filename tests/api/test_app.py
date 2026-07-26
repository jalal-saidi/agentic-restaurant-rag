from __future__ import annotations

import importlib.util
import unittest
from typing import Any

from connoisseur.orchestrators.contracts import (
    BackendStatus,
    ChatInput,
    OrchestrationResult,
)

HAS_API_DEPS = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


class FakeOrchestrator:
    name = "langgraph"

    def __init__(self) -> None:
        self.requests: list[ChatInput] = []

    async def run(self, request: ChatInput) -> OrchestrationResult:
        self.requests.append(request)
        return OrchestrationResult(
            answer="A retrieved recommendation",
            backend="langgraph",
            session_id=request.session_id,
            metadata={"tested": True},
        )

    async def aclose(self) -> None:
        return None


class FakeRegistry:
    def __init__(self) -> None:
        self.backend = FakeOrchestrator()
        self.closed = False

    def statuses(self) -> list[BackendStatus]:
        return [
            BackendStatus("langgraph", True),
            BackendStatus("agno", False, "not configured"),
        ]

    async def readiness(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "backends": [
                {"name": "langgraph", "available": True, "reason": None}
            ],
            "retrieval": {"status": "ok", "active_backend": "lexical"},
        }

    async def get(self, name: str) -> FakeOrchestrator:
        if name != "langgraph":
            raise AssertionError("unexpected backend")
        return self.backend

    async def aclose(self) -> None:
        self.closed = True


@unittest.skipUnless(HAS_API_DEPS, "FastAPI test dependencies are not installed")
class ApiTests(unittest.TestCase):
    def test_health_backends_and_chat_contract(self) -> None:
        from fastapi.testclient import TestClient

        from connoisseur.api.app import create_app

        registry = FakeRegistry()
        with TestClient(create_app(registry=registry)) as client:  # type: ignore[arg-type]
            health = client.get("/healthz")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["status"], "ok")

            readiness = client.get("/readyz")
            self.assertEqual(readiness.status_code, 200)
            self.assertEqual(readiness.json()["status"], "ready")

            backends = client.get("/v1/backends")
            self.assertEqual(backends.status_code, 200)
            self.assertEqual(len(backends.json()), 2)

            response = client.post(
                "/v1/chat",
                json={
                    "message": "Find dinner",
                    "backend": "langgraph",
                    "history": [
                        {"role": "user", "content": "I like Japanese food"}
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            body: dict[str, Any] = response.json()
            self.assertEqual(body["answer"], "A retrieved recommendation")
            self.assertEqual(body["backend"], "langgraph")
            self.assertTrue(body["session_id"])
            self.assertEqual(body["metadata"], {"tested": True})
        self.assertTrue(registry.closed)
