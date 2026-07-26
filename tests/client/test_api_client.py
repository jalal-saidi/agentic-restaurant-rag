from __future__ import annotations

import unittest
from typing import Any

from connoisseur.client.api_client import (
    ApiClientError,
    ConnoisseurApiClient,
    normalize_ui_history,
)


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self.body = body

    def json(self) -> Any:
        return self.body


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.post_path: str | None = None
        self.post_payload: dict[str, Any] | None = None

    async def __aenter__(self) -> FakeHttpClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, path: str, *, json: dict[str, Any]) -> FakeResponse:
        self.post_path = path
        self.post_payload = json
        return self.response


class ApiClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_posts_normalized_history_and_parses_response(self) -> None:
        fake = FakeHttpClient(
            FakeResponse(
                200,
                {
                    "answer": "Try Sakura Garden.",
                    "backend": "agno",
                    "session_id": "session-1",
                    "metadata": {"team_mode": "coordinate"},
                },
            )
        )
        client = ConnoisseurApiClient(
            "http://api:8000/",
            client_factory=lambda **_: fake,
        )

        result = await client.chat(
            "Find dinner",
            backend="agno",
            history=[["Earlier question", "Earlier answer"]],
        )

        self.assertEqual(fake.post_path, "/v1/chat")
        self.assertEqual(
            fake.post_payload["history"],  # type: ignore[index]
            [
                {"role": "user", "content": "Earlier question"},
                {"role": "assistant", "content": "Earlier answer"},
            ],
        )
        self.assertEqual(result.answer, "Try Sakura Garden.")
        self.assertEqual(result.backend, "agno")

    async def test_surfaces_api_error_detail(self) -> None:
        fake = FakeHttpClient(
            FakeResponse(503, {"detail": "Missing required LLM_MODEL"})
        )
        client = ConnoisseurApiClient(
            "http://api:8000",
            client_factory=lambda **_: fake,
        )

        with self.assertRaisesRegex(ApiClientError, "LLM_MODEL"):
            await client.chat("Find dinner", backend="langgraph")

    async def test_caps_history_to_the_configured_recent_messages(self) -> None:
        fake = FakeHttpClient(
            FakeResponse(
                200,
                {
                    "answer": "Done.",
                    "backend": "langgraph",
                    "session_id": "session-1",
                },
            )
        )
        client = ConnoisseurApiClient(
            "http://api:8000",
            max_history_messages=3,
            client_factory=lambda **_: fake,
        )

        await client.chat(
            "Current",
            backend="langgraph",
            history=[
                {"role": "user", "content": f"message-{index}"}
                for index in range(6)
            ],
        )

        self.assertEqual(
            fake.post_payload["history"],  # type: ignore[index]
            [
                {"role": "user", "content": "message-3"},
                {"role": "user", "content": "message-4"},
                {"role": "user", "content": "message-5"},
            ],
        )

    def test_normalizes_message_dictionary_history(self) -> None:
        self.assertEqual(
            normalize_ui_history(
                [
                    {"role": "user", "content": " Hello "},
                    {"role": "tool", "content": "ignored"},
                ]
            ),
            [{"role": "user", "content": "Hello"}],
        )

    def test_rejects_history_limit_above_api_contract(self) -> None:
        with self.assertRaises(ValueError):
            ConnoisseurApiClient(
                "http://api:8000",
                max_history_messages=101,
            )
