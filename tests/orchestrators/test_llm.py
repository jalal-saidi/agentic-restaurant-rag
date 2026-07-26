from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from connoisseur.orchestrators.llm import OpenAICompatibleLLM
from connoisseur.orchestrators.settings import AppSettings


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
        )


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


class OpenAICompatibleLLMTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_configured_token_field_and_omits_blank_temperature(
        self,
    ) -> None:
        client = FakeClient()
        llm = OpenAICompatibleLLM(
            AppSettings(
                llm_model="local-model",
                openai_base_url="http://model.test/v1",
                llm_max_tokens=321,
                llm_max_tokens_parameter="max_tokens",
                llm_temperature=None,
            ),
            client=client,
        )

        answer = await llm.complete(system="system", prompt="prompt")

        self.assertEqual(answer, "answer")
        self.assertEqual(client.chat.completions.kwargs["max_tokens"], 321)
        self.assertNotIn(
            "max_completion_tokens",
            client.chat.completions.kwargs,
        )
        self.assertNotIn("temperature", client.chat.completions.kwargs)

    async def test_includes_explicit_temperature(self) -> None:
        client = FakeClient()
        llm = OpenAICompatibleLLM(
            AppSettings(
                llm_model="model",
                openai_api_key="test-only",
                llm_temperature=0.1,
            ),
            client=client,
        )

        await llm.complete(system="system", prompt="prompt")

        self.assertEqual(client.chat.completions.kwargs["temperature"], 0.1)
        self.assertIn(
            "max_completion_tokens",
            client.chat.completions.kwargs,
        )
