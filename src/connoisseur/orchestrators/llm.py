"""OpenAI-compatible model adapter used by LangGraph nodes."""

from __future__ import annotations

from typing import Any

from .contracts import LanguageModel
from .errors import ModelConfigurationError, ModelInvocationError
from .settings import AppSettings
from .utils import content_to_text


class OpenAICompatibleLLM(LanguageModel):
    """A small lazy adapter over the OpenAI Chat Completions protocol."""

    def __init__(
        self,
        settings: AppSettings,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        configuration_error = self.settings.model_configuration_error()
        if configuration_error:
            raise ModelConfigurationError(configuration_error)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ModelConfigurationError(
                "The 'openai' package is required for the LangGraph backend"
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": self.settings.openai_client_api_key,
            "timeout": self.settings.request_timeout_seconds,
        }
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete(self, *, system: str, prompt: str) -> str:
        client = self._get_client()
        try:
            request: dict[str, Any] = {
                "model": self.settings.llm_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                self.settings.llm_max_tokens_parameter: (
                    self.settings.llm_max_tokens
                ),
            }
            if self.settings.llm_temperature is not None:
                request["temperature"] = self.settings.llm_temperature
            response = await client.chat.completions.create(**request)
            content = response.choices[0].message.content
            text = content_to_text(content)
            if not text:
                raise ModelInvocationError("The model returned an empty response")
            return text
        except ModelInvocationError:
            raise
        except Exception as exc:
            raise ModelInvocationError(
                f"Model request failed: {type(exc).__name__}"
            ) from exc

    async def aclose(self) -> None:
        if self._client is None:
            return
        close = getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if hasattr(result, "__await__"):
            await result
