"""Environment-backed configuration for API and orchestration processes."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field


def _integer(
    values: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _integer_alias(
    values: Mapping[str, str],
    names: tuple[str, ...],
    default: int,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Read the first populated alias, preserving a useful validation error."""

    for name in names:
        if values.get(name, "").strip():
            return _integer(
                values,
                name,
                default,
                minimum=minimum,
                maximum=maximum,
            )
    return default


def _floating(
    values: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float,
) -> float:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _optional_floating(
    values: Mapping[str, str],
    name: str,
    *,
    minimum: float,
) -> float | None:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Runtime settings shared by the API, orchestrators, and web client."""

    llm_model: str = ""
    openai_api_key: str = field(default="", repr=False)
    openai_base_url: str | None = None
    mcp_server_url: str = "http://localhost:8001/mcp"
    api_base_url: str = "http://localhost:8000"
    retrieval_limit: int = 5
    langgraph_max_tool_rounds: int = 3
    max_history_messages: int = 12
    max_context_chars: int = 24_000
    llm_max_tokens: int = 1_200
    llm_max_tokens_parameter: str = "max_completion_tokens"
    llm_temperature: float | None = None
    llm_reasoning_effort: str | None = None
    request_timeout_seconds: float = 90.0
    client_timeout_seconds: float = 300.0

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> AppSettings:
        if environ is None:
            from dotenv import load_dotenv

            load_dotenv()
        values = os.environ if environ is None else environ
        base_url = values.get("OPENAI_BASE_URL", "").strip() or None
        max_tokens_parameter = (
            values.get("LLM_MAX_TOKENS_PARAMETER", "").strip()
            or "max_completion_tokens"
        )
        if max_tokens_parameter not in {"max_completion_tokens", "max_tokens"}:
            raise ValueError(
                "LLM_MAX_TOKENS_PARAMETER must be max_completion_tokens "
                "or max_tokens"
            )
        reasoning_effort = (
            values.get("LLM_REASONING_EFFORT", "").strip().lower() or None
        )
        if reasoning_effort not in {
            None,
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError(
                "LLM_REASONING_EFFORT must be none, minimal, low, medium, "
                "high, xhigh, max, or blank"
            )
        return cls(
            llm_model=values.get("LLM_MODEL", "").strip(),
            openai_api_key=values.get("OPENAI_API_KEY", "").strip(),
            openai_base_url=base_url,
            mcp_server_url=values.get(
                "MCP_SERVER_URL",
                "http://localhost:8001/mcp",
            ).strip(),
            api_base_url=values.get(
                "API_BASE_URL",
                "http://localhost:8000",
            ).strip().rstrip("/"),
            retrieval_limit=_integer_alias(
                values,
                ("RETRIEVAL_TOP_K", "RETRIEVAL_LIMIT"),
                5,
                minimum=1,
                maximum=50,
            ),
            langgraph_max_tool_rounds=_integer(
                values,
                "LANGGRAPH_MAX_TOOL_ROUNDS",
                3,
                minimum=1,
                maximum=10,
            ),
            max_history_messages=_integer(
                values,
                "MAX_HISTORY_MESSAGES",
                12,
                minimum=0,
                maximum=100,
            ),
            max_context_chars=_integer(
                values,
                "MAX_CONTEXT_CHARS",
                24_000,
                minimum=1_000,
            ),
            llm_max_tokens=_integer(
                values,
                "LLM_MAX_TOKENS",
                1_200,
                minimum=1,
            ),
            llm_max_tokens_parameter=max_tokens_parameter,
            llm_temperature=_optional_floating(
                values,
                "LLM_TEMPERATURE",
                minimum=0.0,
            ),
            llm_reasoning_effort=reasoning_effort,
            request_timeout_seconds=_floating(
                values,
                "REQUEST_TIMEOUT_SECONDS",
                90.0,
                minimum=1.0,
            ),
            client_timeout_seconds=_floating(
                values,
                "CLIENT_TIMEOUT_SECONDS",
                300.0,
                minimum=1.0,
            ),
        )

    def model_configuration_error(self) -> str | None:
        missing: list[str] = []
        if not self.llm_model:
            missing.append("LLM_MODEL")
        if not self.openai_api_key and not self.openai_base_url:
            missing.append(
                "OPENAI_API_KEY (or OPENAI_BASE_URL for a keyless local endpoint)"
            )
        if not missing:
            return None
        return f"Missing required environment variable(s): {', '.join(missing)}"

    @property
    def openai_client_api_key(self) -> str:
        """Return the SDK credential or a non-secret local-endpoint placeholder."""

        if self.openai_api_key:
            return self.openai_api_key
        if self.openai_base_url:
            # The OpenAI Python SDK requires a non-empty value even when a local
            # compatible server does not authenticate requests.
            return "not-required"
        return ""
