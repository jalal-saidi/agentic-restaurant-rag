"""Small normalization helpers that do not depend on either framework."""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from .contracts import ChatMessage


def module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def to_jsonable(value: Any) -> Any:
    """Convert common SDK result objects to JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value):
        # ``is_dataclass`` also accepts dataclass classes; runtime SDK values
        # reaching this serializer are instances.
        return to_jsonable(asdict(value))  # type: ignore[arg-type]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_jsonable(model_dump(mode="json"))
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return to_jsonable(dict_method())
    return str(value)


def json_for_prompt(value: Any, *, max_chars: int) -> str:
    serialized = json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    if len(serialized) <= max_chars:
        return serialized
    return serialized[: max_chars - 80] + "\n... [evidence truncated]"


def parse_json_object(value: str) -> dict[str, Any]:
    """Parse a JSON object even when a model wraps it in a code fence."""

    candidate = value.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model response JSON was not an object")
    return parsed


def content_to_text(value: Any) -> str:
    """Extract text from OpenAI/Agno response content variants."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "content", "answer", "response"):
            if key in value:
                text = content_to_text(value[key])
                if text:
                    return text
        return json.dumps(to_jsonable(value), ensure_ascii=False)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return "\n".join(filter(None, (content_to_text(item) for item in value))).strip()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return content_to_text(model_dump(mode="json"))
    return str(value).strip()


def format_history(
    history: Sequence[ChatMessage],
    *,
    max_messages: int,
) -> str:
    if max_messages == 0 or not history:
        return "(no prior conversation)"
    selected = history[-max_messages:]
    return "\n".join(f"{message.role}: {message.content}" for message in selected)
