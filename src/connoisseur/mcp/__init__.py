"""MCP transport for the Connoisseur retrieval service.

Exports are loaded lazily so ``python -m connoisseur.mcp.server`` does not
import the server once through this package and then execute it a second time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["ConnoisseurTools", "create_mcp", "create_service"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    server = import_module(".server", __name__)
    return getattr(server, name)
