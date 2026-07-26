"""Environment-driven configuration for the data and MCP layers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _first_value(
    environ: Mapping[str, str], *names: str, default: str | None = None
) -> str | None:
    for name in names:
        value = environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _as_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _default_project_root() -> Path:
    # src/connoisseur/core/config.py -> repository root
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class Settings:
    """Resolved application settings.

    Relative data paths are resolved against ``data_root``. The short
    deployment environment names are preferred, while ``CONNOISSEUR_*``
    aliases are accepted for local development.
    """

    project_root: Path
    data_root: Path
    restaurant_data_path: Path
    review_data_path: Path
    recipe_data_path: Path
    recipe_image_dir: Path
    chroma_path: Path
    retrieval_mode: str = "semantic"
    chroma_collection_prefix: str = "connoisseur"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8001
    mcp_path: str = "/mcp"
    mcp_transport: str = "http"
    mcp_server_url: str = "http://localhost:8001/mcp"

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        project_root: str | Path | None = None,
    ) -> Settings:
        if environ is None:
            from dotenv import load_dotenv

            load_dotenv()
        env = os.environ if environ is None else environ
        root_value = project_root or _first_value(
            env,
            "PROJECT_ROOT",
            "CONNOISSEUR_PROJECT_ROOT",
            default=str(_default_project_root()),
        )
        if root_value is None:  # The default above makes this defensive only.
            raise ValueError("PROJECT_ROOT could not be resolved")
        root = Path(root_value).expanduser().resolve()

        data_root_value = _first_value(
            env, "DATA_ROOT", "CONNOISSEUR_DATA_ROOT", default=str(root)
        )
        data_root = _as_path(data_root_value or root, root).resolve()

        restaurant_path = _as_path(
            _first_value(
                env,
                "RESTAURANT_DATA_PATH",
                "CONNOISSEUR_RESTAURANT_DATA",
                default="data/restaurants.json",
            )
            or "",
            data_root,
        ).resolve()
        review_path = _as_path(
            _first_value(
                env,
                "REVIEW_DATA_PATH",
                "CONNOISSEUR_REVIEW_DATA",
                default="data/reviews.json",
            )
            or "",
            data_root,
        ).resolve()
        recipe_path = _as_path(
            _first_value(
                env,
                "RECIPE_DATA_PATH",
                "CONNOISSEUR_RECIPE_DATA",
                default="data/recipes.json",
            )
            or "",
            data_root,
        ).resolve()
        image_dir = _as_path(
            _first_value(
                env,
                "RECIPE_IMAGE_DIR",
                "CONNOISSEUR_RECIPE_IMAGES",
                default="data/recipe_images",
            )
            or "",
            data_root,
        ).resolve()
        chroma_path = _as_path(
            _first_value(
                env,
                "CHROMA_PATH",
                "CONNOISSEUR_CHROMA_DIR",
                default=".data/chroma",
            )
            or "",
            data_root,
        ).resolve()

        retrieval_mode = (
            _first_value(
                env,
                "RETRIEVAL_MODE",
                "CONNOISSEUR_RETRIEVAL_BACKEND",
                default="semantic",
            )
            or "semantic"
        ).lower()
        if retrieval_mode == "chroma":
            retrieval_mode = "semantic"
        if retrieval_mode not in {"semantic", "lexical", "auto"}:
            raise ValueError(
                "RETRIEVAL_MODE must be one of: semantic, lexical, auto"
            )

        transport = (
            _first_value(
                env,
                "MCP_TRANSPORT",
                "CONNOISSEUR_MCP_TRANSPORT",
                default="http",
            )
            or "http"
        ).lower()
        if transport not in {"http", "stdio", "sse"}:
            raise ValueError("MCP_TRANSPORT must be one of: http, stdio, sse")

        port_text = _first_value(
            env, "MCP_PORT", "CONNOISSEUR_MCP_PORT", default="8001"
        )
        try:
            port = int(port_text or "8001")
        except ValueError as exc:
            raise ValueError("MCP_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("MCP_PORT must be between 1 and 65535")

        mcp_path = _first_value(
            env, "MCP_PATH", "CONNOISSEUR_MCP_PATH", default="/mcp"
        ) or "/mcp"
        if not mcp_path.startswith("/"):
            mcp_path = f"/{mcp_path}"

        return cls(
            project_root=root,
            data_root=data_root,
            restaurant_data_path=restaurant_path,
            review_data_path=review_path,
            recipe_data_path=recipe_path,
            recipe_image_dir=image_dir,
            chroma_path=chroma_path,
            retrieval_mode=retrieval_mode,
            chroma_collection_prefix=_first_value(
                env,
                "CHROMA_COLLECTION_PREFIX",
                "CONNOISSEUR_CHROMA_COLLECTION_PREFIX",
                default="connoisseur",
            )
            or "connoisseur",
            mcp_host=_first_value(
                env, "MCP_HOST", "CONNOISSEUR_MCP_HOST", default="0.0.0.0"
            )
            or "0.0.0.0",
            mcp_port=port,
            mcp_path=mcp_path,
            mcp_transport=transport,
            mcp_server_url=_first_value(
                env,
                "MCP_SERVER_URL",
                "CONNOISSEUR_MCP_SERVER_URL",
                default=f"http://localhost:{port}{mcp_path}",
            )
            or f"http://localhost:{port}{mcp_path}",
        )
