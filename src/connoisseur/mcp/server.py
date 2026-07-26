"""FastMCP server exposing the shared Connoisseur retrieval layer."""

from __future__ import annotations

import json
from collections.abc import Callable
from threading import Lock
from typing import Any

from connoisseur.core.config import Settings
from connoisseur.core.repository import DataRepository
from connoisseur.core.retrieval import RetrievalService

try:
    from fastmcp import FastMCP
except ImportError:  # Core and unit tests remain usable without server extras.
    FastMCP = None  # type: ignore[assignment,misc]


def create_service(settings: Settings | None = None) -> RetrievalService:
    resolved = settings or Settings.from_env()
    return RetrievalService(DataRepository(resolved), settings=resolved)


class ConnoisseurTools:
    """Dependency-injectable MCP tool implementations."""

    def __init__(self, service: RetrievalService) -> None:
        self.service = service

    def search_restaurants(
        self,
        query: str,
        limit: int = 5,
        cuisine: str | None = None,
        neighborhood: str | None = None,
        min_rating: float | None = None,
        price_range: str | None = None,
    ) -> dict[str, Any]:
        """Semantically search restaurants with optional structured filters."""

        return self.service.search_restaurants(
            query,
            limit=limit,
            cuisine=cuisine,
            neighborhood=neighborhood,
            min_rating=min_rating,
            price_range=price_range,
        )

    def get_restaurant_details(
        self, restaurant_id: str = "", name: str = ""
    ) -> dict[str, Any]:
        """Get one restaurant by a search-result ID or unambiguous name."""

        return self.service.restaurant_details(
            restaurant_id=restaurant_id, name=name
        )

    def search_recipes(
        self,
        query: str,
        limit: int = 5,
        cuisine: str | None = None,
        max_total_minutes: int | None = None,
        ingredients: list[str] | None = None,
    ) -> dict[str, Any]:
        """Semantically search recipes, ingredients, and image descriptions."""

        return self.service.search_recipes(
            query,
            limit=limit,
            cuisine=cuisine,
            max_total_minutes=max_total_minutes,
            ingredients=ingredients,
        )

    def get_restaurant_reviews(
        self, restaurant_id: str = "", restaurant_name: str = ""
    ) -> dict[str, Any]:
        """Retrieve all available reviews for a restaurant."""

        return self.service.restaurant_reviews(
            restaurant_id=restaurant_id,
            restaurant_name=restaurant_name,
        )

    def health_check(self) -> dict[str, Any]:
        """Report corpus availability and semantic fallback state."""

        return self.service.health()

    def corpus_stats(self) -> dict[str, Any]:
        """Return corpus counts and available cuisine/neighborhood facets."""

        return self.service.corpus_stats()


class _LazyTools:
    def __init__(self, provider: Callable[[], RetrievalService]) -> None:
        self._provider = provider

    def current(self) -> ConnoisseurTools:
        return ConnoisseurTools(self._provider())


def _lazy_service_provider(
    service: RetrievalService | None, settings: Settings | None
) -> Callable[[], RetrievalService]:
    instance = service
    lock = Lock()

    def provide() -> RetrievalService:
        nonlocal instance
        if instance is None:
            with lock:
                if instance is None:
                    instance = create_service(settings)
        return instance

    return provide


def create_mcp(
    service: RetrievalService | None = None,
    *,
    settings: Settings | None = None,
) -> Any:
    """Build a FastMCP server without loading data until its first tool call."""

    if FastMCP is None:
        raise RuntimeError(
            "FastMCP is not installed. Install the server dependencies first."
        )
    server = FastMCP("Connoisseur Retrieval Server")
    tools = _LazyTools(_lazy_service_provider(service, settings))

    @server.tool
    def search_restaurants(
        query: str,
        limit: int = 5,
        cuisine: str | None = None,
        neighborhood: str | None = None,
        min_rating: float | None = None,
        price_range: str | None = None,
    ) -> dict[str, Any]:
        """Find restaurants by meaning, cuisine, location, rating, price, or vibe."""

        return tools.current().search_restaurants(
            query,
            limit,
            cuisine,
            neighborhood,
            min_rating,
            price_range,
        )

    @server.tool
    def get_restaurant_details(
        restaurant_id: str = "", name: str = ""
    ) -> dict[str, Any]:
        """Fetch structured restaurant details by stable ID or name."""

        return tools.current().get_restaurant_details(restaurant_id, name)

    @server.tool
    def search_recipes(
        query: str,
        limit: int = 5,
        cuisine: str | None = None,
        max_total_minutes: int | None = None,
        ingredients: list[str] | None = None,
    ) -> dict[str, Any]:
        """Find recipes by meaning, cuisine, duration, and required ingredients."""

        return tools.current().search_recipes(
            query, limit, cuisine, max_total_minutes, ingredients
        )

    @server.tool
    def get_restaurant_reviews(
        restaurant_id: str = "", restaurant_name: str = ""
    ) -> dict[str, Any]:
        """Retrieve reviews by restaurant stable ID or name."""

        return tools.current().get_restaurant_reviews(
            restaurant_id, restaurant_name
        )

    @server.tool
    def health_check() -> dict[str, Any]:
        """Check data readiness and the active retrieval backend."""

        return tools.current().health_check()

    @server.tool
    def corpus_stats() -> dict[str, Any]:
        """Inspect corpus sizes and filter facets."""

        return tools.current().corpus_stats()

    @server.resource("connoisseur://health")
    def health_resource() -> str:
        """Machine-readable snapshot of server and corpus health."""

        return json.dumps(tools.current().health_check(), sort_keys=True)

    @server.custom_route("/healthz", methods=["GET"])
    async def http_health_check(request: Any) -> Any:
        """HTTP readiness probe that initializes and checks the data service."""

        del request
        from starlette.responses import JSONResponse

        try:
            payload = tools.current().health_check()
        except Exception as exc:
            return JSONResponse(
                {
                    "status": "unhealthy",
                    "error": type(exc).__name__,
                },
                status_code=503,
            )
        # Lexical fallback is an intentional usable state, so "degraded"
        # remains ready and returns HTTP 200.
        return JSONResponse(payload, status_code=200)

    return server


# FastMCP CLI and ASGI tooling discover this conventional module-level name.
mcp = create_mcp() if FastMCP is not None else None


def main() -> None:
    if FastMCP is None:
        raise RuntimeError(
            "FastMCP is not installed. Install the server dependencies first."
        )
    settings = Settings.from_env()
    server = mcp or create_mcp(settings=settings)
    if settings.mcp_transport == "stdio":
        server.run(transport="stdio")
        return
    server.run(
        transport=settings.mcp_transport,
        host=settings.mcp_host,
        port=settings.mcp_port,
        path=settings.mcp_path,
    )


if __name__ == "__main__":
    main()
