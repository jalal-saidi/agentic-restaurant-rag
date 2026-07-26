"""High-level retrieval API used by MCP, LangGraph, and Agno."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeVar

from .adapters import normalized_text
from .config import Settings
from .indexes import ChromaIndex, LexicalIndex, SearchDocument, SearchIndex
from .models import Recipe, Restaurant
from .repository import DataRepository


class _Identified(Protocol):
    @property
    def id(self) -> str:
        ...


_T = TypeVar("_T", bound=_Identified)
_DURATION_PATTERN = re.compile(
    r"(?:(?P<hours>\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h))?"
    r"\s*(?:(?P<minutes>\d+)\s*(?:minutes?|mins?|m))?",
    re.IGNORECASE,
)


def duration_minutes(value: str) -> int | None:
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if not match or not any(match.groupdict().values()):
        return None
    hours = float(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    return round(hours * 60 + minutes)


def _bounded_limit(limit: int) -> int:
    if not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    return limit


def _matches(value: str, expected: str | None) -> bool:
    if not expected:
        return True
    return normalized_text(expected) in normalized_text(value)


class RetrievalService:
    """Search and lookup facade with semantic-to-lexical failover."""

    def __init__(
        self,
        repository: DataRepository,
        *,
        settings: Settings | None = None,
        index_factory: Callable[
            [str, tuple[SearchDocument, ...]], SearchIndex
        ]
        | None = None,
    ) -> None:
        self.repository = repository
        self.settings = settings or repository.settings
        self._requested_backend = self.settings.retrieval_mode
        self._fallback_reason: str | None = None

        restaurant_documents = tuple(
            SearchDocument(
                id=restaurant.id,
                text=restaurant.search_text,
                metadata={
                    "kind": "restaurant",
                    "cuisine": restaurant.cuisine,
                    "neighborhood": restaurant.neighborhood,
                    "rating": restaurant.rating,
                    "price_range": restaurant.price_range,
                },
            )
            for restaurant in repository.restaurants
        )
        recipe_documents = tuple(
            SearchDocument(
                id=recipe.id,
                text=recipe.search_text,
                metadata={
                    "kind": "recipe",
                    "source_id": recipe.source_id,
                    "cuisine": recipe.cuisine,
                },
            )
            for recipe in repository.recipes
        )
        if index_factory:
            self._restaurant_index = index_factory(
                "restaurants", restaurant_documents
            )
            self._recipe_index = index_factory("recipes", recipe_documents)
        else:
            self._restaurant_index = self._build_index(
                "restaurants", restaurant_documents
            )
            self._recipe_index = self._build_index("recipes", recipe_documents)

    def _build_index(
        self, corpus: str, documents: tuple[SearchDocument, ...]
    ) -> SearchIndex:
        if self.settings.retrieval_mode == "lexical":
            return LexicalIndex(documents)
        try:
            return ChromaIndex(
                documents,
                persist_path=self.settings.chroma_path,
                collection_name=(
                    f"{self.settings.chroma_collection_prefix}_{corpus}"
                ),
            )
        except Exception as exc:
            self._fallback_reason = f"{type(exc).__name__}: {exc}"
            return LexicalIndex(documents)

    @property
    def active_backend(self) -> str:
        backends = {
            self._restaurant_index.backend_name,
            self._recipe_index.backend_name,
        }
        return next(iter(backends)) if len(backends) == 1 else "mixed"

    def _rank(
        self,
        query: str,
        records: Iterable[_T],
        index: SearchIndex,
        *,
        limit: int,
    ) -> list[tuple[_T, float]]:
        by_id = {record.id: record for record in records}
        hits = index.search(query, limit=limit, allowed_ids=set(by_id))
        return [(by_id[record_id], score) for record_id, score in hits]

    def search_restaurants(
        self,
        query: str,
        *,
        limit: int = 5,
        cuisine: str | None = None,
        neighborhood: str | None = None,
        min_rating: float | None = None,
        price_range: str | None = None,
    ) -> dict[str, Any]:
        limit = _bounded_limit(limit)
        if min_rating is not None and not 0 <= min_rating <= 5:
            raise ValueError("min_rating must be between 0 and 5")
        restaurants = tuple(
            restaurant
            for restaurant in self.repository.restaurants
            if _matches(restaurant.cuisine, cuisine)
            and _matches(restaurant.neighborhood, neighborhood)
            and (min_rating is None or restaurant.rating >= min_rating)
            and (
                not price_range
                or normalized_text(restaurant.price_range)
                == normalized_text(price_range)
            )
        )
        ranked = self._rank(
            query, restaurants, self._restaurant_index, limit=limit
        )
        results = [
            {**restaurant.to_dict(), "relevance_score": round(score, 6)}
            for restaurant, score in ranked
        ]
        return {
            "status": "ok",
            "query": query,
            "backend": self.active_backend,
            "count": len(results),
            "results": results,
        }

    def restaurant_details(
        self, *, restaurant_id: str = "", name: str = ""
    ) -> dict[str, Any]:
        if not restaurant_id and not name.strip():
            raise ValueError("restaurant_id or name is required")
        restaurant = self.repository.restaurant(
            restaurant_id=restaurant_id, name=name
        )
        if not restaurant:
            return {
                "status": "not_found",
                "restaurant_id": restaurant_id or None,
                "name": name or None,
            }
        reviews = self.repository.restaurant_reviews(
            restaurant_id=restaurant.id
        )
        return {
            "status": "found",
            "restaurant": restaurant.to_dict(),
            "review_count": len(reviews),
            "average_review_rating": (
                round(sum(review.rating for review in reviews) / len(reviews), 2)
                if reviews
                else None
            ),
        }

    def search_recipes(
        self,
        query: str,
        *,
        limit: int = 5,
        cuisine: str | None = None,
        max_total_minutes: int | None = None,
        ingredients: list[str] | None = None,
    ) -> dict[str, Any]:
        limit = _bounded_limit(limit)
        if max_total_minutes is not None and max_total_minutes <= 0:
            raise ValueError("max_total_minutes must be positive")
        ingredient_queries = [
            normalized_text(ingredient)
            for ingredient in (ingredients or [])
            if ingredient.strip()
        ]

        def recipe_matches(recipe: Recipe) -> bool:
            if not _matches(recipe.cuisine, cuisine):
                return False
            minutes = duration_minutes(recipe.total_time)
            if (
                max_total_minutes is not None
                and (minutes is None or minutes > max_total_minutes)
            ):
                return False
            searchable_ingredients = normalized_text(" ".join(recipe.ingredients))
            return all(
                ingredient in searchable_ingredients
                for ingredient in ingredient_queries
            )

        recipes = tuple(
            recipe for recipe in self.repository.recipes if recipe_matches(recipe)
        )
        effective_query = " ".join(
            part for part in (query, " ".join(ingredients or [])) if part.strip()
        )
        ranked = self._rank(
            effective_query, recipes, self._recipe_index, limit=limit
        )
        results = [
            {**recipe.to_dict(), "relevance_score": round(score, 6)}
            for recipe, score in ranked
        ]
        return {
            "status": "ok",
            "query": query,
            "backend": self.active_backend,
            "count": len(results),
            "results": results,
        }

    def restaurant_reviews(
        self, *, restaurant_id: str = "", restaurant_name: str = ""
    ) -> dict[str, Any]:
        if not restaurant_id and not restaurant_name.strip():
            raise ValueError("restaurant_id or restaurant_name is required")
        reviews = self.repository.restaurant_reviews(
            restaurant_id=restaurant_id, restaurant_name=restaurant_name
        )
        return {
            "status": "found" if reviews else "not_found",
            "count": len(reviews),
            "reviews": [review.to_dict() for review in reviews],
        }

    def corpus_stats(self) -> dict[str, Any]:
        stats = self.repository.stats()
        stats.update(
            {
                "requested_backend": self._requested_backend,
                "active_backend": self.active_backend,
            }
        )
        return stats

    def health(self) -> dict[str, Any]:
        stats = self.repository.stats()
        degraded = (
            self._requested_backend in {"semantic", "auto"}
            and self.active_backend != "semantic"
        )
        return {
            "status": "degraded" if degraded else "ok",
            "requested_backend": self._requested_backend,
            "active_backend": self.active_backend,
            "fallback_reason": self._fallback_reason,
            "corpora": {
                "restaurants": stats["restaurants"],
                "reviews": stats["reviews"],
                "recipes": stats["recipes"],
                "recipe_images": stats["recipe_images"],
            },
        }


def cuisine_counts(records: Iterable[Restaurant | Recipe]) -> dict[str, int]:
    """Public helper useful to clients building filter menus."""

    return dict(sorted(Counter(record.cuisine for record in records).items()))
