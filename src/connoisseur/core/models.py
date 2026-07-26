"""Canonical domain models shared by both orchestration implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _string_list(values: tuple[str, ...]) -> list[str]:
    return list(values)


@dataclass(frozen=True, slots=True)
class Restaurant:
    id: str
    name: str
    neighborhood: str
    cuisine: str
    restaurant_type: str
    rating: float
    price_range: str
    signature_dish: str
    vibes: tuple[str, ...]
    description: str

    @property
    def search_text(self) -> str:
        return " ".join(
            (
                self.name,
                self.neighborhood,
                self.cuisine,
                self.restaurant_type,
                self.signature_dish,
                *self.vibes,
                self.description,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "neighborhood": self.neighborhood,
            "cuisine": self.cuisine,
            "type": self.restaurant_type,
            "rating": self.rating,
            "price_range": self.price_range,
            "signature_dish": self.signature_dish,
            "vibes": _string_list(self.vibes),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class Review:
    id: str
    restaurant_id: str | None
    restaurant_name: str
    reviewer: str
    rating: float
    review_text: str
    image_description: str
    visit_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "restaurant_id": self.restaurant_id,
            "restaurant_name": self.restaurant_name,
            "reviewer": self.reviewer,
            "rating": self.rating,
            "review_text": self.review_text,
            "image_description": self.image_description,
            "visit_date": self.visit_date,
        }


@dataclass(frozen=True, slots=True)
class Recipe:
    id: str
    source_id: int
    name: str
    cuisine: str
    servings: int
    prep_time: str
    cook_time: str
    total_time: str
    ingredients: tuple[str, ...]
    directions: tuple[str, ...]
    image_description: str
    image_path: Path | None

    @property
    def image_filename(self) -> str | None:
        return self.image_path.name if self.image_path else None

    @property
    def search_text(self) -> str:
        return " ".join(
            (
                self.name,
                self.cuisine,
                *self.ingredients,
                self.image_description,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "name": self.name,
            "cuisine": self.cuisine,
            "servings": self.servings,
            "prep_time": self.prep_time,
            "cook_time": self.cook_time,
            "total_time": self.total_time,
            "ingredients": _string_list(self.ingredients),
            "directions": _string_list(self.directions),
            "image_description": self.image_description,
            "image_filename": self.image_filename,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    id: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "relevance_score": round(self.score, 6)}
