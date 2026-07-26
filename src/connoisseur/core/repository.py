"""Repository that loads and links the application corpus once."""

from __future__ import annotations

from collections import Counter
from threading import RLock
from typing import Any

from .adapters import load_recipes, load_restaurants, load_reviews, normalized_text
from .config import Settings
from .models import Recipe, Restaurant, Review


class DataRepository:
    """Thread-safe, cached access to canonical restaurant and recipe data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = RLock()
        self._restaurants: tuple[Restaurant, ...] | None = None
        self._reviews: tuple[Review, ...] | None = None
        self._recipes: tuple[Recipe, ...] | None = None

    def refresh(self) -> None:
        with self._lock:
            restaurants = load_restaurants(self.settings.restaurant_data_path)
            reviews = load_reviews(self.settings.review_data_path, restaurants)
            recipes = load_recipes(
                self.settings.recipe_data_path, self.settings.recipe_image_dir
            )
            self._restaurants = restaurants
            self._reviews = reviews
            self._recipes = recipes

    def _ensure_loaded(self) -> None:
        if (
            self._restaurants is not None
            and self._reviews is not None
            and self._recipes is not None
        ):
            return
        with self._lock:
            if (
                self._restaurants is None
                or self._reviews is None
                or self._recipes is None
            ):
                self.refresh()

    @property
    def restaurants(self) -> tuple[Restaurant, ...]:
        self._ensure_loaded()
        return self._restaurants or ()

    @property
    def reviews(self) -> tuple[Review, ...]:
        self._ensure_loaded()
        return self._reviews or ()

    @property
    def recipes(self) -> tuple[Recipe, ...]:
        self._ensure_loaded()
        return self._recipes or ()

    def restaurant(self, *, restaurant_id: str = "", name: str = "") -> Restaurant | None:
        if restaurant_id:
            return next(
                (
                    restaurant
                    for restaurant in self.restaurants
                    if restaurant.id == restaurant_id
                ),
                None,
            )
        query = normalized_text(name)
        if not query:
            return None
        exact = next(
            (
                restaurant
                for restaurant in self.restaurants
                if normalized_text(restaurant.name) == query
            ),
            None,
        )
        if exact:
            return exact
        matches = [
            restaurant
            for restaurant in self.restaurants
            if query in normalized_text(restaurant.name)
        ]
        return matches[0] if len(matches) == 1 else None

    def restaurant_reviews(
        self, *, restaurant_id: str = "", restaurant_name: str = ""
    ) -> tuple[Review, ...]:
        restaurant = self.restaurant(
            restaurant_id=restaurant_id, name=restaurant_name
        )
        if restaurant:
            expected_id = restaurant.id
            expected_name = normalized_text(restaurant.name)
            return tuple(
                review
                for review in self.reviews
                if review.restaurant_id == expected_id
                or normalized_text(review.restaurant_name) == expected_name
            )
        query = normalized_text(restaurant_name)
        if not query:
            return ()
        return tuple(
            review
            for review in self.reviews
            if query in normalized_text(review.restaurant_name)
        )

    def stats(self) -> dict[str, Any]:
        restaurants = self.restaurants
        recipes = self.recipes
        return {
            "restaurants": len(restaurants),
            "reviews": len(self.reviews),
            "recipes": len(recipes),
            "recipe_images": sum(
                1 for recipe in recipes if recipe.image_path is not None
            ),
            "restaurant_cuisines": dict(
                sorted(Counter(item.cuisine for item in restaurants).items())
            ),
            "recipe_cuisines": dict(
                sorted(Counter(item.cuisine for item in recipes).items())
            ),
            "neighborhoods": dict(
                sorted(Counter(item.neighborhood for item in restaurants).items())
            ),
        }
