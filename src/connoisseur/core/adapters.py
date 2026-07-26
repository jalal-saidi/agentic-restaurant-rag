"""Adapters from JSON corpus records to canonical domain models."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import Recipe, Restaurant, Review

_IMAGE_PATTERN = re.compile(
    r"^recipe(?P<id>\d+)\.(?P<extension>png|jpe?g|webp)$", re.IGNORECASE
)
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class DataValidationError(ValueError):
    """Raised when a corpus file does not match its expected schema."""


def normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.casefold().split())


def stable_slug(value: str) -> str:
    slug = _SLUG_PATTERN.sub("-", normalized_text(value)).strip("-")
    return slug or "unknown"


def deterministic_id(prefix: str, *parts: object, readable: str = "") -> str:
    canonical = "\x1f".join(normalized_text(str(part)) for part in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    middle = f":{stable_slug(readable)}" if readable else ""
    return f"{prefix}{middle}:{digest}"


def _load_json_array(path: Path) -> list[Mapping[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise DataValidationError(f"Data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DataValidationError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, list):
        raise DataValidationError(f"Expected a JSON array in {path}")
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise DataValidationError(
                f"Expected an object at index {index} in {path}"
            )
    return value


def _required(item: Mapping[str, Any], key: str, source: str) -> Any:
    if key not in item or item[key] is None:
        raise DataValidationError(f"Missing required field '{key}' in {source}")
    return item[key]


def _as_strings(value: Any, key: str, source: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise DataValidationError(f"Field '{key}' must be a list in {source}")
    return tuple(str(item).strip() for item in value if str(item).strip())


def map_recipe_images(image_dir: Path) -> dict[int, Path]:
    """Map recipe IDs to images by parsing the filename's numeric component.

    Iteration order is intentionally irrelevant. Duplicate numeric IDs are
    rejected instead of silently pairing a recipe with an arbitrary image.
    """

    if not image_dir.exists():
        return {}
    if not image_dir.is_dir():
        raise DataValidationError(f"Recipe image path is not a directory: {image_dir}")

    result: dict[int, Path] = {}
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file():
            continue
        match = _IMAGE_PATTERN.fullmatch(path.name)
        if not match:
            continue
        source_id = int(match.group("id"))
        if source_id in result:
            raise DataValidationError(
                f"Duplicate recipe image ID {source_id}: "
                f"{result[source_id].name}, {path.name}"
            )
        result[source_id] = path.resolve()
    return result


def load_restaurants(path: Path) -> tuple[Restaurant, ...]:
    records: list[Restaurant] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(_load_json_array(path)):
        source = f"{path} record {index}"
        name = str(_required(item, "name", source)).strip()
        record_id = deterministic_id("restaurant", name, readable=name)
        if record_id in seen_ids:
            raise DataValidationError(f"Duplicate restaurant name in {source}: {name}")
        seen_ids.add(record_id)
        records.append(
            Restaurant(
                id=record_id,
                name=name,
                neighborhood=str(
                    _required(item, "neighborhood", source)
                ).strip(),
                cuisine=str(_required(item, "cuisine", source)).strip(),
                restaurant_type=str(_required(item, "type", source)).strip(),
                rating=float(_required(item, "rating", source)),
                price_range=str(_required(item, "price_range", source)).strip(),
                signature_dish=str(
                    _required(item, "signature_dish", source)
                ).strip(),
                vibes=_as_strings(
                    _required(item, "vibes", source), "vibes", source
                ),
                description=str(_required(item, "description", source)).strip(),
            )
        )
    return tuple(records)


def load_reviews(
    path: Path, restaurants: Iterable[Restaurant] = ()
) -> tuple[Review, ...]:
    restaurant_ids = {
        normalized_text(restaurant.name): restaurant.id for restaurant in restaurants
    }
    records: list[Review] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(_load_json_array(path)):
        source = f"{path} record {index}"
        restaurant_name = str(
            _required(item, "restaurant_name", source)
        ).strip()
        reviewer = str(_required(item, "reviewer", source)).strip()
        visit_date = str(item.get("visit_date", "")).strip()
        record_id = deterministic_id(
            "review", restaurant_name, reviewer, visit_date
        )
        if record_id in seen_ids:
            raise DataValidationError(f"Duplicate review identity in {source}")
        seen_ids.add(record_id)
        records.append(
            Review(
                id=record_id,
                restaurant_id=restaurant_ids.get(normalized_text(restaurant_name)),
                restaurant_name=restaurant_name,
                reviewer=reviewer,
                rating=float(_required(item, "rating", source)),
                review_text=str(_required(item, "review_text", source)).strip(),
                image_description=str(item.get("image_description", "")).strip(),
                visit_date=visit_date,
            )
        )
    return tuple(records)


def load_recipes(path: Path, image_dir: Path) -> tuple[Recipe, ...]:
    image_by_id = map_recipe_images(image_dir)
    records: list[Recipe] = []
    seen_source_ids: set[int] = set()
    for index, item in enumerate(_load_json_array(path)):
        source = f"{path} record {index}"
        try:
            source_id = int(_required(item, "id", source))
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"Recipe ID must be an integer in {source}") from exc
        if source_id in seen_source_ids:
            raise DataValidationError(f"Duplicate recipe ID {source_id} in {source}")
        seen_source_ids.add(source_id)
        records.append(
            Recipe(
                id=f"recipe:{source_id:04d}",
                source_id=source_id,
                name=str(_required(item, "name", source)).strip(),
                cuisine=str(_required(item, "cuisine", source)).strip(),
                servings=int(_required(item, "servings", source)),
                prep_time=str(_required(item, "prep_time", source)).strip(),
                cook_time=str(_required(item, "cook_time", source)).strip(),
                total_time=str(_required(item, "total_time", source)).strip(),
                ingredients=_as_strings(
                    _required(item, "ingredients", source),
                    "ingredients",
                    source,
                ),
                directions=_as_strings(
                    _required(item, "directions", source), "directions", source
                ),
                image_description=str(
                    item.get("image_description", "")
                ).strip().strip('"'),
                image_path=image_by_id.get(source_id),
            )
        )
    return tuple(records)
