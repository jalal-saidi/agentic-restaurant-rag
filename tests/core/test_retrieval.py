from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from connoisseur.core.config import Settings
from connoisseur.core.repository import DataRepository
from connoisseur.core.retrieval import RetrievalService, duration_minutes


def _write_fixture_data(root: Path) -> Settings:
    data_dir = root / "data"
    images = data_dir / "recipe_images"
    images.mkdir(parents=True)

    (data_dir / "restaurants.json").write_text(
        json.dumps(
            [
                {
                    "name": "Garden Table",
                    "neighborhood": "Silver Lake",
                    "cuisine": "Californian",
                    "type": "bistro",
                    "rating": 4.8,
                    "price_range": "$$$",
                    "signature_dish": "heirloom tomato tart",
                    "vibes": ["romantic", "greenhouse"],
                    "description": "A candlelit garden dining room.",
                },
                {
                    "name": "Noodle Light",
                    "neighborhood": "Downtown",
                    "cuisine": "Japanese",
                    "type": "noodle bar",
                    "rating": 4.1,
                    "price_range": "$$",
                    "signature_dish": "miso ramen",
                    "vibes": ["energetic"],
                    "description": "A bright counter-service room.",
                },
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "reviews.json").write_text(
        json.dumps(
            [
                {
                    "restaurant_name": "Garden Table",
                    "reviewer": "A Reviewer",
                    "rating": 5,
                    "review_text": "Perfect date-night atmosphere.",
                    "image_description": "A tomato tart.",
                    "visit_date": "2025-01-02",
                }
            ]
        ),
        encoding="utf-8",
    )
    (data_dir / "recipes.json").write_text(
        json.dumps(
            [
                {
                    "id": 2,
                    "name": "Fast Tomato Pasta",
                    "cuisine": "Italian",
                    "servings": 2,
                    "prep_time": "10 mins",
                    "cook_time": "15 mins",
                    "total_time": "25 mins",
                    "ingredients": ["tomato", "basil", "pasta"],
                    "directions": ["Boil pasta.", "Add sauce."],
                    "image_description": "Red pasta with fresh basil.",
                },
                {
                    "id": 10,
                    "name": "Slow Bean Stew",
                    "cuisine": "American",
                    "servings": 4,
                    "prep_time": "15 mins",
                    "cook_time": "1 hour 30 mins",
                    "total_time": "1 hour 45 mins",
                    "ingredients": ["beans", "onion"],
                    "directions": ["Simmer."],
                    "image_description": "A bowl of bean stew.",
                },
            ]
        ),
        encoding="utf-8",
    )
    (images / "recipe10.png").touch()
    (images / "recipe2.png").touch()
    return Settings.from_env(
        {
            "DATA_ROOT": str(root),
            "RETRIEVAL_MODE": "lexical",
            "MCP_PORT": "8001",
        },
        project_root=root,
    )


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.settings = _write_fixture_data(self.root)
        self.repository = DataRepository(self.settings)
        self.service = RetrievalService(
            self.repository, settings=self.settings
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_restaurant_search_ranks_and_filters(self) -> None:
        response = self.service.search_restaurants(
            "romantic garden", min_rating=4.5, price_range="$$$"
        )

        self.assertEqual(response["count"], 1)
        self.assertEqual(response["results"][0]["name"], "Garden Table")
        self.assertEqual(response["backend"], "lexical")

    def test_details_and_review_lookup_share_stable_restaurant_id(self) -> None:
        result = self.service.search_restaurants("tomato tart")["results"][0]
        details = self.service.restaurant_details(restaurant_id=result["id"])
        reviews = self.service.restaurant_reviews(
            restaurant_id=result["id"]
        )

        self.assertEqual(details["status"], "found")
        self.assertEqual(details["review_count"], 1)
        self.assertEqual(reviews["reviews"][0]["restaurant_id"], result["id"])

    def test_recipe_search_filters_duration_and_maps_correct_image(self) -> None:
        response = self.service.search_recipes(
            "tomato basil",
            cuisine="Italian",
            max_total_minutes=30,
            ingredients=["basil"],
        )

        self.assertEqual(response["count"], 1)
        self.assertEqual(response["results"][0]["source_id"], 2)
        self.assertEqual(
            response["results"][0]["image_filename"], "recipe2.png"
        )

    def test_health_and_stats_are_offline(self) -> None:
        health = self.service.health()
        stats = self.service.corpus_stats()

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["active_backend"], "lexical")
        self.assertEqual(stats["restaurants"], 2)
        self.assertEqual(stats["recipes"], 2)
        self.assertEqual(stats["recipe_images"], 2)

    def test_duration_parser_supports_hours_and_minutes(self) -> None:
        self.assertEqual(duration_minutes("25 mins"), 25)
        self.assertEqual(duration_minutes("1 hour 45 mins"), 105)
        self.assertIsNone(duration_minutes("varies"))


if __name__ == "__main__":
    unittest.main()
