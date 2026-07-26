from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from connoisseur.core.adapters import (
    DataValidationError,
    load_recipes,
    load_restaurants,
    map_recipe_images,
)


class AdapterTests(unittest.TestCase):
    def test_recipe_images_are_mapped_by_numeric_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            for name in ("recipe10.png", "recipe2.png", "recipe1.png"):
                (image_dir / name).touch()

            mapping = map_recipe_images(image_dir)

            self.assertEqual(mapping[1].name, "recipe1.png")
            self.assertEqual(mapping[2].name, "recipe2.png")
            self.assertEqual(mapping[10].name, "recipe10.png")

    def test_duplicate_numeric_image_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            (image_dir / "recipe2.png").touch()
            (image_dir / "recipe02.jpg").touch()

            with self.assertRaisesRegex(DataValidationError, "Duplicate recipe"):
                map_recipe_images(image_dir)

    def test_recipe_record_uses_matching_image_not_iteration_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            (image_dir / "recipe10.png").touch()
            (image_dir / "recipe2.png").touch()
            recipes_path = root / "recipes.json"
            recipes_path.write_text(
                json.dumps(
                    [
                        {
                            "id": 2,
                            "name": "Two",
                            "cuisine": "Test",
                            "servings": 1,
                            "prep_time": "1 min",
                            "cook_time": "1 min",
                            "total_time": "2 mins",
                            "ingredients": ["two"],
                            "directions": ["cook"],
                            "image_description": "two",
                        },
                        {
                            "id": 10,
                            "name": "Ten",
                            "cuisine": "Test",
                            "servings": 1,
                            "prep_time": "1 min",
                            "cook_time": "1 min",
                            "total_time": "2 mins",
                            "ingredients": ["ten"],
                            "directions": ["cook"],
                            "image_description": "ten",
                        },
                    ]
                ),
                encoding="utf-8",
            )

            recipes = load_recipes(recipes_path, image_dir)

            self.assertEqual(recipes[0].image_filename, "recipe2.png")
            self.assertEqual(recipes[1].image_filename, "recipe10.png")

    def test_restaurant_ids_are_stable_across_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "restaurants.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "name": "Café Example",
                            "neighborhood": "Downtown",
                            "cuisine": "Californian",
                            "type": "bistro",
                            "rating": 4.5,
                            "price_range": "$$$",
                            "signature_dish": "tomato tart",
                            "vibes": ["bright"],
                            "description": "A sunny room.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            first = load_restaurants(path)
            second = load_restaurants(path)

            self.assertEqual(first[0].id, second[0].id)
            self.assertTrue(first[0].id.startswith("restaurant:cafe-example:"))


if __name__ == "__main__":
    unittest.main()
