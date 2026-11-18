import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.validate_catalog import (
    choose_image_check_indexes,
    validate_catalog_items
)


def catalog_item(item_id: str, image_path: str) -> dict:
    return {
        "id": item_id,
        "name": f"Item {item_id}",
        "category": "top",
        "style": None,
        "brand": None,
        "price": None,
        "image_path": image_path,
        "product_url": None
    }


class ValidateCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_image_sample_is_limited_and_repeatable(self):
        first = choose_image_check_indexes(100, image_check_limit=10, seed=7)
        second = choose_image_check_indexes(100, image_check_limit=10, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)

    def test_checks_all_metadata_but_only_selected_images(self):
        items = [
            catalog_item(f"item-{index}", f"missing-{index}.jpg")
            for index in range(20)
        ]

        summary, errors = validate_catalog_items(
            catalog_items=items,
            project_root=self.root,
            image_check_limit=5,
            image_check_seed=42
        )

        self.assertEqual(summary["records_checked"], 20)
        self.assertEqual(summary["image_records_selected"], 5)
        self.assertEqual(summary["image_records_checked"], 5)
        self.assertEqual(summary["missing_images"], 5)
        self.assertEqual(summary["invalid_records"], 5)
        self.assertEqual(len(errors), 5)

    def test_full_image_check_remains_available(self):
        Image.new("RGB", (4, 4), color="blue").save(self.root / "valid.jpg")
        items = [
            catalog_item("valid", "valid.jpg"),
            catalog_item("missing", "missing.jpg")
        ]

        summary, _ = validate_catalog_items(
            catalog_items=items,
            project_root=self.root,
            image_check_limit=None
        )

        self.assertEqual(summary["image_records_checked"], 2)
        self.assertEqual(summary["missing_images"], 1)
        self.assertEqual(summary["broken_images"], 0)


if __name__ == "__main__":
    unittest.main()
