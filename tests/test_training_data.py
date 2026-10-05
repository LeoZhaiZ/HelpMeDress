import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.training_data import FashionTripletDataset


class FashionTripletDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

        image_settings = {
            "anchor.png": ("L", 40),
            "positive.png": ("RGBA", (10, 20, 30, 255)),
            "negative.png": ("RGB", (200, 100, 50))
        }
        for filename, (mode, color) in image_settings.items():
            Image.new(mode, (8, 6), color=color).save(self.root / filename)

        self.manifest = [{
            "anchor": {
                "id": "user-a",
                "image_path": "anchor.png"
            },
            "positive": {
                "id": "shop-a",
                "image_path": "positive.png"
            },
            "negative": {
                "id": "shop-b",
                "image_path": "negative.png"
            },
            "source_category_id": 1,
            "source_category": "short sleeve top"
        }]
        self.manifest_path = self.root / "triplets.json"
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(self.manifest, file)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_all_three_images_as_rgb(self):
        dataset = FashionTripletDataset(
            triplets_path=self.manifest_path,
            project_root=self.root
        )

        self.assertEqual(len(dataset), 1)
        sample = dataset[0]

        self.assertEqual(sample["anchor"].mode, "RGB")
        self.assertEqual(sample["positive"].mode, "RGB")
        self.assertEqual(sample["negative"].mode, "RGB")
        self.assertEqual(sample["anchor_id"], "user-a")
        self.assertEqual(sample["positive_id"], "shop-a")
        self.assertEqual(sample["negative_id"], "shop-b")
        self.assertEqual(sample["source_category_id"], 1)

    def test_applies_the_same_transform_to_every_image(self):
        dataset = FashionTripletDataset(
            triplets_path=self.manifest_path,
            transform=lambda image: (image.mode, image.size),
            project_root=self.root
        )

        sample = dataset[0]

        self.assertEqual(sample["anchor"], ("RGB", (8, 6)))
        self.assertEqual(sample["positive"], ("RGB", (8, 6)))
        self.assertEqual(sample["negative"], ("RGB", (8, 6)))

    def test_rejects_a_manifest_with_a_missing_role(self):
        invalid_manifest_path = self.root / "invalid_triplets.json"
        with invalid_manifest_path.open("w", encoding="utf-8") as file:
            json.dump([{"anchor": self.manifest[0]["anchor"]}], file)

        with self.assertRaisesRegex(ValueError, "positive"):
            FashionTripletDataset(
                triplets_path=invalid_manifest_path,
                project_root=self.root
            )

    def test_reports_a_missing_image_when_sample_is_loaded(self):
        self.manifest[0]["negative"]["image_path"] = "missing.png"
        with self.manifest_path.open("w", encoding="utf-8") as file:
            json.dump(self.manifest, file)

        dataset = FashionTripletDataset(
            triplets_path=self.manifest_path,
            project_root=self.root
        )

        with self.assertRaisesRegex(FileNotFoundError, "negative image"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()

