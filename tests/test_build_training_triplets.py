import unittest

from scripts.build_training_triplets import build_training_triplets


def metadata_item(
    item_id: str,
    group_id: str,
    source_domain: str,
    category_id: int
) -> dict:
    return {
        "id": item_id,
        "image_path": f"data/fake/{item_id}.jpg",
        "item_group_id": group_id,
        "source_domain": source_domain,
        "source_category_id": category_id,
        "source_category": "short sleeve top"
    }


class BuildTrainingTripletsTests(unittest.TestCase):
    def setUp(self):
        self.metadata = [
            metadata_item("user-a", "garment-a", "user", 1),
            metadata_item("shop-a", "garment-a", "shop", 1),
            metadata_item("user-b", "garment-b", "user", 1),
            metadata_item("shop-b", "garment-b", "shop", 1),
            metadata_item("user-c", "garment-c", "user", 2),
            metadata_item("shop-c", "garment-c", "shop", 2)
        ]

    def test_matches_positives_and_uses_different_same_category_negatives(self):
        triplets, summary = build_training_triplets(self.metadata, seed=42)

        self.assertEqual(len(triplets), 2)
        self.assertEqual(summary["anchors_without_negative"], 1)

        for triplet in triplets:
            anchor = triplet["anchor"]
            positive = triplet["positive"]
            negative = triplet["negative"]

            self.assertEqual(
                anchor["item_group_id"],
                positive["item_group_id"]
            )
            self.assertNotEqual(
                anchor["item_group_id"],
                negative["item_group_id"]
            )
            self.assertEqual(
                anchor["source_category_id"],
                positive["source_category_id"]
            )
            self.assertEqual(
                anchor["source_category_id"],
                negative["source_category_id"]
            )
            self.assertEqual(triplet["source_category_id"], 1)

    def test_same_seed_produces_the_same_triplets(self):
        first_triplets, _ = build_training_triplets(self.metadata, seed=7)
        second_triplets, _ = build_training_triplets(self.metadata, seed=7)

        self.assertEqual(first_triplets, second_triplets)

    def test_limit_caps_the_number_of_triplets(self):
        triplets, summary = build_training_triplets(
            self.metadata,
            seed=42,
            limit=1
        )

        self.assertEqual(len(triplets), 1)
        self.assertEqual(summary["triplets_created"], 1)


if __name__ == "__main__":
    unittest.main()
