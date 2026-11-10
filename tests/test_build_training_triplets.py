import unittest

from scripts.build_training_triplets import (
    build_training_triplets,
    split_metadata_by_identity
)


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

    def test_identity_split_has_no_garment_overlap(self):
        metadata = []
        for index in range(20):
            group_id = f"garment-{index:02d}"
            metadata.extend([
                metadata_item(f"user-{index}", group_id, "user", 1),
                metadata_item(f"shop-{index}", group_id, "shop", 1)
            ])

        training_metadata, tuning_metadata, summary = (
            split_metadata_by_identity(
                metadata,
                tuning_fraction=0.2,
                seed=42
            )
        )

        training_groups = {
            item["item_group_id"]
            for item in training_metadata
        }
        tuning_groups = {
            item["item_group_id"]
            for item in tuning_metadata
        }

        self.assertTrue(training_groups)
        self.assertTrue(tuning_groups)
        self.assertTrue(training_groups.isdisjoint(tuning_groups))
        self.assertEqual(summary["training_identity_groups"], 16)
        self.assertEqual(summary["tuning_identity_groups"], 4)

        training_triplets, _ = build_training_triplets(
            training_metadata,
            seed=42
        )
        tuning_triplets, _ = build_training_triplets(
            tuning_metadata,
            seed=43
        )

        training_triplet_groups = {
            item["item_group_id"]
            for triplet in training_triplets
            for item in (
                triplet["anchor"],
                triplet["positive"],
                triplet["negative"]
            )
        }
        tuning_triplet_groups = {
            item["item_group_id"]
            for triplet in tuning_triplets
            for item in (
                triplet["anchor"],
                triplet["positive"],
                triplet["negative"]
            )
        }

        self.assertTrue(training_triplets)
        self.assertTrue(tuning_triplets)
        self.assertTrue(
            training_triplet_groups.isdisjoint(tuning_triplet_groups)
        )

    def test_identity_split_is_repeatable(self):
        first_training, first_tuning, _ = split_metadata_by_identity(
            self.metadata,
            tuning_fraction=0.34,
            seed=7
        )
        second_training, second_tuning, _ = split_metadata_by_identity(
            self.metadata,
            tuning_fraction=0.34,
            seed=7
        )

        self.assertEqual(first_training, second_training)
        self.assertEqual(first_tuning, second_tuning)

    def test_identity_split_requires_multiple_groups(self):
        one_group = [
            metadata_item("user-a", "garment-a", "user", 1),
            metadata_item("shop-a", "garment-a", "shop", 1)
        ]

        with self.assertRaisesRegex(ValueError, "two garment identities"):
            split_metadata_by_identity(one_group)


if __name__ == "__main__":
    unittest.main()
