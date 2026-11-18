import unittest

from scripts.train_clip_triplet import (
    choose_subset_indexes,
    collate_triplets
)


class TrainClipTripletTests(unittest.TestCase):
    def test_subset_is_limited_and_repeatable(self):
        first = choose_subset_indexes(length=100, limit=10, seed=7)
        second = choose_subset_indexes(length=100, limit=10, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(first, sorted(first))

    def test_subset_uses_every_item_when_limit_is_larger(self):
        self.assertEqual(
            choose_subset_indexes(length=3, limit=10, seed=7),
            [0, 1, 2]
        )

    def test_collate_keeps_roles_and_ids_together(self):
        samples = [
            {
                "anchor": "anchor-a",
                "positive": "positive-a",
                "negative": "negative-a",
                "anchor_id": "a",
                "positive_id": "p-a",
                "negative_id": "n-a"
            },
            {
                "anchor": "anchor-b",
                "positive": "positive-b",
                "negative": "negative-b",
                "anchor_id": "b",
                "positive_id": "p-b",
                "negative_id": "n-b"
            }
        ]

        batch = collate_triplets(samples)

        self.assertEqual(batch["anchor"], ["anchor-a", "anchor-b"])
        self.assertEqual(batch["positive_id"], ["p-a", "p-b"])
        self.assertEqual(batch["negative_id"], ["n-a", "n-b"])


if __name__ == "__main__":
    unittest.main()
