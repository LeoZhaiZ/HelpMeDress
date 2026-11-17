import unittest
from uuid import UUID

from src.qdrant_ids import to_qdrant_id


class QdrantIdTests(unittest.TestCase):
    def test_keeps_integer_ids(self):
        self.assertEqual(to_qdrant_id(123), 123)

    def test_converts_numeric_strings_to_integers(self):
        self.assertEqual(to_qdrant_id("123"), 123)

    def test_converts_text_ids_to_repeatable_uuids(self):
        first_id = to_qdrant_id("deepfashion2-train-000001-item1")
        second_id = to_qdrant_id("deepfashion2-train-000001-item1")

        self.assertEqual(first_id, second_id)
        self.assertEqual(str(UUID(first_id)), first_id)


if __name__ == "__main__":
    unittest.main()
