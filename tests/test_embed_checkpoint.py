import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from src.config import CLIP_MODEL_NAME
from src.embed import load_visual_projection_checkpoint


class EmbedCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_loads_projection_weights_and_returns_metrics(self):
        model = SimpleNamespace(
            visual_projection=torch.nn.Linear(3, 2, bias=False)
        )
        expected_weight = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        checkpoint_path = self.root / "checkpoint.pt"
        torch.save({
            "model_name": CLIP_MODEL_NAME,
            "visual_projection_state_dict": {
                "weight": expected_weight
            },
            "projection_weight_change": 0.25
        }, checkpoint_path)

        metadata = load_visual_projection_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path
        )

        self.assertTrue(torch.equal(
            model.visual_projection.weight,
            expected_weight
        ))
        self.assertEqual(metadata["projection_weight_change"], 0.25)
        self.assertNotIn("visual_projection_state_dict", metadata)

    def test_rejects_checkpoint_for_a_different_model(self):
        model = SimpleNamespace(
            visual_projection=torch.nn.Linear(3, 2, bias=False)
        )
        checkpoint_path = self.root / "wrong-model.pt"
        torch.save({
            "model_name": "some-other-model",
            "visual_projection_state_dict": model.visual_projection.state_dict()
        }, checkpoint_path)

        with self.assertRaisesRegex(ValueError, "does not match"):
            load_visual_projection_checkpoint(model, checkpoint_path)

    def test_rejects_projection_with_the_wrong_shape(self):
        model = SimpleNamespace(
            visual_projection=torch.nn.Linear(3, 2, bias=False)
        )
        checkpoint_path = self.root / "wrong-shape.pt"
        torch.save({
            "model_name": CLIP_MODEL_NAME,
            "visual_projection_state_dict": {
                "weight": torch.ones(4, 4)
            }
        }, checkpoint_path)

        with self.assertRaisesRegex(ValueError, "do not match"):
            load_visual_projection_checkpoint(model, checkpoint_path)


if __name__ == "__main__":
    unittest.main()
