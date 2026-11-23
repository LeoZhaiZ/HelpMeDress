from PIL import Image
import torch
import numpy as np
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

from src.config import CLIP_MODEL_NAME, EMBEDDING_SIZE


def load_visual_projection_checkpoint(
    model,
    checkpoint_path: str | Path,
    expected_model_name: str = CLIP_MODEL_NAME
) -> dict:
    """Load a trusted projection-only training checkpoint into CLIP."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Could not find projection checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("Projection checkpoint must contain a dictionary.")

    checkpoint_model_name = checkpoint.get("model_name")
    if checkpoint_model_name != expected_model_name:
        raise ValueError(
            "Checkpoint model does not match the embedding model: "
            f"expected {expected_model_name}, got {checkpoint_model_name}"
        )

    projection_state = checkpoint.get("visual_projection_state_dict")
    if not isinstance(projection_state, dict):
        raise ValueError(
            "Checkpoint is missing visual_projection_state_dict."
        )

    try:
        model.visual_projection.load_state_dict(
            projection_state,
            strict=True
        )
    except RuntimeError as error:
        raise ValueError(
            "Checkpoint projection weights do not match this CLIP model."
        ) from error

    # Keep the useful training measurements without retaining another copy of
    # the tensor state in service metadata.
    return {
        key: value
        for key, value in checkpoint.items()
        if key != "visual_projection_state_dict"
    }


class EmbeddingService:
    """
    Turns clothing images into embeddings.

    An embedding is a list of numbers that represents the visual meaning
    of an image. Similar clothing items should have similar embeddings.
    """

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.model_name = CLIP_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.checkpoint_metadata = None

        print(f"Loading CLIP model on {self.device}...")

        self.model = CLIPModel.from_pretrained(self.model_name)
        if self.checkpoint_path is not None:
            self.checkpoint_metadata = load_visual_projection_checkpoint(
                model=self.model,
                checkpoint_path=self.checkpoint_path,
                expected_model_name=self.model_name
            )
            print(
                "Loaded trained visual projection from: "
                f"{self.checkpoint_path}"
            )

        self.model = self.model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.model.eval()

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        """Create one normalized embedding for each image in a batch."""
        if not images:
            raise ValueError("At least one image is required for embedding.")

        # CLIP expects three-channel RGB images. Converting the whole list here
        # also makes uploads with grayscale or transparent pixels safe to use.
        rgb_images = [image.convert("RGB") for image in images]

        inputs = self.processor(
            images=rgb_images,
            return_tensors="pt"
        ).to(self.device)

        # Passing the complete batch through CLIP at once is much faster than
        # running the model separately for every catalog image.
        with torch.inference_mode():
            image_output = self.model.get_image_features(**inputs)

            # Newer versions of Hugging Face Transformers return
            # a BaseModelOutputWithPooling instead of the tensor directly.
        image_features = image_output.pooler_output

        embeddings = image_features.cpu().numpy()

        # Normalize every vector independently so cosine similarity works
        # properly for each image in the batch.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Embedding norm is zero.")

        normalized_embeddings = embeddings / norms

        return normalized_embeddings.tolist()

    def embed_image(self, image: Image.Image) -> list[float]:
        """Create one normalized embedding for a single image."""
        return self.embed_images([image])[0]

    def get_embedding_size(self) -> int:
        return EMBEDDING_SIZE
