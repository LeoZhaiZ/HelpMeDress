from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

from src.config import CLIP_MODEL_NAME, EMBEDDING_SIZE


class EmbeddingService:
    """
    Turns clothing images into embeddings.

    An embedding is a list of numbers that represents the visual meaning
    of an image. Similar clothing items should have similar embeddings.
    """

    def __init__(self):
        self.model_name = CLIP_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP model on {self.device}...")

        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
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
