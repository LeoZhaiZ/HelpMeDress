from PIL import Image
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel


class EmbeddingService:
    """
    Turns clothing images into embeddings.

    An embedding is a list of numbers that represents the visual meaning
    of an image. Similar clothing items should have similar embeddings.
    """

    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP model on {self.device}...")

        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        self.model.eval()

    def embed_image(self, image: Image.Image) -> list[float]:
        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            image_output = self.model.get_image_features(**inputs)

            # Newer versions of Hugging Face Transformers return
            # a BaseModelOutputWithPooling instead of the tensor directly.
        image_features = image_output.pooler_output

        embedding = image_features.cpu().numpy()[0]

        # Normalize vector so cosine similarity works properly.
        norm = np.linalg.norm(embedding)

        if norm == 0:
            raise ValueError("Embedding norm is zero.")

        embedding = embedding / norm

        return embedding.tolist()

    def get_embedding_size(self) -> int:
        return 512