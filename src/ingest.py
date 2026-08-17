import json
from pathlib import Path

from PIL import Image

from src.embed import EmbeddingService
from src.search import VectorSearchService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata_path = PROJECT_ROOT / "data" / "processed" / "catalog_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {metadata_path}")

    with open(metadata_path, "r") as file:
        catalog_items = json.load(file)

    if not catalog_items:
        raise ValueError("catalog_metadata.json is empty.")

    embedding_service = EmbeddingService()
    search_service = VectorSearchService()

    search_service.create_collection_if_needed(
        vector_size=embedding_service.get_embedding_size()
    )

    for item in catalog_items:
        image_path = PROJECT_ROOT / item["image_path"]

        if not image_path.exists():
            print(f"Skipping missing image: {image_path}")
            continue

        print(f"Embedding and saving: {item['name']}")

        image = Image.open(image_path)
        embedding = embedding_service.embed_image(image)

        search_service.upsert_item(item=item, embedding=embedding)

    print("Catalog ingestion complete.")


if __name__ == "__main__":
    main()