"""Batch-ingest clothing catalog images into Qdrant."""

import argparse
import json
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from PIL import Image
from qdrant_client.models import PointStruct


# Running ``python scripts/batch_ingest.py`` puts the scripts directory on
# Python's import path. Add the project root so imports from ``src`` work too.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EMBEDDING_SIZE  # noqa: E402
from src.embed import EmbeddingService  # noqa: E402
from src.search import VectorSearchService  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-ingest clothing images into Qdrant."
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/catalog_metadata.json",
        help="Path to the catalog metadata JSON file."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images to embed and upsert at once."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of metadata records to attempt."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the Qdrant collection before ingestion."
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative.")

    return args


def to_qdrant_id(item_id: object) -> int | str:
    """Convert a catalog ID into an integer or deterministic UUID string."""
    if isinstance(item_id, int) and not isinstance(item_id, bool):
        return item_id

    if isinstance(item_id, str) and item_id.isdigit():
        return int(item_id)

    return str(uuid5(NAMESPACE_URL, f"helpmedress:{item_id}"))


def load_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        catalog_items = json.load(file)

    if not isinstance(catalog_items, list):
        raise ValueError("Metadata JSON must contain a list of catalog items.")

    return catalog_items


def main():
    args = parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    catalog_items = load_metadata(metadata_path)

    if args.limit is not None:
        catalog_items = catalog_items[:args.limit]

    total_attempted = len(catalog_items)
    successful_count = 0
    skipped_count = 0

    # Load CLIP once, then reuse the same model for every image batch.
    embedding_service = EmbeddingService()
    search_service = VectorSearchService()

    if args.reset:
        search_service.delete_collection_if_exists()

    search_service.create_collection_if_needed(EMBEDDING_SIZE)

    for batch_start in range(0, total_attempted, args.batch_size):
        batch_items = catalog_items[
            batch_start:batch_start + args.batch_size
        ]
        valid_items = []
        images = []

        for item in batch_items:
            try:
                image_path = Path(item["image_path"])
                if not image_path.is_absolute():
                    image_path = PROJECT_ROOT / image_path

                # Converting inside the context manager creates an independent
                # RGB image before the source file is closed.
                with Image.open(image_path) as image:
                    images.append(image.convert("RGB"))

                valid_items.append(item)
            except (KeyError, OSError) as error:
                skipped_count += 1
                print(f"Skipping item with unreadable image: {error}")

        if images:
            embeddings = embedding_service.embed_images(images)

            # A Qdrant point combines an ID, an embedding vector, and the
            # original catalog metadata used by search results.
            points = [
                PointStruct(
                    id=to_qdrant_id(item["id"]),
                    vector=embedding,
                    payload=item
                )
                for item, embedding in zip(valid_items, embeddings)
            ]

            search_service.upsert_points(points)
            successful_count += len(points)

        processed_count = min(
            batch_start + args.batch_size,
            total_attempted
        )
        print(f"Progress: {processed_count}/{total_attempted} attempted")

    print("\nBatch ingestion complete.")
    print(f"Successful: {successful_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total attempted: {total_attempted}")


if __name__ == "__main__":
    main()
