"""Validate catalog metadata and image files before ingestion."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


# Allow this file to be run directly with ``python scripts/validate_catalog.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.batch_ingest import to_qdrant_id 


REQUIRED_FIELDS = {
    "id",
    "name",
    "category",
    "style",
    "brand",
    "price",
    "image_path",
    "product_url"
}

SUPPORTED_CATEGORIES = {
    "top",
    "bottom",
    "shoes",
    "accessory",
    "outerwear"
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate catalog metadata and image files."
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/catalog_metadata.json",
        help="Path to the catalog metadata JSON file."
    )
    return parser.parse_args()


def validate_catalog_items(
    catalog_items: object,
    project_root: Path
) -> tuple[dict[str, int], list[str]]:
    """Validate parsed catalog data and return summary counts and errors."""
    if not isinstance(catalog_items, list):
        return {
            "records_checked": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "missing_images": 0,
            "broken_images": 0,
            "duplicate_ids": 0,
            "qdrant_id_collisions": 0
        }, ["Metadata JSON must contain a list of catalog items."]

    summary = {
        "records_checked": len(catalog_items),
        "valid_records": 0,
        "invalid_records": 0,
        "missing_images": 0,
        "broken_images": 0,
        "duplicate_ids": 0,
        "qdrant_id_collisions": 0
    }
    errors = []
    seen_source_ids = {}
    seen_qdrant_ids = {}

    for index, item in enumerate(catalog_items, start=1):
        item_errors = []

        if not isinstance(item, dict):
            item_errors.append("record must be a JSON object")
        else:
            missing_fields = sorted(REQUIRED_FIELDS - item.keys())
            if missing_fields:
                item_errors.append(
                    f"missing fields: {', '.join(missing_fields)}"
                )

            item_id = item.get("id")
            id_is_valid = (
                isinstance(item_id, (int, str))
                and not isinstance(item_id, bool)
                and not (isinstance(item_id, str) and not item_id.strip())
            )

            if not id_is_valid:
                item_errors.append("id must be a non-empty string or integer")
            else:
                source_id_key = (type(item_id).__name__, str(item_id))
                if source_id_key in seen_source_ids:
                    first_index = seen_source_ids[source_id_key]
                    summary["duplicate_ids"] += 1
                    item_errors.append(
                        f"duplicate id also used by record {first_index}"
                    )
                else:
                    seen_source_ids[source_id_key] = index

                qdrant_id = to_qdrant_id(item_id)
                if qdrant_id in seen_qdrant_ids:
                    first_index = seen_qdrant_ids[qdrant_id]
                    summary["qdrant_id_collisions"] += 1
                    item_errors.append(
                        "converted Qdrant ID collides with "
                        f"record {first_index}"
                    )
                else:
                    seen_qdrant_ids[qdrant_id] = index

            name = item.get("name")
            if not isinstance(name, str) or not name.strip():
                item_errors.append("name must be a non-empty string")

            category = item.get("category")
            if category not in SUPPORTED_CATEGORIES:
                supported = ", ".join(sorted(SUPPORTED_CATEGORIES))
                item_errors.append(
                    f"category must be one of: {supported}"
                )

            for field in ("style", "brand", "product_url"):
                value = item.get(field)
                if value is not None and not isinstance(value, str):
                    item_errors.append(
                        f"{field} must be a string or null"
                    )

            price = item.get("price")
            price_is_number = (
                isinstance(price, (int, float))
                and not isinstance(price, bool)
            )
            if price is not None and not price_is_number:
                item_errors.append("price must be a number or null")
            elif price_is_number and price < 0:
                item_errors.append("price cannot be negative")

            image_path_value = item.get("image_path")
            if (
                not isinstance(image_path_value, str)
                or not image_path_value.strip()
            ):
                item_errors.append("image_path must be a non-empty string")
            else:
                image_path = Path(image_path_value)
                if not image_path.is_absolute():
                    image_path = project_root / image_path

                if not image_path.is_file():
                    summary["missing_images"] += 1
                    item_errors.append(f"image does not exist: {image_path}")
                else:
                    try:
                        # Loading the pixels catches files that exist but are
                        # corrupt or are not actually readable images.
                        with Image.open(image_path) as image:
                            image.load()
                    except OSError as error:
                        summary["broken_images"] += 1
                        item_errors.append(
                            f"image cannot be opened: {error}"
                        )

        if item_errors:
            summary["invalid_records"] += 1
            errors.extend(
                f"Record {index}: {message}"
                for message in item_errors
            )
        else:
            summary["valid_records"] += 1

    return summary, errors


def main() -> int:
    args = parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            catalog_items = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read metadata: {error}")
        return 1

    summary, errors = validate_catalog_items(
        catalog_items=catalog_items,
        project_root=PROJECT_ROOT
    )

    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"- {error}")
        print()

    print("Catalog validation summary")
    print(f"Records checked: {summary['records_checked']}")
    print(f"Valid records: {summary['valid_records']}")
    print(f"Invalid records: {summary['invalid_records']}")
    print(f"Missing images: {summary['missing_images']}")
    print(f"Broken images: {summary['broken_images']}")
    print(f"Duplicate IDs: {summary['duplicate_ids']}")
    print(f"Qdrant ID collisions: {summary['qdrant_id_collisions']}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
