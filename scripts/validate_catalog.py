"""Validate catalog metadata and image files before ingestion."""

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image


# Allow this file to be run directly with ``python scripts/validate_catalog.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.qdrant_ids import to_qdrant_id


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
    "outerwear",
    "dress"
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
    image_check_group = parser.add_mutually_exclusive_group()
    image_check_group.add_argument(
        "--image-check-limit",
        type=int,
        default=1000,
        help=(
            "Maximum number of randomly sampled images to fully open "
            "(default: 1000)."
        )
    )
    image_check_group.add_argument(
        "--check-all-images",
        action="store_true",
        help="Fully open every image instead of checking a sample."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to choose the image sample (default: 42)."
    )

    args = parser.parse_args()
    if args.image_check_limit is not None and args.image_check_limit < 0:
        parser.error("--image-check-limit cannot be negative.")
    return args


def choose_image_check_indexes(
    record_count: int,
    image_check_limit: int | None,
    seed: int
) -> set[int]:
    """Choose repeatable zero-based record indexes for image decoding."""
    if image_check_limit is None or image_check_limit >= record_count:
        return set(range(record_count))
    if image_check_limit <= 0:
        return set()

    random_generator = random.Random(seed)
    return set(random_generator.sample(range(record_count), image_check_limit))


def validate_catalog_items(
    catalog_items: object,
    project_root: Path,
    show_progress: bool = False,
    image_check_limit: int | None = None,
    image_check_seed: int = 42
) -> tuple[dict[str, int], list[str]]:
    """Validate all metadata and fully open a selected image sample."""
    if not isinstance(catalog_items, list):
        return {
            "records_checked": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "image_records_selected": 0,
            "image_records_checked": 0,
            "missing_images": 0,
            "broken_images": 0,
            "duplicate_ids": 0,
            "qdrant_id_collisions": 0
        }, ["Metadata JSON must contain a list of catalog items."]

    summary = {
        "records_checked": len(catalog_items),
        "valid_records": 0,
        "invalid_records": 0,
        "image_records_selected": 0,
        "image_records_checked": 0,
        "missing_images": 0,
        "broken_images": 0,
        "duplicate_ids": 0,
        "qdrant_id_collisions": 0
    }
    errors = []
    seen_source_ids = {}
    seen_qdrant_ids = {}
    image_check_indexes = choose_image_check_indexes(
        record_count=len(catalog_items),
        image_check_limit=image_check_limit,
        seed=image_check_seed
    )
    summary["image_records_selected"] = len(image_check_indexes)

    for zero_based_index, item in enumerate(catalog_items):
        index = zero_based_index + 1
        item_errors = []
        check_image = zero_based_index in image_check_indexes

        if check_image:
            summary["image_records_checked"] += 1

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
            elif check_image:
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

        if show_progress and (
            index % 10000 == 0
            or index == len(catalog_items)
        ):
            print(
                f"Progress: {index}/{len(catalog_items)} metadata records; "
                f"{summary['image_records_checked']}/"
                f"{summary['image_records_selected']} sampled images checked",
                flush=True
            )

    return summary, errors


def main() -> int:
    args = parse_args()

    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    print(f"Loading metadata: {metadata_path}", flush=True)
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            catalog_items = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read metadata: {error}")
        return 1

    if not isinstance(catalog_items, list):
        image_check_limit = None
    elif args.check_all_images:
        image_check_limit = None
        print(
            f"Loaded {len(catalog_items)} records. Every image will be "
            "fully checked.",
            flush=True
        )
    else:
        image_check_limit = args.image_check_limit
        sample_size = min(image_check_limit, len(catalog_items))
        print(
            f"Loaded {len(catalog_items)} records. All metadata will be "
            f"checked; {sample_size} images will be fully opened.",
            flush=True
        )

    summary, errors = validate_catalog_items(
        catalog_items=catalog_items,
        project_root=PROJECT_ROOT,
        show_progress=True,
        image_check_limit=image_check_limit,
        image_check_seed=args.seed
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
    print(
        "Images fully checked: "
        f"{summary['image_records_checked']}/"
        f"{summary['image_records_selected']}"
    )
    print(f"Missing images: {summary['missing_images']}")
    print(f"Broken images: {summary['broken_images']}")
    print(f"Duplicate IDs: {summary['duplicate_ids']}")
    print(f"Qdrant ID collisions: {summary['qdrant_id_collisions']}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
