"""Convert DeepFashion2 annotations into HelpMeDress garment metadata."""

import argparse
import json
import math
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Map DeepFashion2's detailed categories into the broader categories used by
# HelpMeDress while preserving the original category in the metadata.
CATEGORY_MAP = {
    1: "top",        # short sleeve top
    2: "top",        # long sleeve top
    3: "outerwear",  # short sleeve outwear
    4: "outerwear",  # long sleeve outwear
    5: "top",        # vest
    6: "top",        # sling
    7: "bottom",     # shorts
    8: "bottom",     # trousers
    9: "bottom",     # skirt
    10: "dress",     # short sleeve dress
    11: "dress",     # long sleeve dress
    12: "dress",     # vest dress
    13: "dress"      # sling dress
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Crop DeepFashion2 garments and create HelpMeDress metadata."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="data/deepfashion2",
        help="DeepFashion2 directory containing train/ and validation/."
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="validation",
        help="Dataset split to prepare (default: validation)."
    )
    parser.add_argument(
        "--output",
        help=(
            "Output metadata JSON path. Defaults to "
            "data/processed/deepfashion2_<split>_metadata.json."
        )
    )
    parser.add_argument(
        "--crops-dir",
        help=(
            "Directory for garment crops. Defaults to "
            "data/processed/deepfashion2_crops/<split>."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of image annotations to process."
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit cannot be negative.")

    return args


def resolve_from_project(path_value: str) -> Path:
    """Resolve a relative path from the HelpMeDress project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def metadata_image_path(image_path: Path) -> str:
    """Prefer portable project-relative paths in generated metadata."""
    resolved_path = image_path.resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved_path)


def item_annotations(annotation: dict) -> list[tuple[str, dict]]:
    """Return item1, item2, and similar garment annotations in order."""
    items = []

    for key, value in annotation.items():
        if key.startswith("item") and key[4:].isdigit() and isinstance(value, dict):
            items.append((key, value))

    return sorted(items, key=lambda item: int(item[0][4:]))


def crop_box(
    bounding_box: object,
    image_width: int,
    image_height: int
) -> tuple[int, int, int, int] | None:
    """Validate and clamp a DeepFashion2 [x1, y1, x2, y2] box."""
    if not isinstance(bounding_box, list) or len(bounding_box) != 4:
        return None

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in bounding_box
    ):
        return None

    x1, y1, x2, y2 = bounding_box
    left = max(0, math.floor(x1))
    top = max(0, math.floor(y1))
    right = min(image_width, math.ceil(x2))
    bottom = min(image_height, math.ceil(y2))

    if right <= left or bottom <= top:
        return None

    return left, top, right, bottom


def build_metadata_item(
    split: str,
    image_id: str,
    item_key: str,
    annotation: dict,
    item: dict,
    mapped_category: str,
    crop_path: Path,
    bounding_box: tuple[int, int, int, int]
) -> dict:
    """Build one HelpMeDress metadata record for a garment crop."""
    category_name = str(item.get("category_name", "clothing item"))
    pair_id = annotation.get("pair_id")
    source_style = item.get("style")

    # DeepFashion2 defines a positive matching identity with the same pair_id
    # and a style number greater than zero.
    item_group_id = None
    if pair_id is not None and isinstance(source_style, int) and source_style > 0:
        item_group_id = f"deepfashion2-{pair_id}-{source_style}"

    source_id = f"{split}-{image_id}-{item_key}"

    return {
        "id": f"deepfashion2-{source_id}",
        "name": f"DeepFashion2 {category_name.replace('_', ' ').title()}",
        "category": mapped_category,
        "style": None,
        "brand": None,
        "price": None,
        "image_path": metadata_image_path(crop_path),
        "product_url": None,
        "source_dataset": "deepfashion2",
        "source_id": source_id,
        "source_image_id": image_id,
        "source_domain": annotation.get("source"),
        "source_category": category_name,
        "source_category_id": item.get("category_id"),
        "source_style": source_style,
        "pair_id": pair_id,
        "item_group_id": item_group_id,
        "split": split,
        "bounding_box": list(bounding_box)
    }


def prepare_annotation(
    annotation_path: Path,
    image_path: Path,
    crops_dir: Path,
    split: str
) -> tuple[list[dict], dict[str, int]]:
    """Create crops and metadata for all usable garments in one image."""
    counts = {
        "crops_created": 0,
        "unsupported_categories": 0,
        "invalid_items": 0,
        "annotation_errors": 0,
        "missing_or_broken_images": 0
    }

    try:
        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)
    except (OSError, json.JSONDecodeError):
        counts["annotation_errors"] += 1
        return [], counts

    if not isinstance(annotation, dict):
        counts["annotation_errors"] += 1
        return [], counts

    try:
        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
    except OSError:
        counts["missing_or_broken_images"] += 1
        return [], counts

    metadata_items = []
    image_id = annotation_path.stem

    for item_key, item in item_annotations(annotation):
        category_id = item.get("category_id")
        mapped_category = CATEGORY_MAP.get(category_id)

        if mapped_category is None:
            counts["unsupported_categories"] += 1
            continue

        bounding_box = crop_box(
            item.get("bounding_box"),
            image_width=source_image.width,
            image_height=source_image.height
        )
        if bounding_box is None:
            counts["invalid_items"] += 1
            continue

        crop_path = crops_dir / f"{image_id}_{item_key}.jpg"
        garment_crop = source_image.crop(bounding_box)
        garment_crop.save(crop_path, format="JPEG", quality=95)

        metadata_items.append(
            build_metadata_item(
                split=split,
                image_id=image_id,
                item_key=item_key,
                annotation=annotation,
                item=item,
                mapped_category=mapped_category,
                crop_path=crop_path,
                bounding_box=bounding_box
            )
        )
        counts["crops_created"] += 1

    return metadata_items, counts


def main() -> int:
    args = parse_args()
    dataset_root = resolve_from_project(args.dataset_root)
    split_root = dataset_root / args.split
    images_dir = split_root / "image"
    annotations_dir = split_root / "annos"

    if not images_dir.is_dir() or not annotations_dir.is_dir():
        print("Could not find the expected DeepFashion2 directories:")
        print(f"- {images_dir}")
        print(f"- {annotations_dir}")
        return 1

    output_path = resolve_from_project(
        args.output
        or f"data/processed/deepfashion2_{args.split}_metadata.json"
    )
    crops_dir = resolve_from_project(
        args.crops_dir
        or f"data/processed/deepfashion2_crops/{args.split}"
    )

    annotation_paths = sorted(annotations_dir.glob("*.json"))
    if args.limit is not None:
        annotation_paths = annotation_paths[:args.limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    metadata_items = []
    totals = {
        "crops_created": 0,
        "unsupported_categories": 0,
        "invalid_items": 0,
        "annotation_errors": 0,
        "missing_or_broken_images": 0
    }

    total_annotations = len(annotation_paths)
    for index, annotation_path in enumerate(annotation_paths, start=1):
        image_path = images_dir / f"{annotation_path.stem}.jpg"
        items, counts = prepare_annotation(
            annotation_path=annotation_path,
            image_path=image_path,
            crops_dir=crops_dir,
            split=args.split
        )
        metadata_items.extend(items)

        for key, value in counts.items():
            totals[key] += value

        if index % 100 == 0 or index == total_annotations:
            print(f"Progress: {index}/{total_annotations} annotations")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metadata_items, file, indent=2)
        file.write("\n")

    print("\nDeepFashion2 preparation complete.")
    print(f"Annotations attempted: {total_annotations}")
    print(f"Garment crops created: {totals['crops_created']}")
    print(f"Unsupported category items skipped: {totals['unsupported_categories']}")
    print(f"Invalid item annotations skipped: {totals['invalid_items']}")
    print(f"Annotation files skipped: {totals['annotation_errors']}")
    print(
        "Missing or broken source images: "
        f"{totals['missing_or_broken_images']}"
    )
    print(f"Metadata written to: {output_path}")
    print(f"Crops written to: {crops_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
