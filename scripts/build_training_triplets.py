"""Build DeepFashion2 anchor, positive, and negative training examples."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build training triplets from prepared DeepFashion2 metadata."
    )
    parser.add_argument(
        "--metadata",
        default="data/processed/deepfashion2_train_metadata.json",
        help="Metadata made with --split train --source all."
    )
    parser.add_argument(
        "--output",
        default="data/processed/deepfashion2_train_triplets.json",
        help="Path for the generated triplet JSON file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to choose positives and negatives (default: 42)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of triplets to create."
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


def load_metadata(metadata_path: Path) -> list[dict]:
    """Load the prepared metadata and make sure it contains a list."""
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Could not find metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, list):
        raise ValueError("Metadata JSON must contain a list of items.")

    return metadata


def item_sort_key(item: dict) -> tuple[str, str]:
    """Keep option ordering stable before making seeded random choices."""
    return str(item.get("id", "")), str(item.get("image_path", ""))


def training_item(item: dict) -> dict:
    """Keep only the fields the future PyTorch dataset will need."""
    return {
        "id": item["id"],
        "image_path": item["image_path"],
        "item_group_id": item["item_group_id"],
        "source_domain": item["source_domain"],
        "source_category_id": item["source_category_id"],
        "source_category": item.get("source_category")
    }


def build_training_triplets(
    metadata: list[dict],
    seed: int = 42,
    limit: int | None = None
) -> tuple[list[dict], dict[str, int]]:
    """Match user anchors with shop positives and same-category negatives."""
    items_by_group = defaultdict(lambda: {"user": [], "shop": []})
    shop_items_by_category_and_group = defaultdict(lambda: defaultdict(list))
    invalid_records = 0

    for item in metadata:
        if not isinstance(item, dict):
            invalid_records += 1
            continue

        item_id = item.get("id")
        image_path = item.get("image_path")
        group_id = item.get("item_group_id")
        source_domain = item.get("source_domain")
        category_id = item.get("source_category_id")

        item_id_is_valid = (
            isinstance(item_id, (int, str))
            and not isinstance(item_id, bool)
            and not (isinstance(item_id, str) and not item_id.strip())
        )
        category_id_is_valid = (
            isinstance(category_id, int)
            and not isinstance(category_id, bool)
        )

        if (
            not item_id_is_valid
            or not isinstance(image_path, str)
            or not image_path.strip()
            or not isinstance(group_id, str)
            or not group_id.strip()
            or source_domain not in {"user", "shop"}
            or not category_id_is_valid
        ):
            invalid_records += 1
            continue

        items_by_group[group_id][source_domain].append(item)
        if source_domain == "shop":
            shop_items_by_category_and_group[category_id][group_id].append(
                item
            )

    for group in items_by_group.values():
        group["user"].sort(key=item_sort_key)
        group["shop"].sort(key=item_sort_key)

    shop_group_ids_by_category = {}
    shop_group_indexes_by_category = {}
    for category_id, category_groups in shop_items_by_category_and_group.items():
        group_ids = sorted(category_groups)
        shop_group_ids_by_category[category_id] = group_ids
        shop_group_indexes_by_category[category_id] = {
            group_id: index
            for index, group_id in enumerate(group_ids)
        }

        for shop_items in category_groups.values():
            shop_items.sort(key=item_sort_key)

    random_generator = random.Random(seed)
    triplets = []
    anchors_checked = 0
    anchors_without_positive = 0
    anchors_without_negative = 0

    for group_id in sorted(items_by_group):
        group = items_by_group[group_id]

        for anchor in group["user"]:
            if limit is not None and len(triplets) >= limit:
                break

            anchors_checked += 1
            category_id = anchor["source_category_id"]
            category_groups = shop_items_by_category_and_group[category_id]
            positive_options = category_groups.get(group_id, [])

            if not positive_options:
                anchors_without_positive += 1
                continue

            # A useful negative looks similar enough to be challenging: it is
            # a shop item in the same detailed category, but a different
            # garment identity.
            negative_group_ids = shop_group_ids_by_category[category_id]
            current_group_index = shop_group_indexes_by_category[
                category_id
            ].get(group_id)

            if (
                not negative_group_ids
                or (
                    len(negative_group_ids) == 1
                    and current_group_index is not None
                )
            ):
                anchors_without_negative += 1
                continue

            # Select a random group index while skipping the anchor's group.
            # This avoids scanning every shop item for every anchor.
            if current_group_index is None:
                negative_group_index = random_generator.randrange(
                    len(negative_group_ids)
                )
            else:
                negative_group_index = random_generator.randrange(
                    len(negative_group_ids) - 1
                )
                if negative_group_index >= current_group_index:
                    negative_group_index += 1

            negative_group_id = negative_group_ids[negative_group_index]
            negative_options = category_groups[negative_group_id]

            positive = random_generator.choice(positive_options)
            negative = random_generator.choice(negative_options)

            triplets.append({
                "anchor": training_item(anchor),
                "positive": training_item(positive),
                "negative": training_item(negative),
                "source_category_id": category_id,
                "source_category": anchor.get("source_category")
            })

        if limit is not None and len(triplets) >= limit:
            break

    summary = {
        "metadata_records": len(metadata),
        "invalid_records": invalid_records,
        "anchors_checked": anchors_checked,
        "anchors_without_positive": anchors_without_positive,
        "anchors_without_negative": anchors_without_negative,
        "triplets_created": len(triplets)
    }
    return triplets, summary


def main() -> int:
    args = parse_args()
    metadata_path = resolve_from_project(args.metadata)
    output_path = resolve_from_project(args.output)

    try:
        metadata = load_metadata(metadata_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not load training metadata: {error}")
        return 1

    triplets, summary = build_training_triplets(
        metadata=metadata,
        seed=args.seed,
        limit=args.limit
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(triplets, file, indent=2)
        file.write("\n")

    print("Training triplet preparation complete.")
    print(f"Metadata records: {summary['metadata_records']}")
    print(f"Invalid records skipped: {summary['invalid_records']}")
    print(f"User anchors checked: {summary['anchors_checked']}")
    print(
        "Anchors without a shop positive: "
        f"{summary['anchors_without_positive']}"
    )
    print(
        "Anchors without a same-category negative: "
        f"{summary['anchors_without_negative']}"
    )
    print(f"Triplets created: {summary['triplets_created']}")
    print(f"Triplets written to: {output_path}")

    if not triplets:
        print("No usable triplets were found.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
