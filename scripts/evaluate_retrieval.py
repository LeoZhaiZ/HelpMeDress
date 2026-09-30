"""Evaluate zero-shot CLIP retrieval on DeepFashion2 query/gallery pairs."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_deepfashion2 import crop_box
from src.config import CLIP_MODEL_NAME
from src.embed import EmbeddingService


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure zero-shot CLIP retrieval on DeepFashion2."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/deepfashion2",
        help="DeepFashion2 directory containing validation/image."
    )
    parser.add_argument(
        "--evaluation-dir",
        default="data/deepfashion2/evaluation/json_for_validation",
        help="Directory containing val_query.json and val_gallery.json."
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=100,
        help="Maximum number of valid user queries (default: 100)."
    )
    parser.add_argument(
        "--gallery-limit",
        type=int,
        default=1000,
        help="Target number of shop gallery items (default: 1000)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of garment crops embedded together (default: 32)."
    )
    parser.add_argument(
        "--output",
        default="data/processed/retrieval_evaluation.json",
        help="Path for the measured evaluation results."
    )
    parser.add_argument(
        "--grids-dir",
        default="data/processed/retrieval_grids",
        help="Directory for nearest-neighbor image grids."
    )
    parser.add_argument(
        "--grid-count",
        type=int,
        default=20,
        help="Number of diagnostic grids to save (default: 20)."
    )

    args = parser.parse_args()

    if args.query_limit <= 0:
        parser.error("--query-limit must be greater than zero.")
    if args.gallery_limit <= 0:
        parser.error("--gallery-limit must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if args.grid_count < 0:
        parser.error("--grid-count cannot be negative.")

    return args


def resolve_from_project(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_json_list(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Could not find evaluation file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in: {path}")

    return data


def garment_identity(item: dict) -> tuple[object, object] | None:
    """Return the DeepFashion2 identity used to define a correct match."""
    pair_id = item.get("pair_id")
    style = item.get("style")

    if pair_id is None or not isinstance(style, int) or style <= 0:
        return None

    return pair_id, style


def gallery_entry_key(item: dict) -> tuple:
    return (
        item.get("gallery_image_id"),
        tuple(item.get("bbox", [])),
        item.get("pair_id"),
        item.get("style")
    )


def select_evaluation_subset(
    queries: list[dict],
    gallery: list[dict],
    query_limit: int,
    gallery_limit: int
) -> tuple[list[dict], list[dict]]:
    """Select queries and a gallery that includes every query's positives."""
    gallery_by_identity = defaultdict(list)
    for gallery_item in gallery:
        identity = garment_identity(gallery_item)
        if identity is not None:
            gallery_by_identity[identity].append(gallery_item)

    selected_queries = []
    for query in queries:
        identity = garment_identity(query)
        if identity in gallery_by_identity:
            selected_queries.append(query)
        if len(selected_queries) >= query_limit:
            break

    if not selected_queries:
        raise ValueError("No queries with matching gallery items were found.")

    selected_gallery = []
    selected_keys = set()

    # Always include every correct answer for the selected queries.
    for query in selected_queries:
        identity = garment_identity(query)
        for gallery_item in gallery_by_identity[identity]:
            key = gallery_entry_key(gallery_item)
            if key not in selected_keys:
                selected_gallery.append(gallery_item)
                selected_keys.add(key)

    # Fill the remaining gallery with non-matching distractor items.
    for gallery_item in gallery:
        if len(selected_gallery) >= gallery_limit:
            break

        if garment_identity(gallery_item) is None:
            continue

        key = gallery_entry_key(gallery_item)
        if key not in selected_keys:
            selected_gallery.append(gallery_item)
            selected_keys.add(key)

    return selected_queries, selected_gallery


def load_garment_crop(
    item: dict,
    image_id_field: str,
    images_dir: Path
) -> Image.Image:
    """Open one validation image and crop its annotated garment."""
    image_id = int(item[image_id_field])
    image_path = images_dir / f"{image_id:06d}.jpg"

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        bounding_box = crop_box(
            item.get("bbox"),
            image_width=image.width,
            image_height=image.height
        )

        if bounding_box is None:
            raise ValueError(f"Invalid bounding box for image {image_id}")

        return image.crop(bounding_box)


def embed_items(
    items: list[dict],
    image_id_field: str,
    images_dir: Path,
    embedding_service: EmbeddingService,
    batch_size: int,
    label: str
) -> tuple[list[dict], np.ndarray, int]:
    """Crop and embed evaluation items, skipping unreadable images."""
    successful_items = []
    embeddings = []
    skipped_count = 0

    for batch_start in range(0, len(items), batch_size):
        batch_items = items[batch_start:batch_start + batch_size]
        valid_batch_items = []
        images = []

        for item in batch_items:
            try:
                images.append(
                    load_garment_crop(item, image_id_field, images_dir)
                )
                valid_batch_items.append(item)
            except (KeyError, OSError, TypeError, ValueError):
                skipped_count += 1

        if images:
            batch_embeddings = embedding_service.embed_images(images)
            embeddings.extend(batch_embeddings)
            successful_items.extend(valid_batch_items)

        processed_count = min(batch_start + batch_size, len(items))
        print(f"Embedding {label}: {processed_count}/{len(items)}")

    if not embeddings:
        raise ValueError(f"No {label} images could be embedded.")

    return (
        successful_items,
        np.asarray(embeddings, dtype=np.float32),
        skipped_count
    )


def calculate_recall(
    queries: list[dict],
    query_embeddings: np.ndarray,
    gallery: list[dict],
    gallery_embeddings: np.ndarray
) -> tuple[dict[int, float], dict[int, int], list[dict]]:
    """Compare each query with every gallery vector and calculate Recall@K."""
    recall_levels = (1, 5, 10)
    hit_counts = {level: 0 for level in recall_levels}
    evaluated_queries = 0
    query_results = []

    gallery_identities = [garment_identity(item) for item in gallery]
    available_identities = set(gallery_identities)

    for query, query_embedding in zip(queries, query_embeddings):
        correct_identity = garment_identity(query)
        if correct_identity not in available_identities:
            continue

        # The vectors are normalized, so their dot product is cosine
        # similarity. Every gallery vector is compared with this query.
        similarity_scores = gallery_embeddings @ query_embedding
        ranked_indices = np.argsort(-similarity_scores)

        first_correct_rank = next(
            (
                rank
                for rank, gallery_index in enumerate(ranked_indices, start=1)
                if gallery_identities[gallery_index] == correct_identity
            ),
            None
        )

        for level in recall_levels:
            top_indices = ranked_indices[:level]
            if any(
                gallery_identities[index] == correct_identity
                for index in top_indices
            ):
                hit_counts[level] += 1

        top_results = []
        for rank, gallery_index in enumerate(ranked_indices[:10], start=1):
            gallery_item = gallery[gallery_index]
            top_results.append({
                "rank": rank,
                "similarity_score": float(similarity_scores[gallery_index]),
                "is_correct": (
                    gallery_identities[gallery_index] == correct_identity
                ),
                "gallery_image_id": gallery_item.get("gallery_image_id"),
                "bbox": gallery_item.get("bbox"),
                "pair_id": gallery_item.get("pair_id"),
                "style": gallery_item.get("style")
            })

        query_results.append({
            "query_image_id": query.get("query_image_id"),
            "bbox": query.get("bbox"),
            "pair_id": query.get("pair_id"),
            "style": query.get("style"),
            "category_id": query.get("cls"),
            "first_correct_rank": first_correct_rank,
            "top_results": top_results
        })
        evaluated_queries += 1

    if evaluated_queries == 0:
        raise ValueError("No queries retained a valid gallery match.")

    recall = {
        level: hit_counts[level] / evaluated_queries
        for level in recall_levels
    }
    return recall, hit_counts, query_results


def draw_crop(
    canvas: Image.Image,
    crop: Image.Image,
    column: int,
    border_color: str,
    title: str,
    detail: str
):
    """Draw one labeled garment crop in a diagnostic grid."""
    cell_width = 170
    image_size = 150
    left = column * cell_width + 10
    top = 28

    crop = crop.copy()
    crop.thumbnail((image_size, image_size))
    image_left = left + (image_size - crop.width) // 2
    image_top = top + (image_size - crop.height) // 2
    canvas.paste(crop, (image_left, image_top))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (left, top, left + image_size, top + image_size),
        outline=border_color,
        width=4
    )
    font = ImageFont.load_default()
    draw.text((left, 8), title, fill="black", font=font)
    draw.text((left, top + image_size + 8), detail, fill="black", font=font)


def save_retrieval_grid(
    query_result: dict,
    images_dir: Path,
    output_path: Path
):
    """Save one query beside its ten highest-ranked shop results."""
    top_results = query_result["top_results"]
    canvas = Image.new(
        "RGB",
        ((len(top_results) + 1) * 170, 215),
        color="white"
    )

    query_crop = load_garment_crop(
        query_result,
        image_id_field="query_image_id",
        images_dir=images_dir
    )
    correct_rank = query_result["first_correct_rank"]
    rank_text = str(correct_rank) if correct_rank is not None else "not found"
    draw_crop(
        canvas=canvas,
        crop=query_crop,
        column=0,
        border_color="blue",
        title="QUERY",
        detail=f"Correct rank: {rank_text}"
    )

    for column, result in enumerate(top_results, start=1):
        result_crop = load_garment_crop(
            result,
            image_id_field="gallery_image_id",
            images_dir=images_dir
        )
        border_color = "green" if result["is_correct"] else "red"
        match_text = "MATCH" if result["is_correct"] else "wrong"
        draw_crop(
            canvas=canvas,
            crop=result_crop,
            column=column,
            border_color=border_color,
            title=f"RANK {result['rank']}",
            detail=(
                f"{match_text}  score={result['similarity_score']:.3f}"
            )
        )

    canvas.save(output_path, format="JPEG", quality=92)


def save_diagnostic_grids(
    query_results: list[dict],
    images_dir: Path,
    grids_dir: Path,
    grid_count: int
) -> int:
    """Save failures first so the grids are useful for error analysis."""
    if grid_count == 0:
        return 0

    def priority(result: dict) -> tuple[int, int]:
        rank = result["first_correct_rank"]
        if rank is None:
            return 0, 0
        if rank > 10:
            return 0, rank
        if rank > 1:
            return 1, rank
        return 2, rank

    grids_dir.mkdir(parents=True, exist_ok=True)
    selected_results = sorted(query_results, key=priority)[:grid_count]
    saved_count = 0

    for index, query_result in enumerate(selected_results, start=1):
        image_id = int(query_result["query_image_id"])
        correct_rank = query_result["first_correct_rank"]
        rank_label = correct_rank if correct_rank is not None else "not_found"
        output_path = grids_dir / (
            f"{index:03d}_query_{image_id:06d}_rank_{rank_label}.jpg"
        )

        try:
            save_retrieval_grid(query_result, images_dir, output_path)
            saved_count += 1
        except (KeyError, OSError, TypeError, ValueError) as error:
            print(f"Skipping diagnostic grid for query {image_id}: {error}")

    return saved_count


def main() -> int:
    args = parse_args()
    dataset_root = resolve_from_project(args.dataset_root)
    evaluation_dir = resolve_from_project(args.evaluation_dir)
    output_path = resolve_from_project(args.output)
    grids_dir = resolve_from_project(args.grids_dir)
    images_dir = dataset_root / "validation" / "image"

    queries = load_json_list(evaluation_dir / "val_query.json")
    gallery = load_json_list(evaluation_dir / "val_gallery.json")
    selected_queries, selected_gallery = select_evaluation_subset(
        queries=queries,
        gallery=gallery,
        query_limit=args.query_limit,
        gallery_limit=args.gallery_limit
    )

    print(f"Selected queries: {len(selected_queries)}")
    print(f"Selected gallery items: {len(selected_gallery)}")

    embedding_service = EmbeddingService()
    gallery_items, gallery_embeddings, skipped_gallery = embed_items(
        items=selected_gallery,
        image_id_field="gallery_image_id",
        images_dir=images_dir,
        embedding_service=embedding_service,
        batch_size=args.batch_size,
        label="gallery"
    )
    query_items, query_embeddings, skipped_queries = embed_items(
        items=selected_queries,
        image_id_field="query_image_id",
        images_dir=images_dir,
        embedding_service=embedding_service,
        batch_size=args.batch_size,
        label="queries"
    )

    recall, hit_counts, query_results = calculate_recall(
        queries=query_items,
        query_embeddings=query_embeddings,
        gallery=gallery_items,
        gallery_embeddings=gallery_embeddings
    )

    evaluated_query_count = len(query_results)
    saved_grid_count = save_diagnostic_grids(
        query_results=query_results,
        images_dir=images_dir,
        grids_dir=grids_dir,
        grid_count=args.grid_count
    )

    results = {
        "model": CLIP_MODEL_NAME,
        "query_count": evaluated_query_count,
        "gallery_count": len(gallery_items),
        "skipped_queries": skipped_queries,
        "skipped_gallery_items": skipped_gallery,
        "recall_at_1": recall[1],
        "recall_at_5": recall[5],
        "recall_at_10": recall[10],
        "queries": query_results
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    print("\nZero-shot CLIP retrieval evaluation")
    print(f"Queries evaluated: {evaluated_query_count}")
    print(f"Gallery items: {len(gallery_items)}")
    for level in (1, 5, 10):
        percentage = recall[level] * 100
        print(
            f"Recall@{level}: {hit_counts[level]}/"
            f"{evaluated_query_count} ({percentage:.2f}%)"
        )
    print(f"Results written to: {output_path}")
    print(f"Diagnostic grids saved: {saved_grid_count}")
    if saved_grid_count:
        print(f"Diagnostic grids directory: {grids_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
