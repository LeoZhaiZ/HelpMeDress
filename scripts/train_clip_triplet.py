"""Smoke-train CLIP's visual projection with fashion triplet loss."""

import argparse
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TRIPLET_ROLES = ("anchor", "positive", "negative")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train CLIP's visual projection on prepared fashion triplets."
        )
    )
    parser.add_argument(
        "--train-triplets",
        default="data/processed/deepfashion2_train_triplets.json",
        help="Training triplet manifest."
    )
    parser.add_argument(
        "--tuning-triplets",
        default="data/processed/deepfashion2_tuning_triplets.json",
        help="Identity-separated tuning triplet manifest."
    )
    parser.add_argument(
        "--output",
        default="data/processed/checkpoints/clip_triplet_smoke.pt",
        help="Where to save the trained visual projection checkpoint."
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=64,
        help="Training triplets used by this run (default: 64)."
    )
    parser.add_argument(
        "--tuning-limit",
        type=int,
        default=32,
        help="Tuning triplets used by this run (default: 32)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Triplets per batch; CLIP sees three times this many images."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes through the selected training triplets."
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate for the visual projection."
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
        help="Required cosine-distance gap between positive and negative."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for repeatable sampling and batch order."
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Training device (default: auto)."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0 for macOS safety)."
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Print training progress every N batches."
    )

    args = parser.parse_args()
    positive_integer_fields = (
        "train_limit",
        "tuning_limit",
        "batch_size",
        "epochs",
        "log_every"
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.margin < 0:
        parser.error("--margin cannot be negative.")

    return args


def resolve_from_project(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def choose_subset_indexes(length: int, limit: int, seed: int) -> list[int]:
    """Choose a repeatable sample without changing the source manifest."""
    if limit >= length:
        return list(range(length))

    random_generator = random.Random(seed)
    return sorted(random_generator.sample(range(length), limit))


def collate_triplets(samples: list[dict]) -> dict[str, list]:
    """Keep PIL images in lists until the CLIP processor batches them."""
    batch = {
        role: [sample[role] for sample in samples]
        for role in TRIPLET_ROLES
    }
    for role in TRIPLET_ROLES:
        batch[f"{role}_id"] = [
            sample[f"{role}_id"]
            for sample in samples
        ]
    return batch


def choose_device(torch, requested_device: str):
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")

    if requested_device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")

    if requested_device == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def freeze_everything_except_projection(model) -> tuple[int, int]:
    """Freeze the ViT and leave only the final image projection trainable."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.visual_projection.parameters():
        parameter.requires_grad = True

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total


def encode_triplet_batch(torch, functional, model, processor, batch, device):
    """Encode all three triplet roles with the same model weights."""
    batch_size = len(batch["anchor"])
    images = [
        image
        for role in TRIPLET_ROLES
        for image in batch[role]
    ]
    pixel_values = processor(
        images=images,
        return_tensors="pt"
    ).pixel_values.to(device)

    outputs = model(pixel_values=pixel_values)
    embeddings = functional.normalize(outputs.image_embeds, p=2, dim=-1)
    return embeddings.split(batch_size, dim=0)


def cosine_triplet_loss(torch, functional, anchor, positive, negative, margin):
    """Return mean triplet loss plus distances used for reporting."""
    positive_distances = 1 - (anchor * positive).sum(dim=-1)
    negative_distances = 1 - (anchor * negative).sum(dim=-1)
    losses = functional.relu(
        positive_distances - negative_distances + margin
    )
    return losses.mean(), positive_distances, negative_distances, losses


def run_epoch(
    torch,
    functional,
    model,
    processor,
    loader,
    device,
    margin: float,
    optimizer=None,
    log_every: int = 5
) -> dict[str, float]:
    """Train when an optimizer is provided; otherwise only measure loss."""
    is_training = optimizer is not None
    model.train(is_training)

    # The ViT is frozen, so keeping it in evaluation mode makes its output
    # stable while the projection matrix learns.
    model.vision_model.eval()

    total_loss = 0.0
    total_positive_distance = 0.0
    total_negative_distance = 0.0
    total_active_triplets = 0
    total_triplets = 0

    for batch_number, batch in enumerate(loader, start=1):
        if is_training:
            # PyTorch accumulates gradients by default. Clearing them here
            # ensures this batch updates the weights using only its own loss.
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            anchor, positive, negative = encode_triplet_batch(
                torch=torch,
                functional=functional,
                model=model,
                processor=processor,
                batch=batch,
                device=device
            )
            loss, positive_distances, negative_distances, losses = (
                cosine_triplet_loss(
                    torch=torch,
                    functional=functional,
                    anchor=anchor,
                    positive=positive,
                    negative=negative,
                    margin=margin
                )
            )

        if is_training:
            # Backpropagation follows every operation that produced `loss`
            # and stores d(loss)/d(weight) in each trainable parameter's grad.
            loss.backward()

            # AdamW uses those gradients to take one small step that should
            # reduce the loss on future batches.
            optimizer.step()

        batch_triplets = len(batch["anchor"])
        total_triplets += batch_triplets
        total_loss += loss.item() * batch_triplets
        total_positive_distance += positive_distances.sum().item()
        total_negative_distance += negative_distances.sum().item()
        total_active_triplets += (losses > 0).sum().item()

        if is_training and (
            batch_number % log_every == 0
            or batch_number == len(loader)
        ):
            print(
                f"Training batch {batch_number}/{len(loader)} | "
                f"loss {total_loss / total_triplets:.4f}",
                flush=True
            )

    return {
        "loss": total_loss / total_triplets,
        "positive_distance": total_positive_distance / total_triplets,
        "negative_distance": total_negative_distance / total_triplets,
        "active_triplet_fraction": total_active_triplets / total_triplets,
        "triplets": total_triplets
    }


def print_metrics(label: str, metrics: dict[str, float]):
    print(label)
    print(f"  Triplets: {metrics['triplets']}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(
        "  Anchor-positive cosine distance: "
        f"{metrics['positive_distance']:.4f}"
    )
    print(
        "  Anchor-negative cosine distance: "
        f"{metrics['negative_distance']:.4f}"
    )
    print(
        "  Triplets still violating the margin: "
        f"{metrics['active_triplet_fraction']:.1%}"
    )


def main() -> int:
    args = parse_args()

    # Heavy ML imports happen after argument parsing so `--help` stays fast and
    # environment problems are easier to locate.
    print("Importing PyTorch and Transformers...", flush=True)
    import torch
    import torch.nn.functional as functional
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Subset
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    from src.config import CLIP_MODEL_NAME
    from src.training_data import FashionTripletDataset

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device(torch, args.device)
    print(f"Using device: {device}", flush=True)

    training_dataset = FashionTripletDataset(
        resolve_from_project(args.train_triplets)
    )
    tuning_dataset = FashionTripletDataset(
        resolve_from_project(args.tuning_triplets)
    )
    training_indexes = choose_subset_indexes(
        len(training_dataset),
        args.train_limit,
        args.seed
    )
    tuning_indexes = choose_subset_indexes(
        len(tuning_dataset),
        args.tuning_limit,
        args.seed + 1
    )

    generator = torch.Generator().manual_seed(args.seed)
    loader_settings = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_triplets,
        "pin_memory": device.type == "cuda"
    }
    training_loader = DataLoader(
        Subset(training_dataset, training_indexes),
        shuffle=True,
        generator=generator,
        **loader_settings
    )
    tuning_loader = DataLoader(
        Subset(tuning_dataset, tuning_indexes),
        shuffle=False,
        **loader_settings
    )

    print(f"Loading pretrained model: {CLIP_MODEL_NAME}", flush=True)
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME)
    model = CLIPVisionModelWithProjection.from_pretrained(
        CLIP_MODEL_NAME
    ).to(device)
    trainable_parameters, total_parameters = (
        freeze_everything_except_projection(model)
    )
    print(
        f"Trainable parameters: {trainable_parameters:,}/"
        f"{total_parameters:,} (visual projection only)",
        flush=True
    )

    optimizer = AdamW(
        model.visual_projection.parameters(),
        lr=args.learning_rate
    )
    initial_projection_weight = (
        model.visual_projection.weight.detach().cpu().clone()
    )

    baseline_metrics = run_epoch(
        torch=torch,
        functional=functional,
        model=model,
        processor=processor,
        loader=tuning_loader,
        device=device,
        margin=args.margin
    )
    print_metrics("Tuning metrics before training", baseline_metrics)

    training_metrics = None
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        training_metrics = run_epoch(
            torch=torch,
            functional=functional,
            model=model,
            processor=processor,
            loader=training_loader,
            device=device,
            margin=args.margin,
            optimizer=optimizer,
            log_every=args.log_every
        )
        print_metrics("Training metrics", training_metrics)

    final_tuning_metrics = run_epoch(
        torch=torch,
        functional=functional,
        model=model,
        processor=processor,
        loader=tuning_loader,
        device=device,
        margin=args.margin
    )
    print_metrics("Tuning metrics after training", final_tuning_metrics)
    projection_weight_change = torch.linalg.vector_norm(
        model.visual_projection.weight.detach().cpu()
        - initial_projection_weight
    ).item()
    print(
        "Visual projection weight change (L2 norm): "
        f"{projection_weight_change:.6f}"
    )

    output_path = resolve_from_project(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    projection_state = {
        key: value.detach().cpu()
        for key, value in model.visual_projection.state_dict().items()
    }
    torch.save({
        "model_name": CLIP_MODEL_NAME,
        "visual_projection_state_dict": projection_state,
        "margin": args.margin,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "training_metrics": training_metrics,
        "baseline_tuning_metrics": baseline_metrics,
        "final_tuning_metrics": final_tuning_metrics,
        "projection_weight_change": projection_weight_change
    }, output_path)
    print(f"Checkpoint written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
