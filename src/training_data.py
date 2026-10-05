"""Load prepared image triplets for future model training."""

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRIPLET_ROLES = ("anchor", "positive", "negative")


class FashionTripletDataset(Dataset):
    """Open the three images belonging to each prepared training example."""

    def __init__(
        self,
        triplets_path: str | Path,
        transform: Callable[[Image.Image], object] | None = None,
        project_root: str | Path = PROJECT_ROOT
    ):
        self.project_root = Path(project_root)
        self.transform = transform

        manifest_path = Path(triplets_path)
        if not manifest_path.is_absolute():
            manifest_path = self.project_root / manifest_path

        self.triplets = self._load_triplets(manifest_path)

    @staticmethod
    def _load_triplets(manifest_path: Path) -> list[dict]:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Could not find triplet manifest: {manifest_path}"
            )

        with manifest_path.open("r", encoding="utf-8") as file:
            triplets = json.load(file)

        if not isinstance(triplets, list):
            raise ValueError("Triplet manifest must contain a JSON list.")

        for index, triplet in enumerate(triplets):
            if not isinstance(triplet, dict):
                raise ValueError(f"Triplet {index} must be a JSON object.")

            for role in TRIPLET_ROLES:
                item = triplet.get(role)
                if not isinstance(item, dict):
                    raise ValueError(
                        f"Triplet {index} is missing a valid {role} item."
                    )

                image_path = item.get("image_path")
                if not isinstance(image_path, str) or not image_path.strip():
                    raise ValueError(
                        f"Triplet {index} has no {role} image path."
                    )

        return triplets

    def __len__(self) -> int:
        return len(self.triplets)

    def _open_image(self, image_path_value: str, role: str, index: int):
        image_path = Path(image_path_value)
        if not image_path.is_absolute():
            image_path = self.project_root / image_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Could not find {role} image for triplet {index}: "
                f"{image_path}"
            )

        try:
            with Image.open(image_path) as image:
                loaded_image = image.convert("RGB")
        except OSError as error:
            raise OSError(
                f"Could not open {role} image for triplet {index}: "
                f"{image_path}"
            ) from error

        if self.transform is not None:
            return self.transform(loaded_image)

        return loaded_image

    def __getitem__(self, index: int) -> dict:
        triplet = self.triplets[index]

        sample = {
            role: self._open_image(
                image_path_value=triplet[role]["image_path"],
                role=role,
                index=index
            )
            for role in TRIPLET_ROLES
        }

        # Keep the IDs in each sample so a bad training example can be traced
        # back to the generated manifest and its source crops.
        for role in TRIPLET_ROLES:
            sample[f"{role}_id"] = triplet[role].get("id")

        sample["source_category_id"] = triplet.get("source_category_id")
        sample["source_category"] = triplet.get("source_category")

        return sample

