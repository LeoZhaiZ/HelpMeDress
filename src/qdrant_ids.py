"""Convert catalog IDs into values accepted by Qdrant."""

from uuid import NAMESPACE_URL, uuid5


def to_qdrant_id(item_id: object) -> int | str:
    """Return an integer ID or a deterministic UUID string."""
    if isinstance(item_id, int) and not isinstance(item_id, bool):
        return item_id

    if isinstance(item_id, str) and item_id.isdigit():
        return int(item_id)

    return str(uuid5(NAMESPACE_URL, f"helpmedress:{item_id}"))

