class OutfitBuilder:
    """
    Simple rule-based outfit builder.

    This is not advanced AI yet. It just decides what clothing slots
    are missing based on the item the user uploaded.
    """

    def get_needed_slots(self, anchor_category: str) -> list[str]:
        if anchor_category == "top":
            return ["bottom", "shoes", "accessory"]

        if anchor_category == "bottom":
            return ["top", "shoes", "accessory"]

        if anchor_category == "shoes":
            return ["top", "bottom", "accessory"]

        if anchor_category == "outerwear":
            return ["top", "bottom", "shoes"]

        if anchor_category == "accessory":
            return ["top", "bottom", "shoes"]

        return ["top", "bottom", "shoes"]