"""Central configuration values for HelpMeDress."""

# Qdrant stores and searches the catalog's image embeddings.
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION_NAME = "helpmedress_items"

# CLIP converts each clothing image into a 512-dimensional vector.
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_SIZE = 512
