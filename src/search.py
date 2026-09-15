from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)

from src.config import (
    QDRANT_COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT
)


class VectorSearchService:
    """
    Handles Qdrant vector database logic.

    Qdrant stores clothing image embeddings and lets us search for
    the most visually similar items.
    """

    def __init__(self):
        self.collection_name = QDRANT_COLLECTION_NAME
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def create_collection_if_needed(self, vector_size: int):
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            print(f"Creating Qdrant collection: {self.collection_name}")

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
        else:
            print(f"Qdrant collection already exists: {self.collection_name}")

    def delete_collection_if_exists(self):
        """Delete the current collection so it can be rebuilt from scratch."""
        if self.client.collection_exists(self.collection_name):
            print(f"Deleting Qdrant collection: {self.collection_name}")
            self.client.delete_collection(
                collection_name=self.collection_name
            )
        else:
            print(f"Qdrant collection does not exist: {self.collection_name}")

    def upsert_item(self, item: dict, embedding: list[float]):
        point = PointStruct(
            id=item["id"],
            vector=embedding,
            payload=item
        )

        self.upsert_points([point])

    def upsert_points(self, points: list[PointStruct]):
        """Insert or update several Qdrant points in one request."""
        if not points:
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def count_points(self) -> int:
        """Return the exact number of points stored in the collection."""
        count_result = self.client.count(
            collection_name=self.collection_name,
            exact=True
        )

        return count_result.count

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        category: str | None = None
    ) -> list[dict]:
        query_filter = None

        if category:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category)
                    )
                ]
            )

        search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        )

        results = search_response.points

        formatted_results = []

        for result in results:
            item = dict(result.payload)
            item["similarity_score"] = result.score
            formatted_results.append(item)

        return formatted_results
