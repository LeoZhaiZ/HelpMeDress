from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)


class VectorSearchService:
    """
    Handles Qdrant vector database logic.

    Qdrant stores clothing image embeddings and lets us search for
    the most visually similar items.
    """

    def __init__(self):
        self.collection_name = "helpmedress_items"
        self.client = QdrantClient(host="localhost", port=6333)

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

    def upsert_item(self, item: dict, embedding: list[float]):
        point = PointStruct(
            id=item["id"],
            vector=embedding,
            payload=item
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

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

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
        )

        formatted_results = []

        for result in results:
            item = dict(result.payload)
            item["similarity_score"] = result.score
            formatted_results.append(item)

        return formatted_results