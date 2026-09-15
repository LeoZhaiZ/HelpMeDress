# HelpMeDress

Made by Leo Zhai.

HelpMeDress is an AI fashion recommendation MVP. A user uploads a clothing
image, CLIP converts it into an embedding, Qdrant finds visually similar
catalog items, and Streamlit displays the results.

## How to Run

### 1. Activate the virtual environment

```bash
source venv/bin/activate
```

### 2. Start Qdrant

Make sure Docker is running, then start the local vector database:

```bash
docker compose up -d
```

### 3. Ingest catalog items

The batch ingestion script reads `data/processed/catalog_metadata.json`,
embeds the images referenced in `data/raw`, and stores their vectors and
metadata in Qdrant.

For a quick test, rebuild the collection with at most five catalog items:

```bash
python scripts/batch_ingest.py --limit 5 --reset
```

For a complete clean rebuild of the collection:

```bash
python scripts/batch_ingest.py --reset
```

To add or update catalog items without deleting the existing collection:

```bash
python scripts/batch_ingest.py
```

The `--reset` option deletes the existing Qdrant collection before recreating
and ingesting it. It removes indexed vectors and payloads, but it does not
delete the image files or metadata JSON.

Optional ingestion arguments:

```text
--metadata PATH     Metadata JSON path (default: data/processed/catalog_metadata.json)
--batch-size NUMBER Images processed together (default: 32)
--limit NUMBER      Maximum number of metadata records to attempt
--reset             Rebuild the Qdrant collection from scratch
```

### 4. Start the FastAPI backend

```bash
uvicorn src.api:app --reload
```

The backend runs at `http://127.0.0.1:8000`.

Useful endpoints:

```text
GET  /health
GET  /items/count
POST /search/similar
POST /outfit/generate
```

For example, check the number of indexed items with:

```bash
curl http://127.0.0.1:8000/items/count
```

### 5. Start the Streamlit frontend

Open a second terminal, activate the virtual environment again, then run:

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

The sidebar shows whether the API is reachable and how many catalog items are
currently indexed. Upload a clothing image to find similar items or generate
outfit recommendations.

## Current MVP Flow

```text
Image upload
→ batched or single-image CLIP embedding
→ Qdrant vector search
→ FastAPI response
→ Streamlit results
```

Qdrant stores item IDs, embedding vectors, and metadata payloads. The actual
image files remain in `data/raw`.

## Future Work

The following features are planned but are not implemented yet:

- Prepare larger datasets such as DeepFashion2 or Consumer-to-Shop.
- Measure retrieval with Recall@1, Recall@5, Recall@10, and neighbor grids.
- Train or fine-tune a visual style embedding model with Triplet Loss.
- Add SAM-based garment segmentation.
- Add reference-based outfit recommendations and shoppable alternatives.
- Replace Streamlit with a React frontend.
- Containerize the backend and deploy the application on AWS.

Target metrics and catalog sizes will only be reported after they are actually
measured.
