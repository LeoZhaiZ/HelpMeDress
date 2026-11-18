# HelpMeDress

Made by Leo Zhai.

HelpMeDress is an AI fashion-retrieval MVP. A user uploads a clothing image,
CLIP converts it into an embedding, Qdrant finds visually similar catalog
items, FastAPI returns the matches, and Streamlit displays them.

For the complete DeepFashion2 preparation, ingestion, and evaluation process,
see [PIPELINE.md](PIPELINE.md).

## What Is Implemented

- Batched, normalized 512-dimensional CLIP image embeddings
- Batched catalog ingestion into Qdrant
- Metadata and image validation before ingestion
- DeepFashion2 garment cropping and metadata conversion
- DeepFashion2 mappings for tops, bottoms, outerwear, and dresses
- Optional user/shop source filtering during dataset preparation
- FastAPI similarity search and indexed-item count endpoints
- Streamlit API status, similarity search, and outfit UI
- Exact in-memory retrieval evaluation with Recall@1, Recall@5, Recall@10,
  per-query rankings, and nearest-neighbor grids
- Optional category-filtered retrieval evaluation
- Projection-only CLIP triplet-training smoke test with held-out tuning metrics

## Training Status

The project currently uses the pretrained
`openai/clip-vit-base-patch32` model for the application. Preparing images,
creating embeddings, and storing them in Qdrant still do not update CLIP's
weights. `scripts/train_clip_triplet.py` is the first isolated training step:
it freezes the ViT backbone, updates the visual projection with triplet loss,
and saves a checkpoint. The application does not load that checkpoint yet.

For a small training smoke test using the identity-separated manifests:

```bash
python scripts/train_clip_triplet.py
```

The default run uses 64 training triplets and 32 tuning triplets. Increase
`--train-limit` and `--tuning-limit` only after this small run succeeds.

## Quick Start

Run all commands from the project root.

### 1. Set up Python

If the virtual environment already exists:

```bash
source venv/bin/activate
python -m pip install -r requirements.txt
```

If the project is inside an iCloud-synced Documents folder, keep the virtual
environment outside that folder. iCloud can evict package files and leave
`dataless` placeholders, which makes imports appear frozen. One option is:

```bash
python3.11 -m venv ~/.venvs/HelpMeDress
source ~/.venvs/HelpMeDress/bin/activate
python -m pip install -r requirements.txt
```

### 2. Start Qdrant

Make sure Docker is running, then run:

```bash
docker compose up -d
```

### 3. Validate the catalog

The default command checks `data/processed/catalog_metadata.json`:

```bash
python scripts/validate_catalog.py
```

To validate prepared DeepFashion2 shop metadata instead:

```bash
python scripts/validate_catalog.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json
```

The validator checks every metadata record and fully opens a repeatable random
sample of up to 1,000 images by default. This keeps large DeepFashion2 runs
practical. Use `--image-check-limit 5000` for a larger sample, or
`--check-all-images` only when an exhaustive image check is worth the time.

### 4. Ingest the catalog

For a five-item smoke test that rebuilds the Qdrant collection:

```bash
python scripts/batch_ingest.py --limit 5 --reset
```

For a complete clean rebuild using the default catalog:

```bash
python scripts/batch_ingest.py --reset
```

For a prepared DeepFashion2 shop catalog:

```bash
python scripts/batch_ingest.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json \
  --reset
```

`--reset` deletes and recreates only the Qdrant collection. It does not delete
source images, garment crops, or metadata files. Without `--reset`, points are
added or updated using their IDs.

### 5. Start the backend

```bash
uvicorn src.api:app --reload
```

The API runs at `http://127.0.0.1:8000` and provides:

```text
GET  /health
GET  /items/count
POST /search/similar
POST /outfit/generate
```

### 6. Start the frontend

In a second terminal:

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

## Current Retrieval Flow

```text
Catalog image
  -> optional garment crop
  -> batched CLIP embedding
  -> Qdrant vector + metadata payload

Uploaded query image
  -> CLIP embedding
  -> optional category filter
  -> Qdrant cosine-similarity search
  -> FastAPI JSON
  -> Streamlit results
```

Qdrant stores the item ID, embedding vector, and metadata payload. The actual
image remains on disk; its path is stored in the payload.

## Measured Evaluation

The current evaluation uses the first 100 valid DeepFashion2 validation
queries and a selected gallery of 1,000 shop items. These are measured local
results for that subset, not claims about the complete dataset:

| Evaluation | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| CLIP, all garment categories | 17% | 28% | 39% |
| CLIP, same detailed DeepFashion2 category only | 26% | 62% | 79% |

See [PIPELINE.md](PIPELINE.md) for the dataset layout, evaluation commands,
output files, and an explanation of what these numbers mean.

## Remaining Work

- Fine-tune a fashion-specific embedding model and compare it with the CLIP
  baseline
- Improve exact-item matching and reduce background/person interference
- Add garment segmentation
- Evaluate on larger query and gallery subsets
- Replace the rule-based outfit builder with reference-based recommendations
- Replace Streamlit with a React frontend
- Containerize the complete application and deploy it
