# HelpMeDress pipeline notes

This is the practical version of how the project works and how to run it. It
also covers the problems I ran into while moving from a tiny test catalog to
DeepFashion2.

uses the pretrained `openai/clip-vit-base-patch32` model to create embeddings. Preparing images,
creating vectors, and putting those vectors in Qdrant are data preparation and
inference, not training

The current pipeline is basically:


DeepFashion2 images and annotations
1. crop out each garment
2. create catalog metadata
3. check the metadata and images
4. run the crops through CLIP
5. store the vectors and metadata in Qdrant
6. search through FastAPI and Streamlit

Getting DeepFashion2 ready

For the work so far, I used `validation.zip` and
`json_for_validation.zip`. The validation file contains the images and their
annotations, while the other file tells us which consumer photos should match
which shop photos.

`train.zip` is not required for the current evaluation, but it will be needed
when I build a larger catalog or start actual fine-tuning. `test.zip` is not
needed right now.

ignore dataset on git because of size

The original images are not ideal inputs for retrieval. They often contain a
person, a background, a phone, and sometimes more than one piece of clothing.
The preparation script uses DeepFashion2's `[x1, y1, x2, y2]` bounding box to
cut out just the garment. That smaller image is what I mean by a garment
crop.

I normally test the script on a few annotations first:

```bash
python scripts/prepare_deepfashion2.py \
  --split validation \
  --source shop \
  --limit 10
```

An image can have no supported garments or more
than one garment, so the crop count will not always equal the limit.

Once the test works, this prepares every shop image in the validation split:

```bash
python scripts/prepare_deepfashion2.py \
  --split validation \
  --source shop
```

The output is going to be put in

```text
data/processed/deepfashion2_validation_shop_metadata.json
data/processed/deepfashion2_crops/validation/
```

I added the `--source` option after noticing that consumer photos and clean
shop photos were being mixed together. `--source shop` is useful for building
the searchable catalog, while `--source user` can be used when I specifically
want consumer images.

Another problem was that dress annotations were being skipped. DeepFashion2
uses category IDs 10 through 13 for different kinds of dresses, so I added all
four to the category map. The current broader categories are:

- IDs 1, 2, 5, and 6 become `top`
- IDs 3 and 4 become `outerwear`
- IDs 7, 8, and 9 become `bottom`
- IDs 10, 11, 12, and 13 become `dress`

The original detailed category is still kept in the metadata, so mapping it to
a simpler app category does not throw that information away.

Before sending thousands of records into Qdrant, I run the validator:

```bash
python scripts/validate_catalog.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json
```

This catches missing fields, missing or broken images, unsupported categories,
duplicate IDs, and Qdrant ID collisions. I added it so i can catch the issue in the beginning

to start qdrant

```bash
docker compose up -d
```

Then i tried a five-item ingestion to just test it

```bash
python scripts/batch_ingest.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json \
  --limit 5 \
  --reset
```

If that works, remove the limit:

```bash
python scripts/batch_ingest.py \
  --metadata data/processed/deepfashion2_validation_shop_metadata.json \
  --reset
```

`--reset` only deletes and rebuilds the Qdrant collection. It does not delete
the dataset, the crops, or the metadata file. I use it after a small test so
the five test points do not remain mixed into the full rebuild. If I leave it
off, existing points are updated and new points are added.

The first version embedded one image at a time. That worked for a tiny catalog
but would waste a lot of time on a larger dataset, so I changed it to send a
batch of images through CLIP together and then upsert the whole batch into
Qdrant. CLIP is also loaded only once for the entire run.

I also had to deal with Qdrant's ID rules. It accepts integers or UUIDs, but
dataset IDs can be arbitrary strings. Numeric strings are converted to
integers, and other strings are converted to deterministic UUIDs. This means
the same catalog item gets the same Qdrant ID every time the script runs.

Qdrant stores the ID, the 512-number CLIP vector, and the metadata. It does not
store the actual JPEG; the metadata points back to the crop on disk.

How to run the app

Start the API in one terminal:

```bash
source venv/bin/activate
uvicorn src.api:app --reload
```

Start Streamlit in a second terminal:

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

I added `GET /items/count` and the Streamlit status panel because it was hard
to tell whether the backend was actually connected or whether Qdrant contained
anything.

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/items/count
```

When someone uploads an image, the API converts it to RGB, creates a normalized
CLIP vector, and asks Qdrant for the nearest catalog vectors using cosine
similarity. The optional category choice filters the catalog before returning
the highest-ranked items.

How I tested whether retrieval was actually working:

At first, the app could return results, but that did not tell me whether those
results were correct. I added evaluate_retrieval.py so there would be
an actual baseline

In the evaluation, a **query** is a consumer photo of a garment. The gallery is
the collection of shop garments it is searched against. DeepFashion2 tells us
which shop garment is the real match through its pair and style IDs.

The evaluator crops the query and gallery garments, embeds them with CLIP, and
compares the query vector directly with every gallery vector in memory. Since
the vectors are normalized, their dot product is cosine similarity. This is
called exact in-memory search because every selected gallery item is checked;
Qdrant's approximate index is not involved.

Run the original baseline with:

```bash
python scripts/evaluate_retrieval.py
```

By default, that uses 100 valid queries and 1,000 gallery items. It saves the
overall Recall@1, Recall@5, and Recall@10 results, each query's rankings, and 20
image grids under `data/processed/`.

Looking at those grids showed that CLIP often understood the general color,
pattern, or clothing type but still missed the exact garment. It also sometimes
ranked a completely different type of clothing above the correct result.

To check how much of the failure was caused by that category confusion, I added:

```bash
python scripts/evaluate_retrieval.py --category-filter
```

This version only compares a query with gallery items from the same detailed
DeepFashion2 category. It is separate from the broader `top`, `bottom`,
`outerwear`, and `dress` mapping assigned to this DeepFashion2 catalog.

These are the results measured on the same 100-query/1,000-gallery subset:

# ALL TRAININGS AND STUFF
First training:
  All categories:
    Recall@1: 17%
    Recall@5: 28%
    Recall@10: 39%
  Same detailed category only:
    Recall@1: 26%
    Recall@5: 62%
    Recall@10: 79%

Recall@5 means that at least one correct shop match appeared in the first five
results. The jump from 28% to 62% showed that category confusion caused a lot
of failures. The category filter helped, but the 26% Recall@1 also showed that
general-purpose CLIP still has trouble recognizing the exact same garment.

This is all i have done for now :salute: its gna be updated the more i work on it. also i think training stats are going to be put here too


