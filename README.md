# HelpMeDress
Made by Leo Zhai (Github email commit fix)
# HelpMeDress

HelpMeDress is an AI fashion recommendation MVP. It lets a user upload a clothing image, converts it into a CLIP embedding, searches Qdrant for visually similar clothing items, and displays the results in a Streamlit frontend.

## How to Run

### 1. Activate virtual environment

```bash
source venv/bin/activate
```

### 2. Start Qdrant

Make sure Docker is open, then run:

```bash
docker compose up -d
```

### 3. Ingest catalog items

This embeds the clothing images in `data/raw` and saves them into Qdrant.

```bash
python -m src.ingest
```

### 4. Start the FastAPI backend

```bash
uvicorn src.api:app --reload
```

The backend should now be running at:

```text
http://127.0.0.1:8000
```

You can test it at:

```text
http://127.0.0.1:8000/health
```

### 5. Start the Streamlit frontend

Open a second terminal, activate the virtual environment again, then run:

```bash
source venv/bin/activate
streamlit run frontend/app.py
```

### 6. Use the app

Upload a clothing image, then click:

```text
Find Similar Items
```

or:

```text
Generate Outfit
```

## Current MVP Flow

```text
Image upload
→ CLIP embedding
→ Qdrant vector search
→ FastAPI response
→ Streamlit results
```
