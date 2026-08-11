from io import BytesIO
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.embed import EmbeddingService
from src.search import VectorSearchService
from src.outfit_builder import OutfitBuilder


app = FastAPI(title="HelpMeDress API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_service = EmbeddingService()
search_service = VectorSearchService()
outfit_builder = OutfitBuilder()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "HelpMeDress API is running"
    }


@app.post("/search/similar")
async def search_similar(
    file: UploadFile = File(...),
    top_k: int = Form(5),
    category: Optional[str] = Form(None)
):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        query_embedding = embedding_service.embed_image(image)

        results = search_service.search_similar(
            query_vector=query_embedding,
            top_k=top_k,
            category=category if category else None
        )

        return {
            "query_filename": file.filename,
            "results": results
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/outfit/generate")
async def generate_outfit(
    file: UploadFile = File(...),
    anchor_category: str = Form("top")
):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        query_embedding = embedding_service.embed_image(image)

        needed_slots = outfit_builder.get_needed_slots(anchor_category)

        outfit = {
            "anchor_category": anchor_category,
            "needed_slots": needed_slots,
            "items": {}
        }

        for slot in needed_slots:
            matches = search_service.search_similar(
                query_vector=query_embedding,
                top_k=3,
                category=slot
            )

            outfit["items"][slot] = matches

        return outfit

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))