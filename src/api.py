from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from src.search_index import SearchIndex

load_dotenv()
INDEX_PATH = os.environ.get("INDEX_PATH", "outputs/index.faiss")
META_PATH = os.environ.get("META_PATH", "outputs/meta.npy")

app = FastAPI(title="Semantic Video Search API")

try:
    si = SearchIndex(INDEX_PATH, META_PATH)
except Exception as e:
    si = None

class Query(BaseModel):
    q: str
    k: int = 5

@app.get("/health")
def health():
    return {"ready": si is not None}

@app.post("/search")
def search(query: Query):
    if si is None:
        raise HTTPException(status_code=503, detail="Index not loaded. Run ingestion first.")
    results = si.search(query.q, k=query.k)
    return {"query": query.q, "results": results}
