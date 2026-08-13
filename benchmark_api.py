from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speculative Vector Search Benchmark API")

# Configuration (matches chunker.py)
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "msmarco_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Global state
model = None
qdrant = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResponse(BaseModel):
    latency_ms: float
    results: list
    message: str = "Success"

@app.on_event("startup")
async def startup_event():
    global model, qdrant
    logger.info("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Connecting to Qdrant...")
    qdrant = QdrantClient(path=QDRANT_PATH)

@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if not model or not qdrant:
        raise HTTPException(status_code=500, detail="Models not loaded")

    start_time = time.perf_counter()
    
    # 1. Embed query
    vector = model.encode(req.query).tolist()
    
    # 2. Search Qdrant
    search_results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=req.top_k
    )
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000
    
    results = []
    for hit in search_results:
        results.append({
            "score": hit.score,
            "child_chunk": hit.payload.get("child_chunk"),
            "parent_context": hit.payload.get("parent_context"),
            "language": hit.payload.get("language")
        })
        
    return SearchResponse(
        latency_ms=latency_ms,
        results=results,
        message=f"Search completed in {latency_ms:.2f} ms"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
