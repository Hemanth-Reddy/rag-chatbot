# api/main.py
# Feature 8: REST API
# FastAPI server exposing all RAG functionality as HTTP endpoints.
# Warms up slow components (spaCy, ChromaDB) at startup — not per request.

import logging
import os
import shutil
import sys
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from guardrails.pii_guard import get_guardrail
from ingest.document_ingester import ingest_file, ingest_text
from rag.rag_chain import run_rag_query
from agent.agent_chain import run_agentic_query
from vectordb.chroma_store import get_chroma_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s -- %(message)s",
)
logger = logging.getLogger(__name__)


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    apply_guardrails: bool = Field(True)


class QueryResponse(BaseModel):
    answer: str
    sanitized_query: str
    source_documents: list[dict]
    retrieved_chunks: int
    retrieval_scores: list[float]
    guardrails: dict


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10)
    source_name: str = Field("manual_input")


class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    apply_guardrails: bool = Field(True)


class AgentQueryResponse(BaseModel):
    answer: str
    sanitized_query: str
    tool_calls: list[dict]
    guardrails: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    collection: str


# ── Lifespan: warm up on startup ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up -- warming components...")
    get_guardrail()     # loads spaCy NLP model
    get_chroma_store()  # opens ChromaDB
    logger.info("RAG API is ready.")
    yield
    logger.info("Shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation with PII guardrails, ChromaDB, and Gemini",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error("Unhandled exception:\n%s", traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — confirms API is running."""
    return HealthResponse(
        status="ok",
        model=settings.llm_model,
        collection=settings.chroma_collection_name,
    )


@app.get("/stats", tags=["System"])
async def stats():
    """Return vector store statistics — doc count, collection name, location."""
    return get_chroma_store().get_stats()


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Query the RAG system.
    PII is scrubbed from both the query and the response automatically.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = run_rag_query(
        user_query=request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        apply_guardrails=request.apply_guardrails,
    )

    return QueryResponse(
        answer=result.answer,
        sanitized_query=result.sanitized_query,
        source_documents=result.source_documents,
        retrieved_chunks=result.retrieved_chunks,
        retrieval_scores=result.retrieval_scores,
        guardrails={
            "input_pii_detected": result.input_pii_detected,
            "input_pii_entities": result.input_pii_entities,
            "output_pii_detected": result.output_pii_detected,
            "output_pii_entities": result.output_pii_entities,
        },
    )


@app.post("/agent/query", response_model=AgentQueryResponse, tags=["Agentic RAG"])
async def agent_query(request: AgentQueryRequest):
    """
    Agentic RAG query. The LLM decides which tools to call:
    search_knowledge_base, web_search, or get_current_date.
    Falls back to web search when the local knowledge base has no relevant results.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = run_agentic_query(
        user_query=request.query,
        apply_guardrails=request.apply_guardrails,
    )

    return AgentQueryResponse(
        answer=result.answer,
        sanitized_query=result.sanitized_query,
        tool_calls=result.tool_calls,
        guardrails={
            "input_pii_detected": result.input_pii_detected,
            "input_pii_entities": result.input_pii_entities,
            "output_pii_detected": result.output_pii_detected,
            "output_pii_entities": result.output_pii_entities,
        },
    )


@app.post("/ingest/file", tags=["Ingestion"])
async def ingest_file_endpoint(file: UploadFile = File(...)):
    """Upload a PDF, TXT, DOCX, or MD file and ingest it into ChromaDB."""
    allowed = {".pdf", ".txt", ".docx", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = ingest_file(tmp_path)
        result["original_filename"] = file.filename
        return result
    finally:
        os.unlink(tmp_path)


@app.post("/ingest/text", tags=["Ingestion"])
async def ingest_text_endpoint(request: IngestTextRequest):
    """Ingest raw text directly into ChromaDB."""
    return ingest_text(
        text=request.text,
        metadata={"source": request.source_name},
    )


@app.delete("/collection", tags=["System"])
async def reset_collection():
    """Delete all documents from the vector store. Development only."""
    get_chroma_store().delete_collection()
    return {"status": "reset", "collection": settings.chroma_collection_name}


# ── Direct run ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )