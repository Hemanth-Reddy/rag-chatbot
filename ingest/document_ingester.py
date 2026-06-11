# ingest/document_ingester.py
# Feature 6: Document Ingestion
# Loads files, splits into overlapping chunks, stores in ChromaDB.
# Supports: .txt, .pdf, .docx, .md
# Chunk size 512 chars with 64 overlap — good balance of context vs precision.

import logging
import os
from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from vectordb.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx", ".md"}

CHUNK_SIZE = 512       # characters per chunk (~128 tokens)
CHUNK_OVERLAP = 64     # overlap between chunks — preserves context at boundaries
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _load_file(file_path: str) -> list[Document]:
    """Pick the right loader based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def ingest_file(file_path: str) -> dict:
    """
    Ingest a single file into ChromaDB.

    Steps:
        1. Load file with the appropriate loader
        2. Tag each page with source metadata
        3. Split into overlapping chunks
        4. Store chunks in ChromaDB

    Args:
        file_path: Path to the document file.

    Returns:
        Summary dict with chunk count and status.
    """
    file_path = os.path.abspath(file_path)
    ext = Path(file_path).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Ingesting: %s", file_path)
    raw_docs = _load_file(file_path)

    # Tag every page with its source file
    for doc in raw_docs:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["file_path"] = file_path

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        length_function=len,
    )
    chunks = splitter.split_documents(raw_docs)

    # Add chunk position for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(chunks)

    # Store in ChromaDB
    store = get_chroma_store()
    store.add_documents(chunks)

    result = {
        "file": os.path.basename(file_path),
        "raw_pages": len(raw_docs),
        "chunks_created": len(chunks),
        "status": "success",
    }
    logger.info("Ingested '%s': %d chunks.", file_path, len(chunks))
    return result


def ingest_directory(dir_path: str) -> list[dict]:
    """Recursively ingest all supported files from a directory."""
    results = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(root, fname)
                try:
                    results.append(ingest_file(fpath))
                except Exception as e:
                    logger.error("Failed to ingest '%s': %s", fpath, e)
                    results.append({
                        "file": fname,
                        "status": "error",
                        "error": str(e),
                    })
    return results


def ingest_text(text: str, metadata: dict | None = None) -> dict:
    """
    Ingest raw text directly — used by the API's /ingest/text endpoint.

    Args:
        text:     The text content to ingest.
        metadata: Optional source label and other metadata.
    """
    doc = Document(
        page_content=text,
        metadata=metadata or {"source": "direct_input"},
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents([doc])
    store = get_chroma_store()
    store.add_documents(chunks)
    return {
        "chunks_created": len(chunks),
        "status": "success",
    }


def ingest_sample_data() -> None:
    """
    Ingest built-in sample texts for quick testing without real files.
    Run directly:
        python -c "from ingest.document_ingester import ingest_sample_data; ingest_sample_data()"
    """
    samples = [
        (
            "LangChain is an open-source framework for building LLM-powered applications. "
            "It provides tools for chaining prompts, connecting to data sources, and building agents. "
            "Key features include document loaders, text splitters, vector store integrations, "
            "and retrieval chains.",
            {"source": "sample_langchain.txt"},
        ),
        (
            "ChromaDB is a lightweight, open-source vector database designed for AI applications. "
            "It stores text embeddings and supports fast similarity search. "
            "ChromaDB can run fully locally on disk or in-memory, making it ideal for development.",
            {"source": "sample_chromadb.txt"},
        ),
        (
            "Retrieval-Augmented Generation (RAG) is a technique that improves LLM responses "
            "by retrieving relevant documents from a knowledge base before generating an answer. "
            "This prevents hallucinations and keeps answers grounded in factual source material.",
            {"source": "sample_rag.txt"},
        ),
        (
            "FastAPI is a modern Python web framework for building REST APIs. "
            "It uses Python type hints for automatic request validation and OpenAPI docs generation. "
            "FastAPI is built on Starlette and Pydantic, making it fast and developer-friendly.",
            {"source": "sample_fastapi.txt"},
        ),
        (
            "Google Gemini Flash Lite is a lightweight multimodal language model. "
            "It is available through the Gemini API with a generous free tier. "
            "It supports text inputs and is optimized for speed and efficiency.",
            {"source": "sample_gemini.txt"},
        ),
    ]

    store = get_chroma_store()
    total_chunks = 0
    for text, meta in samples:
        doc = Document(page_content=text, metadata=meta)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents([doc])
        store.add_documents(chunks)
        total_chunks += len(chunks)
        logger.info("Ingested: %s", meta["source"])

    print(f"Sample data ready: {len(samples)} documents, {total_chunks} chunks.")