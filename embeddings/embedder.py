# embeddings/embedder.py
# Feature 3: Local Embeddings
# Converts text to vectors using a local sentence-transformer model.
# No API key required. Model downloads once (~90MB) and is cached locally.

import logging
from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings

logger = logging.getLogger(__name__)

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return the shared embedding model instance.
    First call downloads the model (~90MB) — takes 30-60 seconds.
    Every call after that is instant — model stays in memory.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Embedding model loaded successfully.")
    return _embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings.
    Returns a list of float vectors — one vector per input string.
    Used when ingesting documents into ChromaDB.
    """
    return get_embeddings().embed_documents(texts)


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Returns one float vector.
    Used at query time to find similar document chunks.
    """
    return get_embeddings().embed_query(query)