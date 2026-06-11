# vectordb/chroma_store.py
# Feature 4: ChromaDB Vector Store
# Stores document embeddings on disk and retrieves similar chunks at query time.
# Data persists between sessions — ingest once, query forever.

import logging
import os
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from embeddings.embedder import get_embeddings

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Local ChromaDB vector store.

    Public methods:
        add_documents(docs)      -> embed and persist document chunks
        similarity_search(query) -> retrieve top-k similar chunks with scores
        get_retriever()          -> LangChain-compatible retriever object
        get_stats()              -> collection info (doc count, location)
        delete_collection()      -> wipe everything (dev/test only)
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        self._store: Optional[Chroma] = None

    def _get_store(self) -> Chroma:
        """Lazy-initialize — only opens ChromaDB when first needed."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=get_embeddings(),
                persist_directory=self.persist_dir,
                collection_metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB ready — collection: '%s', dir: '%s'",
                self.collection_name,
                self.persist_dir,
            )
        return self._store

    # ── Write ──────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Embed and persist a list of Document objects.
        Returns the generated IDs for each stored chunk.
        """
        store = self._get_store()
        ids = store.add_documents(documents)
        logger.info("Stored %d chunks in ChromaDB.", len(documents))
        return ids

    def delete_collection(self) -> None:
        """
        Wipe the entire collection from disk.
        Use only during development or testing — irreversible.
        """
        client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(self.collection_name)
            logger.info("Deleted collection: %s", self.collection_name)
        except Exception:
            logger.warning(
                "Collection '%s' did not exist — nothing deleted.",
                self.collection_name,
            )
        self._store = None  # force re-init on next use

    # ── Read ───────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[tuple[Document, float]]:
        """
        Find the top-k document chunks most similar to the query.

        Returns list of (Document, score) tuples.
        Score is cosine similarity 0.0-1.0. Higher = more relevant.
        Chunks scoring below score_threshold are filtered out.
        """
        store = self._get_store()
        top_k = k or settings.rag_top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else settings.rag_score_threshold
        )

        raw_results = store.similarity_search_with_relevance_scores(
            query, k=top_k
        )
        filtered = [
            (doc, score) for doc, score in raw_results if score >= threshold
        ]

        logger.debug(
            "Query matched %d/%d chunks above threshold %.2f",
            len(filtered), len(raw_results), threshold,
        )
        return filtered

    def get_retriever(self, k: Optional[int] = None):
        """
        Return a LangChain-compatible retriever.
        Used by LangChain chains — not needed for our custom RAG chain.
        """
        store = self._get_store()
        return store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k or settings.rag_top_k,
                "score_threshold": settings.rag_score_threshold,
            },
        )

    def get_stats(self) -> dict:
        """Return current collection statistics."""
        store = self._get_store()
        count = store._collection.count()
        return {
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "document_count": count,
        }


# Module-level singleton
_chroma_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """Return the shared ChromaStore instance (lazy-initialized)."""
    global _chroma_store
    if _chroma_store is None:
        _chroma_store = ChromaStore()
    return _chroma_store