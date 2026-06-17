"""
agent/tools.py
Tool implementations for the agentic RAG loop.
Each function is called by the agent when Gemini requests it.
"""

import logging
from datetime import datetime

from duckduckgo_search import DDGS

from vectordb.chroma_store import get_chroma_store

logger = logging.getLogger(__name__)

# Max chars returned per tool to avoid filling the context window
_KB_CHUNK_PREVIEW = 500
_WEB_BODY_PREVIEW = 400
_WEB_MAX_RESULTS = 5
_KB_MAX_RESULTS = 5


def search_knowledge_base(query: str) -> str:
    """
    Search the local ChromaDB knowledge base.
    Returns formatted chunks with source and relevance score.
    """
    store = get_chroma_store()
    results = store.similarity_search(query, k=_KB_MAX_RESULTS)

    if not results:
        return "No relevant documents found in the knowledge base for this query."

    parts = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(
            f"[Chunk {i}] Source: {source} | Relevance: {score:.2f}\n"
            f"{doc.page_content[:_KB_CHUNK_PREVIEW]}"
        )

    logger.info("search_knowledge_base: %d chunks returned for query: %s", len(results), query[:80])
    return "\n\n".join(parts)


def web_search(query: str) -> str:
    """
    Search the web via DuckDuckGo (no API key required).
    Returns top results with title, URL, and snippet.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=_WEB_MAX_RESULTS))

        if not results:
            return "No web results found for this query."

        parts = []
        for r in results:
            parts.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('href', 'N/A')}\n"
                f"{r.get('body', '')[:_WEB_BODY_PREVIEW]}"
            )

        logger.info("web_search: %d results returned for query: %s", len(results), query[:80])
        return "\n\n".join(parts)

    except Exception as e:
        logger.error("web_search failed: %s", e)
        return f"Web search failed: {e}"


def get_current_date() -> str:
    """Return the current date and time in ISO format."""
    now = datetime.now().isoformat()
    logger.info("get_current_date called: %s", now)
    return now


# Registry mapping Gemini function names → Python callables
TOOL_REGISTRY: dict[str, callable] = {
    "search_knowledge_base": search_knowledge_base,
    "web_search": web_search,
    "get_current_date": get_current_date,
}
