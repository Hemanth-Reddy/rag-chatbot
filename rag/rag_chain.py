# rag/rag_chain.py
# Feature 7: RAG Chain
# Orchestrates all components into one pipeline:
#   PII scrub → ChromaDB retrieve → Gemini LLM → PII scrub → return

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document

from guardrails.pii_guard import get_guardrail, GuardResult
from vectordb.chroma_store import get_chroma_store
# from llm.llm_client import invoke_llm
from llm.provider import get_invoke_fn

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Full structured response from the RAG pipeline.
    Every field is populated — nothing hidden from the caller.
    """
    # The final answer shown to the user (PII scrubbed)
    answer: str

    # The query actually sent to the LLM (PII scrubbed)
    sanitized_query: str

    # Source document chunks used as context
    source_documents: list[dict] = field(default_factory=list)

    # Retrieval stats
    retrieved_chunks: int = 0
    retrieval_scores: list[float] = field(default_factory=list)

    # Guardrail metadata — what PII was found and where
    input_pii_detected: bool = False
    input_pii_entities: list[dict] = field(default_factory=list)
    output_pii_detected: bool = False
    output_pii_entities: list[dict] = field(default_factory=list)


def _format_context(docs_with_scores: list[tuple[Document, float]]) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.
    Each chunk is labeled with its source file and relevance score.
    """
    parts = []
    for i, (doc, score) in enumerate(docs_with_scores, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_str = f", page {page}" if page != "" else ""
        parts.append(
            f"[Document {i} -- {source}{page_str} -- relevance: {score:.2f}]\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def run_rag_query(
    user_query: str,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    apply_guardrails: bool = True,
) -> RAGResponse:
    """
    Execute the full RAG pipeline.

    Steps:
        1. Scrub PII from user query
        2. Search ChromaDB for relevant chunks
        3. Format context string
        4. Call Gemini with context + query
        5. Scrub PII from LLM response
        6. Return structured RAGResponse

    Args:
        user_query:       Raw question from the user.
        top_k:            How many chunks to retrieve (overrides settings).
        score_threshold:  Minimum relevance score 0-1 (overrides settings).
        apply_guardrails: Set False only for debugging — always True in production.
    """
    guardrail = get_guardrail()
    store = get_chroma_store()

    # ── 1. PII scrub on input ──────────────────────────────────────────
    if apply_guardrails:
        input_guard = guardrail.scrub_input(user_query)
        effective_query = input_guard.sanitized_text
    else:
        input_guard = GuardResult(
            original_text=user_query,
            sanitized_text=user_query,
            pii_detected=False,
        )
        effective_query = user_query

    logger.info("Query (post-guardrail): %s", effective_query[:120])

    # ── 2. Retrieve from ChromaDB ──────────────────────────────────────
    docs_with_scores = store.similarity_search(
        query=effective_query,
        k=top_k,
        score_threshold=score_threshold,
    )

    if not docs_with_scores:
        logger.warning("No relevant documents found for query.")
        return RAGResponse(
            answer=(
                "I couldn't find relevant information in the knowledge base "
                "to answer your question. Please try rephrasing, or ingest "
                "documents related to your topic first."
            ),
            sanitized_query=effective_query,
            input_pii_detected=input_guard.pii_detected,
            input_pii_entities=input_guard.detected_entities,
            retrieved_chunks=0,
        )

    # ── 3. Format context ──────────────────────────────────────────────
    context = _format_context(docs_with_scores)

    # ── 4. Call LLM ───────────────────────────────────────────────────
    # raw_answer = invoke_llm(effective_query, context)
    invoke_fn = get_invoke_fn()
    raw_answer = invoke_fn(effective_query, context)

    # ── 5. PII scrub on output ─────────────────────────────────────────
    if apply_guardrails:
        output_guard = guardrail.scrub_output(raw_answer)
        final_answer = output_guard.sanitized_text
    else:
        output_guard = GuardResult(
            original_text=raw_answer,
            sanitized_text=raw_answer,
            pii_detected=False,
        )
        final_answer = raw_answer

    # ── 6. Build and return response ───────────────────────────────────
    source_docs = [
        {
            "content_preview": doc.page_content[:200],
            "metadata": doc.metadata,
            "relevance_score": round(score, 4),
        }
        for doc, score in docs_with_scores
    ]

    return RAGResponse(
        answer=final_answer,
        sanitized_query=effective_query,
        source_documents=source_docs,
        retrieved_chunks=len(docs_with_scores),
        retrieval_scores=[round(s, 4) for _, s in docs_with_scores],
        input_pii_detected=input_guard.pii_detected,
        input_pii_entities=input_guard.detected_entities,
        output_pii_detected=output_guard.pii_detected,
        output_pii_entities=output_guard.detected_entities,
    )