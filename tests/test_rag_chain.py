# tests/test_rag_chain.py
# Feature 7 test — LLM is mocked, no API calls needed.
# ChromaDB uses an isolated temp store pre-loaded with known test docs.

import os
import sys
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.documents import Document
from vectordb.chroma_store import ChromaStore

MOCK_ANSWER = "LangChain is a framework for building LLM-powered applications."


@pytest.fixture(scope="module")
def populated_store():
    """Isolated ChromaStore pre-loaded with known test documents."""
    tmpdir = tempfile.mkdtemp()
    store = ChromaStore(persist_dir=tmpdir, collection_name="test_rag")
    store.add_documents([
        Document(
            page_content="LangChain is an open-source framework for LLM applications.",
            metadata={"source": "langchain.txt"},
        ),
        Document(
            page_content="ChromaDB stores embeddings for fast similarity search.",
            metadata={"source": "chromadb.txt"},
        ),
        Document(
            page_content="RAG combines retrieval with generation to reduce hallucinations.",
            metadata={"source": "rag.txt"},
        ),
        Document(
            page_content="FastAPI is a modern Python framework for building REST APIs.",
            metadata={"source": "fastapi.txt"},
        ),
    ])
    yield store
    try:
        store._store = None
        import gc
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


class TestRAGResponseStructure:
    def test_response_has_answer(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query("What is LangChain?", apply_guardrails=False)
        assert result.answer == MOCK_ANSWER

    def test_response_has_sanitized_query(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query("What is LangChain?", apply_guardrails=False)
        assert result.sanitized_query == "What is LangChain?"

    def test_response_has_source_documents(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query(
                    "LangChain framework", score_threshold=0.0, apply_guardrails=False
                )
        assert isinstance(result.source_documents, list)
        if result.source_documents:
            assert "content_preview" in result.source_documents[0]
            assert "relevance_score" in result.source_documents[0]
            assert "metadata" in result.source_documents[0]

    def test_response_has_retrieval_stats(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query(
                    "ChromaDB", score_threshold=0.0, apply_guardrails=False
                )
        assert isinstance(result.retrieved_chunks, int)
        assert isinstance(result.retrieval_scores, list)


class TestRAGGuardrails:
    def test_pii_scrubbed_from_input(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query(
                    "My name is John Doe, what is LangChain?",
                    apply_guardrails=True,
                )
        assert result.input_pii_detected
        assert "John Doe" not in result.sanitized_query

    def test_pii_scrubbed_from_output(self, populated_store):
        pii_answer = "Contact Alice Smith at alice@example.com for more info."
        with patch("rag.rag_chain.invoke_llm", return_value=pii_answer):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query("LangChain", apply_guardrails=True)
        assert "alice@example.com" not in result.answer

    def test_guardrails_off_preserves_original_query(self, populated_store):
        query = "What is LangChain?"
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query(query, apply_guardrails=False)
        assert result.sanitized_query == query
        assert not result.input_pii_detected


class TestRAGNoResults:
    def test_graceful_when_no_docs_match(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                # Score threshold so high nothing passes
                result = run_rag_query(
                    "xyzzy quantum unicorn flux",
                    score_threshold=0.999,
                    apply_guardrails=False,
                )
        assert result.retrieved_chunks == 0
        assert "couldn't find" in result.answer.lower()

    def test_no_results_still_returns_ragresponse(self, populated_store):
        with patch("rag.rag_chain.invoke_llm", return_value=MOCK_ANSWER):
            with patch("rag.rag_chain.get_chroma_store", return_value=populated_store):
                from rag.rag_chain import run_rag_query
                result = run_rag_query(
                    "xyzzy quantum unicorn flux",
                    score_threshold=0.999,
                    apply_guardrails=False,
                )
        assert hasattr(result, "answer")
        assert hasattr(result, "sanitized_query")
        assert hasattr(result, "source_documents")