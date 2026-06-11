# tests/test_vectordb.py
# Feature 4 test — no LLM, no internet needed.
# Uses a temporary directory — never touches your real chroma_store/.
import sys
import shutil
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.documents import Document
from vectordb.chroma_store import ChromaStore


@pytest.fixture(scope="module")
def store():
    """Fresh isolated ChromaStore in a temp dir for all tests."""
    import shutil
    tmpdir = tempfile.mkdtemp()
    s = ChromaStore(persist_dir=tmpdir, collection_name="test_col")
    yield s
    # Explicitly close ChromaDB before deleting on Windows
    try:
        s._store = None
        import gc
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass  # Windows may still hold the lock — temp dir cleans up on reboot

@pytest.fixture(scope="module")
def populated_store(store):
    """Add known test documents once, reuse across all tests."""
    docs = [
        Document(
            page_content="Python is a high-level programming language known for simplicity.",
            metadata={"source": "python.txt"},
        ),
        Document(
            page_content="LangChain is a framework for building LLM-powered applications.",
            metadata={"source": "langchain.txt"},
        ),
        Document(
            page_content="ChromaDB is an open-source vector database for AI applications.",
            metadata={"source": "chroma.txt"},
        ),
        Document(
            page_content="FastAPI is a modern Python web framework for building REST APIs.",
            metadata={"source": "fastapi.txt"},
        ),
        Document(
            page_content="Gemini is Google's multimodal AI model with a free API tier.",
            metadata={"source": "gemini.txt"},
        ),
    ]
    store.add_documents(docs)
    return store


class TestAddDocuments:
    def test_documents_stored(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["document_count"] >= 5

    def test_stats_fields_present(self, populated_store):
        stats = populated_store.get_stats()
        assert "document_count" in stats
        assert "collection_name" in stats
        assert "persist_dir" in stats

    def test_collection_name_correct(self, populated_store):
        stats = populated_store.get_stats()
        assert stats["collection_name"] == "test_col"


class TestSimilaritySearch:
    def test_returns_results(self, populated_store):
        results = populated_store.similarity_search(
            "Python programming", k=3, score_threshold=0.0
        )
        assert len(results) > 0

    def test_result_is_document_and_score(self, populated_store):
        results = populated_store.similarity_search(
            "vector database", k=2, score_threshold=0.0
        )
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_most_relevant_doc_ranks_first(self, populated_store):
        results = populated_store.similarity_search(
            "LangChain LLM framework", k=5, score_threshold=0.0
        )
        top_doc, _ = results[0]
        assert "LangChain" in top_doc.page_content

    def test_chroma_doc_relevant_for_vector_query(self, populated_store):
        results = populated_store.similarity_search(
            "vector database ChromaDB", k=5, score_threshold=0.0
        )
        contents = [doc.page_content for doc, _ in results]
        assert any("ChromaDB" in c or "vector" in c for c in contents)

    def test_high_threshold_returns_fewer(self, populated_store):
        low = populated_store.similarity_search(
            "Python", k=5, score_threshold=0.0
        )
        high = populated_store.similarity_search(
            "Python", k=5, score_threshold=0.95
        )
        assert len(low) >= len(high)

    def test_k_limits_results(self, populated_store):
        results = populated_store.similarity_search(
            "programming", k=2, score_threshold=0.0
        )
        assert len(results) <= 2

    def test_metadata_preserved(self, populated_store):
        results = populated_store.similarity_search(
            "Python language", k=3, score_threshold=0.0
        )
        for doc, _ in results:
            assert "source" in doc.metadata


class TestRetriever:
    def test_retriever_created(self, populated_store):
        retriever = populated_store.get_retriever(k=3)
        assert retriever is not None

    def test_retriever_returns_documents(self, populated_store):
        retriever = populated_store.get_retriever(k=3)
        docs = retriever.invoke("FastAPI REST API")
        assert isinstance(docs, list)