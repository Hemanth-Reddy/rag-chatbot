# tests/test_api.py
# Feature 8 test — all heavy dependencies are mocked.
# No API key, no ChromaDB, no internet needed.

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient


def make_mock_rag_response(answer="Test answer."):
    """Build a realistic mock RAGResponse object."""
    mock = MagicMock()
    mock.answer = answer
    mock.sanitized_query = "test query"
    mock.source_documents = [
        {
            "content_preview": "Sample content about LangChain.",
            "metadata": {"source": "test.txt"},
            "relevance_score": 0.85,
        }
    ]
    mock.retrieved_chunks = 1
    mock.retrieval_scores = [0.85]
    mock.input_pii_detected = False
    mock.input_pii_entities = []
    mock.output_pii_detected = False
    mock.output_pii_entities = []
    return mock


@pytest.fixture(scope="module")
def client():
    """TestClient with all heavy dependencies mocked at startup."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "fake-key-for-testing",
            "CHROMA_PERSIST_DIR": tmpdir,
        }):
            with patch("api.main.get_guardrail"), \
                 patch("api.main.get_chroma_store"):
                from api.main import app
                with TestClient(app, raise_server_exceptions=False) as c:
                    yield c


class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_has_model_field(self, client):
        resp = client.get("/health")
        assert "model" in resp.json()

    def test_has_collection_field(self, client):
        resp = client.get("/health")
        assert "collection" in resp.json()


class TestQueryEndpoint:
    def test_valid_query_returns_200(self, client):
        with patch("api.main.run_rag_query", return_value=make_mock_rag_response()):
            resp = client.post("/query", json={"query": "What is LangChain?"})
        assert resp.status_code == 200

    def test_response_has_answer(self, client):
        with patch("api.main.run_rag_query", return_value=make_mock_rag_response("Good answer.")):
            resp = client.post("/query", json={"query": "What is LangChain?"})
        assert resp.json()["answer"] == "Good answer."

    def test_response_has_all_fields(self, client):
        with patch("api.main.run_rag_query", return_value=make_mock_rag_response()):
            resp = client.post("/query", json={"query": "What is LangChain?"})
        data = resp.json()
        assert "answer" in data
        assert "sanitized_query" in data
        assert "source_documents" in data
        assert "guardrails" in data
        assert "retrieved_chunks" in data
        assert "retrieval_scores" in data

    def test_empty_query_returns_400(self, client):
        resp = client.post("/query", json={"query": "   "})
        assert resp.status_code == 400

    def test_query_too_long_returns_422(self, client):
        resp = client.post("/query", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_pii_detected_in_guardrails(self, client):
        mock = make_mock_rag_response()
        mock.input_pii_detected = True
        mock.input_pii_entities = [{"entity_type": "EMAIL_ADDRESS"}]
        with patch("api.main.run_rag_query", return_value=mock):
            resp = client.post("/query", json={"query": "my email is test@test.com"})
        assert resp.json()["guardrails"]["input_pii_detected"] is True

    def test_custom_top_k_passed_through(self, client):
        with patch("api.main.run_rag_query", return_value=make_mock_rag_response()) as mock_fn:
            client.post("/query", json={"query": "test question", "top_k": 10})
        call_kwargs = mock_fn.call_args.kwargs
        assert call_kwargs.get("top_k") == 10


class TestIngestTextEndpoint:
    def test_valid_text_returns_200(self, client):
        with patch("api.main.ingest_text", return_value={"chunks_created": 2, "status": "success"}):
            resp = client.post("/ingest/text", json={
                "text": "This is test content about AI and machine learning.",
                "source_name": "test_source",
            })
        assert resp.status_code == 200