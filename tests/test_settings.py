# tests/test_settings.py
# Feature 1 test — runs with no internet, no LLM, no ChromaDB.
# Just confirms .env loads correctly into the Settings object.

import sys
from pathlib import Path

# Make sure Python can find the project root regardless of where pytest runs from
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings


def test_api_key_is_loaded():
    """GOOGLE_API_KEY must be present and non-empty."""
    assert settings.google_api_key, "GOOGLE_API_KEY is missing from .env"
    assert len(settings.google_api_key) > 10, "GOOGLE_API_KEY looks too short"


def test_llm_defaults():
    """LLM settings load with correct defaults."""
    assert settings.llm_model == "gemini-flash-lite-latest"
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 2048


def test_chroma_defaults():
    """ChromaDB settings load with correct defaults."""
    assert settings.chroma_persist_dir == "./chroma_store"
    assert settings.chroma_collection_name == "rag_documents"


def test_embedding_model_default():
    """Embedding model name loads correctly."""
    assert settings.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


def test_rag_defaults():
    """RAG retrieval settings load with correct defaults."""
    assert settings.rag_top_k == 5
    assert settings.rag_score_threshold == 0.3


def test_api_defaults():
    """API server settings load with correct defaults."""
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000


def test_settings_is_singleton():
    """Importing settings twice returns the same object."""
    from config.settings import settings as settings2
    assert settings is settings2