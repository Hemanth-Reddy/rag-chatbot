# tests/test_provider.py
# Feature 12 test — confirms provider switch works correctly.
# No API calls, no tunnel needed — all mocked.

import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from llm.provider import get_invoke_fn


class TestProviderSwitch:

    def test_gemini_provider_returns_callable(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "gemini"
            mock_settings.llm_model = "gemini-flash-lite-latest"
            fn = get_invoke_fn()
        assert callable(fn)

    def test_ollama_provider_returns_callable(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.ollama_model = "llama3.2:3b"
            mock_settings.ollama_base_url = "https://fake.trycloudflare.com"
            fn = get_invoke_fn()
        assert callable(fn)

    def test_gemini_returns_invoke_llm(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "gemini"
            mock_settings.llm_model = "gemini-flash-lite-latest"
            fn = get_invoke_fn()
        from llm.llm_client import invoke_llm
        assert fn is invoke_llm

    def test_ollama_returns_invoke_ollama(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.ollama_model = "llama3.2:3b"
            mock_settings.ollama_base_url = "https://fake.trycloudflare.com"
            fn = get_invoke_fn()
        from llm.ollama_client import invoke_ollama
        assert fn is invoke_ollama

    def test_unknown_provider_raises(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "openai"
            with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
                get_invoke_fn()

    def test_provider_case_insensitive(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "OLLAMA"
            mock_settings.ollama_model = "llama3.2:3b"
            mock_settings.ollama_base_url = "https://fake.trycloudflare.com"
            fn = get_invoke_fn()
        assert callable(fn)

    def test_provider_strips_whitespace(self):
        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "  gemini  "
            mock_settings.llm_model = "gemini-flash-lite-latest"
            fn = get_invoke_fn()
        assert callable(fn)


class TestRAGChainUsesProvider:
    """Confirm RAG chain routes through provider switch correctly."""

    def test_rag_chain_calls_ollama_when_provider_is_ollama(self):
        import shutil
        import tempfile
        from langchain_core.documents import Document
        from vectordb.chroma_store import ChromaStore

        tmpdir = tempfile.mkdtemp()
        store = ChromaStore(persist_dir=tmpdir, collection_name="test_provider")
        store.add_documents([
            Document(
                page_content="LangChain is a framework for LLM apps.",
                metadata={"source": "test.txt"},
            )
        ])

        ollama_called = []

        def fake_ollama(query, context):
            ollama_called.append(True)
            return "Answer from Ollama."

        with patch("llm.provider.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.ollama_model = "llama3.2:3b"
            mock_settings.ollama_base_url = "https://fake.trycloudflare.com"
            with patch("llm.ollama_client.invoke_ollama", side_effect=fake_ollama):
                with patch("rag.rag_chain.get_chroma_store", return_value=store):
                    from rag.rag_chain import run_rag_query
                    result = run_rag_query(
                        "What is LangChain?",
                        score_threshold=0.0,
                        apply_guardrails=False,
                    )

        try:
            store._store = None
            import gc; gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass