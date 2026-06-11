# tests/test_llm_client.py
# Feature 5 test — makes ONE real Gemini API call.
# Requires GOOGLE_API_KEY in .env and internet access.
# All other tests in this file use mocks — no API calls.

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from llm.llm_client import get_llm, invoke_llm, SYSTEM_PROMPT


class TestLLMLoads:
    def test_llm_instance_created(self):
        llm = get_llm()
        assert llm is not None

    def test_llm_is_singleton(self):
        """Calling get_llm() twice returns the same object."""
        llm1 = get_llm()
        llm2 = get_llm()
        assert llm1 is llm2

    def test_correct_model_configured(self):
        llm = get_llm()
        assert "gemini" in llm.model.lower()


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_has_key_rules(self):
        """Confirm the prompt contains our safety instructions."""
        assert "ONLY" in SYSTEM_PROMPT
        assert "not" in SYSTEM_PROMPT.lower()


class TestInvokeLLMMocked:
    """These tests mock the LLM — no API calls, always fast."""

    def test_invoke_returns_string(self):
        mock_response = MagicMock()
        mock_response.content = "ChromaDB is a vector database."
        with patch("llm.llm_client.get_llm") as mock_get:
            mock_get.return_value.invoke.return_value = mock_response
            result = invoke_llm("What is ChromaDB?", "ChromaDB stores embeddings.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invoke_passes_query_and_context(self):
        mock_response = MagicMock()
        mock_response.content = "Test answer."
        with patch("llm.llm_client.get_llm") as mock_get:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = mock_response
            mock_get.return_value = mock_llm
            invoke_llm("My question", "My context")
            # Confirm invoke was called once with our messages
            assert mock_llm.invoke.call_count == 1
            call_args = mock_llm.invoke.call_args[0][0]
            # Should be a list of messages
            assert isinstance(call_args, list)
            assert len(call_args) == 2

    def test_invoke_returns_llm_content(self):
        mock_response = MagicMock()
        mock_response.content = "Specific answer from mock."
        with patch("llm.llm_client.get_llm") as mock_get:
            mock_get.return_value.invoke.return_value = mock_response
            result = invoke_llm("question", "context")
        assert result == "Specific answer from mock."


class TestInvokeLLMReal:
    """One real API call — confirms your key and network work."""

    def test_real_gemini_call(self):
        result = invoke_llm(
            user_query="What is ChromaDB?",
            context="ChromaDB is an open-source vector database used for AI applications. It stores embeddings and supports fast similarity search."
        )
        assert isinstance(result, str)
        assert len(result) > 20
        # Gemini should mention ChromaDB or vector in its answer
        assert any(
            word in result.lower()
            for word in ["chromadb", "vector", "database", "embeddings"]
        )