# tests/test_ollama_client.py
# Feature 11 test — mocked HTTP calls, no Colab tunnel needed for most tests.
# One real call test at the end — requires tunnel to be live.

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import httpx
from llm.ollama_client import invoke_ollama, SYSTEM_PROMPT


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert SYSTEM_PROMPT is not None
        assert len(SYSTEM_PROMPT) > 50

    def test_system_prompt_has_safety_rules(self):
        assert "ONLY" in SYSTEM_PROMPT
        assert "not" in SYSTEM_PROMPT.lower()


class TestInvokeOllamaMocked:
    """All mocked — no tunnel needed."""

    def test_returns_string(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "LangChain is a framework."}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.ollama_client.httpx.post", return_value=mock_response):
            result = invoke_ollama("What is LangChain?", "LangChain is a framework.")
        assert isinstance(result, str)
        assert result == "LangChain is a framework."

    def test_returns_response_field(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Specific answer here."}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.ollama_client.httpx.post", return_value=mock_response):
            result = invoke_ollama("question", "context")
        assert result == "Specific answer here."

    def test_calls_correct_endpoint(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.ollama_client.httpx.post", return_value=mock_response) as mock_post:
            invoke_ollama("question", "context")
            call_url = mock_post.call_args[0][0]
            assert "/api/generate" in call_url

    def test_sends_correct_model(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.ollama_client.httpx.post", return_value=mock_response) as mock_post:
            invoke_ollama("question", "context")
            call_body = mock_post.call_args.kwargs["json"]
            assert call_body["model"] == "llama3.2:3b"

    def test_stream_is_false(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.ollama_client.httpx.post", return_value=mock_response) as mock_post:
            invoke_ollama("question", "context")
            call_body = mock_post.call_args.kwargs["json"]
            assert call_body["stream"] is False

    def test_raises_when_url_not_set(self):
        with patch("llm.ollama_client.settings") as mock_settings:
            mock_settings.ollama_base_url = ""
            with pytest.raises(ValueError, match="OLLAMA_BASE_URL"):
                invoke_ollama("question", "context")

    def test_strips_whitespace_from_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "  answer with spaces  "}
        mock_response.raise_for_status = MagicMock()

        with patch("llm.ollama_client.httpx.post", return_value=mock_response):
            result = invoke_ollama("question", "context")
        assert result == "answer with spaces"


class TestInvokeOllamaReal:
    """Real call — requires Colab tunnel to be live."""

    def test_real_ollama_call(self):
        """
        Makes a real HTTP call to your Colab Ollama server.
        Skip this test if tunnel is down:
            pytest tests/test_ollama_client.py -v -k "not real"
        """
        result = invoke_ollama(
            user_query="What is ChromaDB?",
            context=(
                "ChromaDB is an open-source vector database used for AI applications. "
                "It stores embeddings and supports fast similarity search."
            ),
        )
        assert isinstance(result, str)
        assert len(result) > 20
        assert any(
            word in result.lower()
            for word in ["chromadb", "vector", "database", "embeddings", "similarity"]
        )