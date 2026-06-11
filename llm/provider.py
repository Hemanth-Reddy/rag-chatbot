# llm/provider.py
# Feature 12: Config-driven LLM provider switch
# Returns the correct invoke function based on LLM_PROVIDER in .env
# "gemini" -> uses Gemini API (Feature 5)
# "ollama" -> uses Colab Llama server (Feature 11)
# The RAG chain calls get_invoke_fn() — it never needs to know which provider.

import logging
from typing import Callable
from config.settings import settings

logger = logging.getLogger(__name__)


def get_invoke_fn() -> Callable[[str, str], str]:
    """
    Return the correct LLM invoke function based on LLM_PROVIDER setting.

    Returns:
        A callable with signature: fn(user_query: str, context: str) -> str
    """
    provider = settings.llm_provider.strip().lower()

    if provider == "ollama":
        logger.info("LLM provider: Ollama (%s @ %s)",
                    settings.ollama_model, settings.ollama_base_url)
        from llm.ollama_client import invoke_ollama
        return invoke_ollama

    elif provider == "gemini":
        logger.info("LLM provider: Gemini (%s)", settings.llm_model)
        from llm.llm_client import invoke_llm
        return invoke_llm

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid options: 'gemini', 'ollama'"
        )