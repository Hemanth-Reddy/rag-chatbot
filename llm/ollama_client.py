# llm/ollama_client.py
# Feature 11: Ollama LLM Client
# Calls the Llama model running on your Colab server via HTTP.
# Drop-in replacement for llm_client.py — same invoke_ollama() signature.

import logging
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

# Same system prompt as Gemini client — consistent behaviour across providers
SYSTEM_PROMPT = """You are a helpful, accurate, and concise AI assistant.
Answer questions using ONLY the context provided below.
Rules:
- If the context does not contain enough information, say exactly:
  "I don't have enough information in the provided documents to answer that."
- Do NOT make up facts or invent information not in the context.
- Do NOT include personal information you may see in the context.
- Cite which source document your answer comes from when available.
- Be concise and factual."""


def invoke_ollama(user_query: str, context: str) -> str:
    """
    Send query + context to Llama running on Colab via Ollama API.

    Args:
        user_query: The sanitized (PII-scrubbed) question from the user.
        context:    Retrieved document chunks formatted as a single string.

    Returns:
        Raw text response from Llama.

    Raises:
        httpx.ConnectError:   If the Colab tunnel is down.
        httpx.TimeoutException: If Llama takes too long to respond.
    """
    if not settings.ollama_base_url:
        raise ValueError(
            "OLLAMA_BASE_URL is not set in .env. "
            "Start your Colab tunnel and paste the URL."
        )

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT FROM DOCUMENTS:\n{context}\n\n"
        f"USER QUESTION:\n{user_query}\n\n"
        "Answer based only on the context above."
    )

    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"

    logger.info("Calling Ollama at: %s", url)
    logger.info("Model: %s", settings.ollama_model)

    response = httpx.post(
        url,
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,        # get full response at once
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens,
            },
        },
        timeout=120.0,              # Llama can be slow — 2 min timeout
    )

    response.raise_for_status()
    data = response.json()
    answer = data.get("response", "").strip()

    logger.info("Ollama responded with %d characters.", len(answer))
    return answer