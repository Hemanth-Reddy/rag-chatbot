"""
Google Gemini LLM client via LangChain.
Free tier: https://aistudio.google.com
"""

import logging
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings

logger = logging.getLogger(__name__)

_llm: Optional[ChatGoogleGenerativeAI] = None

SYSTEM_PROMPT = """You are a helpful, accurate, and concise AI assistant.
Answer questions using ONLY the provided context below.
If the context does not contain enough information to answer, say:
"I don't have enough information in the provided documents to answer that."
Do NOT make up facts. Do NOT reference personal information you may have seen.
Always cite which document/source your answer is from if metadata is available."""


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a cached Gemini LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            convert_system_message_to_human=False,
        )
        logger.info("LLM initialized: %s", settings.llm_model)
    return _llm


def invoke_llm(user_query: str, context: str) -> str:
    """
    Call the LLM with a system prompt + assembled RAG context.
    Returns the raw text response.
    """
    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"""CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{user_query}

Provide a comprehensive, factual answer based only on the context above."""
        ),
    ]
    response = llm.invoke(messages)
    return response.content