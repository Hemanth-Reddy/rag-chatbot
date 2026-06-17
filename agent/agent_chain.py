"""
agent/agent_chain.py
Wraps the raw agent loop with PII guardrails.
Mirrors the structure of rag/rag_chain.py for consistency.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from guardrails.pii_guard import get_guardrail, GuardResult
from agent.agent import run_agent

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Structured response from the agentic RAG pipeline."""
    answer: str
    sanitized_query: str
    tool_calls: list[dict] = field(default_factory=list)
    input_pii_detected: bool = False
    input_pii_entities: list[dict] = field(default_factory=list)
    output_pii_detected: bool = False
    output_pii_entities: list[dict] = field(default_factory=list)


def run_agentic_query(
    user_query: str,
    apply_guardrails: bool = True,
) -> AgentResponse:
    """
    Execute the full agentic RAG pipeline.

    Steps:
        1. Scrub PII from user query
        2. Run Gemini tool-calling agent loop
        3. Scrub PII from final answer
        4. Return structured AgentResponse
    """
    guardrail = get_guardrail()

    # ── 1. PII scrub on input ──────────────────────────────────────────
    if apply_guardrails:
        input_guard = guardrail.scrub_input(user_query)
        effective_query = input_guard.sanitized_text
    else:
        input_guard = GuardResult(
            original_text=user_query,
            sanitized_text=user_query,
            pii_detected=False,
        )
        effective_query = user_query

    logger.info("Agentic query (post-guardrail): %s", effective_query[:120])

    # ── 2. Run agent loop ──────────────────────────────────────────────
    result = run_agent(effective_query)

    # ── 3. PII scrub on output ─────────────────────────────────────────
    if apply_guardrails:
        output_guard = guardrail.scrub_output(result["answer"])
        final_answer = output_guard.sanitized_text
    else:
        output_guard = GuardResult(
            original_text=result["answer"],
            sanitized_text=result["answer"],
            pii_detected=False,
        )
        final_answer = result["answer"]

    return AgentResponse(
        answer=final_answer,
        sanitized_query=effective_query,
        tool_calls=result["tool_calls"],
        input_pii_detected=input_guard.pii_detected,
        input_pii_entities=input_guard.detected_entities,
        output_pii_detected=output_guard.pii_detected,
        output_pii_entities=output_guard.detected_entities,
    )
