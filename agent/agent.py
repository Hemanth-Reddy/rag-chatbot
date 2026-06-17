"""
agent/agent.py
Raw Gemini tool-calling loop.

Flow:
  1. Send user query + tool definitions to Gemini
  2. If Gemini returns a FunctionCall → execute tool → send result back
  3. Repeat until Gemini returns plain text (no more tool calls)
  4. Return final answer + list of tool calls made
"""

import logging

import google.generativeai as genai

from config.settings import settings
from agent.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5

# Truncate tool results before sending back to Gemini to prevent context overflow
_MAX_TOOL_RESULT_CHARS = 3000

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a local knowledge base and web search. "
    "Always call search_knowledge_base first for any factual question. "
    "Use web_search only if the knowledge base has no relevant information or the question "
    "requires current/live data. "
    "Use get_current_date when the question involves today's date, current time, or recent events. "
    "After gathering information from tools, synthesize a clear, accurate answer. "
    "Cite the source (document name or URL) when available."
)

# ── Tool schema definitions ────────────────────────────────────────────────────

_TOOL_DECLARATIONS = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="search_knowledge_base",
            description=(
                "Search the local knowledge base of ingested documents. "
                "Use this first for any question about uploaded or domain-specific content. "
                "Returns relevant text chunks with source citations and relevance scores."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "query": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The search query to find relevant document chunks.",
                    ),
                },
                required=["query"],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the web for current information not available in the knowledge base. "
                "Use when the knowledge base has no relevant results, or the question requires "
                "live, recent, or real-time data. Returns titles, URLs, and snippets."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "query": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="The web search query.",
                    ),
                },
                required=["query"],
            ),
        ),
        genai.protos.FunctionDeclaration(
            name="get_current_date",
            description=(
                "Get the current date and time. Use when the question involves "
                "'today', 'now', 'current date', 'this year', or any time-sensitive context."
            ),
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={},
            ),
        ),
    ]
)


# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(query: str) -> dict:
    """
    Execute the Gemini tool-calling loop for a given query.

    Returns:
        {
            "answer": str,           final text answer from Gemini
            "tool_calls": list[dict] each call: {tool, args, result_preview}
        }
    """
    genai.configure(api_key=settings.google_api_key)

    model = genai.GenerativeModel(
        model_name=settings.llm_model,
        tools=[_TOOL_DECLARATIONS],
        system_instruction=SYSTEM_PROMPT,
    )

    chat = model.start_chat()
    response = chat.send_message(query)

    tool_calls: list[dict] = []
    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1

        # Find a function_call part in the response
        fc_part = None
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call.name:
                fc_part = part
                break

        if fc_part is None:
            # No tool call requested — Gemini has the final answer
            break

        fn_name = fc_part.function_call.name
        fn_args = dict(fc_part.function_call.args)

        logger.info("Agent iteration %d: calling tool '%s' with args %s", iterations, fn_name, fn_args)

        # Execute tool
        tool_fn = TOOL_REGISTRY.get(fn_name)
        if tool_fn:
            try:
                raw_result = tool_fn(**fn_args)
            except Exception as e:
                raw_result = f"Tool '{fn_name}' raised an error: {e}"
                logger.error("Tool '%s' error: %s", fn_name, e)
        else:
            raw_result = f"Unknown tool requested: {fn_name}"
            logger.warning("Unknown tool: %s", fn_name)

        tool_calls.append({
            "tool": fn_name,
            "args": fn_args,
            "result_preview": str(raw_result)[:200],
        })

        # Send tool result back to Gemini (truncated to protect context window)
        response = chat.send_message(
            genai.protos.Content(
                parts=[
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn_name,
                            response={"result": str(raw_result)[:_MAX_TOOL_RESULT_CHARS]},
                        )
                    )
                ]
            )
        )

    # Extract text from final response
    answer_parts = []
    for part in response.candidates[0].content.parts:
        if hasattr(part, "text") and part.text:
            answer_parts.append(part.text)

    answer = "".join(answer_parts).strip()

    if not answer:
        answer = "I was unable to produce an answer after searching available sources."

    logger.info(
        "Agent completed in %d iteration(s). Tools used: %s",
        iterations,
        [c["tool"] for c in tool_calls],
    )

    return {"answer": answer, "tool_calls": tool_calls}
