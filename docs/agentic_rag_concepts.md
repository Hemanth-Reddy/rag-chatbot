# Agentic RAG — Concepts, Why, How, Difficulties

---

## 1. What is Agentic RAG?

Standard RAG is a fixed pipeline:
```
query → retrieve from vector DB → LLM generates answer
```
The LLM has no control. It always gets exactly one retrieval pass from one source.

Agentic RAG gives the LLM **agency** — it decides:
- Which tools to call
- In what order
- Whether to call more tools based on what it got back
- When it has enough information to answer

```
query → LLM thinks → calls tool A → LLM thinks → calls tool B → LLM answers
```

The LLM is no longer just a text generator. It becomes an **orchestrator**.

---

## 2. Why Standard RAG Fails (and Agentic RAG Fixes It)

| Problem with Standard RAG | Agentic Fix |
|---|---|
| Single vector DB miss = "I don't know" | Fall back to web search |
| Query about today's date/events = hallucination | `get_current_date` tool |
| Multi-hop questions need 2+ retrievals | Agent loops and calls again |
| Wrong source used for question type | Agent picks the right tool |
| All questions treated the same | Agent routes based on question type |

**Example:** "What did Elon Musk say yesterday about Tesla?"
- Standard RAG: searches local KB → finds nothing → fails
- Agentic RAG: LLM sees query needs current info → calls `web_search` → gets answer

---

## 3. Core Concept: Function Calling / Tool Use

This is the foundation. LLMs like Gemini support **function calling** natively:

1. You define tools as JSON schemas (name, description, parameters)
2. LLM receives these definitions alongside the user query
3. Instead of generating text, LLM returns a structured `FunctionCall` object
4. Your code executes the function
5. You send the result back to LLM
6. LLM either calls another tool or generates the final answer

```
You → [query + tool definitions] → Gemini
                                        ↓
                              FunctionCall { name: "search_kb", args: {...} }
You → [execute tool, get result] → Gemini
                                        ↓
                              FunctionCall { name: "web_search", args: {...} }
You → [execute tool, get result] → Gemini
                                        ↓
                              "Here is the answer: ..."  ← text, loop ends
```

The LLM never directly executes tools. It just **requests** them. Your code does the actual execution. This is an important security boundary.

---

## 4. ReAct Pattern (Reason + Act)

The mental model behind agentic systems. Each step:

1. **Reason** — LLM thinks: "The user asked X. I need Y to answer. I should call Z."
2. **Act** — LLM calls a tool
3. **Observe** — LLM receives tool output
4. Repeat until answer is ready

```
Thought: User wants latest news. My KB won't have this. Use web_search.
Action: web_search("Tesla news today")
Observation: [search results]
Thought: I have enough info now.
Answer: Based on today's news...
```

Gemini doesn't literally output "Thought/Action/Observation" tags — that's LangChain's abstraction. Raw Gemini tool-calling achieves the same loop via multi-turn chat.

---

## 5. How Gemini Tool Calling Works (Raw SDK)

```python
import google.generativeai as genai

# Step 1: Define tools as FunctionDeclarations
tools = [genai.protos.Tool(function_declarations=[
    genai.protos.FunctionDeclaration(
        name="search_knowledge_base",
        description="Search local documents for relevant information",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={"query": genai.protos.Schema(type=genai.protos.Type.STRING)},
            required=["query"]
        )
    )
])]

# Step 2: Create model with tools
model = genai.GenerativeModel("gemini-1.5-flash", tools=tools)
chat = model.start_chat()

# Step 3: Send user message
response = chat.send_message("What is the capital of France?")

# Step 4: Check if LLM wants to call a tool
part = response.candidates[0].content.parts[0]
if hasattr(part, "function_call"):
    fn_name = part.function_call.name       # "search_knowledge_base"
    fn_args = dict(part.function_call.args) # {"query": "capital of France"}

    # Step 5: Execute tool
    result = your_tool_function(**fn_args)

    # Step 6: Send result back
    response = chat.send_message(genai.protos.Content(parts=[
        genai.protos.Part(function_response=genai.protos.FunctionResponse(
            name=fn_name,
            response={"result": result}
        ))
    ]))

# Step 7: Now response.text is the final answer
```

---

## 6. The Three Tools in This App

### `search_knowledge_base(query: str)`
- Calls ChromaDB similarity search
- Returns top-k chunks with relevance scores
- Used when: question is about ingested documents
- The "traditional RAG" tool

### `web_search(query: str)`
- Calls DuckDuckGo (no API key needed)
- Returns top search results as text
- Used when: KB has no relevant info, or question needs current data
- Fallback when local knowledge fails

### `get_current_date()`
- Returns `datetime.now().isoformat()`
- Used when: question involves "today", "current", "latest", "this year"
- Prevents date hallucination — LLMs have training cutoffs

---

## 7. Tool Selection — How the LLM Decides

The LLM picks tools based on **tool descriptions**. This means:

- Description quality = agent quality
- Bad: `"search_knowledge_base — searches documents"`
- Good: `"search_knowledge_base — Search the local knowledge base of ingested documents. Use this first for any question about uploaded content. Returns relevant text chunks with source citations."`

The LLM reads these descriptions at inference time and reasons about which fits. You are essentially **prompting the LLM** through tool descriptions.

---

## 8. Multi-Hop Retrieval

Some questions require chaining tools:

**Question:** "Compare what the uploaded report says about Q3 revenue vs what analysts are saying now."

1. `search_knowledge_base("Q3 revenue")` → gets internal report data
2. `web_search("Q3 revenue analyst opinions 2025")` → gets external data
3. LLM synthesizes both → final answer

Standard RAG cannot do this. It has one retrieval pass. Agentic RAG iterates.

---

## 9. Safety: The Iteration Cap

Without a limit, a buggy agent can loop forever (tool fails → LLM retries → tool fails...).

Always set a **max_iterations** cap (e.g., 5). If reached:
- Return best answer so far
- Or return explicit "could not answer after N steps" message

This is not optional — it is essential for production systems.

---

## 10. Agentic RAG vs Standard RAG vs Full Agents

| | Standard RAG | Agentic RAG | Full Agent (e.g., AutoGPT) |
|---|---|---|---|
| Retrieval sources | 1 (vector DB) | Multiple tools | Any tool, any API |
| LLM control | None | Tool selection | Full autonomy |
| Loop | No | Bounded (max N) | Unbounded |
| Predictability | High | Medium | Low |
| Cost per query | Low | Medium | High |
| Best for | Known domain Q&A | Mixed knowledge queries | Open-ended tasks |

Agentic RAG sits in the sweet spot: more capable than standard RAG, more predictable than full agents.

---

## 11. Where It Fits in System Design

```
User Query
    ↓
PII Guardrail (scrub sensitive data before LLM sees it)
    ↓
Agentic Loop
    ├── search_knowledge_base → ChromaDB
    ├── web_search → DuckDuckGo
    └── get_current_date → system clock
    ↓
LLM synthesizes final answer
    ↓
PII Guardrail (scrub sensitive data from response)
    ↓
Response to user + tool_calls metadata (for transparency)
```

Guardrails wrap the agent — PII scrub happens before the loop starts and after it ends.

---

## 12. Real-World Use Cases

| Use Case | Why Agentic RAG |
|---|---|
| Enterprise Q&A bot | Internal KB + live data (prices, inventory) |
| Legal research assistant | Case law DB + current regulatory updates |
| Medical info tool | Clinical guidelines + latest research papers |
| Customer support | Product docs + live order status API |
| Financial analyst | Reports + live market data |
| Dev documentation bot | Local codebase + Stack Overflow / GitHub |

Common pattern: **structured internal knowledge + dynamic external data**.

---

## 13. Key Difficulties

### 1. Tool Description Quality
LLM picks tools from descriptions. Vague descriptions = wrong tool selected = wrong answer. Requires iteration and testing.

### 2. Hallucinated Tool Arguments
LLM may pass malformed args (wrong types, missing required fields). Always validate tool inputs before executing.

### 3. Infinite Loop Risk
Covered by iteration cap. But also: LLM might call the same tool repeatedly with slightly different queries. Detect and break this pattern.

### 4. Latency
Each tool call = one extra LLM round trip + tool execution time. A 2-tool chain can be 3-5x slower than standard RAG. Show progress to users.

### 5. Cost
More LLM calls = more tokens = higher API cost. Each tool result is fed back into context, growing the conversation. Monitor token usage.

### 6. Context Window Exhaustion
Long tool results (e.g., 10 web search snippets) consume context. Truncate tool outputs before returning to LLM.

### 7. Evaluation is Hard
Standard RAG: did the retrieved chunk contain the answer? Agentic RAG: did the agent pick the right tools in the right order AND answer correctly? Harder to score automatically.

### 8. PII in Tool Results
Web search may return PII in results that get fed to the LLM. Guardrails must run on tool outputs too, not just user input/output.

---

## 14. Interview Talking Points

**"What is Agentic RAG?"**
RAG where the LLM orchestrates retrieval instead of a fixed pipeline. LLM uses function calling to decide which tools to invoke, iterating until it has enough information to answer.

**"How is it different from standard RAG?"**
Standard RAG: always one vector DB retrieval. Agentic: LLM chooses tools dynamically, can chain multiple calls, falls back to web search if KB misses.

**"What patterns does it use?"**
ReAct (Reason + Act), function calling / tool use, multi-turn conversation loop.

**"What are the tradeoffs?"**
Higher latency, higher cost, harder to evaluate, less predictable than standard RAG. But handles knowledge gaps and time-sensitive queries that standard RAG fails on.

**"How do you prevent infinite loops?"**
Max iterations cap (e.g., 5). Also: track tool call history within a session and detect repeated identical calls.

**"How do you handle PII in an agentic system?"**
Scrub before the loop (user query), scrub after the loop (final answer). Also sanitize tool results before feeding back to LLM — web search results may contain PII.
