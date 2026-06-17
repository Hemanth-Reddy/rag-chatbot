# RAG System with PII Guardrails + Agentic Mode

A production-grade **Retrieval-Augmented Generation (RAG)** system for private document Q&A. Supports dual LLM providers (Google Gemini and Ollama/Llama), local embeddings, ChromaDB vector storage, bidirectional PII detection via Microsoft Presidio, and an **Agentic Mode** where the LLM orchestrates tool calls across the knowledge base, web search, and system time.

---

## Two Modes

### Standard RAG (Agentic Mode OFF)
Fixed pipeline — always retrieves from local ChromaDB, then generates.

```
User Query
    │
    ▼
[PII Scrub — Input]        ← Presidio + spaCy (16 entity types)
    │
    ▼
[Embed Query]              ← sentence-transformers/all-MiniLM-L6-v2 (local)
    │
    ▼
[ChromaDB Retrieval]       ← cosine similarity, top-k chunks
    │
    ▼
[LLM — Gemini / Ollama]   ← context-grounded system prompt
    │
    ▼
[PII Scrub — Output]
    │
    ▼
Response + Sources + Retrieval Metadata
```

### Agentic RAG (Agentic Mode ON)
The LLM decides which tools to call, in what order, iterating until it has enough information.

```
User Query
    │
    ▼
[PII Scrub — Input]
    │
    ▼
[Gemini Tool-Calling Loop]  ← max 5 iterations
    ├── search_knowledge_base(query)  → ChromaDB
    ├── web_search(query)             → DuckDuckGo (no API key)
    └── get_current_date()            → system clock
    │
    ▼
[PII Scrub — Output]
    │
    ▼
Response + Tools Used Metadata
```

The LLM picks tools based on the question type:
- Question about uploaded docs → `search_knowledge_base`
- No KB results / needs live data → `web_search`
- "Today", "current", "latest" → `get_current_date`
- Multi-hop questions → multiple tools in sequence

---

## Components

| Layer | Technology |
|---|---|
| REST API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| LLM (cloud) | Google Gemini (`langchain-google-genai` + raw `google-generativeai`) |
| LLM (local) | Llama 3.2 via Ollama (HTTP, supports Colab tunnels) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector DB | ChromaDB (persistent local store) |
| Web Search | DuckDuckGo via `duckduckgo-search` (no API key required) |
| PII Detection | Microsoft Presidio + spaCy `en_core_web_sm` |
| Config | Pydantic Settings + `.env` |

---

## Features

- **Agentic Mode** — LLM orchestrates tool calls (KB search, web search, date); falls back to web when local knowledge is insufficient
- **Standard RAG** — classic fixed pipeline: retrieve → generate; unchanged when Agentic Mode is off
- **Dual LLM provider** — switch between Gemini and Ollama via one env var; same invoke interface
- **Raw Gemini tool-calling** — uses `google-generativeai` SDK directly, not a framework abstraction; demonstrates the ReAct loop at the protocol level
- **Local embeddings** — no API cost; ~90 MB model downloaded once and cached
- **PII guardrails** — bidirectional scrubbing on both modes (query before agent/LLM, response after); 16 entity types
- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown
- **Full source traceability** — standard mode shows source docs + relevance scores; agentic mode shows tool call chain
- **Persistent vector store** — ingest once, query forever across restarts

---

## Project Structure

```
RAG/
├── agent/                # Agentic RAG layer
│   ├── tools.py          #   Tool implementations: KB search, web search, date
│   ├── agent.py          #   Raw Gemini tool-calling loop (ReAct pattern)
│   └── agent_chain.py    #   PII guardrail wrapper around the agent loop
├── api/                  # FastAPI server (health, query, agent/query, ingest, stats, reset)
├── config/               # Pydantic-based settings loaded from .env
├── docs/                 # Concept documentation
│   └── agentic_rag_concepts.md
├── embeddings/           # Sentence Transformers wrapper (lazy singleton)
├── guardrails/           # Presidio PII detection and anonymization
├── ingest/               # Document loaders, chunking (512 chars / 64 overlap)
├── llm/                  # Gemini client, Ollama client, provider abstraction
├── rag/                  # Standard RAG pipeline orchestrator
├── vectordb/             # ChromaDB wrapper (add, search, delete, stats)
├── ui/                   # Streamlit chat interface
├── tests/                # Unit tests for all components
├── chroma_store/         # Persisted vector database (git-ignored)
├── .env.example          # Configuration template
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.10+
- Google Gemini API key — free tier at [aistudio.google.com](https://aistudio.google.com)
- (Optional) Ollama for local LLM

### Install

```bash
git clone <repo-url>
cd RAG
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure

```bash
# Windows (PowerShell)
Copy-Item .env.example .env
# macOS/Linux
cp .env.example .env
```

Edit `.env`:

```env
# LLM provider: "gemini" or "ollama"
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_google_gemini_api_key_here

# Ollama (only needed if LLM_PROVIDER=ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Vector store
CHROMA_PERSIST_DIR=./chroma_store
CHROMA_COLLECTION_NAME=rag_documents

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM settings
LLM_MODEL=gemini-flash-lite-latest
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048

# Retrieval settings
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.3

# API server
API_HOST=0.0.0.0
API_PORT=8000
```

---

## Running

### API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs at `http://localhost:8000/docs`.

### Web UI

```bash
python -m streamlit run ui/app.py
```

Opens at `http://localhost:8501`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Vector store statistics |
| `POST` | `/query` | Standard RAG query (ChromaDB only) |
| `POST` | `/agent/query` | Agentic RAG query (LLM picks tools) |
| `POST` | `/ingest/file` | Upload PDF, DOCX, TXT, or Markdown |
| `POST` | `/ingest/text` | Ingest raw text directly |
| `DELETE` | `/collection` | Reset the entire vector store |

### Example: Standard Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the refund terms?", "apply_guardrails": true, "top_k": 5}'
```

```json
{
  "answer": "...",
  "sanitized_query": "...",
  "source_documents": [...],
  "retrieved_chunks": 3,
  "retrieval_scores": [0.82, 0.74, 0.61],
  "guardrails": {
    "input_pii_detected": false,
    "input_pii_entities": [],
    "output_pii_detected": false,
    "output_pii_entities": []
  }
}
```

### Example: Agentic Query

```bash
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the latest news about RAG systems?", "apply_guardrails": true}'
```

```json
{
  "answer": "...",
  "sanitized_query": "...",
  "tool_calls": [
    {
      "tool": "search_knowledge_base",
      "args": {"query": "RAG systems"},
      "result_preview": "No relevant documents found..."
    },
    {
      "tool": "web_search",
      "args": {"query": "latest news RAG systems 2025"},
      "result_preview": "Title: ..."
    }
  ],
  "guardrails": {
    "input_pii_detected": false,
    "input_pii_entities": [],
    "output_pii_detected": false,
    "output_pii_entities": []
  }
}
```

### Example: Ingest a File

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/document.pdf"
```

---

## Agentic RAG — How It Works

The agent loop in `agent/agent.py` uses **raw Gemini function calling** (not LangChain agents):

1. Tool schemas are defined as `FunctionDeclaration` objects and sent to Gemini alongside the query
2. Gemini returns a `FunctionCall` object (tool name + args) instead of text when it needs more information
3. The app executes the tool and sends the result back via `FunctionResponse`
4. This repeats until Gemini returns plain text — the final answer
5. A hard cap of **5 iterations** prevents infinite loops

This is the **ReAct pattern** (Reason + Act) at the raw protocol level. The UI expander shows exactly which tools were called and with what arguments.

### Tool Descriptions Drive Tool Selection

The LLM selects tools based on their descriptions — not hard-coded routing. Changing a description changes agent behavior. This is the key design lever for agentic systems.

---

## PII Guardrails

Both modes share the same guardrail pipeline. Detected entity types (replaced with `<ENTITY_TYPE>`):

`PERSON` · `EMAIL_ADDRESS` · `PHONE_NUMBER` · `US_SSN` · `CREDIT_CARD` · `IP_ADDRESS` · `LOCATION` · `DATE_TIME` · `US_PASSPORT` · `US_DRIVER_LICENSE` · `IBAN_CODE` · `MEDICAL_LICENSE` · `URL` · `US_BANK_NUMBER` · `CRYPTO` · `NRP`

Toggle per-request via `apply_guardrails` in the payload, or from the Streamlit sidebar.

---

## LLM Provider Switching

```env
# Gemini (requires GOOGLE_API_KEY)
LLM_PROVIDER=gemini

# Local Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

Ollama supports any locally pulled model. `OLLAMA_BASE_URL` can point to a remote HTTP endpoint (e.g., a Colab tunnel).

> **Note:** Agentic Mode requires Gemini (`LLM_PROVIDER=gemini`) — it uses the `google-generativeai` SDK for native function calling. Ollama does not support the agentic endpoint.

---

## Tests

```bash
pytest tests/ -v
```

---

## License

MIT
