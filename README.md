# RAG System with PII Guardrails

A production-grade **Retrieval-Augmented Generation (RAG)** system for private document Q&A. Supports dual LLM providers (Google Gemini and Ollama/Llama), local embeddings, ChromaDB vector storage, and bidirectional PII detection via Microsoft Presidio.

---

## Architecture

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
[PII Scrub — Output]       ← same Presidio pipeline
    │
    ▼
Response + Sources + Retrieval Metadata
```

### Components

| Layer | Technology |
|---|---|
| REST API | FastAPI + Uvicorn |
| Web UI | Streamlit |
| LLM (cloud) | Google Gemini Flash Lite (`langchain-google-genai`) |
| LLM (local) | Llama 3.2 via Ollama (HTTP, supports Colab tunnels) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector DB | ChromaDB (persistent local store) |
| PII Detection | Microsoft Presidio + spaCy `en_core_web_sm` |
| Config | Pydantic Settings + `.env` |

---

## Features

- **Dual LLM provider** — switch between Gemini and Ollama via a single env var; both use the same invoke interface
- **Local embeddings** — no API cost; ~90 MB model downloaded once and cached
- **PII guardrails** — bidirectional scrubbing (query before LLM, response after); detects persons, emails, phones, SSNs, credit cards, IPs, passports, IBANs, crypto addresses, and more
- **Multi-format ingestion** — PDF, DOCX, TXT, Markdown; recursive directory support
- **Full source traceability** — every response includes source documents, relevance scores, and retrieval stats
- **Persistent vector store** — ingest once, query forever across restarts
- **Toggleable guardrails** — enable/disable PII scrubbing per request from the UI or API

---

## Project Structure

```
RAG/
├── api/                  # FastAPI server (health, query, ingest, stats, reset)
├── config/               # Pydantic-based settings loaded from .env
├── embeddings/           # Sentence Transformers wrapper (lazy singleton)
├── guardrails/           # Presidio PII detection and anonymization
├── ingest/               # Document loaders, chunking (512 chars / 64 overlap)
├── llm/                  # Gemini client, Ollama client, provider abstraction
├── rag/                  # RAG pipeline orchestrator (RAGResponse dataclass)
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
- (Optional) Google Gemini API key — free tier at [aistudio.google.com](https://aistudio.google.com)
- (Optional) Ollama running locally or via a remote HTTP endpoint

### Install

```bash
git clone <repo-url>
cd RAG
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Configure

```bash
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
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### Web UI

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/stats` | Vector store statistics |
| `POST` | `/query` | RAG query with optional PII scrubbing |
| `POST` | `/ingest/file` | Upload PDF, DOCX, TXT, or Markdown |
| `POST` | `/ingest/text` | Ingest raw text directly |
| `DELETE` | `/collection` | Reset the entire vector store |

### Example: Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the refund terms?", "use_guardrails": true, "top_k": 5}'
```

Response:

```json
{
  "answer": "...",
  "sources": [...],
  "retrieval_stats": {
    "chunks_retrieved": 3,
    "avg_score": 0.71
  },
  "pii_detected": false
}
```

### Example: Ingest a File

```bash
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@/path/to/document.pdf"
```

---

## PII Guardrails

Detected entity types (replaced with `<ENTITY_TYPE>` labels):

`PERSON` · `EMAIL_ADDRESS` · `PHONE_NUMBER` · `US_SSN` · `CREDIT_CARD` · `IP_ADDRESS` · `LOCATION` · `DATE_TIME` · `US_PASSPORT` · `US_DRIVER_LICENSE` · `IBAN_CODE` · `MEDICAL_LICENSE` · `URL` · `US_BANK_NUMBER` · `CRYPTO` · `NRP`

Guardrails can be toggled per-request via the `use_guardrails` field in the query payload, or from the settings sidebar in the Streamlit UI.

---

## Tests

```bash
pytest tests/ -v
```

---

## LLM Provider Switching

Change `LLM_PROVIDER` in `.env` — no code changes needed:

```env
# Use Gemini (requires GOOGLE_API_KEY)
LLM_PROVIDER=gemini

# Use local Ollama
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

Ollama supports any model available locally (`ollama pull llama3.2`). The `OLLAMA_BASE_URL` can also point to a remote HTTP endpoint (e.g., a Colab tunnel).

---

## License

MIT
