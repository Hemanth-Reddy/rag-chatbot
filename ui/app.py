# ui/app.py
# Feature 13: Streamlit UI
# Connects to the RAG API at http://localhost:8000
# Run with: streamlit run ui/app.py

import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Assistant")
st.caption("Retrieval-Augmented Generation with PII Guardrails")

# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    agentic_mode = st.toggle("Agentic Mode", value=False, help="LLM picks tools: KB search + web search + date")
    apply_guardrails = st.toggle("PII Guardrails", value=True)
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5, disabled=agentic_mode)
    score_threshold = st.slider(
        "Relevance threshold", min_value=0.0, max_value=1.0, value=0.3, step=0.05, disabled=agentic_mode
    )
    if agentic_mode:
        st.info("Agentic: LLM searches KB, web, and date as needed.")

    st.divider()

    # ── Health check ───────────────────────────────────────────────
    st.header("🖥️ Server Status")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            st.success("API is running")
            st.code(f"Model : {data['model']}\nStore : {data['collection']}")
        else:
            st.error("API returned error")
    except Exception:
        st.error("API is offline\nStart with: python -m api.main")

    st.divider()

    # ── Stats ──────────────────────────────────────────────────────
    st.header("📊 Knowledge Base")
    try:
        resp = requests.get(f"{API_URL}/stats", timeout=3)
        if resp.status_code == 200:
            stats = resp.json()
            st.metric("Documents", stats.get("document_count", 0))
            st.caption(f"Collection: {stats.get('collection_name', '')}")
    except Exception:
        st.caption("Could not load stats")

    st.divider()

    # ── Ingest ─────────────────────────────────────────────────────
    st.header("📥 Ingest")
    ingest_tab, text_tab = st.tabs(["File", "Text"])

    with ingest_tab:
        uploaded_file = st.file_uploader(
            "Upload document", type=["pdf", "txt", "docx", "md"]
        )
        if st.button("Ingest File"):
            if uploaded_file:
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest/file",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        st.success(f"Ingested {resp.json()['chunks_created']} chunks from {uploaded_file.name}!")
                    else:
                        st.error(f"Ingest failed: {resp.json().get('detail', 'unknown error')}")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Upload a file first.")

    with text_tab:
        ingest_text = st.text_area("Paste text to add to knowledge base", height=100)
        ingest_source = st.text_input("Source name", value="manual_input")
        if st.button("Ingest Text"):
            if ingest_text.strip():
                try:
                    resp = requests.post(
                        f"{API_URL}/ingest/text",
                        json={"text": ingest_text, "source_name": ingest_source},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        st.success(f"Ingested {resp.json()['chunks_created']} chunks!")
                    else:
                        st.error("Ingest failed")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter some text first.")

# ── Chat history ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            if meta.get("mode") == "agentic":
                with st.expander("🤖 Agent Details"):
                    col1, col2 = st.columns(2)
                    col1.metric("PII in query", "Yes" if meta["input_pii"] else "No")
                    col2.metric("PII in response", "Yes" if meta["output_pii"] else "No")
                    if meta.get("tool_calls"):
                        st.subheader("Tools used")
                        for tc in meta["tool_calls"]:
                            st.markdown(f"**{tc['tool']}** — args: `{tc['args']}`")
                            st.caption(tc["result_preview"])
                    if meta["input_pii_entities"]:
                        st.warning(f"PII in query: {[e['entity_type'] for e in meta['input_pii_entities']]}")
            else:
                with st.expander("📎 Sources & Details"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Chunks retrieved", meta["retrieved_chunks"])
                    col2.metric("PII in query", "Yes" if meta["input_pii"] else "No")
                    col3.metric("PII in response", "Yes" if meta["output_pii"] else "No")

                    if meta["sources"]:
                        st.subheader("Source documents")
                        for i, src in enumerate(meta["sources"], 1):
                            st.markdown(
                                f"**{i}. {src['metadata'].get('source', 'unknown')}** "
                                f"— relevance: `{src['relevance_score']}`"
                            )
                            st.caption(src["content_preview"])

                    if meta["input_pii_entities"]:
                        st.warning(
                            f"PII detected in your query: "
                            f"{[e['entity_type'] for e in meta['input_pii_entities']]}"
                        )

# ── Query input ────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about your documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call API
    with st.chat_message("assistant"):
        spinner_msg = "Agent thinking..." if agentic_mode else "Thinking..."
        with st.spinner(spinner_msg):
            try:
                if agentic_mode:
                    resp = requests.post(
                        f"{API_URL}/agent/query",
                        json={
                            "query": prompt,
                            "apply_guardrails": apply_guardrails,
                        },
                        timeout=120,
                    )
                else:
                    resp = requests.post(
                        f"{API_URL}/query",
                        json={
                            "query": prompt,
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                            "apply_guardrails": apply_guardrails,
                        },
                        timeout=120,
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    st.markdown(answer)

                    if agentic_mode:
                        meta = {
                            "mode": "agentic",
                            "input_pii": data["guardrails"]["input_pii_detected"],
                            "output_pii": data["guardrails"]["output_pii_detected"],
                            "input_pii_entities": data["guardrails"]["input_pii_entities"],
                            "tool_calls": data["tool_calls"],
                        }
                        with st.expander("🤖 Agent Details"):
                            col1, col2 = st.columns(2)
                            col1.metric("PII in query", "Yes" if meta["input_pii"] else "No")
                            col2.metric("PII in response", "Yes" if meta["output_pii"] else "No")
                            if meta["tool_calls"]:
                                st.subheader("Tools used")
                                for tc in meta["tool_calls"]:
                                    st.markdown(f"**{tc['tool']}** — args: `{tc['args']}`")
                                    st.caption(tc["result_preview"])
                            if meta["input_pii_entities"]:
                                st.warning(f"PII in query: {[e['entity_type'] for e in meta['input_pii_entities']]}")
                    else:
                        meta = {
                            "mode": "standard",
                            "retrieved_chunks": data["retrieved_chunks"],
                            "input_pii": data["guardrails"]["input_pii_detected"],
                            "output_pii": data["guardrails"]["output_pii_detected"],
                            "input_pii_entities": data["guardrails"]["input_pii_entities"],
                            "sources": data["source_documents"],
                        }
                        with st.expander("📎 Sources & Details"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Chunks retrieved", meta["retrieved_chunks"])
                            col2.metric("PII in query", "Yes" if meta["input_pii"] else "No")
                            col3.metric("PII in response", "Yes" if meta["output_pii"] else "No")

                            if meta["sources"]:
                                st.subheader("Source documents")
                                for i, src in enumerate(meta["sources"], 1):
                                    st.markdown(
                                        f"**{i}. {src['metadata'].get('source', 'unknown')}** "
                                        f"— relevance: `{src['relevance_score']}`"
                                    )
                                    st.caption(src["content_preview"])

                            if meta["input_pii_entities"]:
                                st.warning(
                                    f"PII detected in your query: "
                                    f"{[e['entity_type'] for e in meta['input_pii_entities']]}"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": meta,
                    })

                else:
                    error = resp.json().get("detail", "Unknown error")
                    st.error(f"API error: {error}")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach API. Start it with: python -m api.main")
            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")