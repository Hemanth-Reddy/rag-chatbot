# tests/test_ingestion.py
# Feature 6 test — no LLM, no internet needed.
# Creates real temp files and ingests them into an isolated ChromaDB.

import os
import shutil
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.documents import Document
from unittest.mock import patch

from ingest.document_ingester import (
    ingest_file,
    ingest_text,
    ingest_sample_data,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
)
from vectordb.chroma_store import ChromaStore


@pytest.fixture(scope="module")
def isolated_store():
    """Isolated ChromaStore so ingestion tests don't touch real data."""
    tmpdir = tempfile.mkdtemp()
    store = ChromaStore(persist_dir=tmpdir, collection_name="test_ingest")
    with patch("ingest.document_ingester.get_chroma_store", return_value=store):
        yield store
    try:
        store._store = None
        import gc
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def sample_txt_file():
    """Create a real temporary .txt file for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(
            "LangChain is a framework for building LLM applications.\n\n"
            "It provides document loaders, text splitters, and vector store integrations.\n\n"
            "ChromaDB is a vector database that works well with LangChain.\n\n"
            "FastAPI is used to build REST APIs in Python quickly and efficiently."
        )
        return f.name


class TestSupportedExtensions:
    def test_txt_supported(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_pdf_supported(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_docx_supported(self):
        assert ".docx" in SUPPORTED_EXTENSIONS

    def test_md_supported(self):
        assert ".md" in SUPPORTED_EXTENSIONS

    def test_xlsx_not_supported(self):
        assert ".xlsx" not in SUPPORTED_EXTENSIONS


class TestChunkSettings:
    def test_chunk_size_reasonable(self):
        assert 256 <= CHUNK_SIZE <= 1024

    def test_overlap_less_than_chunk(self):
        assert CHUNK_OVERLAP < CHUNK_SIZE

    def test_overlap_positive(self):
        assert CHUNK_OVERLAP > 0


class TestIngestFile:
    def test_ingest_txt_returns_success(self, isolated_store, sample_txt_file):
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_file(sample_txt_file)
        assert result["status"] == "success"

    def test_ingest_txt_creates_chunks(self, isolated_store, sample_txt_file):
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_file(sample_txt_file)
        assert result["chunks_created"] >= 1

    def test_ingest_result_has_filename(self, isolated_store, sample_txt_file):
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_file(sample_txt_file)
        assert result["file"] == os.path.basename(sample_txt_file)

    def test_ingest_result_has_raw_pages(self, isolated_store, sample_txt_file):
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_file(sample_txt_file)
        assert result["raw_pages"] >= 1

    def test_unsupported_extension_raises(self, isolated_store):
        with pytest.raises(ValueError, match="Unsupported file type"):
            ingest_file("somefile.xlsx")

    def test_chunks_stored_in_chroma(self, isolated_store, sample_txt_file):
        before = isolated_store.get_stats()["document_count"]
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_file(sample_txt_file)
        after = isolated_store.get_stats()["document_count"]
        assert after > before


class TestIngestText:
    def test_ingest_text_returns_success(self, isolated_store):
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_text(
                "RAG stands for Retrieval Augmented Generation. "
                "It combines document retrieval with LLM generation.",
                metadata={"source": "test_direct"},
            )
        assert result["status"] == "success"

    def test_ingest_text_creates_chunks(self, isolated_store):
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_text(
                "This is test content for direct text ingestion.",
                metadata={"source": "test_direct"},
            )
        assert result["chunks_created"] >= 1

    def test_ingest_text_default_metadata(self, isolated_store):
        """Ingesting without metadata should not raise."""
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            result = ingest_text("Some plain text content here.")
        assert result["status"] == "success"


class TestIngestSampleData:
    def test_sample_data_ingests_successfully(self, isolated_store):
        before = isolated_store.get_stats()["document_count"]
        with patch("ingest.document_ingester.get_chroma_store", return_value=isolated_store):
            ingest_sample_data()
        after = isolated_store.get_stats()["document_count"]
        assert after > before