# tests/test_embeddings.py
# Feature 3 test — no LLM, no ChromaDB, no internet after first run.
# First run downloads the model (~90MB). Subsequent runs are instant.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from embeddings.embedder import embed_query, embed_texts, get_embeddings


@pytest.fixture(scope="module")
def embedder():
    """Load the model once for all tests in this file."""
    return get_embeddings()


class TestModelLoads:
    def test_embedder_loads(self, embedder):
        assert embedder is not None

    def test_singleton_same_instance(self, embedder):
        """Calling get_embeddings() twice returns the exact same object."""
        from embeddings.embedder import get_embeddings
        second = get_embeddings()
        assert embedder is second


class TestEmbedQuery:
    def test_returns_list(self, embedder):
        vec = embed_query("What is ChromaDB?")
        assert isinstance(vec, list)

    def test_correct_dimensions(self, embedder):
        """all-MiniLM-L6-v2 always produces 384-dimensional vectors."""
        vec = embed_query("Hello world")
        assert len(vec) == 384

    def test_values_are_floats(self, embedder):
        vec = embed_query("Hello world")
        assert all(isinstance(v, float) for v in vec)

    def test_different_texts_different_vectors(self, embedder):
        vec1 = embed_query("Python programming language")
        vec2 = embed_query("Chocolate chip cookies recipe")
        assert vec1 != vec2

    def test_similar_texts_similar_vectors(self, embedder):
        """Semantically similar sentences should have high cosine similarity."""
        import math
        vec1 = embed_query("What is machine learning?")
        vec2 = embed_query("Explain machine learning to me.")
        # Cosine similarity of normalized vectors = dot product
        dot = sum(a * b for a, b in zip(vec1, vec2))
        assert dot > 0.8, f"Expected similar vectors, got similarity: {dot:.3f}"


class TestEmbedTexts:
    def test_embed_multiple_texts(self, embedder):
        texts = [
            "ChromaDB is a vector database.",
            "LangChain is a framework for LLMs.",
            "FastAPI is a web framework.",
        ]
        vecs = embed_texts(texts)
        assert len(vecs) == 3

    def test_each_vector_correct_dimensions(self, embedder):
        texts = ["First sentence.", "Second sentence."]
        vecs = embed_texts(texts)
        for vec in vecs:
            assert len(vec) == 384

    def test_single_text_list(self, embedder):
        vecs = embed_texts(["Just one sentence."])
        assert len(vecs) == 1
        assert len(vecs[0]) == 384