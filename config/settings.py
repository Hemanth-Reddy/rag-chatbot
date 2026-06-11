# # config/settings.py
# # Feature 1: Configuration Loading
# # Reads all settings from .env file.
# # Every other module will import `settings` from here — never hardcode values.

# from pydantic_settings import BaseSettings
# from pydantic import Field


# class Settings(BaseSettings):

#     # ── LLM ───────────────────────────────────────────────────────────
#     google_api_key: str = Field(..., env="GOOGLE_API_KEY")
#     llm_model: str = Field("gemini-flash-lite-latest", env="LLM_MODEL")
#     llm_temperature: float = Field(0.2, env="LLM_TEMPERATURE")
#     llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")

#     # ── ChromaDB ──────────────────────────────────────────────────────
#     chroma_persist_dir: str = Field("./chroma_store", env="CHROMA_PERSIST_DIR")
#     chroma_collection_name: str = Field("rag_documents", env="CHROMA_COLLECTION_NAME")

#     # ── Embeddings ────────────────────────────────────────────────────
#     embedding_model: str = Field(
#         "sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
#     )

#     # ── RAG retrieval ─────────────────────────────────────────────────
#     rag_top_k: int = Field(5, env="RAG_TOP_K")
#     rag_score_threshold: float = Field(0.3, env="RAG_SCORE_THRESHOLD")

#     # ── API server ────────────────────────────────────────────────────
#     api_host: str = Field("0.0.0.0", env="API_HOST")
#     api_port: int = Field(8000, env="API_PORT")

#     class Config:
#         env_file = ".env"
#         env_file_encoding = "utf-8"


# # Single shared instance.
# # Import this object everywhere — never instantiate Settings() again.
# settings = Settings()

# config/settings.py
# Feature 1: Configuration Loading
# Feature 11: Added Ollama settings

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):

    # ── LLM (Gemini) ──────────────────────────────────────────────────
    google_api_key: str = Field("", env="GOOGLE_API_KEY")
    llm_model: str = Field("gemini-flash-lite-latest", env="LLM_MODEL")
    llm_temperature: float = Field(0.2, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(2048, env="LLM_MAX_TOKENS")

    # ── Ollama (Colab / self-hosted) ───────────────────────────────────
    ollama_base_url: str = Field("", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.2:3b", env="OLLAMA_MODEL")

    # ── LLM provider switch ───────────────────────────────────────────
    # Set to "ollama" to use Colab Llama, "gemini" to use Gemini API
    llm_provider: str = Field("gemini", env="LLM_PROVIDER")

    # ── ChromaDB ──────────────────────────────────────────────────────
    chroma_persist_dir: str = Field("./chroma_store", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("rag_documents", env="CHROMA_COLLECTION_NAME")

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # ── RAG retrieval ─────────────────────────────────────────────────
    rag_top_k: int = Field(5, env="RAG_TOP_K")
    rag_score_threshold: float = Field(0.3, env="RAG_SCORE_THRESHOLD")

    # ── API server ────────────────────────────────────────────────────
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()