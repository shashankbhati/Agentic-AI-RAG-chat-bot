import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_collection")
    EMBED_MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "phi3:mini")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    TOP_K: int = int(os.getenv("TOP_K", "3"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))

    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:4173"
    ).split(",")

    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
    RATE_LIMIT_WINDOW: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
    SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "24"))
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "50"))

    # ── Auth ──────────────────────────────────────────────────────────────────
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me-in-production-please")


settings = Settings()
