import logging
from typing import List
from app.config import settings

logger = logging.getLogger(__name__)


class EmbedderService:
    """Singleton embedding service — model loaded once at startup."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = None
        return cls._instance

    def load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {settings.EMBED_MODEL_NAME}")
            self._model = SentenceTransformer(settings.EMBED_MODEL_NAME)
            logger.info("Embedding model loaded successfully")

    def encode(self, text: str) -> List[float]:
        if self._model is None:
            self.load()
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            self.load()
        embeddings = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()


embedder = EmbedderService()
