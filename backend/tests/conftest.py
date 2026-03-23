import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

DUMMY_VECTOR = [0.1] * 384


def _make_mock_model():
    mock_model = MagicMock()
    arr = np.array([DUMMY_VECTOR])
    mock_model.encode.return_value = arr
    return mock_model


def _make_mock_qdrant_client():
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock(collections=[])
    mock_client.query_points.return_value = MagicMock(points=[])
    mock_client.scroll.return_value = ([], None)
    return mock_client


@pytest.fixture(scope="session")
def mock_services():
    """Mock all external services so tests run without Qdrant/Ollama."""
    mock_model = _make_mock_model()
    mock_qdrant = _make_mock_qdrant_client()

    # Patch at the source modules (lazy imports inside methods)
    with (
        patch("sentence_transformers.SentenceTransformer", return_value=mock_model),
        patch("qdrant_client.QdrantClient", return_value=mock_qdrant),
        patch("ollama.chat", return_value={"message": {"content": "Test answer"}}),
        patch("ollama.list", return_value=[]),
    ):
        yield {"model": mock_model, "qdrant": mock_qdrant}


@pytest.fixture(scope="session")
def client(mock_services):
    from app.main import app
    with TestClient(app) as c:
        yield c
