def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "RAG Chatbot API" in data["name"]
    assert "version" in data


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "qdrant" in data
    assert "llm_provider" in data
    assert "embed_model" in data
    assert "version" in data
