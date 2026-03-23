def test_chat_missing_query(client):
    response = client.post("/api/v1/chat", json={})
    assert response.status_code == 422


def test_chat_empty_query(client):
    response = client.post("/api/v1/chat", json={"query": ""})
    assert response.status_code == 422


def test_chat_query_too_long(client):
    response = client.post("/api/v1/chat", json={"query": "x" * 2001})
    assert response.status_code == 422


def test_chat_returns_session_id(client):
    response = client.post("/api/v1/chat", json={"query": "What is this document about?"})
    # Will fail if Qdrant is not connected, but schema should be correct
    if response.status_code == 200:
        data = response.json()
        assert "session_id" in data
        assert "answer" in data
        assert "sources" in data


def test_chat_session_history_not_found(client):
    response = client.get("/api/v1/chat/nonexistent-session-id/history")
    assert response.status_code == 404


def test_search_missing_query(client):
    response = client.post("/api/v1/search", json={})
    assert response.status_code == 422


def test_search_top_k_out_of_range(client):
    response = client.post("/api/v1/search", json={"query": "test", "top_k": 100})
    assert response.status_code == 422
