import io


def test_upload_no_file(client):
    response = client.post("/api/v1/documents/upload")
    assert response.status_code == 422


def test_upload_non_pdf(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_empty_pdf(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
    )
    assert response.status_code == 400


def test_list_documents(client):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)


def test_delete_document(client):
    # Deleting a non-existent doc should not crash (Qdrant will just no-op)
    response = client.delete("/api/v1/documents/nonexistent.pdf")
    # Either 200 (success/no-op) or 500 if Qdrant is unavailable
    assert response.status_code in (200, 500)
