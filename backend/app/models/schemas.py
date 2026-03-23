from pydantic import BaseModel, Field
from typing import Optional, List


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Session ID for conversation history")
    stream: bool = Field(False, description="Enable streaming response")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=3, ge=1, le=20)


class SearchResult(BaseModel):
    score: float
    text: str
    filename: str
    chunk_index: int


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class ChatResponse(BaseModel):
    session_id: str
    query: str
    answer: str
    retrieved_chunks: List[str]
    sources: List[str]


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[ChatMessage]


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_created: int


class DeleteResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm_provider: str
    embed_model: str
    version: str
    collection_info: dict
