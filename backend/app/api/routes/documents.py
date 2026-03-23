import logging
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from app.models.schemas import UploadResponse, DocumentListResponse, DocumentInfo, DeleteResponse
from app.services.ingest_service import (
    ingest_document,
    ingest_document_to_collection,
    list_documents,
    list_documents_for_collection,
    delete_document,
)
from app.services.usage_tracker import track
from app.api.dependencies import optional_api_key
from app.database.session import get_db
from app.database.models import APIKey
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse, summary="Upload and index a PDF")
async def upload_document(
    file: UploadFile = File(...),
    api_key: Optional[APIKey] = Depends(optional_api_key),
    db: Session = Depends(get_db),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    if len(file_bytes) > settings.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large (max {settings.MAX_UPLOAD_MB}MB)")

    try:
        if api_key:
            result = ingest_document_to_collection(file_bytes, file.filename, api_key.collection_name)
            track(db, api_key.id, "upload", file.filename)
        else:
            result = ingest_document(file_bytes, file.filename)

        return UploadResponse(
            message=f"'{file.filename}' uploaded and indexed successfully",
            filename=result["filename"],
            chunks_created=result["chunk_count"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error for {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process document")


@router.get("/documents", response_model=DocumentListResponse, summary="List indexed documents")
def get_documents(
    api_key: Optional[APIKey] = Depends(optional_api_key),
):
    try:
        if api_key:
            docs = list_documents_for_collection(api_key.collection_name)
        else:
            docs = list_documents()
        return DocumentListResponse(documents=[DocumentInfo(**d) for d in docs], total=len(docs))
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document list")


@router.delete("/documents/{filename:path}", response_model=DeleteResponse, summary="Delete a document")
def remove_document(
    filename: str,
    api_key: Optional[APIKey] = Depends(optional_api_key),
    db: Session = Depends(get_db),
):
    try:
        collection = api_key.collection_name if api_key else None
        delete_document(filename, collection)
        if api_key:
            track(db, api_key.id, "delete", filename)
        return DeleteResponse(message=f"'{filename}' deleted successfully")
    except Exception as e:
        logger.error(f"Delete error for {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete document")
