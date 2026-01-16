"""
Document Upload API Endpoints.

Provides endpoints for dynamic document management:
- Upload new documents to the knowledge base
- List uploaded documents
- Delete documents
- Reprocess/reindex documents

This enables live document updates without code changes.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.services.document_processor import DocumentProcessor, get_document_processor
from src.services.rag import get_rag_pipeline
from src.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["Document Management"])


# =============================================================================
# Response Models
# =============================================================================

class UploadResponse(BaseModel):
    """Response for document upload."""
    success: bool
    filename: str
    chunks_created: int
    message: str


class FileInfo(BaseModel):
    """Information about an uploaded file."""
    filename: str
    size_bytes: int
    modified_at: str
    path: str


class FileListResponse(BaseModel):
    """Response for listing files."""
    files: List[FileInfo]
    total_count: int


class DeleteResponse(BaseModel):
    """Response for file deletion."""
    success: bool
    filename: str
    message: str


class ReindexResponse(BaseModel):
    """Response for reindex operation."""
    success: bool
    documents_indexed: int
    message: str


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload (txt, md)"),
    category: Optional[str] = Form(None, description="Document category"),
    index_immediately: bool = Form(True, description="Add to vector index immediately")
):
    """
    Upload a document to the knowledge base.
    
    The document will be:
    1. Saved to the uploads directory
    2. Chunked into smaller pieces
    3. Embedded and indexed in the vector database (if index_immediately=True)
    
    Supported formats: .txt, .md
    
    **Example:**
    ```bash
    curl -X POST "http://localhost:8000/api/documents/upload" \\
         -F "file=@policy.md" \\
         -F "category=Domain Policies" \\
         -F "index_immediately=true"
    ```
    """
    # Validate file type
    allowed_extensions = {'.txt', '.md', '.markdown'}
    file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Process the document
        processor = get_document_processor()
        documents, file_path = processor.process_uploaded_file(
            file.filename,
            content,
            category
        )
        
        # Index if requested
        if index_immediately and documents:
            vector_store = get_vector_store()
            vector_store.add_documents(documents)
        
        return UploadResponse(
            success=True,
            filename=file.filename,
            chunks_created=len(documents),
            message=f"Successfully uploaded and {'indexed' if index_immediately else 'saved'}"
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/files", response_model=FileListResponse)
async def list_files():
    """
    List all uploaded documents.
    
    Returns metadata about each file including size and modification time.
    """
    processor = get_document_processor()
    files = processor.list_uploaded_files()
    
    return FileListResponse(
        files=[FileInfo(**f) for f in files],
        total_count=len(files)
    )


@router.delete("/files/{filename}", response_model=DeleteResponse)
async def delete_file(filename: str):
    """
    Delete an uploaded document.
    
    Note: This removes the file from disk. The document chunks
    may still exist in the vector index until reindexing.
    
    Args:
        filename: Name of the file to delete.
    """
    processor = get_document_processor()
    
    if processor.delete_file(filename):
        return DeleteResponse(
            success=True,
            filename=filename,
            message="File deleted successfully"
        )
    else:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_all():
    """
    Reindex all uploaded documents.
    
    This clears the current vector index and rebuilds it from:
    1. The synthetic knowledge base (built-in policies)
    2. All uploaded documents
    
    Use this after deleting files or if the index becomes corrupted.
    """
    try:
        from src.data.knowledge_base import get_knowledge_base
        
        # Get all documents
        base_docs = get_knowledge_base()
        
        # Process uploaded files
        processor = get_document_processor()
        uploaded_files = processor.list_uploaded_files()
        
        uploaded_docs = []
        for file_info in uploaded_files:
            try:
                with open(file_info["path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                docs = processor.process_text(content, file_info["filename"])
                uploaded_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to process {file_info['filename']}: {e}")
        
        # Rebuild index
        all_docs = base_docs + uploaded_docs
        
        vector_store = get_vector_store()
        vector_store.clear()
        vector_store.add_documents(all_docs)
        
        return ReindexResponse(
            success=True,
            documents_indexed=len(all_docs),
            message=f"Reindexed {len(base_docs)} base + {len(uploaded_docs)} uploaded documents"
        )
        
    except Exception as e:
        logger.error(f"Reindex failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


@router.get("/stats")
async def get_index_stats():
    """
    Get statistics about the vector index.
    
    Returns information about the number of indexed documents,
    embedding dimensions, and other metrics.
    """
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    processor = get_document_processor()
    uploaded_files = processor.list_uploaded_files()
    
    return {
        **stats,
        "uploaded_files_count": len(uploaded_files),
        "upload_directory": str(processor.upload_dir)
    }
