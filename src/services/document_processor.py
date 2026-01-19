"""
Document Processing Service.

Handles document ingestion, chunking, and indexing for the RAG pipeline.
Supports dynamic document uploads from users.

Features:
- SEMANTIC CHUNKING: Splits by topic/meaning, not character count
- Fallback simple chunking for efficiency
- Metadata extraction
- Multiple file format support (txt, md, pdf coming soon)
- Automatic re-indexing on document changes

Chunking Strategies:
1. Semantic Chunking (default): Uses embeddings to detect topic boundaries
2. Simple Chunking: Character-based with sentence awareness (faster)
"""

import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from src.models.schemas import Document
from src.services.vector_store import FAISSVectorStore, get_vector_store

logger = logging.getLogger(__name__)

# Default chunk settings
DEFAULT_CHUNK_SIZE = 500  # characters for simple chunking
DEFAULT_CHUNK_OVERLAP = 50  # characters
SEMANTIC_MIN_CHUNK = 100  # min chars for semantic chunks
SEMANTIC_MAX_CHUNK = 1500  # max chars for semantic chunks


class DocumentProcessor:
    """
    Processes and chunks documents for indexing in the vector database.
    
    This enables dynamic document uploads instead of static knowledge bases.
    
    Supports two chunking strategies:
    1. "semantic" (default): Uses embeddings to detect topic boundaries
       - Better for complex documents with multiple topics
       - Preserves context and meaning
       - Slower but higher quality
       
    2. "simple": Character-based with sentence awareness
       - Faster processing
       - Good for uniform documents
       - Use when speed matters more than precision
    """
    
    def __init__(
        self,
        chunk_strategy: Literal["semantic", "simple"] = "semantic",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        upload_dir: str = "./uploads",
        semantic_threshold: float = 0.5
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_strategy: "semantic" or "simple" chunking.
            chunk_size: Maximum characters per chunk (for simple chunking).
            chunk_overlap: Overlap between chunks for context.
            upload_dir: Directory to store uploaded files.
            semantic_threshold: Similarity threshold for semantic chunking.
        """
        self.chunk_strategy = chunk_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_threshold = semantic_threshold
        
        # Lazy load semantic chunker (expensive to initialize)
        self._semantic_chunker = None
        
        # Track processed documents
        self._document_hashes: dict[str, str] = {}
    
    def _get_semantic_chunker(self):
        """Lazy load semantic chunker."""
        if self._semantic_chunker is None:
            from src.services.semantic_chunker import SemanticChunker
            self._semantic_chunker = SemanticChunker(
                similarity_threshold=self.semantic_threshold,
                min_chunk_size=SEMANTIC_MIN_CHUNK,
                max_chunk_size=SEMANTIC_MAX_CHUNK
            )
        return self._semantic_chunker
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using semantic chunking.

        Args:
            text: The text to chunk.

        Returns:
            List of semantically coherent chunks.
        """
        # Always use semantic chunking for better quality
        return self._semantic_chunk(text)
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Split text using semantic chunking.
        
        Uses embeddings to detect topic boundaries and creates
        coherent chunks that preserve meaning.
        
        Args:
            text: Text to chunk.
            
        Returns:
            List of semantically coherent chunks.
        """
        try:
            chunker = self._get_semantic_chunker()
            semantic_chunks = chunker.chunk_with_overlap(text, overlap_sentences=1)
            return [chunk.text for chunk in semantic_chunks]
        except Exception as e:
            logger.warning(f"Semantic chunking failed, falling back to simple: {e}")
            return self._simple_chunk(text)
    
    def _simple_chunk(
        self,
        text: str,
        chunk_size: int | None = None,
        overlap: int | None = None
    ) -> List[str]:
        """
        Simple character-based chunking with sentence awareness.
        
        Args:
            text: The text to chunk.
            chunk_size: Max characters per chunk.
            overlap: Overlap between chunks.
            
        Returns:
            List of text chunks.
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in ['. ', '.\n', '! ', '? ', '\n\n']:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - overlap if end < len(text) else len(text)
        
        return chunks
    
    def extract_metadata(self, content: str, filename: str) -> dict:
        """
        Extract metadata from document content.
        
        Args:
            content: Document content.
            filename: Original filename.
            
        Returns:
            Metadata dictionary.
        """
        # Extract title from first line or filename
        lines = content.strip().split('\n')
        title = lines[0].strip('#').strip() if lines else filename
        
        # Detect category from content keywords
        category = "General"
        category_keywords = {
            "Domain Policies": ["suspend", "domain", "registr", "whois"],
            "Billing": ["payment", "refund", "invoice", "billing", "renew"],
            "Technical": ["dns", "nameserver", "mx record", "cname", "ip address"],
            "Security": ["abuse", "phishing", "malware", "security", "hack"],
            "Account": ["password", "login", "account", "authentication"],
        }
        
        content_lower = content.lower()
        for cat, keywords in category_keywords.items():
            if any(kw in content_lower for kw in keywords):
                category = cat
                break
        
        return {
            "title": title[:100],  # Limit title length
            "category": category,
            "filename": filename,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": hashlib.md5(content.encode()).hexdigest()
        }
    
    def process_text(
        self,
        content: str,
        filename: str,
        category: str | None = None
    ) -> List[Document]:
        """
        Process raw text into chunked documents.
        
        Args:
            content: Raw text content.
            filename: Source filename.
            category: Optional category override.
            
        Returns:
            List of Document objects ready for indexing.
        """
        # Extract metadata
        metadata = self.extract_metadata(content, filename)
        if category:
            metadata["category"] = category
        
        # Chunk the content
        chunks = self.chunk_text(content)
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}-chunk-{i}"
            section = f"Chunk {i+1} of {len(chunks)}"
            
            documents.append(Document(
                id=doc_id,
                title=metadata["title"],
                content=chunk,
                category=metadata["category"],
                section=section
            ))
        
        logger.info(f"Processed '{filename}' into {len(documents)} chunks")
        return documents
    
    def save_uploaded_file(self, filename: str, content: bytes) -> Path:
        """
        Save an uploaded file to disk.
        
        Args:
            filename: Original filename.
            content: File content bytes.
            
        Returns:
            Path to saved file.
        """
        # Sanitize filename
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        file_path = self.upload_dir / safe_filename
        
        file_path.write_bytes(content)
        logger.info(f"Saved uploaded file: {file_path}")
        
        return file_path
    
    def process_uploaded_file(
        self,
        filename: str,
        content: bytes,
        category: str | None = None
    ) -> Tuple[List[Document], str]:
        """
        Process an uploaded file into documents.
        
        Args:
            filename: Original filename.
            content: File content bytes.
            category: Optional category.
            
        Returns:
            Tuple of (documents, file_path).
        """
        # Save file
        file_path = self.save_uploaded_file(filename, content)
        
        # Read and process content
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            text_content = content.decode('latin-1')
        
        documents = self.process_text(text_content, filename, category)
        
        return documents, str(file_path)
    
    def list_uploaded_files(self) -> List[dict]:
        """List all uploaded files with metadata."""
        files = []
        for file_path in self.upload_dir.glob('*'):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": str(file_path)
                })
        return files
    
    def delete_file(self, filename: str) -> bool:
        """
        Delete an uploaded file.
        
        Args:
            filename: File to delete.
            
        Returns:
            True if deleted, False if not found.
        """
        file_path = self.upload_dir / filename
        if file_path.exists():
            file_path.unlink()
            logger.info(f"Deleted file: {filename}")
            return True
        return False


# Singleton instance
_document_processor: DocumentProcessor | None = None


def get_document_processor() -> DocumentProcessor:
    """Get or create the singleton document processor."""
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor
