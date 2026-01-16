"""
Vector store service using FAISS.

FAISS (Facebook AI Similarity Search) is the vector database used for
efficient similarity search in the RAG pipeline.

Features:
- Stores document embeddings for semantic search
- Supports persistence (save/load index to disk)
- Efficient cosine similarity using Inner Product
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from src.config import get_settings
from src.models.schemas import Document, RetrievedContext
from src.services.embedding import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based Vector Database for document similarity search.
    
    This is the vector database component of the RAG pipeline.
    FAISS stores document embeddings and enables fast similarity search
    to retrieve relevant context for customer queries.
    
    Features:
    - Efficient cosine similarity search using IndexFlatIP
    - Batch document indexing with automatic embedding generation
    - Persistence support (save/load to disk)
    - Metadata storage for document retrieval
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        dimension: int | None = None,
        index_path: str | None = None
    ):
        """
        Initialize the FAISS vector database.
        
        Args:
            embedding_service: Service for generating text embeddings.
            dimension: Embedding vector dimension (auto-detected if not provided).
            index_path: Optional path to load existing index from disk.
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.dimension = dimension or self.embedding_service.get_embedding_dimension()
        
        # Initialize FAISS index
        # Using IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        
        # Document metadata storage (FAISS only stores vectors, not metadata)
        self.documents: List[Document] = []
        
        # Load existing index if path provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
        
        logger.info(f"Initialized FAISS vector database with dimension {self.dimension}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity.
        
        With normalized vectors, Inner Product equals Cosine Similarity.
        This allows FAISS to perform cosine similarity search efficiently.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Prevent division by zero
        return vectors / norms
    
    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector database.
        
        This method:
        1. Generates embeddings for each document using Sentence Transformers
        2. Normalizes embeddings for cosine similarity
        3. Adds vectors to the FAISS index
        4. Stores document metadata for retrieval
        
        Args:
            documents: List of Document objects to index.
            
        Returns:
            Number of documents added.
        """
        if not documents:
            return 0
        
        # Generate embeddings for document content
        # Combine title and content for better semantic representation
        texts = [f"{doc.title}\n{doc.content}" for doc in documents]
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Normalize vectors for cosine similarity
        normalized = self._normalize_vectors(embeddings)
        
        # Add to FAISS index
        self.index.add(normalized.astype(np.float32))
        
        # Store document metadata
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to FAISS. Total indexed: {self.index.ntotal}")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None
    ) -> List[RetrievedContext]:
        """
        Search for similar documents in the vector database.
        
        This method:
        1. Generates embedding for the query
        2. Performs similarity search in FAISS
        3. Filters results by similarity threshold
        4. Returns documents with similarity scores
        
        Args:
            query: The search query text.
            top_k: Maximum number of results to return.
            threshold: Minimum similarity score (0.0 to 1.0).
            
        Returns:
            List of RetrievedContext with documents and scores.
        """
        settings = get_settings()
        top_k = top_k or settings.top_k_results
        threshold = threshold if threshold is not None else settings.similarity_threshold
        
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty - no documents indexed")
            return []
        
        # Generate and normalize query embedding
        query_embedding = self.embedding_service.embed_text(query)
        query_normalized = self._normalize_vectors(query_embedding)
        
        # Search FAISS index
        # scores: similarity scores (cosine similarity for normalized vectors)
        # indices: indices of matching documents
        scores, indices = self.index.search(
            query_normalized.astype(np.float32),
            min(top_k, self.index.ntotal)
        )
        
        # Build results with threshold filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            if score < threshold:
                continue
            
            results.append(RetrievedContext(
                document=self.documents[idx],
                similarity_score=float(score)
            ))
        
        logger.info(f"FAISS search returned {len(results)} documents (threshold: {threshold})")
        return results
    
    def save(self, directory: str) -> None:
        """
        Save the vector database to disk.
        
        Saves both the FAISS index and document metadata.
        
        Args:
            directory: Directory path to save the index.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save document metadata
        metadata_path = path / "documents.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.documents, f)
        
        logger.info(f"Saved FAISS index to {directory} ({self.index.ntotal} vectors)")
    
    def load(self, directory: str) -> None:
        """
        Load the vector database from disk.
        
        Args:
            directory: Directory path containing saved index.
        """
        path = Path(directory)
        
        # Load FAISS index
        index_path = path / "faiss.index"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load document metadata
        metadata_path = path / "documents.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.documents = pickle.load(f)
        
        logger.info(f"Loaded FAISS index from {directory} ({self.index.ntotal} vectors)")
    
    def clear(self) -> None:
        """Clear all documents from the vector database."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents = []
        logger.info("FAISS vector database cleared")
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector database."""
        return len(self.documents)
    
    def get_stats(self) -> dict:
        """Get vector database statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_type": "IndexFlatIP (Cosine Similarity)"
        }


# Backward compatible alias
VectorStore = FAISSVectorStore

# Singleton instance
_vector_store: FAISSVectorStore | None = None


def get_vector_store() -> FAISSVectorStore:
    """Get or create the singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    return _vector_store


def initialize_vector_store(documents: List[Document], force_reinit: bool = False) -> FAISSVectorStore:
    """
    Initialize the vector database with documents.

    This is the main entry point for setting up the RAG retrieval system.
    Documents are embedded and indexed in FAISS for similarity search.

    Args:
        documents: List of documents to index in the vector database.
        force_reinit: If True, clears existing documents. If False, adds to existing.

    Returns:
        Initialized FAISSVectorStore instance.
    """
    global _vector_store

    if _vector_store is None:
        # Create new vector store
        _vector_store = FAISSVectorStore()
        _vector_store.add_documents(documents)
    elif force_reinit:
        # Force reinitialization - clear and add only base documents
        _vector_store.clear()
        _vector_store.add_documents(documents)
    else:
        # Check if base documents are already indexed
        existing_ids = {doc.id for doc in _vector_store.documents}
        new_docs = [doc for doc in documents if doc.id not in existing_ids]
        if new_docs:
            _vector_store.add_documents(new_docs)
            logger.info(f"Added {len(new_docs)} new base documents to existing vector store")
        else:
            logger.info("Base documents already indexed, preserving existing documents")

    return _vector_store
