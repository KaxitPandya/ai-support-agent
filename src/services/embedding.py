"""
Embedding service using Sentence Transformers.

Provides text embedding functionality for the RAG pipeline.
Uses the all-MiniLM-L6-v2 model by default for good balance of speed and quality.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using Sentence Transformers.
    
    The embedding model is loaded lazily on first use to reduce startup time.
    """
    
    def __init__(self, model_name: str | None = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Optional model name override. Defaults to config setting.
        """
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model: SentenceTransformer | None = None
        
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded successfully")
        return self._model
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed.
            
        Returns:
            Numpy array of the embedding vector.
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        if not texts:
            return np.array([])
        
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()


# Singleton instance for reuse
_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
