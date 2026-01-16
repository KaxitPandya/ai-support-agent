"""
Unit tests for the embedding service.
"""

import numpy as np
import pytest

from src.services.embedding import EmbeddingService, get_embedding_service


class TestEmbeddingService:
    """Tests for EmbeddingService."""
    
    @pytest.fixture
    def embedding_service(self):
        """Create an embedding service for testing."""
        return EmbeddingService()
    
    def test_embed_text_returns_numpy_array(self, embedding_service):
        """Test that embed_text returns a numpy array."""
        text = "This is a test sentence for embedding."
        embedding = embedding_service.embed_text(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.ndim == 1
        assert len(embedding) > 0
    
    def test_embed_text_consistent(self, embedding_service):
        """Test that same text produces same embedding."""
        text = "Domain registration and transfer policies."
        
        embedding1 = embedding_service.embed_text(text)
        embedding2 = embedding_service.embed_text(text)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_embed_texts_batch(self, embedding_service):
        """Test batch embedding of multiple texts."""
        texts = [
            "My domain was suspended.",
            "How do I renew my domain?",
            "DNS settings are not working."
        ]
        
        embeddings = embedding_service.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.ndim == 2
    
    def test_embed_texts_empty_list(self, embedding_service):
        """Test embedding empty list."""
        embeddings = embedding_service.embed_texts([])
        
        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 0
    
    def test_get_embedding_dimension(self, embedding_service):
        """Test getting embedding dimension."""
        dimension = embedding_service.get_embedding_dimension()
        
        assert isinstance(dimension, int)
        assert dimension > 0
        # all-MiniLM-L6-v2 has 384 dimensions
        assert dimension == 384
    
    def test_similar_texts_have_similar_embeddings(self, embedding_service):
        """Test that semantically similar texts have similar embeddings."""
        text1 = "My domain name was suspended yesterday."
        text2 = "The domain has been suspended."
        text3 = "I want to order pizza tonight."
        
        emb1 = embedding_service.embed_text(text1)
        emb2 = embedding_service.embed_text(text2)
        emb3 = embedding_service.embed_text(text3)
        
        # Normalize and compute cosine similarity
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim_12 = cosine_sim(emb1, emb2)
        sim_13 = cosine_sim(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13
    
    def test_singleton_embedding_service(self):
        """Test that get_embedding_service returns singleton."""
        service1 = get_embedding_service()
        service2 = get_embedding_service()
        
        assert service1 is service2
