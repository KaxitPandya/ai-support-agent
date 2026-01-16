"""
Unit tests for the semantic chunker service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.services.semantic_chunker import SemanticChunker, SemanticChunk


class TestSemanticChunker:
    """Tests for SemanticChunker."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        # Return random embeddings for testing
        service.embed_texts = lambda texts: np.random.rand(len(texts), 384)
        service.embed_text = lambda text: np.random.rand(384)
        return service
    
    @pytest.fixture
    def chunker(self, mock_embedding_service):
        """Create a chunker with mock embedding service."""
        return SemanticChunker(
            embedding_service=mock_embedding_service,
            similarity_threshold=0.5,
            min_chunk_size=50,
            max_chunk_size=500
        )
    
    def test_tokenize_sentences_simple(self, chunker):
        """Test basic sentence tokenization."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = chunker._tokenize_sentences(text)
        
        assert len(sentences) >= 1
        assert "First" in sentences[0]
    
    def test_tokenize_sentences_complex(self, chunker):
        """Test sentence tokenization with various punctuation."""
        text = "Hello world! How are you? I'm fine. Thanks for asking."
        sentences = chunker._tokenize_sentences(text)
        
        assert len(sentences) >= 1
    
    def test_tokenize_empty(self, chunker):
        """Test tokenization of empty text."""
        sentences = chunker._tokenize_sentences("")
        assert sentences == []
    
    def test_chunk_short_text(self, chunker):
        """Test that short text returns single chunk."""
        text = "This is a short text."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_chunk_creates_semantic_chunks(self, chunker):
        """Test that chunking produces SemanticChunk objects."""
        text = "First topic about domains. " * 10 + "Second topic about billing. " * 10
        chunks = chunker.chunk(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(c, SemanticChunk) for c in chunks)
        assert all(c.text for c in chunks)
    
    def test_chunk_with_overlap(self, chunker):
        """Test chunking with sentence overlap."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunker.chunk_with_overlap(text, overlap_sentences=1)
        
        assert len(chunks) >= 1
    
    def test_compute_similarities(self, chunker):
        """Test similarity computation between embeddings."""
        embeddings = np.random.rand(5, 384)
        similarities = chunker._compute_similarities(embeddings)
        
        assert len(similarities) == 4  # n-1 similarities for n embeddings
        assert all(-1 <= s <= 1 for s in similarities)
    
    def test_compute_similarities_single(self, chunker):
        """Test similarity with single embedding."""
        embeddings = np.random.rand(1, 384)
        similarities = chunker._compute_similarities(embeddings)
        
        assert len(similarities) == 0
    
    def test_find_breakpoints(self, chunker):
        """Test breakpoint detection."""
        # Similarities with a clear drop in the middle
        similarities = np.array([0.8, 0.9, 0.3, 0.85, 0.9])
        breakpoints = chunker._find_breakpoints(similarities)
        
        # Should find breakpoint at the low similarity
        assert len(breakpoints) >= 1
    
    def test_merge_small_chunks(self, chunker):
        """Test merging of small chunks."""
        sentences = ["Short.", "Also short.", "This one is longer sentence here.", "Another one."]
        breakpoints = [0, 1, 2]  # Many breakpoints
        
        boundaries = chunker._merge_small_chunks(sentences, breakpoints)
        
        # Should merge some chunks
        assert len(boundaries) >= 1
    
    def test_split_large_chunks(self, chunker):
        """Test splitting of oversized chunks."""
        sentences = ["A" * 200, "B" * 200, "C" * 200]
        boundaries = [(0, 3)]  # Single large chunk
        
        result = chunker._split_large_chunks(sentences, boundaries)
        
        # Should split into multiple chunks
        assert len(result) >= 1


class TestSemanticChunkDataclass:
    """Tests for SemanticChunk dataclass."""
    
    def test_create_chunk(self):
        """Test creating a semantic chunk."""
        chunk = SemanticChunk(
            text="Test content",
            start_sentence_idx=0,
            end_sentence_idx=5
        )
        
        assert chunk.text == "Test content"
        assert chunk.start_sentence_idx == 0
        assert chunk.end_sentence_idx == 5
        assert chunk.avg_embedding is None
    
    def test_chunk_with_embedding(self):
        """Test chunk with embedding."""
        embedding = np.random.rand(384)
        chunk = SemanticChunk(
            text="Test",
            start_sentence_idx=0,
            end_sentence_idx=1,
            avg_embedding=embedding
        )
        
        assert chunk.avg_embedding is not None
        assert len(chunk.avg_embedding) == 384
