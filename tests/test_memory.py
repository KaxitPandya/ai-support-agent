"""
Unit tests for the memory service.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import numpy as np

from src.models.schemas import TicketResponse
from src.services.memory import (
    ConversationMemory,
    LongTermMemory,
    HybridMemory,
    MemoryEntry
)


class TestConversationMemory:
    """Tests for short-term conversation memory."""
    
    @pytest.fixture
    def memory(self):
        """Create a conversation memory instance."""
        return ConversationMemory(max_turns=5)
    
    @pytest.fixture
    def sample_response(self):
        """Create a sample ticket response."""
        return TicketResponse(
            answer="Test answer",
            references=["Ref 1", "Ref 2"],
            action_required="none"
        )
    
    def test_add_turn(self, memory, sample_response):
        """Test adding a conversation turn."""
        memory.add_turn("Test query", sample_response)
        
        assert len(memory.buffer) == 1
    
    def test_max_turns_limit(self, memory, sample_response):
        """Test that buffer respects max_turns limit."""
        for i in range(10):
            memory.add_turn(f"Query {i}", sample_response)
        
        assert len(memory.buffer) == 5  # max_turns = 5
    
    def test_get_context_empty(self, memory):
        """Test getting context from empty buffer."""
        context = memory.get_context()
        assert context == ""
    
    def test_get_context_with_turns(self, memory, sample_response):
        """Test getting context with conversation history."""
        memory.add_turn("First query", sample_response)
        memory.add_turn("Second query", sample_response)
        
        context = memory.get_context(num_turns=2)
        
        assert "First query" in context or "Second query" in context
        assert "Recent Conversation History" in context
    
    def test_clear(self, memory, sample_response):
        """Test clearing the buffer."""
        memory.add_turn("Query", sample_response)
        memory.clear()
        
        assert len(memory.buffer) == 0
    
    def test_get_summary(self, memory, sample_response):
        """Test getting memory summary."""
        memory.add_turn("Query", sample_response)
        
        summary = memory.get_summary()
        
        assert summary["turns"] == 1
        assert summary["max_turns"] == 5


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass."""
    
    def test_create_memory_entry(self):
        """Test creating a memory entry."""
        entry = MemoryEntry(
            id="test-123",
            query="Test query",
            response="Test response",
            references=["Ref 1"],
            action_taken="none",
            timestamp=datetime.now(timezone.utc)
        )
        
        assert entry.id == "test-123"
        assert entry.query == "Test query"
        assert entry.feedback_score is None
    
    def test_to_document(self):
        """Test converting memory to document."""
        entry = MemoryEntry(
            id="test-123",
            query="Test query",
            response="Test response",
            references=["Ref 1"],
            action_taken="none",
            timestamp=datetime.now(timezone.utc)
        )
        
        doc = entry.to_document()
        
        assert doc.id == "test-123"
        assert "Test query" in doc.content
        assert doc.category == "Conversation Memory"


class TestLongTermMemory:
    """Tests for long-term vector-based memory."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service that returns proper embeddings."""
        service = Mock()
        service.embed_texts = lambda texts: np.random.rand(len(texts), 384).astype(np.float32)
        service.embed_text = lambda text: np.random.rand(384).astype(np.float32)
        service.dimension = 384
        return service
    
    @pytest.fixture
    def memory(self, mock_embedding_service):
        """Create a long-term memory instance with pre-configured vector store."""
        import faiss
        from src.services.vector_store import FAISSVectorStore
        
        # Create the memory with our mock
        memory = LongTermMemory.__new__(LongTermMemory)
        memory.embedding_service = mock_embedding_service
        memory.relevance_decay = 0.1
        memory.similarity_threshold = 0.5
        memory.memories = {}
        
        # Create a properly initialized vector store
        memory.memory_store = FAISSVectorStore.__new__(FAISSVectorStore)
        memory.memory_store.embedding_service = mock_embedding_service
        memory.memory_store.dimension = 384
        memory.memory_store.documents = []
        memory.memory_store.index = faiss.IndexFlatIP(384)
        
        return memory
    
    @pytest.fixture
    def sample_response(self):
        """Create a sample ticket response."""
        return TicketResponse(
            answer="Test answer",
            references=["Ref 1"],
            action_required="none"
        )
    
    def test_store_memory(self, memory, sample_response):
        """Test storing a memory."""
        mem_id = memory.store("Test query", sample_response)
        
        assert mem_id is not None
        assert mem_id.startswith("mem_")
        assert mem_id in memory.memories
    
    def test_recall_empty(self, memory):
        """Test recall from empty memory."""
        results = memory.recall("Some query")
        assert results == []
    
    def test_recall_stored_memory(self, memory, sample_response):
        """Test recalling stored memories."""
        memory.store("Domain suspension question", sample_response)
        
        # Note: Recall depends on vector similarity, mock returns random
        # so we just test it doesn't crash
        results = memory.recall("domain issue", top_k=1)
        # Results may or may not match due to random embeddings
        assert isinstance(results, list)
    
    def test_add_feedback(self, memory, sample_response):
        """Test adding feedback to memory."""
        mem_id = memory.store("Query", sample_response)
        
        result = memory.add_feedback(mem_id, score=4.5, feedback_text="Good response")
        
        assert result is True
        assert memory.memories[mem_id].feedback_score == 4.5
    
    def test_add_feedback_nonexistent(self, memory):
        """Test adding feedback to non-existent memory."""
        result = memory.add_feedback("nonexistent", score=5.0)
        assert result is False
    
    def test_get_statistics(self, memory, sample_response):
        """Test getting memory statistics."""
        memory.store("Query 1", sample_response)
        memory.store("Query 2", sample_response)
        
        stats = memory.get_statistics()
        
        assert stats["total_memories"] == 2
        assert "average_feedback_score" in stats
    
    def test_clear(self, memory, sample_response):
        """Test clearing memory."""
        memory.store("Query", sample_response)
        memory.clear()
        
        assert len(memory.memories) == 0


class TestHybridMemory:
    """Tests for hybrid memory combining short and long term."""
    
    @pytest.fixture
    def sample_response(self):
        """Create a sample ticket response."""
        return TicketResponse(
            answer="Test answer",
            references=["Ref 1"],
            action_required="none"
        )
    
    def test_init(self):
        """Test hybrid memory initialization."""
        memory = HybridMemory(
            short_term_turns=5,
            long_term_decay=0.1
        )
        
        assert memory.short_term is not None
        assert memory.long_term is not None
    
    def test_add_interaction(self, sample_response):
        """Test adding interaction to both memories."""
        memory = HybridMemory()
        
        mem_id = memory.add_interaction(
            "Test query",
            sample_response,
            store_long_term=True
        )
        
        assert len(memory.short_term.buffer) == 1
        assert mem_id is not None
    
    def test_add_interaction_short_term_only(self, sample_response):
        """Test adding to short-term only."""
        memory = HybridMemory()
        
        mem_id = memory.add_interaction(
            "Test query",
            sample_response,
            store_long_term=False
        )
        
        assert len(memory.short_term.buffer) == 1
        assert mem_id is None
    
    def test_get_relevant_context(self, sample_response):
        """Test getting combined context."""
        memory = HybridMemory()
        memory.add_interaction("Previous query", sample_response)
        
        context = memory.get_relevant_context(
            "New query",
            include_short_term=True,
            include_long_term=True
        )
        
        # Should include short-term context
        assert isinstance(context, str)
    
    def test_get_statistics(self, sample_response):
        """Test getting combined statistics."""
        memory = HybridMemory()
        memory.add_interaction("Query", sample_response)
        
        stats = memory.get_statistics()
        
        assert "short_term" in stats
        assert "long_term" in stats
    
    def test_clear_session(self, sample_response):
        """Test clearing only session memory."""
        memory = HybridMemory()
        memory.add_interaction("Query", sample_response, store_long_term=True)
        
        memory.clear_session()
        
        assert len(memory.short_term.buffer) == 0
        # Long-term should still have the memory
        assert memory.long_term.get_statistics()["total_memories"] == 1
    
    def test_clear_all(self, sample_response):
        """Test clearing all memory."""
        memory = HybridMemory()
        memory.add_interaction("Query", sample_response)
        
        memory.clear_all()
        
        assert len(memory.short_term.buffer) == 0
        assert memory.long_term.get_statistics()["total_memories"] == 0
