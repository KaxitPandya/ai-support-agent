"""
Unit tests for the FAISS vector database service.
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.models.schemas import Document
from src.services.vector_store import (
    FAISSVectorStore,
    VectorStore,
    initialize_vector_store,
    get_vector_store
)
from src.services import vector_store as vector_store_module


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore (the vector database)."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc-1",
                title="Domain Suspension Policy",
                content="Domains may be suspended for WHOIS verification failure or policy violations.",
                category="Policies",
                section="Section 4.1"
            ),
            Document(
                id="doc-2",
                title="Domain Renewal Process",
                content="Domain renewals can be done 1-10 years in advance. Auto-renewal is enabled by default.",
                category="Billing",
                section="Section 5.1"
            ),
            Document(
                id="doc-3",
                title="DNS Configuration",
                content="Configure your DNS settings by updating nameservers or adding A, CNAME, and MX records.",
                category="Technical",
                section="Section 3.1"
            ),
            Document(
                id="doc-4",
                title="WHOIS Privacy",
                content="WHOIS privacy protection hides your personal information from public WHOIS lookups.",
                category="Privacy",
                section="Section 2.3"
            ),
        ]
    
    @pytest.fixture
    def vector_store(self, sample_documents):
        """Create a vector store with sample documents."""
        store = VectorStore()
        store.add_documents(sample_documents)
        return store
    
    def test_add_documents(self, sample_documents):
        """Test adding documents to vector store."""
        store = VectorStore()
        
        assert store.get_document_count() == 0
        store.add_documents(sample_documents)
        assert store.get_document_count() == len(sample_documents)
    
    def test_add_empty_documents(self):
        """Test adding empty document list."""
        store = VectorStore()
        store.add_documents([])
        
        assert store.get_document_count() == 0
    
    def test_search_returns_relevant_documents(self, vector_store):
        """Test that search returns relevant documents."""
        query = "My domain was suspended, what should I do?"
        results = vector_store.search(query, top_k=2)
        
        assert len(results) > 0
        assert len(results) <= 2
        
        # First result should be about suspension
        assert "suspension" in results[0].document.title.lower() or \
               "suspension" in results[0].document.content.lower()
    
    def test_search_with_threshold(self, vector_store):
        """Test search with similarity threshold."""
        query = "domain suspension"
        
        # Low threshold - should return more results
        results_low = vector_store.search(query, threshold=0.1)
        
        # High threshold - should return fewer results
        results_high = vector_store.search(query, threshold=0.8)
        
        assert len(results_low) >= len(results_high)
    
    def test_search_returns_similarity_scores(self, vector_store):
        """Test that search results include similarity scores."""
        query = "How do I renew my domain?"
        results = vector_store.search(query)
        
        for result in results:
            assert hasattr(result, 'similarity_score')
            assert 0.0 <= result.similarity_score <= 1.0
    
    def test_search_results_ordered_by_relevance(self, vector_store):
        """Test that results are ordered by similarity score."""
        query = "DNS configuration nameserver"
        results = vector_store.search(query, top_k=4)
        
        if len(results) > 1:
            scores = [r.similarity_score for r in results]
            assert scores == sorted(scores, reverse=True)
    
    def test_search_empty_store(self):
        """Test searching empty vector store."""
        store = VectorStore()
        results = store.search("test query")
        
        assert results == []
    
    def test_clear_store(self, vector_store):
        """Test clearing the vector store."""
        assert vector_store.get_document_count() > 0
        
        vector_store.clear()
        
        assert vector_store.get_document_count() == 0
        assert vector_store.search("test") == []
    
    def test_search_different_queries(self, vector_store):
        """Test that different queries return different results."""
        query_suspension = "domain suspended"
        query_dns = "DNS nameserver configuration"
        
        results_suspension = vector_store.search(query_suspension, top_k=1)
        results_dns = vector_store.search(query_dns, top_k=1)
        
        assert results_suspension[0].document.id != results_dns[0].document.id
    
    def test_get_stats(self, vector_store, sample_documents):
        """Test that get_stats returns correct information."""
        stats = vector_store.get_stats()
        
        assert stats["total_vectors"] == len(sample_documents)
        assert stats["total_documents"] == len(sample_documents)
        assert stats["dimension"] == 384  # all-MiniLM-L6-v2 dimension
        assert "IndexFlatIP" in stats["index_type"]
    
    def test_backward_compatible_alias(self):
        """Test that VectorStore alias works."""
        # VectorStore should be an alias for FAISSVectorStore
        store = VectorStore()
        assert isinstance(store, FAISSVectorStore)

    def test_save_and_load(self, sample_documents):
        """Test saving and loading vector store."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate store
            store1 = FAISSVectorStore()
            store1.add_documents(sample_documents)

            # Save to disk
            store1.save(tmpdir)

            # Verify files were created
            assert os.path.exists(os.path.join(tmpdir, "faiss.index"))
            assert os.path.exists(os.path.join(tmpdir, "documents.pkl"))

            # Load in new store
            store2 = FAISSVectorStore()
            store2.load(tmpdir)

            # Verify loaded store has same documents
            assert store2.get_document_count() == len(sample_documents)
            assert len(store2.documents) == len(sample_documents)

    def test_load_at_init(self, sample_documents):
        """Test loading existing index at initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save store
            store1 = FAISSVectorStore()
            store1.add_documents(sample_documents)
            store1.save(tmpdir)

            # Create new store with index_path
            store2 = FAISSVectorStore(index_path=tmpdir)

            # Should have loaded documents
            assert store2.get_document_count() == len(sample_documents)

    def test_load_nonexistent_path(self):
        """Test loading from nonexistent path doesn't crash."""
        store = FAISSVectorStore(index_path="/nonexistent/path")
        # Should initialize empty
        assert store.get_document_count() == 0


class TestVectorStoreSingleton:
    """Tests for vector store singleton functions."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="test-1",
                title="Test Doc 1",
                content="Test content 1",
                category="Test"
            ),
            Document(
                id="test-2",
                title="Test Doc 2",
                content="Test content 2",
                category="Test"
            ),
        ]

    def test_initialize_vector_store(self, sample_documents):
        """Test initializing vector store singleton."""
        # Reset singleton
        vector_store_module._vector_store = None

        store = initialize_vector_store(sample_documents)

        assert store is not None
        assert store.get_document_count() == len(sample_documents)

    def test_initialize_vector_store_idempotent(self, sample_documents):
        """Test that initialize_vector_store doesn't re-add existing docs."""
        # Reset singleton
        vector_store_module._vector_store = None

        store1 = initialize_vector_store(sample_documents)
        initial_count = store1.get_document_count()

        # Initialize again
        store2 = initialize_vector_store(sample_documents)

        # Should be same instance and same count
        assert store1 is store2
        assert store2.get_document_count() == initial_count

    def test_initialize_vector_store_force_reinit(self, sample_documents):
        """Test force reinitialization clears existing docs."""
        # Reset singleton
        vector_store_module._vector_store = None

        # Add some docs
        store1 = initialize_vector_store(sample_documents)

        # Add more docs manually
        extra_doc = Document(
            id="extra",
            title="Extra",
            content="Extra document",
            category="Test"
        )
        store1.add_documents([extra_doc])
        count_with_extra = store1.get_document_count()

        # Force reinit should clear and only add base docs
        store2 = initialize_vector_store(sample_documents, force_reinit=True)

        assert store1 is store2
        assert store2.get_document_count() == len(sample_documents)
        assert store2.get_document_count() < count_with_extra

    def test_get_vector_store(self, sample_documents):
        """Test getting vector store singleton."""
        # Reset singleton
        vector_store_module._vector_store = None

        # Initialize first
        initialize_vector_store(sample_documents)

        # Get should return same instance
        store1 = get_vector_store()
        store2 = get_vector_store()

        assert store1 is store2

    def test_initialize_adds_new_docs_only(self, sample_documents):
        """Test that reinitializing only adds new base documents."""
        # Reset singleton
        vector_store_module._vector_store = None

        # Initialize with 2 docs
        store1 = initialize_vector_store(sample_documents)
        assert store1.get_document_count() == 2

        # Add a third doc that's NOT in base
        extra_doc = Document(
            id="extra-new",
            title="Extra New",
            content="New extra content",
            category="Extra"
        )
        store1.add_documents([extra_doc])
        assert store1.get_document_count() == 3

        # Reinitialize with same base docs - should preserve extra doc
        store2 = initialize_vector_store(sample_documents)

        assert store1 is store2
        assert store2.get_document_count() == 3  # Base 2 + extra 1
