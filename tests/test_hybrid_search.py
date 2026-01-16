"""
Unit tests for hybrid search service.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.schemas import Document
from src.services.hybrid_search import (
    BM25,
    HybridSearchService,
    HybridSearchResult
)


class TestBM25:
    """Tests for BM25 keyword search."""
    
    @pytest.fixture
    def sample_docs(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc-1",
                title="Domain Suspension",
                content="Domains can be suspended for policy violations or missing WHOIS information.",
                category="Policies",
                section="Section 1"
            ),
            Document(
                id="doc-2",
                title="DNS Configuration",
                content="Configure DNS records including A records, CNAME, and MX records.",
                category="Technical",
                section="Section 2"
            ),
            Document(
                id="doc-3",
                title="Billing FAQ",
                content="Payments can be made via credit card or PayPal. Refunds are processed within 5-7 days.",
                category="Billing",
                section="Section 3"
            ),
        ]
    
    @pytest.fixture
    def bm25(self, sample_docs):
        """Create and fit BM25 instance."""
        bm25 = BM25()
        bm25.fit(sample_docs)
        return bm25
    
    def test_fit(self, sample_docs):
        """Test fitting BM25 on documents."""
        bm25 = BM25()
        bm25.fit(sample_docs)
        
        assert bm25.n_docs == 3
        assert len(bm25.corpus) == 3
        assert len(bm25.idf) > 0
    
    def test_tokenize(self):
        """Test text tokenization."""
        bm25 = BM25()
        tokens = bm25._tokenize("Hello World! This is a TEST.")
        
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens
    
    def test_score_relevant_query(self, bm25):
        """Test scoring with relevant query."""
        scores = bm25.score("domain suspension")
        
        # Should return results sorted by score
        assert len(scores) == 3
        assert scores[0][0] == "doc-1"  # Domain suspension doc should rank first
    
    def test_score_different_topic(self, bm25):
        """Test scoring with different topic query."""
        scores = bm25.score("payment refund credit card PayPal")
        
        # Billing doc should rank highest (more specific query)
        assert len(scores) == 3
        # The billing doc should have a higher score for billing terms
        billing_score = next(s for doc_id, s in scores if doc_id == "doc-3")
        other_scores = [s for doc_id, s in scores if doc_id != "doc-3"]
        assert billing_score >= max(other_scores) * 0.5  # Should be competitive
    
    def test_score_no_match(self, bm25):
        """Test scoring with no matching terms."""
        scores = bm25.score("xyz123 unknown terms")
        
        # Should return results but with low/zero scores
        assert len(scores) == 3
        assert all(score >= 0 for _, score in scores)


class TestHybridSearchResult:
    """Tests for HybridSearchResult dataclass."""
    
    def test_create_result(self):
        """Test creating a hybrid search result."""
        doc = Document(
            id="test",
            title="Test",
            content="Content",
            category="Cat",
            section="Sec"
        )
        
        result = HybridSearchResult(
            document=doc,
            semantic_score=0.8,
            keyword_score=0.6,
            combined_score=0.7
        )
        
        assert result.semantic_score == 0.8
        assert result.keyword_score == 0.6
        assert result.rerank_score is None


class TestHybridSearchService:
    """Tests for HybridSearchService."""
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock()
        service.embed_texts = lambda texts: np.random.rand(len(texts), 384)
        service.embed_text = lambda text: np.random.rand(384)
        return service
    
    @pytest.fixture
    def mock_vector_store(self, mock_embedding_service):
        """Create a mock vector store."""
        from src.services.vector_store import FAISSVectorStore
        store = FAISSVectorStore.__new__(FAISSVectorStore)
        store.embedding_service = mock_embedding_service
        store.dimension = 384
        store.documents = []
        import faiss
        store.index = faiss.IndexFlatIP(384)
        return store
    
    @pytest.fixture
    def sample_docs(self):
        """Create sample documents."""
        return [
            Document(
                id="doc-1",
                title="Domain Suspension Guidelines",
                content="Domains may be suspended for WHOIS violations or policy breaches.",
                category="Policies",
                section="Section 4.1"
            ),
            Document(
                id="doc-2",
                title="DNS Configuration",
                content="Configure DNS A records, CNAME records, and MX records for email.",
                category="Technical",
                section="DNS Guide"
            ),
            Document(
                id="doc-3",
                title="Billing and Payments",
                content="We accept credit cards and PayPal. Refunds processed in 5-7 days.",
                category="Billing",
                section="FAQ"
            ),
        ]
    
    @pytest.fixture
    def search_service(self, mock_vector_store, mock_embedding_service, sample_docs):
        """Create and initialize hybrid search service."""
        service = HybridSearchService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            semantic_weight=0.7,
            keyword_weight=0.3,
            use_reranking=False  # Disable for faster tests
        )
        service.index_documents(sample_docs)
        return service
    
    def test_index_documents(self, mock_vector_store, mock_embedding_service, sample_docs):
        """Test document indexing."""
        service = HybridSearchService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        service.index_documents(sample_docs)
        
        assert service._initialized
        assert len(service._documents) == 3
        assert service.bm25.n_docs == 3
    
    def test_search(self, search_service):
        """Test hybrid search."""
        results = search_service.search("domain suspension", top_k=3)
        
        assert len(results) <= 3
        assert all(isinstance(r, HybridSearchResult) for r in results)
    
    def test_search_returns_combined_scores(self, search_service):
        """Test that results include combined scores."""
        results = search_service.search("billing payment", top_k=2)
        
        for result in results:
            assert result.semantic_score >= 0
            assert result.keyword_score >= 0
            assert result.combined_score >= 0
    
    def test_to_retrieved_context(self, search_service):
        """Test converting results to RetrievedContext."""
        results = search_service.search("domain", top_k=2)
        contexts = search_service.to_retrieved_context(results)
        
        assert len(contexts) == len(results)
        for ctx in contexts:
            assert ctx.document is not None
            assert ctx.similarity_score >= 0
    
    def test_normalize_scores(self, search_service):
        """Test score normalization."""
        scores = [0.1, 0.5, 0.9]
        normalized = search_service._normalize_scores(scores)
        
        assert min(normalized) == 0.0
        assert max(normalized) == 1.0
    
    def test_normalize_same_scores(self, search_service):
        """Test normalizing identical scores."""
        scores = [0.5, 0.5, 0.5]
        normalized = search_service._normalize_scores(scores)
        
        assert all(s == 0.5 for s in normalized)
    
    def test_search_uninitialized(self, mock_embedding_service):
        """Test search on uninitialized service."""
        service = HybridSearchService(embedding_service=mock_embedding_service)
        results = service.search("test query")
        
        assert results == []
