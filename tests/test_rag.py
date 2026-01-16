"""
Unit tests for the RAG pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models.schemas import Document, RetrievedContext, TicketResponse
from src.services.rag import RAGPipeline


class TestRAGPipeline:
    """Tests for RAGPipeline."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc-1",
                title="Domain Suspension Guidelines",
                content="Domains are suspended for WHOIS issues or policy violations. "
                       "To reactivate, verify your email and update WHOIS information.",
                category="Domain Policies",
                section="Section 4.2"
            ),
            Document(
                id="doc-2",
                title="Billing FAQ",
                content="Domain renewals are processed automatically. "
                       "To request a refund, contact billing within 5 days.",
                category="Billing",
                section="Section 5.4"
            ),
        ]
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock = MagicMock()
        mock.generate_json.return_value = {
            "answer": "Your domain was suspended due to WHOIS verification issues. Please verify your email.",
            "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
            "action_required": "customer_action_required"
        }
        return mock
    
    @pytest.fixture
    def rag_pipeline(self, sample_documents, mock_llm_service):
        """Create a RAG pipeline with mocked LLM."""
        pipeline = RAGPipeline(
            llm_service=mock_llm_service,
            documents=sample_documents
        )
        pipeline.initialize()
        return pipeline
    
    def test_pipeline_initialization(self, sample_documents):
        """Test that pipeline initializes correctly."""
        pipeline = RAGPipeline(documents=sample_documents)
        
        assert not pipeline._initialized
        pipeline.initialize()
        assert pipeline._initialized
        assert pipeline.vector_store.get_document_count() == len(sample_documents)
    
    def test_retrieve_context(self, rag_pipeline):
        """Test context retrieval."""
        query = "My domain was suspended"
        contexts = rag_pipeline.retrieve_context(query)
        
        assert len(contexts) > 0
        assert all(isinstance(ctx, RetrievedContext) for ctx in contexts)
    
    def test_resolve_ticket_returns_response(self, rag_pipeline):
        """Test that resolve_ticket returns proper response."""
        ticket_text = "My domain was suspended and I didn't get any notice."
        response = rag_pipeline.resolve_ticket(ticket_text)
        
        assert isinstance(response, TicketResponse)
        assert response.answer != ""
        assert isinstance(response.references, list)
        assert response.action_required in [
            "none", "escalate_to_abuse_team", "escalate_to_billing",
            "escalate_to_technical", "customer_action_required", "follow_up_required"
        ]
    
    def test_resolve_ticket_calls_llm(self, rag_pipeline, mock_llm_service):
        """Test that resolve_ticket calls the LLM service."""
        ticket_text = "How do I renew my domain?"
        rag_pipeline.resolve_ticket(ticket_text)
        
        mock_llm_service.generate_json.assert_called_once()
    
    def test_parse_response_valid_data(self, rag_pipeline):
        """Test parsing valid response data."""
        response_data = {
            "answer": "Test answer",
            "references": ["Ref 1", "Ref 2"],
            "action_required": "escalate_to_billing"
        }
        
        response = rag_pipeline._parse_response(response_data, [])
        
        assert response.answer == "Test answer"
        assert response.references == ["Ref 1", "Ref 2"]
        assert response.action_required == "escalate_to_billing"
    
    def test_parse_response_invalid_action(self, rag_pipeline):
        """Test parsing response with invalid action defaults to 'none'."""
        response_data = {
            "answer": "Test answer",
            "references": [],
            "action_required": "invalid_action"
        }
        
        response = rag_pipeline._parse_response(response_data, [])
        
        assert response.action_required == "none"
    
    def test_parse_response_missing_fields(self, rag_pipeline):
        """Test parsing response with missing fields."""
        response_data = {}
        
        response = rag_pipeline._parse_response(response_data, [])
        
        assert "unable to provide" in response.answer.lower()
        assert response.references == []
        assert response.action_required == "none"
    
    def test_llm_error_returns_fallback(self, sample_documents):
        """Test that LLM errors return fallback response."""
        mock_llm = MagicMock()
        mock_llm.generate_json.side_effect = Exception("LLM error")
        
        pipeline = RAGPipeline(
            llm_service=mock_llm,
            documents=sample_documents
        )
        pipeline.initialize()
        
        response = pipeline.resolve_ticket("Test ticket")
        
        assert "unable to process" in response.answer.lower()
        assert response.action_required == "escalate_to_technical"
    
    def test_double_initialization(self, sample_documents):
        """Test that double initialization is handled."""
        pipeline = RAGPipeline(documents=sample_documents)
        pipeline.initialize()
        
        initial_count = pipeline.vector_store.get_document_count()
        pipeline.initialize()  # Should not add documents again
        
        assert pipeline.vector_store.get_document_count() == initial_count
