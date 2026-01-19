"""
Unit tests for the FastAPI endpoints.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models.schemas import TicketResponse


class TestAPI:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def mock_rag_pipeline(self):
        """Create a mock RAG pipeline."""
        mock = MagicMock()
        mock.resolve_ticket.return_value = TicketResponse(
            answer="Your domain was suspended due to WHOIS verification issues.",
            references=["Policy: Domain Suspension Guidelines, Section 4.2"],
            action_required="customer_action_required"
        )
        return mock
    
    @pytest.fixture
    def client(self, mock_rag_pipeline):
        """Create a test client with mocked RAG pipeline."""
        with patch('src.main.get_rag_pipeline', return_value=mock_rag_pipeline):
            with patch('src.main.initialize_rag_pipeline', return_value=mock_rag_pipeline):
                from src.main import app
                yield TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "llm_provider" in data
        assert "embedding_model" in data
    
    def test_resolve_ticket_success(self, client, mock_rag_pipeline):
        """Test successful ticket resolution."""
        request_data = {
            "ticket_text": "My domain was suspended and I didn't get any notice."
        }
        
        response = client.post("/resolve-ticket", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "references" in data
        assert "action_required" in data
    
    def test_resolve_ticket_response_format(self, client, mock_rag_pipeline):
        """Test that response matches expected MCP format."""
        request_data = {
            "ticket_text": "How do I transfer my domain?"
        }
        
        response = client.post("/resolve-ticket", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify MCP-compliant structure
        assert isinstance(data["answer"], str)
        assert isinstance(data["references"], list)
        assert isinstance(data["action_required"], str)
    
    def test_resolve_ticket_empty_text(self, client):
        """Test ticket with empty text returns validation error."""
        request_data = {"ticket_text": ""}
        
        response = client.post("/resolve-ticket", json=request_data)
        
        # Should fail validation (min_length=5)
        assert response.status_code == 422
    
    def test_resolve_ticket_short_text(self, client):
        """Test ticket with too short text returns validation error."""
        request_data = {"ticket_text": "Hi"}
        
        response = client.post("/resolve-ticket", json=request_data)
        
        # Should fail validation (min_length=5)
        assert response.status_code == 422
    
    def test_resolve_ticket_missing_field(self, client):
        """Test ticket without required field returns validation error."""
        request_data = {}
        
        response = client.post("/resolve-ticket", json=request_data)
        
        assert response.status_code == 422
    
    def test_resolve_ticket_calls_pipeline(self, client, mock_rag_pipeline):
        """Test that endpoint calls the RAG pipeline."""
        request_data = {
            "ticket_text": "I need help with DNS settings."
        }
        
        client.post("/resolve-ticket", json=request_data)
        
        mock_rag_pipeline.resolve_ticket.assert_called_once_with(request_data["ticket_text"])
    
    def test_resolve_ticket_long_text(self, client, mock_rag_pipeline):
        """Test ticket with long text is accepted."""
        long_text = "Hello, I have a problem. " * 100  # ~2500 chars
        request_data = {"ticket_text": long_text}
        
        response = client.post("/resolve-ticket", json=request_data)
        
        assert response.status_code == 200


    def test_resolve_ticket_error_handling(self):
        """Test that exceptions in pipeline are caught and return 500."""
        mock_pipeline = MagicMock()
        mock_pipeline.resolve_ticket.side_effect = Exception("Pipeline error")

        with patch('src.main.get_rag_pipeline', return_value=mock_pipeline):
            with patch('src.main.initialize_rag_pipeline'):
                from src.main import app
                client = TestClient(app)

                request_data = {"ticket_text": "Test ticket"}
                response = client.post("/resolve-ticket", json=request_data)

                assert response.status_code == 500
                assert "error" in response.json()["detail"].lower()


class TestAPIValidation:
    """Tests for API input validation."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked pipeline."""
        mock = MagicMock()
        mock.resolve_ticket.return_value = TicketResponse(
            answer="Test answer",
            references=[],
            action_required="none"
        )

        with patch('src.main.get_rag_pipeline', return_value=mock):
            with patch('src.main.initialize_rag_pipeline', return_value=mock):
                from src.main import app
                yield TestClient(app)

    def test_invalid_json(self, client):
        """Test that invalid JSON returns error."""
        response = client.post(
            "/resolve-ticket",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type(self, client):
        """Test that wrong content type is handled."""
        response = client.post(
            "/resolve-ticket",
            data="ticket_text=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422


class TestLifespan:
    """Tests for application lifespan events."""

    def test_lifespan_startup_success(self):
        """Test successful lifespan startup."""
        with patch('src.main.initialize_rag_pipeline'):
            from src.main import app
            # Creating TestClient triggers lifespan events
            client = TestClient(app)
            assert client is not None

