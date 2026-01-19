"""
Unit tests for the LLM service.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from src.services.llm import LLMService, get_llm_service, reset_llm_service


class TestLLMService:
    """Tests for LLM Service."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_initialization_with_api_key(self):
        """Test LLM service initialization with API key."""
        llm = LLMService(api_key="test-key-123")

        assert llm.api_key == "test-key-123"
        assert llm.model is not None
        assert llm.temperature is not None
        assert llm.max_tokens is not None

    @patch.dict(os.environ, {}, clear=True)
    @patch('src.services.llm.get_settings')
    def test_initialization_without_api_key_raises_error(self, mock_get_settings):
        """Test that initialization without API key raises ValueError."""
        # Mock settings to return empty API key
        mock_settings = MagicMock()
        mock_settings.openai_api_key = ""
        mock_settings.openai_model = "gpt-3.5-turbo"
        mock_settings.openai_temperature = 0.7
        mock_settings.openai_max_tokens = 1000
        mock_get_settings.return_value = mock_settings

        with pytest.raises(ValueError, match="OpenAI API key is required"):
            LLMService()

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        llm = LLMService(
            api_key="test-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2000
        )

        assert llm.model == "gpt-4"
        assert llm.temperature == 0.5
        assert llm.max_tokens == 2000

    @patch('src.services.llm.OpenAI')
    def test_generate_success(self, mock_openai_class):
        """Test successful text generation."""
        # Setup mock
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ]

        response = llm.generate(messages)

        assert response == "This is a test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('src.services.llm.OpenAI')
    def test_generate_with_custom_params(self, mock_openai_class):
        """Test generation with custom parameters."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [{"role": "user", "content": "Test"}]
        llm.generate(messages, temperature=0.9, max_tokens=500)

        # Check that custom params were passed
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['temperature'] == 0.9
        assert call_args.kwargs['max_tokens'] == 500

    @patch('src.services.llm.OpenAI')
    def test_generate_api_error(self, mock_openai_class):
        """Test that API errors are properly raised."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception, match="API Error"):
            llm.generate(messages)

    @patch('src.services.llm.OpenAI')
    def test_generate_json_success(self, mock_openai_class):
        """Test successful JSON generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()

        json_content = json.dumps({
            "answer": "Test answer",
            "references": ["ref1", "ref2"],
            "action_required": "none"
        })

        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json_content
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [{"role": "user", "content": "Test"}]
        result = llm.generate_json(messages)

        assert isinstance(result, dict)
        assert result["answer"] == "Test answer"
        assert result["references"] == ["ref1", "ref2"]
        assert result["action_required"] == "none"

    @patch('src.services.llm.OpenAI')
    def test_generate_json_uses_json_mode(self, mock_openai_class):
        """Test that JSON mode is enabled in API call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"test": "value"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [{"role": "user", "content": "Test"}]
        llm.generate_json(messages)

        # Verify JSON mode was set
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['response_format'] == {"type": "json_object"}

    @patch('src.services.llm.OpenAI')
    def test_generate_json_parse_error_fallback(self, mock_openai_class):
        """Test fallback when JSON parsing fails."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Not valid JSON"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [{"role": "user", "content": "Test"}]
        result = llm.generate_json(messages)

        # Should return fallback response
        assert isinstance(result, dict)
        assert "answer" in result
        assert "I apologize" in result["answer"]
        assert result["action_required"] == "escalate_to_technical"

    @patch('src.services.llm.OpenAI')
    def test_generate_json_api_error(self, mock_openai_class):
        """Test that API errors in JSON mode are raised."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        llm = LLMService(api_key="test-key")

        messages = [{"role": "user", "content": "Test"}]

        with pytest.raises(Exception, match="API Error"):
            llm.generate_json(messages)


class TestLLMSingleton:
    """Tests for LLM singleton functions."""

    def test_get_llm_service(self):
        """Test getting singleton LLM service."""
        reset_llm_service()  # Start fresh

        llm1 = get_llm_service()
        llm2 = get_llm_service()

        # Should return same instance
        assert llm1 is llm2

    def test_reset_llm_service(self):
        """Test resetting singleton LLM service."""
        reset_llm_service()

        llm1 = get_llm_service()

        # Reset creates new instance on next call
        reset_llm_service()
        llm2 = get_llm_service()

        # Should be different instances
        assert llm1 is not llm2
