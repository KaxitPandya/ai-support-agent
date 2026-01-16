"""
LLM Service - OpenAI Integration.

Provides a clean interface for generating responses using OpenAI's GPT models.
Supports both text and JSON output modes.
"""

import json
import logging
import os
from typing import Any, Dict, List

from openai import OpenAI

from src.config import get_settings

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM Service using OpenAI.
    
    Features:
    - Text generation with GPT models
    - JSON mode for structured outputs
    - Configurable temperature and max tokens
    - Error handling and fallbacks
    """
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None
    ):
        """
        Initialize the LLM service.
        
        Args:
            api_key: OpenAI API key (defaults to env var).
            model: Model to use (defaults to config).
            temperature: Sampling temperature (defaults to config).
            max_tokens: Maximum tokens in response (defaults to config).
        """
        settings = get_settings()
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or settings.openai_api_key
        self.model = model or os.getenv("OPENAI_MODEL") or settings.openai_model
        self.temperature = temperature if temperature is not None else float(os.getenv("OPENAI_TEMPERATURE", settings.openai_temperature))
        self.max_tokens = max_tokens or int(os.getenv("OPENAI_MAX_TOKENS", settings.openai_max_tokens))
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        
        logger.info(f"LLM Service initialized with model: {self.model}")
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a text response.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            Generated text response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def generate_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Generate a JSON response.
        
        Uses OpenAI's JSON mode for reliable structured output.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional parameters for the API call.
            
        Returns:
            Parsed JSON response as a dictionary.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {response.choices[0].message.content}")
            # Return a fallback response
            return {
                "answer": "I apologize, but I encountered an error processing your request.",
                "references": [],
                "action_required": "escalate_to_technical"
            }
        except Exception as e:
            logger.error(f"OpenAI JSON generation failed: {e}")
            raise


# Singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """Get or create the singleton LLM service."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_llm_service() -> None:
    """Reset the LLM service (useful for testing or reconfiguration)."""
    global _llm_service
    _llm_service = None
