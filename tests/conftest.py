"""
Pytest configuration and shared fixtures.
"""

import os
import pytest

# Set test environment variables
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("DEBUG", "false")


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test."""
    from src.services import rag, llm, vector_store, embedding
    
    # Reset singletons
    rag._rag_pipeline = None
    llm._llm_service = None
    vector_store._vector_store = None
    
    # Note: We don't reset embedding service as model loading is slow
    
    yield
    
    # Cleanup after test
    rag._rag_pipeline = None
    llm._llm_service = None
    vector_store._vector_store = None


@pytest.fixture
def sample_ticket_texts():
    """Sample ticket texts for testing."""
    return [
        "My domain was suspended and I didn't get any notice. How can I reactivate it?",
        "I want to transfer my domain to another registrar. What's the process?",
        "My website is not loading. The DNS seems to be misconfigured.",
        "I need a refund for my domain renewal. I renewed by mistake.",
        "How do I enable WHOIS privacy for my domain?",
        "I think my domain was hijacked. Please help!",
    ]


@pytest.fixture
def expected_actions():
    """Expected action mappings for sample tickets."""
    return {
        "suspended": "customer_action_required",
        "transfer": "none",
        "DNS": "escalate_to_technical",
        "refund": "escalate_to_billing",
        "privacy": "none",
        "hijacked": "escalate_to_abuse_team",
    }
