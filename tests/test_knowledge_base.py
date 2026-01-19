"""
Unit tests for the knowledge base.
"""

import pytest
from src.data.knowledge_base import get_knowledge_base
from src.models.schemas import Document


class TestKnowledgeBase:
    """Tests for knowledge base module."""

    def test_get_knowledge_base_returns_list(self):
        """Test that get_knowledge_base returns a list of documents."""
        docs = get_knowledge_base()

        assert isinstance(docs, list)
        assert len(docs) > 0

    def test_get_knowledge_base_returns_documents(self):
        """Test that all items are Document objects."""
        docs = get_knowledge_base()

        for doc in docs:
            assert isinstance(doc, Document)
            assert hasattr(doc, 'id')
            assert hasattr(doc, 'title')
            assert hasattr(doc, 'category')
            assert hasattr(doc, 'content')

    def test_knowledge_base_has_required_categories(self):
        """Test that knowledge base includes key categories."""
        docs = get_knowledge_base()
        categories = {doc.category for doc in docs}

        # Check for key categories
        assert "Domain Policies" in categories
        assert "WHOIS Information" in categories
        assert "Billing & Payments" in categories
        assert "DNS & Technical" in categories

    def test_knowledge_base_documents_have_content(self):
        """Test that all documents have non-empty content."""
        docs = get_knowledge_base()

        for doc in docs:
            assert doc.content.strip() != ""
            assert len(doc.content) > 50  # Reasonable content length

    def test_knowledge_base_documents_have_unique_ids(self):
        """Test that all document IDs are unique."""
        docs = get_knowledge_base()
        ids = [doc.id for doc in docs]

        assert len(ids) == len(set(ids))  # All IDs are unique
