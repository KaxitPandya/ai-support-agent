"""
Unit tests for MCP prompt templates.
"""

import pytest

from src.models.schemas import Document, RetrievedContext
from src.prompts.mcp_prompt import (
    SYSTEM_ROLE,
    OUTPUT_SCHEMA,
    OUTPUT_SCHEMA_STR,
    ACTION_TYPES,
    build_context_section,
    build_mcp_prompt,
    build_simple_prompt,
    get_mcp_structure_info,
)


class TestMCPPrompts:
    """Tests for MCP prompt templates."""
    
    @pytest.fixture
    def sample_contexts(self):
        """Create sample retrieved contexts."""
        return [
            RetrievedContext(
                document=Document(
                    id="doc-1",
                    title="Domain Suspension Guidelines",
                    content="Domains are suspended for WHOIS issues or violations.",
                    category="Domain Policies",
                    section="Section 4.2"
                ),
                similarity_score=0.85
            ),
            RetrievedContext(
                document=Document(
                    id="doc-2",
                    title="Reactivation Process",
                    content="To reactivate, update WHOIS and verify email.",
                    category="Domain Policies",
                    section="Section 4.3"
                ),
                similarity_score=0.72
            ),
        ]
    
    def test_system_role_content(self):
        """Test that system role contains expected elements."""
        assert "customer support assistant" in SYSTEM_ROLE.lower()
        assert "domain" in SYSTEM_ROLE.lower()
        assert "guidelines" in SYSTEM_ROLE.lower()
    
    def test_output_schema_is_valid_structure(self):
        """Test that output schema contains required fields."""
        assert "answer" in OUTPUT_SCHEMA
        assert "references" in OUTPUT_SCHEMA
        assert "action_required" in OUTPUT_SCHEMA
    
    def test_output_schema_str_is_json_template(self):
        """Test that output schema string is valid JSON template."""
        assert "answer" in OUTPUT_SCHEMA_STR
        assert "references" in OUTPUT_SCHEMA_STR
        assert "action_required" in OUTPUT_SCHEMA_STR
        assert "{" in OUTPUT_SCHEMA_STR
        assert "}" in OUTPUT_SCHEMA_STR
    
    def test_action_types_defined(self):
        """Test that all action types are defined."""
        expected_actions = [
            "none",
            "escalate_to_abuse_team",
            "escalate_to_billing",
            "escalate_to_technical",
            "customer_action_required",
            "follow_up_required"
        ]
        for action in expected_actions:
            assert action in ACTION_TYPES
    
    def test_build_context_section_with_contexts(self, sample_contexts):
        """Test building context section with documents."""
        context = build_context_section(sample_contexts)
        
        assert "Document 1" in context
        assert "Document 2" in context
        assert "Domain Suspension Guidelines" in context
        assert "Similarity Score" in context
        assert "85" in context  # 0.85 formatted as percentage
    
    def test_build_context_section_empty(self):
        """Test building context section with no documents."""
        context = build_context_section([])
        
        assert "No relevant documentation" in context
        assert "general knowledge" in context.lower()
    
    def test_build_mcp_prompt_structure(self, sample_contexts):
        """Test that MCP prompt has correct structure."""
        ticket_text = "My domain was suspended."
        messages = build_mcp_prompt(ticket_text, sample_contexts)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
    
    def test_build_mcp_prompt_contains_ticket(self, sample_contexts):
        """Test that MCP prompt contains the ticket text."""
        ticket_text = "My domain xyz.com was suspended yesterday."
        messages = build_mcp_prompt(ticket_text, sample_contexts)
        
        user_content = messages[1]["content"]
        assert ticket_text in user_content
    
    def test_build_mcp_prompt_contains_context(self, sample_contexts):
        """Test that MCP prompt contains context section."""
        ticket_text = "My domain was suspended."
        messages = build_mcp_prompt(ticket_text, sample_contexts)
        
        user_content = messages[1]["content"]
        assert "CONTEXT" in user_content
        assert "Domain Suspension Guidelines" in user_content
    
    def test_build_mcp_prompt_contains_task(self, sample_contexts):
        """Test that MCP prompt contains task section."""
        ticket_text = "My domain was suspended."
        messages = build_mcp_prompt(ticket_text, sample_contexts)
        
        user_content = messages[1]["content"]
        assert "TASK" in user_content
    
    def test_build_mcp_prompt_contains_output_schema(self, sample_contexts):
        """Test that MCP prompt contains output schema section."""
        ticket_text = "My domain was suspended."
        messages = build_mcp_prompt(ticket_text, sample_contexts)
        
        user_content = messages[1]["content"]
        assert "OUTPUT" in user_content
        assert "SCHEMA" in user_content
        assert "JSON" in user_content
        assert "answer" in user_content
        assert "references" in user_content
        assert "action_required" in user_content
    
    def test_build_mcp_prompt_contains_action_options(self, sample_contexts):
        """Test that MCP prompt lists action options."""
        ticket_text = "Test ticket"
        messages = build_mcp_prompt(ticket_text, sample_contexts)
        
        user_content = messages[1]["content"]
        assert "escalate_to_abuse_team" in user_content
        assert "escalate_to_billing" in user_content
        assert "customer_action_required" in user_content
    
    def test_build_simple_prompt_structure(self):
        """Test simple prompt structure."""
        ticket_text = "Help with my domain"
        messages = build_simple_prompt(ticket_text)
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert ticket_text in messages[1]["content"]
    
    def test_build_simple_prompt_has_task_and_output(self):
        """Test that simple prompt has task and output sections."""
        ticket_text = "Help needed"
        messages = build_simple_prompt(ticket_text)
        
        user_content = messages[1]["content"]
        assert "TASK" in user_content
        assert "OUTPUT" in user_content
    
    def test_get_mcp_structure_info(self):
        """Test MCP structure info helper."""
        info = get_mcp_structure_info()

        assert "pattern" in info
        assert "sections" in info
        assert "output_format" in info
        assert "action_types" in info

        # Verify sections
        sections = info["sections"]
        assert "role" in sections
        assert "context" in sections
        assert "task" in sections
        assert "output_schema" in sections

    def test_build_mcp_prompt_with_memory_context(self, sample_contexts):
        """Test that MCP prompt includes memory section when provided."""
        ticket_text = "Can you remind me what we discussed?"
        memory_context = "Previous conversation about domain suspension..."

        messages = build_mcp_prompt(ticket_text, sample_contexts, memory_context)

        user_content = messages[1]["content"]
        assert "MEMORY SECTION" in user_content
        assert memory_context in user_content

    def test_build_mcp_prompt_without_memory_context(self, sample_contexts):
        """Test that MCP prompt works without memory section."""
        ticket_text = "My domain was suspended."

        messages = build_mcp_prompt(ticket_text, sample_contexts, "")

        user_content = messages[1]["content"]
        assert "MEMORY SECTION" not in user_content
