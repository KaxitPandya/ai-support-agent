"""
Unit tests for the simple memory service.
"""

import pytest
from datetime import datetime
from src.services.simple_memory import (
    SessionMemory, ConversationTurn,
    get_session_memory, reset_session_memory
)


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            query="Test query",
            answer="Test answer",
            references=["ref1", "ref2"],
            action_required="none"
        )

        assert turn.query == "Test query"
        assert turn.answer == "Test answer"
        assert turn.references == ["ref1", "ref2"]
        assert turn.action_required == "none"
        assert isinstance(turn.timestamp, datetime)


class TestSessionMemory:
    """Tests for SessionMemory class."""

    @pytest.fixture
    def memory(self):
        """Create a fresh memory instance."""
        return SessionMemory(max_turns=5, context_window=3)

    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory.max_turns == 5
        assert memory.context_window == 3
        assert len(memory.turns) == 0
        assert memory.is_empty()

    def test_add_turn(self, memory):
        """Test adding a turn to memory."""
        memory.add_turn(
            query="How do I reset my password?",
            answer="You can reset your password by...",
            references=["KB: Password Reset"],
            action_required="none"
        )

        assert len(memory.turns) == 1
        assert not memory.is_empty()

    def test_add_multiple_turns(self, memory):
        """Test adding multiple turns."""
        for i in range(3):
            memory.add_turn(
                query=f"Question {i+1}",
                answer=f"Answer {i+1}",
                references=[],
                action_required="none"
            )

        assert len(memory.turns) == 3

    def test_max_capacity_enforcement(self, memory):
        """Test that memory respects max_turns limit."""
        # Add more turns than max_turns
        for i in range(7):
            memory.add_turn(
                query=f"Question {i+1}",
                answer=f"Answer {i+1}",
                references=[],
                action_required="none"
            )

        # Should only keep last 5 (max_turns=5)
        assert len(memory.turns) == 5
        # First turn should be "Question 3" (oldest was dropped)
        assert memory.turns[0].query == "Question 3"
        assert memory.turns[-1].query == "Question 7"

    def test_get_context_for_prompt_empty(self, memory):
        """Test context generation with empty memory."""
        context = memory.get_context_for_prompt()
        assert context == ""

    def test_get_context_for_prompt_with_turns(self, memory):
        """Test context generation with conversation history."""
        memory.add_turn(
            query="How do I suspend a domain?",
            answer="To suspend a domain, you need to access the control panel...",
            references=["KB: Domain Management"],
            action_required="none"
        )
        memory.add_turn(
            query="Can I unsuspend it later?",
            answer="Yes, you can unsuspend the domain at any time...",
            references=["KB: Domain Management"],
            action_required="none"
        )

        context = memory.get_context_for_prompt()

        assert "Recent Conversation History" in context
        assert "How do I suspend a domain?" in context
        assert "Can I unsuspend it later?" in context
        assert "Turn 1:" in context
        assert "Turn 2:" in context

    def test_get_context_for_prompt_truncates_long_answers(self, memory):
        """Test that long answers are truncated in context."""
        long_answer = "A" * 300  # 300 characters
        memory.add_turn(
            query="Test",
            answer=long_answer,
            references=[],
            action_required="none"
        )

        context = memory.get_context_for_prompt()

        # Should truncate to 200 chars + "..."
        assert "..." in context
        assert long_answer not in context  # Full answer should not be present

    def test_get_context_for_prompt_respects_context_window(self, memory):
        """Test that context window limits returned turns."""
        # Add 5 turns
        for i in range(5):
            memory.add_turn(
                query=f"Question {i+1}",
                answer=f"Answer {i+1}",
                references=[],
                action_required="none"
            )

        # Get context with default window (3 turns)
        context = memory.get_context_for_prompt()

        # Should only include last 3 turns
        assert "Question 3" in context
        assert "Question 4" in context
        assert "Question 5" in context
        assert "Question 1" not in context
        assert "Question 2" not in context

    def test_get_context_for_prompt_custom_num_turns(self, memory):
        """Test custom number of turns in context."""
        for i in range(5):
            memory.add_turn(
                query=f"Question {i+1}",
                answer=f"Answer {i+1}",
                references=[],
                action_required="none"
            )

        # Request only 2 most recent turns
        context = memory.get_context_for_prompt(num_turns=2)

        assert "Question 4" in context
        assert "Question 5" in context
        assert "Question 3" not in context

    def test_get_turns_list(self, memory):
        """Test getting turns as list of dictionaries."""
        memory.add_turn(
            query="Test query",
            answer="Test answer",
            references=["ref1"],
            action_required="customer_action_required"
        )

        turns_list = memory.get_turns_list()

        assert len(turns_list) == 1
        assert turns_list[0]["query"] == "Test query"
        assert turns_list[0]["answer"] == "Test answer"
        assert turns_list[0]["references"] == ["ref1"]
        assert turns_list[0]["action_required"] == "customer_action_required"
        assert "timestamp" in turns_list[0]

    def test_get_statistics(self, memory):
        """Test getting memory statistics."""
        memory.add_turn(
            query="Q1",
            answer="A1",
            references=[],
            action_required="none"
        )
        memory.add_turn(
            query="Q2",
            answer="A2",
            references=[],
            action_required="none"
        )

        stats = memory.get_statistics()

        assert stats["total_turns"] == 2
        assert stats["max_capacity"] == 5
        assert stats["context_window"] == 3
        assert stats["memory_enabled"] is True
        assert stats["persistence"] == "session-only"
        assert "session_duration_seconds" in stats
        assert stats["session_duration_seconds"] >= 0

    def test_clear(self, memory):
        """Test clearing memory."""
        # Add some turns
        for i in range(3):
            memory.add_turn(
                query=f"Q{i}",
                answer=f"A{i}",
                references=[],
                action_required="none"
            )

        assert len(memory.turns) == 3

        # Clear memory
        memory.clear()

        assert len(memory.turns) == 0
        assert memory.is_empty()

    def test_is_empty(self, memory):
        """Test is_empty method."""
        assert memory.is_empty()

        memory.add_turn(
            query="Test",
            answer="Test",
            references=[],
            action_required="none"
        )

        assert not memory.is_empty()


class TestSingletonMemory:
    """Tests for singleton memory functions."""

    def test_get_session_memory(self):
        """Test getting singleton memory instance."""
        reset_session_memory()  # Start fresh

        memory1 = get_session_memory()
        memory2 = get_session_memory()

        # Should return same instance
        assert memory1 is memory2

    def test_singleton_persists_data(self):
        """Test that singleton maintains data across calls."""
        reset_session_memory()  # Start fresh

        memory1 = get_session_memory()
        memory1.add_turn(
            query="Test",
            answer="Test",
            references=[],
            action_required="none"
        )

        memory2 = get_session_memory()

        # Should have the turn we added
        assert len(memory2.turns) == 1
        assert memory2.turns[0].query == "Test"

    def test_reset_session_memory(self):
        """Test resetting singleton memory."""
        reset_session_memory()

        memory1 = get_session_memory()
        memory1.add_turn(
            query="Test",
            answer="Test",
            references=[],
            action_required="none"
        )

        # Reset creates new instance
        reset_session_memory()
        memory2 = get_session_memory()

        # Should be different instance
        assert memory1 is not memory2
        # Should be empty
        assert len(memory2.turns) == 0
