"""
Simple Session-Based Memory System.

Optimized for Streamlit Cloud - uses session state only (no file persistence).
Works reliably and shows memory context visually in the UI.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single conversation turn (query + response)."""
    query: str
    answer: str
    references: List[str]
    action_required: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMemory:
    """
    Session-based conversation memory for Streamlit.

    Stores recent conversation history in memory (no file persistence).
    Perfect for Streamlit Cloud where filesystem is ephemeral.

    Features:
    - Sliding window buffer (keeps last N conversations)
    - Context formatting for LLM prompts
    - Session statistics
    - Clear visualization in UI
    """

    def __init__(self, max_turns: int = 10, context_window: int = 3):
        """
        Initialize session memory.

        Args:
            max_turns: Maximum conversation turns to store (default: 10)
            context_window: Number of recent turns to include in prompts (default: 3)
        """
        self.max_turns = max_turns
        self.context_window = context_window
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self.session_start = datetime.now(timezone.utc)

    def add_turn(
        self,
        query: str,
        answer: str,
        references: List[str],
        action_required: str
    ) -> None:
        """
        Add a conversation turn to memory.

        Args:
            query: User's question
            answer: AI's response
            references: Document references used
            action_required: Required action type
        """
        turn = ConversationTurn(
            query=query,
            answer=answer,
            references=references,
            action_required=action_required
        )
        self.turns.append(turn)
        logger.info(f"Added turn to memory. Total turns: {len(self.turns)}")

    def get_context_for_prompt(self, num_turns: Optional[int] = None) -> str:
        """
        Get formatted conversation context for LLM prompts.

        Args:
            num_turns: Number of recent turns to include (default: context_window)

        Returns:
            Formatted context string for inclusion in prompts
        """
        if not self.turns:
            return ""

        num_turns = num_turns or self.context_window
        recent_turns = list(self.turns)[-num_turns:]

        context_parts = ["## Recent Conversation History\n"]

        for i, turn in enumerate(recent_turns, 1):
            # Truncate answer for context (first 200 chars)
            answer_preview = turn.answer[:200] + "..." if len(turn.answer) > 200 else turn.answer

            context_parts.append(f"""
### Turn {i}:
**Customer Query:** {turn.query}
**Your Previous Response:** {answer_preview}
**Action Taken:** {turn.action_required}
""")

        context_parts.append("\nUse this conversation history to maintain continuity and avoid repeating information.\n")

        return "\n".join(context_parts)

    def get_turns_list(self) -> List[Dict[str, Any]]:
        """
        Get all conversation turns as a list of dictionaries.

        Returns:
            List of turn dictionaries for display in UI
        """
        return [
            {
                "query": turn.query,
                "answer": turn.answer,
                "references": turn.references,
                "action_required": turn.action_required,
                "timestamp": turn.timestamp.isoformat()
            }
            for turn in self.turns
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory stats
        """
        return {
            "total_turns": len(self.turns),
            "max_capacity": self.max_turns,
            "context_window": self.context_window,
            "session_duration_seconds": (datetime.now(timezone.utc) - self.session_start).total_seconds(),
            "memory_enabled": True,
            "persistence": "session-only"
        }

    def clear(self) -> None:
        """Clear all conversation history."""
        self.turns.clear()
        self.session_start = datetime.now(timezone.utc)
        logger.info("Session memory cleared")

    def is_empty(self) -> bool:
        """Check if memory has no turns."""
        return len(self.turns) == 0


# Singleton instance for backward compatibility
_session_memory: Optional[SessionMemory] = None


def get_session_memory() -> SessionMemory:
    """Get singleton session memory instance."""
    global _session_memory
    if _session_memory is None:
        _session_memory = SessionMemory()
    return _session_memory


def reset_session_memory() -> None:
    """Reset the singleton instance (useful for testing)."""
    global _session_memory
    _session_memory = None