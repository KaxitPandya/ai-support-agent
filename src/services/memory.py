"""
Conversation Memory Service.

Implements a memory buffer that stores conversation history in the vector store.
This enables the system to:
1. Learn from past interactions
2. Reference similar previous conversations
3. Provide more consistent responses over time
4. Detect and handle repeat queries

Memory Architecture:
- Each Q&A pair is stored as a searchable document
- Memories have timestamps for relevance decay
- Similar memories can be retrieved to inform new responses
- Feedback can be stored to improve future responses

Based on research from:
- LangChain Memory modules
- RAG best practices for conversational AI
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from collections import deque

import numpy as np

from src.models.schemas import Document, RetrievedContext, TicketResponse
from src.services.embedding import EmbeddingService, get_embedding_service
from src.services.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single conversation memory entry."""
    id: str
    query: str
    response: str
    references: List[str]
    action_taken: str
    timestamp: datetime
    feedback_score: Optional[float] = None  # 1-5 rating
    feedback_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_document(self) -> Document:
        """Convert memory to a searchable document."""
        content = f"""Query: {self.query}

Response: {self.response}

References: {', '.join(self.references) if self.references else 'None'}

Action: {self.action_taken}

Timestamp: {self.timestamp.isoformat()}"""
        
        return Document(
            id=self.id,
            title=f"Memory: {self.query[:50]}...",
            content=content,
            category="Conversation Memory",
            section=f"Memory from {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
        )


class ConversationMemory:
    """
    Short-term conversation buffer for maintaining context within a session.
    
    Uses a sliding window approach to keep recent exchanges in context.
    """
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_turns: Maximum conversation turns to remember.
        """
        self.max_turns = max_turns
        self.buffer: deque = deque(maxlen=max_turns)
        self.session_id: Optional[str] = None
    
    def add_turn(self, query: str, response: TicketResponse) -> None:
        """Add a conversation turn to the buffer."""
        self.buffer.append({
            "query": query,
            "response": response.answer,
            "references": response.references,
            "action": response.action_required,
            "timestamp": datetime.now(timezone.utc)
        })
    
    def get_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context as a formatted string.
        
        Args:
            num_turns: Number of recent turns to include.
            
        Returns:
            Formatted conversation history.
        """
        if not self.buffer:
            return ""
        
        recent = list(self.buffer)[-num_turns:]
        
        context_parts = ["## Recent Conversation History\n"]
        for i, turn in enumerate(recent, 1):
            context_parts.append(f"""
### Turn {i}:
**Customer:** {turn['query']}
**Response:** {turn['response'][:200]}...
**Action:** {turn['action']}
""")
        
        return "\n".join(context_parts)
    
    def clear(self) -> None:
        """Clear the conversation buffer."""
        self.buffer.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current conversation."""
        return {
            "turns": len(self.buffer),
            "max_turns": self.max_turns,
            "session_id": self.session_id
        }


class LongTermMemory:
    """
    Long-term memory stored in the vector database.
    
    Enables:
    - Learning from past interactions
    - Finding similar previous queries
    - Improving responses over time with feedback
    - Detecting repeat/similar queries
    
    This is inspired by:
    - Episodic memory in cognitive architectures
    - LangChain's VectorStoreRetrieverMemory
    """
    
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        relevance_decay: float = 0.1,  # How much older memories decay in relevance
        similarity_threshold: float = 0.75  # Threshold for "similar" queries
    ):
        """
        Initialize long-term memory.
        
        Args:
            embedding_service: Service for embeddings.
            relevance_decay: Decay factor for older memories (per day).
            similarity_threshold: Threshold for finding similar memories.
        """
        self.embedding_service = embedding_service or get_embedding_service()
        self.relevance_decay = relevance_decay
        self.similarity_threshold = similarity_threshold
        
        # Separate vector store for memories
        self.memory_store = FAISSVectorStore(embedding_service=self.embedding_service)
        self.memories: Dict[str, MemoryEntry] = {}
    
    def _generate_memory_id(self, query: str, timestamp: datetime) -> str:
        """Generate unique ID for a memory."""
        content = f"{query}{timestamp.isoformat()}"
        return f"mem_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _apply_relevance_decay(
        self,
        memories: List[RetrievedContext],
        current_time: datetime
    ) -> List[RetrievedContext]:
        """
        Apply time-based relevance decay to memories.
        
        Older memories get lower scores to prioritize recent information.
        """
        decayed = []
        
        for mem in memories:
            # Get memory timestamp
            mem_id = mem.document.id
            if mem_id in self.memories:
                mem_time = self.memories[mem_id].timestamp
                days_old = (current_time - mem_time).days
                
                # Apply exponential decay
                decay_factor = np.exp(-self.relevance_decay * days_old)
                adjusted_score = mem.similarity_score * decay_factor
                
                decayed.append(RetrievedContext(
                    document=mem.document,
                    similarity_score=adjusted_score
                ))
            else:
                decayed.append(mem)
        
        # Re-sort by adjusted score
        decayed.sort(key=lambda x: x.similarity_score, reverse=True)
        return decayed
    
    def store(
        self,
        query: str,
        response: TicketResponse,
        metadata: Dict[str, Any] | None = None
    ) -> str:
        """
        Store a conversation in long-term memory.
        
        Args:
            query: The customer query.
            response: The generated response.
            metadata: Optional additional metadata.
            
        Returns:
            Memory ID.
        """
        timestamp = datetime.now(timezone.utc)
        mem_id = self._generate_memory_id(query, timestamp)
        
        memory = MemoryEntry(
            id=mem_id,
            query=query,
            response=response.answer,
            references=response.references,
            action_taken=response.action_required,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        
        # Store in dictionary
        self.memories[mem_id] = memory
        
        # Store in vector store for retrieval
        doc = memory.to_document()
        self.memory_store.add_documents([doc])
        
        logger.info(f"Stored memory: {mem_id}")
        return mem_id
    
    def recall(
        self,
        query: str,
        top_k: int = 3,
        include_context: bool = True
    ) -> List[MemoryEntry]:
        """
        Recall relevant memories for a query.
        
        Args:
            query: The query to find memories for.
            top_k: Maximum memories to return.
            include_context: Whether to apply relevance decay.
            
        Returns:
            List of relevant MemoryEntry objects.
        """
        if self.memory_store.get_document_count() == 0:
            return []
        
        # Search memory store
        results = self.memory_store.search(
            query,
            top_k=top_k,
            threshold=self.similarity_threshold
        )
        
        # Apply relevance decay
        if include_context:
            results = self._apply_relevance_decay(
                results,
                datetime.now(timezone.utc)
            )
        
        # Convert to MemoryEntry objects
        memories = []
        for ctx in results:
            mem_id = ctx.document.id
            if mem_id in self.memories:
                memories.append(self.memories[mem_id])
        
        return memories
    
    def find_similar_query(self, query: str) -> Optional[MemoryEntry]:
        """
        Find if a very similar query was asked before.
        
        Useful for:
        - Detecting repeat questions
        - Providing consistent responses
        - Quick retrieval for common queries
        
        Args:
            query: The query to check.
            
        Returns:
            MemoryEntry if similar query found, None otherwise.
        """
        memories = self.recall(query, top_k=1)
        
        if memories:
            # Check if it's highly similar (potential repeat)
            results = self.memory_store.search(query, top_k=1, threshold=0.9)
            if results and results[0].similarity_score > 0.9:
                return memories[0]
        
        return None
    
    def add_feedback(
        self,
        memory_id: str,
        score: float,
        feedback_text: str | None = None
    ) -> bool:
        """
        Add feedback to a memory for learning.
        
        Args:
            memory_id: ID of the memory to update.
            score: Rating (1-5).
            feedback_text: Optional feedback text.
            
        Returns:
            True if updated, False if memory not found.
        """
        if memory_id not in self.memories:
            return False
        
        self.memories[memory_id].feedback_score = score
        self.memories[memory_id].feedback_text = feedback_text
        
        logger.info(f"Added feedback to memory {memory_id}: score={score}")
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total = len(self.memories)
        with_feedback = sum(1 for m in self.memories.values() if m.feedback_score)
        avg_score = np.mean([
            m.feedback_score for m in self.memories.values() 
            if m.feedback_score is not None
        ]) if with_feedback else 0
        
        return {
            "total_memories": total,
            "memories_with_feedback": with_feedback,
            "average_feedback_score": float(avg_score),
            "vector_store_size": self.memory_store.get_document_count()
        }
    
    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self.memory_store.clear()
        logger.info("Long-term memory cleared")


class HybridMemory:
    """
    Combines short-term conversation buffer with long-term vector memory.
    
    This is the recommended memory implementation that provides:
    - Session continuity (short-term)
    - Learning from history (long-term)
    - Similar query detection
    - Feedback integration
    """
    
    def __init__(
        self,
        short_term_turns: int = 10,
        long_term_decay: float = 0.1,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize hybrid memory.
        
        Args:
            short_term_turns: Max turns in short-term buffer.
            long_term_decay: Decay factor for long-term memories.
            similarity_threshold: Threshold for memory retrieval.
        """
        self.short_term = ConversationMemory(max_turns=short_term_turns)
        self.long_term = LongTermMemory(
            relevance_decay=long_term_decay,
            similarity_threshold=similarity_threshold
        )
    
    def add_interaction(
        self,
        query: str,
        response: TicketResponse,
        store_long_term: bool = True
    ) -> Optional[str]:
        """
        Add an interaction to memory.
        
        Args:
            query: Customer query.
            response: Generated response.
            store_long_term: Whether to persist to long-term memory.
            
        Returns:
            Memory ID if stored long-term, None otherwise.
        """
        # Always add to short-term
        self.short_term.add_turn(query, response)
        
        # Optionally store in long-term
        if store_long_term:
            return self.long_term.store(query, response)
        
        return None
    
    def get_relevant_context(
        self,
        query: str,
        include_short_term: bool = True,
        include_long_term: bool = True,
        max_long_term: int = 2
    ) -> str:
        """
        Get relevant memory context for a query.
        
        Combines short-term conversation history with
        relevant long-term memories.
        
        Args:
            query: The current query.
            include_short_term: Include recent conversation.
            include_long_term: Include relevant past memories.
            max_long_term: Max long-term memories to include.
            
        Returns:
            Formatted context string.
        """
        context_parts = []
        
        # Short-term context
        if include_short_term:
            short_ctx = self.short_term.get_context()
            if short_ctx:
                context_parts.append(short_ctx)
        
        # Long-term relevant memories
        if include_long_term:
            memories = self.long_term.recall(query, top_k=max_long_term)
            if memories:
                context_parts.append("\n## Relevant Past Interactions\n")
                for mem in memories:
                    context_parts.append(f"""
**Previous Query:** {mem.query}
**Previous Response:** {mem.response[:200]}...
**Action Taken:** {mem.action_taken}
""")
        
        return "\n".join(context_parts)
    
    def check_similar_query(self, query: str) -> Optional[MemoryEntry]:
        """Check if a similar query was asked before."""
        return self.long_term.find_similar_query(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined memory statistics."""
        return {
            "short_term": self.short_term.get_summary(),
            "long_term": self.long_term.get_statistics()
        }
    
    def clear_session(self) -> None:
        """Clear only the short-term session buffer."""
        self.short_term.clear()
    
    def clear_all(self) -> None:
        """Clear all memory (both short and long term)."""
        self.short_term.clear()
        self.long_term.clear()


# Singleton instances
_conversation_memory: ConversationMemory | None = None
_hybrid_memory: HybridMemory | None = None


def get_conversation_memory() -> ConversationMemory:
    """Get singleton conversation memory."""
    global _conversation_memory
    if _conversation_memory is None:
        _conversation_memory = ConversationMemory()
    return _conversation_memory


def get_hybrid_memory() -> HybridMemory:
    """Get singleton hybrid memory."""
    global _hybrid_memory
    if _hybrid_memory is None:
        _hybrid_memory = HybridMemory()
    return _hybrid_memory
