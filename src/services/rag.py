"""
RAG (Retrieval-Augmented Generation) pipeline orchestrator.

Combines document retrieval with LLM generation for accurate,
context-aware ticket resolution.

Features:
- Hybrid Search: Combines semantic + keyword search for better results
- Conversation Memory: Learns from past interactions
- Reranking: Uses cross-encoder for improved relevance
- Similar Query Detection: Provides consistent responses for repeat queries
"""

import logging
from typing import List, Literal, Optional

from src.config import get_settings
from src.data.knowledge_base import get_knowledge_base
from src.models.schemas import Document, RetrievedContext, TicketResponse
from src.prompts.mcp_prompt import build_mcp_prompt
from src.services.llm import LLMService, get_llm_service
from src.services.vector_store import FAISSVectorStore, get_vector_store, initialize_vector_store

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline for ticket resolution.
    
    Enhanced Flow:
    1. Receive ticket text
    2. Check memory for similar previous queries
    3. Retrieve relevant documents using hybrid search
    4. Build MCP-compliant prompt with context + memory
    5. Generate response using LLM
    6. Parse and validate response
    7. Store interaction in memory for learning
    """
    
    def __init__(
        self,
        vector_store: FAISSVectorStore | None = None,
        llm_service: LLMService | None = None,
        documents: List[Document] | None = None,
        use_hybrid_search: bool = True,
        use_memory: bool = True,
        search_mode: Literal["semantic", "hybrid"] = "hybrid"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Optional vector store instance.
            llm_service: Optional LLM service instance.
            documents: Optional list of documents to index.
            use_hybrid_search: Enable hybrid (semantic + keyword) search.
            use_memory: Enable conversation memory.
            search_mode: "semantic" for vector-only, "hybrid" for combined.
        """
        self.settings = get_settings()
        self.llm_service = llm_service
        self.vector_store = vector_store
        self.use_hybrid_search = use_hybrid_search
        self.use_memory = use_memory
        self.search_mode = search_mode
        self._initialized = False
        self._custom_documents = documents
        
        # Lazy-loaded components
        self._hybrid_search = None
        self._memory = None
    
    def _get_hybrid_search(self):
        """Lazy load hybrid search service."""
        if self._hybrid_search is None and self.use_hybrid_search:
            from src.services.hybrid_search import HybridSearchService
            self._hybrid_search = HybridSearchService(
                vector_store=self.vector_store,
                use_reranking=True
            )
        return self._hybrid_search
    
    def _get_memory(self):
        """Lazy load memory service."""
        if self._memory is None and self.use_memory:
            from src.services.memory import HybridMemory
            self._memory = HybridMemory(
                short_term_turns=10,
                long_term_decay=0.1,
                similarity_threshold=0.75
            )
        return self._memory
    
    def initialize(self) -> None:
        """
        Initialize the pipeline with knowledge base.

        This should be called once at application startup.
        Preserves any documents that were already added to the vector store
        (e.g., uploaded documents).
        """
        if self._initialized:
            logger.info("RAG pipeline already initialized")
            return

        # Initialize LLM service
        if self.llm_service is None:
            self.llm_service = get_llm_service()

        # Load base knowledge documents
        base_documents = self._custom_documents or get_knowledge_base()

        # Use existing vector store if available, preserving uploaded documents
        if self.vector_store is None:
            # Get the singleton vector store (may already have uploaded docs)
            self.vector_store = get_vector_store()

        # Add base documents if not already present (preserves uploaded docs)
        existing_ids = {doc.id for doc in self.vector_store.documents}
        new_base_docs = [doc for doc in base_documents if doc.id not in existing_ids]

        if new_base_docs:
            self.vector_store.add_documents(new_base_docs)
            logger.info(f"Added {len(new_base_docs)} base documents to vector store")

        total_docs = self.vector_store.get_document_count()

        # Initialize hybrid search if enabled
        if self.use_hybrid_search and self.search_mode == "hybrid":
            hybrid = self._get_hybrid_search()
            if hybrid:
                # Index ALL documents (base + uploaded)
                hybrid.index_documents(self.vector_store.documents)
                logger.info("Hybrid search initialized with all documents")

        self._initialized = True
        logger.info(f"RAG pipeline initialized. Total documents in vector store: {total_docs}")
        if self.use_memory:
            logger.info("Conversation memory enabled")
    
    def retrieve_context(self, query: str) -> List[RetrievedContext]:
        """
        Retrieve relevant documents for a query.
        
        Uses hybrid search (semantic + keyword) when enabled,
        otherwise falls back to pure semantic search.
        
        Args:
            query: The search query.
            
        Returns:
            List of retrieved contexts with similarity scores.
        """
        if not self._initialized:
            self.initialize()
        
        # Use hybrid search if enabled
        if self.use_hybrid_search and self.search_mode == "hybrid":
            hybrid = self._get_hybrid_search()
            if hybrid and hybrid._initialized:
                results = hybrid.search(
                    query,
                    top_k=self.settings.top_k_results,
                    semantic_threshold=self.settings.similarity_threshold
                )
                return hybrid.to_retrieved_context(results)
        
        # Fallback to pure semantic search
        return self.vector_store.search(
            query,
            top_k=self.settings.top_k_results,
            threshold=self.settings.similarity_threshold
        )
    
    def resolve_ticket(
        self,
        ticket_text: str,
        include_memory_context: bool = True,
        store_in_memory: bool = True
    ) -> TicketResponse:
        """
        Resolve a customer support ticket.
        
        Args:
            ticket_text: The ticket text from the customer.
            include_memory_context: Include relevant past conversations.
            store_in_memory: Store this interaction for future learning.
            
        Returns:
            TicketResponse with answer, references, and action_required.
        """
        if not self._initialized:
            self.initialize()
        
        logger.info(f"Processing ticket: {ticket_text[:100]}...")
        
        # Step 1: Check for similar previous query (for consistency)
        memory = self._get_memory() if self.use_memory else None
        similar_query = None
        
        if memory and include_memory_context:
            similar_query = memory.check_similar_query(ticket_text)
            if similar_query:
                logger.info(f"Found similar previous query: {similar_query.query[:50]}...")
        
        # Step 2: Retrieve relevant context
        contexts = self.retrieve_context(ticket_text)
        logger.info(f"Retrieved {len(contexts)} relevant documents")
        
        # Step 3: Build MCP prompt with optional memory context
        memory_context = ""
        if memory and include_memory_context:
            memory_context = memory.get_relevant_context(
                ticket_text,
                include_short_term=True,
                include_long_term=True,
                max_long_term=2
            )
        
        messages = build_mcp_prompt(ticket_text, contexts, memory_context)
        
        # Step 4: Generate response
        try:
            response_data = self.llm_service.generate_json(messages)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            # Return a fallback response
            return TicketResponse(
                answer="I apologize, but I'm unable to process your request at the moment. Please contact support directly for assistance.",
                references=[],
                action_required="escalate_to_technical"
            )
        
        # Step 5: Parse and validate response
        response = self._parse_response(response_data, contexts)
        
        # Step 6: Store in memory for learning
        if memory and store_in_memory:
            memory.add_interaction(ticket_text, response, store_long_term=True)
            logger.info("Interaction stored in memory")
        
        logger.info(f"Ticket resolved. Action required: {response.action_required}")
        return response
    
    def _parse_response(
        self,
        response_data: dict,
        contexts: List[RetrievedContext]
    ) -> TicketResponse:
        """
        Parse and validate the LLM response.
        
        Args:
            response_data: Raw response from LLM.
            contexts: Retrieved contexts for reference building.
            
        Returns:
            Validated TicketResponse.
        """
        # Extract answer
        answer = response_data.get("answer", "")
        if not answer:
            answer = "I'm unable to provide a specific answer. Please contact support for assistance."
        
        # Extract references
        references = response_data.get("references", [])
        if not isinstance(references, list):
            references = [str(references)] if references else []
        
        # If no references provided, use retrieved document references
        if not references and contexts:
            references = [
                f"{ctx.document.category}: {ctx.document.title}, {ctx.document.section}"
                for ctx in contexts[:3]
                if ctx.document.section
            ]
        
        # Extract and validate action_required
        action_required = response_data.get("action_required", "none")
        valid_actions = {
            "none", "escalate_to_abuse_team", "escalate_to_billing",
            "escalate_to_technical", "customer_action_required", "follow_up_required"
        }
        if action_required not in valid_actions:
            action_required = "none"
        
        return TicketResponse(
            answer=answer,
            references=references,
            action_required=action_required
        )
    
    def get_memory_stats(self) -> dict:
        """Get statistics about the memory system."""
        memory = self._get_memory()
        if memory:
            return memory.get_statistics()
        return {"memory_enabled": False}
    
    def clear_session_memory(self) -> None:
        """Clear the short-term session memory."""
        memory = self._get_memory()
        if memory:
            memory.clear_session()
            logger.info("Session memory cleared")
    
    def add_feedback(self, memory_id: str, score: float, feedback_text: str | None = None) -> bool:
        """
        Add feedback to a memory entry for learning.
        
        Args:
            memory_id: ID of the memory to update.
            score: Rating (1-5).
            feedback_text: Optional feedback.
            
        Returns:
            True if updated, False otherwise.
        """
        memory = self._get_memory()
        if memory:
            return memory.long_term.add_feedback(memory_id, score, feedback_text)
        return False


# Singleton instance
_rag_pipeline: RAGPipeline | None = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or create the singleton RAG pipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline


def initialize_rag_pipeline() -> RAGPipeline:
    """Initialize and return the RAG pipeline."""
    pipeline = get_rag_pipeline()
    pipeline.initialize()
    return pipeline


def reset_rag_pipeline() -> None:
    """Reset the RAG pipeline (useful for testing)."""
    global _rag_pipeline
    _rag_pipeline = None
