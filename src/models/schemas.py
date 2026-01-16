"""
Pydantic schemas for API request/response validation.
Follows MCP (Model Context Protocol) output structure.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ActionRequired(str, Enum):
    """Possible actions that may be required after ticket resolution."""
    
    NONE = "none"
    ESCALATE_TO_ABUSE_TEAM = "escalate_to_abuse_team"
    ESCALATE_TO_BILLING = "escalate_to_billing"
    ESCALATE_TO_TECHNICAL = "escalate_to_technical"
    CUSTOMER_ACTION_REQUIRED = "customer_action_required"
    FOLLOW_UP_REQUIRED = "follow_up_required"


class TicketRequest(BaseModel):
    """Request schema for the /resolve-ticket endpoint."""
    
    ticket_text: str = Field(
        ...,
        min_length=5,
        max_length=5000,
        description="The customer support ticket text to analyze",
        examples=["My domain was suspended and I didn't get any notice. How can I reactivate it?"]
    )


class TicketResponse(BaseModel):
    """
    MCP-compliant response schema for ticket resolution.
    
    This follows the Model Context Protocol structure with:
    - answer: The helpful response for the customer
    - references: List of relevant policy/documentation references
    - action_required: The action needed (if any)
    """
    
    answer: str = Field(
        ...,
        description="The generated response to help resolve the customer's issue"
    )
    references: List[str] = Field(
        default_factory=list,
        description="List of relevant policy documents or FAQ sections referenced"
    )
    action_required: str = Field(
        default=ActionRequired.NONE.value,
        description="Action required to resolve the ticket"
    )


class Document(BaseModel):
    """A document in the knowledge base."""
    
    id: str = Field(..., description="Unique identifier for the document")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content")
    category: str = Field(..., description="Document category")
    section: Optional[str] = Field(None, description="Section reference")


class RetrievedContext(BaseModel):
    """Context retrieved from the vector store."""
    
    document: Document
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(default="healthy")
    version: str
    llm_provider: str
    embedding_model: str
