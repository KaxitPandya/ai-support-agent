"""
Model Context Protocol (MCP) Prompt Templates.

MCP is a structured prompt engineering pattern that ensures consistent,
high-quality LLM outputs by organizing prompts into clear sections:

1. ROLE - Who the AI is and its capabilities
2. CONTEXT - Retrieved information relevant to the task
3. TASK - What the AI needs to do
4. OUTPUT SCHEMA - Expected format of the response

This pattern improves:
- Response accuracy by providing relevant context
- Output consistency through explicit schema definition
- Maintainability through clear structure
"""

from typing import List

from src.models.schemas import RetrievedContext


# =============================================================================
# ROLE DEFINITION
# =============================================================================
# The system prompt defines WHO the AI is and its expertise areas.
# This sets expectations and boundaries for the AI's responses.

SYSTEM_ROLE = """You are an expert customer support assistant for a domain registrar company.
Your role is to help support agents resolve customer tickets efficiently and accurately.

## Your Expertise Areas:
- Domain registration, renewal, and transfer processes
- WHOIS information requirements and privacy protection
- DNS configuration and troubleshooting
- Billing, payments, and refund policies
- Domain suspension and reactivation procedures
- Security features and abuse policy enforcement

## Response Guidelines:
1. Provide clear, accurate, and helpful responses
2. Reference specific policies and documentation sections when applicable
3. Identify when escalation to specialized teams is needed
4. Be empathetic to customer concerns while following policy
5. Include actionable next steps for resolution"""


# =============================================================================
# OUTPUT SCHEMA
# =============================================================================
# Defines the exact JSON structure expected from the LLM.
# This ensures consistent, parseable outputs for downstream processing.

OUTPUT_SCHEMA = {
    "answer": "A clear, helpful response addressing the customer's issue with specific steps",
    "references": ["List of policy documents or FAQ sections used to formulate the answer"],
    "action_required": "One of the predefined action types (see below)"
}

OUTPUT_SCHEMA_STR = """{
  "answer": "A clear, helpful response addressing the customer's issue. Include specific steps they can take.",
  "references": ["List of relevant policy documents or FAQ sections used to formulate the answer"],
  "action_required": "One of: 'none', 'escalate_to_abuse_team', 'escalate_to_billing', 'escalate_to_technical', 'customer_action_required', 'follow_up_required'"
}"""

ACTION_TYPES = {
    "none": "Issue can be resolved with the provided answer",
    "escalate_to_abuse_team": "Abuse, policy violation, or security issue requiring abuse team review",
    "escalate_to_billing": "Payment, refund, or billing-related issue requiring billing team",
    "escalate_to_technical": "Complex DNS, hosting, or technical issue requiring engineering",
    "customer_action_required": "Customer needs to take action (verify email, update WHOIS, pay invoice, etc.)",
    "follow_up_required": "Support agent should follow up after customer takes action"
}


# =============================================================================
# CONTEXT BUILDER
# =============================================================================
# Formats retrieved documents into the CONTEXT section of the prompt.

def build_context_section(contexts: List[RetrievedContext]) -> str:
    """
    Build the CONTEXT section from RAG-retrieved documents.
    
    This section provides the LLM with relevant information from
    the knowledge base to ground its response in actual documentation.
    
    Args:
        contexts: Documents retrieved from the vector database.
        
    Returns:
        Formatted context string with references and content.
    """
    if not contexts:
        return """No relevant documentation was found in the knowledge base.
Please use your general knowledge of domain registrar policies and best practices.
Note: Response should still follow standard domain registrar procedures."""
    
    context_parts = []
    for i, ctx in enumerate(contexts, 1):
        doc = ctx.document
        reference = f"{doc.category}: {doc.title}"
        if doc.section:
            reference += f", {doc.section}"
        
        context_parts.append(f"""
### Document {i}: {reference}
**Similarity Score:** {ctx.similarity_score:.2%}

{doc.content.strip()}
""")
    
    return "\n".join(context_parts)


def build_action_options() -> str:
    """Build the action options section for the prompt."""
    lines = ["### Available Action Types:"]
    for action, description in ACTION_TYPES.items():
        lines.append(f"- `{action}` - {description}")
    return "\n".join(lines)


# =============================================================================
# MCP PROMPT BUILDER
# =============================================================================

def build_mcp_prompt(
    ticket_text: str,
    contexts: List[RetrievedContext],
    memory_context: str = ""
) -> List[dict]:
    """
    Build a complete MCP-structured prompt for ticket resolution.
    
    The prompt follows the Model Context Protocol pattern:
    
    ┌─────────────────────────────────────────────────────────────┐
    │ SYSTEM MESSAGE (ROLE)                                       │
    │ - Who the AI is                                             │
    │ - Expertise areas                                           │
    │ - Response guidelines                                       │
    └─────────────────────────────────────────────────────────────┘
    ┌─────────────────────────────────────────────────────────────┐
    │ USER MESSAGE                                                │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │ MEMORY (Past Conversations)                             │ │
    │ │ - Relevant previous interactions                        │ │
    │ │ - Session context                                       │ │
    │ └─────────────────────────────────────────────────────────┘ │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │ CONTEXT (Retrieved Documents)                           │ │
    │ │ - Relevant policies from vector database                │ │
    │ │ - Similarity scores                                     │ │
    │ └─────────────────────────────────────────────────────────┘ │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │ TASK (What to do)                                       │ │
    │ │ - The customer ticket                                   │ │
    │ │ - Instructions for analysis                             │ │
    │ └─────────────────────────────────────────────────────────┘ │
    │ ┌─────────────────────────────────────────────────────────┐ │
    │ │ OUTPUT SCHEMA (Expected format)                         │ │
    │ │ - JSON structure                                        │ │
    │ │ - Field descriptions                                    │ │
    │ │ - Valid action types                                    │ │
    │ └─────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────┘
    
    Args:
        ticket_text: The customer support ticket to analyze.
        contexts: Documents retrieved from the vector database.
        memory_context: Optional context from conversation memory.
        
    Returns:
        List of message dicts ready for LLM API call.
    """
    
    # Build CONTEXT section from retrieved documents
    context_section = build_context_section(contexts)
    
    # Build optional memory section
    memory_section = ""
    if memory_context:
        memory_section = f"""
================================================================================
                             MEMORY SECTION
                  (Relevant Past Conversations for Context)
================================================================================

{memory_context}
"""
    
    # Build the complete user message with MCP structure
    user_message = f"""{memory_section}
================================================================================
                              CONTEXT SECTION
                    (Retrieved from Knowledge Base via RAG)
================================================================================

{context_section}

================================================================================
                               TASK SECTION
================================================================================

Analyze the following customer support ticket and provide a helpful response
based on the context provided above.

### Customer Ticket:
\"\"\"
{ticket_text}
\"\"\"

### Analysis Instructions:
1. Read the customer's issue carefully and identify the core problem
2. Use the provided context documents to formulate an accurate response
3. Cite specific policy sections when relevant (use exact references)
4. Determine if any escalation or follow-up action is required
5. Provide clear, actionable steps for resolution

================================================================================
                            OUTPUT SCHEMA SECTION
================================================================================

You MUST respond with a valid JSON object matching this exact schema:

{OUTPUT_SCHEMA_STR}

{build_action_options()}

### Response Requirements:
- Output ONLY valid JSON, no additional text or markdown
- The "answer" field should be comprehensive and actionable
- The "references" field should list actual document references from context
- The "action_required" field MUST be one of the predefined action types
"""

    return [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": user_message}
    ]


def build_simple_prompt(ticket_text: str) -> List[dict]:
    """
    Build a simplified MCP prompt without retrieved context.
    
    This is a fallback when no relevant documents are found.
    
    Args:
        ticket_text: The customer support ticket to analyze.
        
    Returns:
        List of message dicts for LLM API call.
    """
    user_message = f"""
================================================================================
                               TASK SECTION
================================================================================

Analyze the following customer support ticket for a domain registrar
and provide a helpful response.

### Customer Ticket:
\"\"\"
{ticket_text}
\"\"\"

================================================================================
                            OUTPUT SCHEMA SECTION
================================================================================

Respond with a valid JSON object matching this exact schema:

{OUTPUT_SCHEMA_STR}

{build_action_options()}

Output ONLY valid JSON, no additional text.
"""

    return [
        {"role": "system", "content": SYSTEM_ROLE},
        {"role": "user", "content": user_message}
    ]


# =============================================================================
# PROMPT UTILITIES
# =============================================================================

def get_mcp_structure_info() -> dict:
    """Return information about the MCP prompt structure."""
    return {
        "pattern": "Model Context Protocol (MCP)",
        "sections": {
            "role": "System message defining AI identity and guidelines",
            "context": "Retrieved documents from vector database (RAG)",
            "task": "The customer ticket and analysis instructions",
            "output_schema": "JSON structure specification"
        },
        "output_format": OUTPUT_SCHEMA,
        "action_types": ACTION_TYPES
    }
