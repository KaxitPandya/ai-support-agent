"""
FastAPI application for the Knowledge Assistant.

Provides:
- POST /resolve-ticket - Customer support ticket resolution
- POST /api/documents/upload - Dynamic document upload
- GET /api/documents/files - List uploaded documents
- DELETE /api/documents/files/{filename} - Delete documents
- POST /api/documents/reindex - Rebuild vector index

Note: The main UI is provided via Streamlit (streamlit_app.py)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.upload import router as upload_router
from src.config import get_settings
from src.models.schemas import HealthResponse, TicketRequest, TicketResponse
from src.services.rag import get_rag_pipeline, initialize_rag_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes the RAG pipeline on startup and cleans up on shutdown.
    """
    # Startup
    logger.info("Starting Knowledge Assistant...")
    try:
        initialize_rag_pipeline()
        logger.info("Knowledge Assistant ready!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Knowledge Assistant...")


# Get settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Knowledge Assistant API

A RAG-powered support ticket resolution system that helps support teams
respond to customer queries efficiently using relevant documentation.

### Features
- **Semantic Search**: Finds relevant documentation based on ticket content
- **Context-Aware Responses**: Uses retrieved context for accurate answers
- **MCP-Compliant Output**: Structured JSON responses with references and actions
- **Multiple LLM Support**: Works with OpenAI and Ollama
- **Dynamic Documents**: Upload new documents without restarting

### Usage

**Resolve a ticket:**
```bash
POST /resolve-ticket
{"ticket_text": "My domain was suspended..."}
```

**Upload a new document:**
```bash
POST /api/documents/upload
Content-Type: multipart/form-data
file=@policy.md
```
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include document management API router
app.include_router(upload_router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "ui": "Run 'streamlit run streamlit_app.py' for the web UI"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the application and its components.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model
    )


@app.post(
    "/resolve-ticket",
    response_model=TicketResponse,
    tags=["Ticket Resolution"],
    summary="Resolve a customer support ticket",
    response_description="MCP-compliant structured response with answer, references, and action required"
)
async def resolve_ticket(request: TicketRequest) -> TicketResponse:
    """
    Analyze and resolve a customer support ticket.
    
    This endpoint uses RAG (Retrieval-Augmented Generation) to:
    1. Search for relevant documentation based on the ticket content
    2. Build a context-aware prompt with retrieved information
    3. Generate a helpful response using the configured LLM
    4. Return a structured JSON response following MCP format
    
    **Example Request:**
    ```json
    {
        "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
    }
    ```
    
    **Example Response:**
    ```json
    {
        "answer": "Your domain may have been suspended due to...",
        "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
        "action_required": "customer_action_required"
    }
    ```
    """
    try:
        pipeline = get_rag_pipeline()
        response = pipeline.resolve_ticket(request.ticket_text)
        return response
    except Exception as e:
        logger.error(f"Error resolving ticket: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request. Please try again."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
