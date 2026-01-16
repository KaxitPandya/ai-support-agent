# 🧠 Knowledge Assistant - RAG-Powered Support Ticket Resolution

A production-ready **Retrieval-Augmented Generation (RAG)** system that helps support teams respond to customer tickets efficiently using relevant documentation.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([https://your-app.streamlit.app](https://ai-support-agent1.streamlit.app/))
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [🚀 Deploy to Streamlit Cloud](#deploy-to-streamlit-cloud)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Development](#development)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)

---

## Overview

The Knowledge Assistant analyzes customer support queries and returns structured, helpful responses by:

1. **Retrieving** relevant documentation from a vector database
2. **Augmenting** prompts with retrieved context
3. **Generating** accurate, policy-compliant responses using LLMs

### Sample Input

```json
{
  "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
}
```

### Sample Output (MCP-compliant)

```json
{
  "answer": "Your domain may have been suspended due to a WHOIS verification failure or policy violation. To reactivate, please log into your account, navigate to 'My Domains', and check the suspension reason. For WHOIS issues, update your contact information and verify your email. The domain should be reactivated within 24-48 hours after verification.",
  "references": [
    "Policy: Domain Suspension Guidelines, Section 4.2 - Reactivation Process",
    "Policy: Domain Suspension Guidelines, Section 4.3 - Communication"
  ],
  "action_required": "customer_action_required"
}
```

---

## Features

| Feature | Description |
|---------|-------------|
| **🌐 Streamlit Web UI** | Beautiful, interactive chat interface - deploy to Streamlit Cloud |
| **RAG Pipeline** | FAISS-based vector search with Sentence Transformers embeddings |
| **Hybrid Search** | Combined semantic + BM25 keyword search with cross-encoder reranking |
| **Semantic Chunking** | Topic-aware document splitting using embeddings (not character-based) |
| **Conversation Memory** | Short-term buffer + long-term vector memory for learning |
| **OpenAI Integration** | Works with GPT-4, GPT-4o-mini, GPT-3.5-turbo |
| **MCP-Compliant** | Structured prompts with role, context, task, and output schema |
| **Dynamic Document Upload** | Upload new documents via UI without restarting |
| **FastAPI Backend** | Modern async API with automatic OpenAPI documentation |
| **Docker Ready** | Multi-stage Dockerfile and Docker Compose for easy deployment |
| **Comprehensive Tests** | 114 tests with 76% coverage |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                              │
│           POST /resolve-ticket  │  POST /api/documents/upload           │
└───────────────────────┬─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline (Enhanced)                         │
│                                                                          │
│  ┌─────────────────┐     ┌──────────────────────────────────────────┐   │
│  │ Conversation    │     │           Hybrid Search                   │   │
│  │ Memory          │     │  ┌─────────────┐   ┌─────────────────┐   │   │
│  │                 │     │  │  Semantic   │   │    BM25         │   │   │
│  │ • Short-term    │     │  │  (FAISS/    │ + │   Keyword       │   │   │
│  │   buffer        │     │  │   FAISS)    │   │   Search        │   │   │
│  │ • Long-term     │     │  └──────┬──────┘   └───────┬─────────┘   │   │
│  │   vector store  │     │         │                  │             │   │
│  └────────┬────────┘     │         └───────┬──────────┘             │   │
│           │              │                 ▼                         │   │
│           │              │    ┌───────────────────────┐             │   │
│           │              │    │  Cross-Encoder        │             │   │
│           │              │    │  Reranking            │             │   │
│           │              │    └───────────┬───────────┘             │   │
│           │              └────────────────┼─────────────────────────┘   │
│           │                               │                              │
│           └───────────────────────────────┼──────────────────────────┐  │
│                                           ▼                          │  │
│  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │                    MCP Prompt Builder                          │  │  │
│  │  Memory Context + Retrieved Docs + Task + Output Schema        │  │  │
│  └───────────────────────────────┬────────────────────────────────┘  │  │
└──────────────────────────────────┼───────────────────────────────────┘  │
                                   ▼                                       │
┌─────────────────────────────────────────────────────────────────────────┐
│                           LLM Service (OpenAI)                          │
│                    ┌─────────────────────────────┐                      │
│                    │  GPT-4o-mini / GPT-4 / 3.5  │                      │
│                    └─────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Advanced Features

#### 🔍 Hybrid Search
Combines **semantic search** (vector similarity) with **BM25 keyword search** for superior results:
- Semantic search finds conceptually similar documents
- BM25 finds exact term matches
- Cross-encoder reranking improves final ordering

#### 🧠 Semantic Chunking
Unlike simple character-based chunking, our system:
- Uses embeddings to detect topic boundaries
- Splits at semantic breakpoints (topic changes)
- Preserves context and meaning within chunks
- Automatically adjusts chunk sizes to content structure

#### 💾 Conversation Memory
The system learns from interactions:
- **Short-term**: Maintains session context (recent exchanges)
- **Long-term**: Stores conversations in vector database
- **Similar query detection**: Provides consistent responses
- **Feedback integration**: Improves with user ratings

#### 🗄️ Vector Database
- **FAISS**: Fast, efficient, in-memory vector search with persistence support

### Component Details

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Embedding Service** | Sentence Transformers (all-MiniLM-L6-v2) | Text → Vector conversion |
| **Vector Store** | FAISS (IndexFlatIP) | Fast similarity search |
| **LLM Service** | OpenAI / Ollama | Response generation |
| **API Layer** | FastAPI | HTTP endpoint handling |

---

## Quick Start

### Option 1: Streamlit Web UI (Easiest)

```bash
# 1. Clone the repository
git clone <your-fork-url>
cd interview-exercise-ai

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment file and add your API key
cp env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here

# 5. Run Streamlit app
streamlit run streamlit_app.py

# Open http://localhost:8501 in your browser
```

### Option 2: Docker

```bash
# 1. Clone the repository
git clone <your-fork-url>
cd interview-exercise-ai

# 2. Copy environment file and add your API key
cp env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here

# 3. Start the application
docker-compose up --build

# 4. Test the API
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "My domain was suspended. How can I reactivate it?"}'
```

---

## 🚀 Deploy to Streamlit Cloud

The easiest way to deploy this application is via **Streamlit Cloud** - it's free and connects directly to GitHub!

### Step 1: Fork & Push to GitHub

```bash
# Fork this repository and clone your fork
git clone https://github.com/YOUR_USERNAME/interview-exercise-ai.git
cd interview-exercise-ai
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository and branch
5. Set the main file path: `streamlit_app.py`
6. Click **"Deploy!"**

### Step 3: Configure Secrets

In your Streamlit Cloud app dashboard:

1. Click **"Manage app"** → **"Settings"** → **"Secrets"**
2. Add your secrets in TOML format:

```toml
OPENAI_API_KEY = "sk-your-api-key-here"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_TEMPERATURE = "0.3"
OPENAI_MAX_TOKENS = "1024"
TOP_K_RESULTS = "5"
SIMILARITY_THRESHOLD = "0.3"
```

3. Click **"Save"** - your app will automatically restart!

### That's it! 🎉

Your Knowledge Assistant is now live at `https://your-app.streamlit.app`

---

## Configuration

All configuration is done via environment variables. See `env.example` for all options.

### Required Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | - |
| `LLM_PROVIDER` | LLM to use: `openai` or `ollama` | `openai` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Response creativity (0-1) | `0.3` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Ollama model name | `llama2` |
| `EMBEDDING_MODEL` | Sentence Transformer model | `all-MiniLM-L6-v2` |
| `TOP_K_RESULTS` | Documents to retrieve | `3` |
| `SIMILARITY_THRESHOLD` | Min similarity score | `0.3` |

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### `GET /`
Returns API information.

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "llm_provider": "openai",
  "embedding_model": "all-MiniLM-L6-v2"
}
```

#### `POST /resolve-ticket`
Resolve a customer support ticket.

**Request:**
```json
{
  "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
}
```

**Response:**
```json
{
  "answer": "Your domain may have been suspended due to...",
  "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
  "action_required": "customer_action_required"
}
```

**Action Values:**
- `none` - Issue resolved
- `escalate_to_abuse_team` - Security/abuse issue
- `escalate_to_billing` - Payment/refund issue
- `escalate_to_technical` - Technical issue requiring engineering
- `customer_action_required` - Customer needs to take action
- `follow_up_required` - Agent should follow up

### Document Management Endpoints

#### `POST /api/documents/upload`
Upload a new document to the knowledge base.

```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@new_policy.md" \
  -F "category=Domain Policies" \
  -F "index_immediately=true"
```

#### `GET /api/documents/files`
List all uploaded documents.

#### `DELETE /api/documents/files/{filename}`
Delete an uploaded document.

#### `POST /api/documents/reindex`
Rebuild the vector index from all documents.

#### `GET /api/documents/stats`
Get vector database statistics.

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Development

### Prerequisites

- Python 3.10+
- pip or poetry
- Docker (optional)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies with dev extras
pip install -r requirements.txt
pip install pytest pytest-asyncio pytest-cov

# Run in development mode
uvicorn src.main:app --reload
```

### Code Style

The project follows PEP 8 and uses type hints throughout. Key conventions:

- **Modules**: Lowercase with underscores (`vector_store.py`)
- **Classes**: PascalCase (`VectorStore`)
- **Functions**: Lowercase with underscores (`embed_text`)
- **Constants**: Uppercase (`SYSTEM_PROMPT`)

---

## Testing

### Run All Tests

```bash
# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_rag.py -v

# Run specific test
pytest tests/test_api.py::TestAPI::test_resolve_ticket_success -v
```

### Test Coverage

The test suite covers:

- ✅ Embedding service (text embedding, similarity)
- ✅ Vector store (CRUD, similarity search)
- ✅ RAG pipeline (context retrieval, response generation)
- ✅ API endpoints (validation, error handling)
- ✅ MCP prompt templates (structure, content)

---

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   └── schemas.py       # Pydantic models (request/response)
│   ├── services/
│   │   ├── embedding.py     # Sentence Transformers embedding
│   │   ├── vector_store.py  # FAISS vector operations
│   │   ├── llm.py           # LLM client (OpenAI/Ollama)
│   │   └── rag.py           # RAG pipeline orchestrator
│   ├── data/
│   │   └── knowledge_base.py # Synthetic documentation
│   └── prompts/
│       └── mcp_prompt.py    # MCP-compliant prompt templates
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_embedding.py
│   ├── test_vector_store.py
│   ├── test_rag.py
│   ├── test_api.py
│   └── test_prompts.py
├── Dockerfile               # Multi-stage Docker build
├── docker-compose.yml       # Production compose
├── docker-compose.dev.yml   # Development compose
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
└── README.md               # This file
```

---

## Design Decisions

### 1. RAG Architecture & Vector Database

**Choice**: FAISS as the Vector Database + Sentence Transformers for Embeddings

**Architecture Flow**:
```
Source Documents (knowledge_base.py)
         │
         ▼
    Embedding Service (Sentence Transformers)
         │
         ▼
    FAISS Vector Database ◄─── This IS the vector DB
         │
         ▼
    Similarity Search Results
```

**Why FAISS is a Vector Database**:
- FAISS (Facebook AI Similarity Search) stores document embeddings as vectors
- Performs efficient similarity search (cosine similarity via Inner Product)
- Supports persistence (save/load index to disk)
- Used by companies like Facebook, Spotify, and Airbnb at massive scale

**Knowledge Base vs Vector Store**:
- `knowledge_base.py` = **Source data** (like documents in a CMS)
- `vector_store.py` = **FAISS vector database** that indexes and searches embeddings

**Alternatives Considered**: Qdrant, Weaviate, Chroma
**Why FAISS**: No external dependencies, battle-tested, excellent for this scale, easy deployment

### 2. LLM Integration

**Choice**: Dual support (OpenAI + Ollama)

**Rationale**:
- **OpenAI**: Production-ready, best quality, easy to use.
- **Ollama**: Local inference for privacy/cost-sensitive scenarios.
- **Abstraction Layer**: `LLMService` class abstracts provider details.

### 3. MCP (Model Context Protocol) Prompt Design

**What MCP Means Here**:

In this project, MCP refers to a **structured prompt engineering pattern**, NOT Anthropic's tool integration protocol. The interview defines MCP as:
> "Prompt should have clearly defined role, context, task, and output schema"

**MCP Prompt Structure**:
```
┌─────────────────────────────────────────────────────┐
│ SYSTEM MESSAGE (ROLE)                               │
│ - AI identity: "Expert support assistant"           │
│ - Expertise areas                                   │
│ - Response guidelines                               │
└─────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────┐
│ USER MESSAGE                                        │
│ ├── CONTEXT: Retrieved docs from FAISS             │
│ ├── TASK: Customer ticket + instructions           │
│ └── OUTPUT SCHEMA: JSON format specification       │
└─────────────────────────────────────────────────────┘
```

**Benefits of MCP Pattern**:
- **Consistency**: Same structure for every request
- **Grounding**: LLM uses retrieved context, not hallucinations
- **Parseable Output**: Strict JSON schema for downstream processing
- **Maintainability**: Clear separation of concerns

### 4. Knowledge Base

**Choice**: Synthetic domain registrar documentation

**Rationale**:
- Realistic scenario for Tucows (domain registrar)
- Covers common support topics: suspension, billing, DNS, transfers
- Demonstrates understanding of the domain

### 5. Testing Strategy

**Choice**: Mocked LLM for unit tests

**Rationale**:
- Fast, deterministic tests
- No API costs during development
- Integration tests can use real LLM when needed

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FAISS](https://faiss.ai/) for vector search
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenAI](https://openai.com/) for LLM capabilities
