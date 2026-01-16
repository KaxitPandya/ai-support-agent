# 🧠 AI Support Agent - Knowledge Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-00a393.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready RAG (Retrieval-Augmented Generation) system that helps support teams resolve customer tickets efficiently using AI and relevant documentation.**

Built with FastAPI, OpenAI GPT-4o, FAISS vector database, and Streamlit - featuring advanced hybrid search, conversation memory, and Model Context Protocol (MCP) structured prompting.

---

## 🎯 What This Does

Transform customer support tickets into accurate, policy-compliant responses using AI:

**Input:**
```json
{
  "ticket_text": "My domain was suspended and I didn't get any notice. How can I reactivate it?"
}
```

**Output (MCP-Compliant):**
```json
{
  "answer": "Your domain may have been suspended due to WHOIS verification failure or policy violation. To reactivate: 1) Log into your domain management portal, 2) Navigate to 'My Domains' and check suspension details, 3) Update your WHOIS information and verify your email. Reactivation typically takes 24-48 hours after verification.",
  "references": [
    "Policy: Domain Suspension Guidelines, Section 4.2 - Reactivation Process",
    "Policy: Domain Suspension Guidelines, Section 4.3 - Communication"
  ],
  "action_required": "customer_action_required"
}
```

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Hybrid Search** | Combines semantic (FAISS) + keyword (BM25) search with cross-encoder reranking |
| 🧠 **Conversation Memory** | Short-term + long-term memory for consistent, context-aware responses |
| 📚 **Dynamic Knowledge Base** | Upload documents via UI or API without code changes |
| 🎨 **Beautiful Web UI** | Professional Streamlit interface with analytics and debugging tools |
| 🤖 **MCP-Compliant** | Structured prompt engineering with role, context, task, and output schema |
| 📊 **RAG Inspector** | Debug and visualize the retrieval pipeline in real-time |
| 🧪 **Semantic Chunking** | Topic-aware document splitting using embeddings (not character-based) |
| 🚀 **Production Ready** | 114+ unit tests, Docker support, comprehensive error handling |
| ⚡ **FastAPI Backend** | Async API with automatic OpenAPI docs at `/docs` |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Customer Support Ticket                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                              │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐            │
│  │  Query     │→ │   Hybrid    │→ │   Context    │            │
│  │ Embedding  │  │   Search    │  │  Augmented   │            │
│  │            │  │ (Semantic+  │  │    Prompt    │            │
│  │            │  │   BM25)     │  │    (MCP)     │            │
│  └────────────┘  └─────────────┘  └──────────────┘            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   OpenAI GPT-4o / GPT-4o-mini                   │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│        Structured JSON Response (Answer + References)           │
└─────────────────────────────────────────────────────────────────┘
```

**Core Technologies:**
- **LLM:** OpenAI GPT-4o / GPT-4o-mini
- **Vector DB:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
- **API:** FastAPI (async Python web framework)
- **UI:** Streamlit (interactive data apps)
- **Search:** Hybrid (semantic + BM25 keyword + cross-encoder reranking)

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/KaxitPandya/ai-support-agent.git
cd ai-support-agent

# 2. Create .env file and add your OpenAI API key
cp env.example .env
# Edit .env and set: OPENAI_API_KEY=sk-your-key-here

# 3. Start with Docker Compose
docker-compose up --build

# 4. Access the application
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Option 2: Local Python Environment

```bash
# 1. Clone the repository
git clone https://github.com/KaxitPandya/ai-support-agent.git
cd ai-support-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp env.example .env
# Edit .env and add your OpenAI API key

# 5. Run Streamlit UI (easiest way to start)
streamlit run streamlit_app.py
# Opens at http://localhost:8501

# OR run FastAPI backend
uvicorn src.main:app --reload --port 8000
# API at http://localhost:8000
```

---

## 🌐 Deployment Options

### Deploy to Streamlit Cloud (Free, 1-Click)

1. **Fork this repository** to your GitHub account

2. **Go to [share.streamlit.io](https://share.streamlit.io)** and click "New app"

3. **Select your repository:**
   - Repository: `YOUR-USERNAME/ai-support-agent`
   - Branch: `main`
   - Main file: `streamlit_app.py`

4. **Add secrets** in Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   OPENAI_API_KEY = "sk-your-actual-key-here"
   OPENAI_MODEL = "gpt-4o-mini"
   OPENAI_TEMPERATURE = "0.3"
   OPENAI_MAX_TOKENS = "1024"
   TOP_K_RESULTS = "5"
   SIMILARITY_THRESHOLD = "0.3"
   ```

5. **Click Deploy** - Your app will be live at `https://your-app.streamlit.app` 🎉

### Deploy with Docker

See the [Docker Deployment](#-docker-deployment) section below.

---

## 📖 API Reference

### Resolve a Support Ticket

**Endpoint:** `POST /resolve-ticket`

**Request:**
```bash
curl -X POST http://localhost:8000/resolve-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_text": "My domain was suspended. How can I reactivate it?"
  }'
```

**Response:**
```json
{
  "answer": "Your domain may have been suspended due to...",
  "references": ["Policy: Domain Suspension Guidelines, Section 4.2"],
  "action_required": "customer_action_required"
}
```

### Upload a Document

**Endpoint:** `POST /api/documents/upload`

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@policy.md" \
  -F "category=Domain Policies" \
  -F "index_immediately=true"
```

### Additional Endpoints

- **Health Check:** `GET /health`
- **List Uploaded Files:** `GET /api/documents/files`
- **Delete File:** `DELETE /api/documents/files/{filename}`
- **Reindex All:** `POST /api/documents/reindex`
- **Get Stats:** `GET /api/documents/stats`

**Interactive Documentation:** http://localhost:8000/docs

---

## 🎨 Web UI Features

The Streamlit interface provides:

### 1. 🎫 Ticket Resolution
- Resolve customer tickets with AI-powered responses
- Quick examples for common scenarios
- Real-time RAG pipeline visualization
- View retrieved documents and similarity scores

### 2. 📚 Knowledge Base Management
- Upload new documents (.txt, .md)
- Browse indexed documents by category
- Delete and reindex documents
- Track upload history

### 3. 🔬 RAG Inspector
- Test the retrieval pipeline with custom queries
- View MCP prompt structure
- Debug similarity scores and document ranking
- Understand how the AI generates responses

### 4. 📊 Analytics Dashboard
- Total documents indexed
- Tickets resolved count
- System configuration overview
- Performance metrics

### 5. ⚙️ Settings
- Adjust RAG parameters (top-k, threshold)
- Configure LLM settings (model, temperature, max tokens)
- Reset pipeline and clear session

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t ai-support-agent .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e OPENAI_API_KEY=sk-your-key-here \
  -e OPENAI_MODEL=gpt-4o-mini \
  -v $(pwd)/data:/app/data \
  --name support-agent \
  ai-support-agent

# Check logs
docker logs -f support-agent

# Stop the container
docker stop support-agent
```

### Docker Compose (Multi-Service)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild after changes
docker-compose up --build
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run all 114 tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_rag.py -v

# Run with verbose output
pytest -vv
```

### Test Coverage

- ✅ **RAG Pipeline:** Context retrieval, response generation, error handling
- ✅ **Vector Store:** FAISS operations, similarity search, persistence
- ✅ **Embeddings:** Text embedding, similarity calculation
- ✅ **Hybrid Search:** Semantic + keyword search, reranking
- ✅ **Memory System:** Short-term buffer, long-term storage
- ✅ **API Endpoints:** Request validation, error responses
- ✅ **MCP Prompts:** Prompt structure, context injection

---

## 📁 Project Structure

```
ai-support-agent/
├── src/
│   ├── api/
│   │   └── upload.py              # Document upload endpoints
│   ├── data/
│   │   └── knowledge_base.py      # Sample support docs
│   ├── models/
│   │   └── schemas.py             # Pydantic models
│   ├── prompts/
│   │   └── mcp_prompt.py          # MCP prompt templates
│   ├── services/
│   │   ├── rag.py                 # RAG pipeline orchestrator
│   │   ├── vector_store.py        # FAISS vector database
│   │   ├── embedding.py           # Sentence Transformers
│   │   ├── llm.py                 # OpenAI integration
│   │   ├── hybrid_search.py       # Hybrid search engine
│   │   ├── memory.py              # Conversation memory
│   │   ├── semantic_chunker.py    # Topic-aware chunking
│   │   └── document_processor.py  # Document processing
│   ├── config.py                  # Configuration management
│   └── main.py                    # FastAPI application
├── tests/                         # 114+ unit tests
├── streamlit_app.py               # Streamlit web UI
├── Dockerfile                     # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── requirements.txt               # Python dependencies
├── env.example                    # Environment template
└── README.md                      # This file
```

---

## ⚙️ Configuration

All settings are managed via environment variables. See [env.example](env.example) for all options.

### Required Settings

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | `sk-...` |

### Optional Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_MODEL` | OpenAI model name | `gpt-4o-mini` |
| `OPENAI_TEMPERATURE` | Response creativity (0-1) | `0.3` |
| `OPENAI_MAX_TOKENS` | Max response length | `1024` |
| `TOP_K_RESULTS` | Documents to retrieve | `5` |
| `SIMILARITY_THRESHOLD` | Min similarity score | `0.3` |
| `EMBEDDING_MODEL` | Sentence Transformer model | `all-MiniLM-L6-v2` |

---

## 🧩 How It Works

### 1. Document Indexing (One-Time Setup)
```
Documents → Chunking → Embedding → FAISS Vector Database
```
- Documents are split into semantic chunks (topic-aware)
- Each chunk is embedded using Sentence Transformers
- Embeddings stored in FAISS for fast similarity search

### 2. Ticket Resolution (Per Query)
```
Ticket → Embed → Search (Hybrid) → Rerank → Build Prompt (MCP) → LLM → Response
```
- Customer ticket is embedded
- Hybrid search retrieves relevant docs (semantic + keyword)
- Cross-encoder reranks results
- MCP prompt built with context
- OpenAI generates structured response

### 3. Model Context Protocol (MCP)

MCP is a structured prompt engineering pattern with four sections:

```
┌─────────────────────────────────────────────┐
│ ROLE: Expert support assistant identity     │
├─────────────────────────────────────────────┤
│ CONTEXT: Retrieved documents from RAG       │
├─────────────────────────────────────────────┤
│ TASK: Customer ticket + instructions        │
├─────────────────────────────────────────────┤
│ OUTPUT: JSON schema specification           │
└─────────────────────────────────────────────┘
```

This ensures:
- **Consistency:** Same structure every time
- **Grounding:** Responses based on actual documentation
- **Parseable:** Structured JSON for downstream processing

---

## 🎓 Design Decisions

### Why FAISS?
- **Fast:** Optimized for billion-scale similarity search
- **Simple:** No external database server required
- **Battle-Tested:** Used by Facebook, Spotify, Airbnb
- **Persistent:** Can save/load index to disk

### Why Hybrid Search?
- **Semantic Search:** Finds conceptually similar content
- **Keyword Search (BM25):** Finds exact term matches
- **Cross-Encoder Reranking:** Improves final ranking
- **Result:** Better retrieval accuracy than either alone

### Why Semantic Chunking?
- **Topic-Aware:** Splits at semantic boundaries, not arbitrary character limits
- **Context Preservation:** Keeps related information together
- **Better Retrieval:** More meaningful chunks = better search results

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [OpenAI](https://openai.com/) for GPT models
- [FAISS](https://faiss.ai/) by Facebook AI for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the UI framework

---

## 📧 Contact

**Kaxit Pandya** - [GitHub](https://github.com/KaxitPandya)

**Project Link:** [https://github.com/KaxitPandya/ai-support-agent](https://github.com/KaxitPandya/ai-support-agent)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Built with ❤️ using OpenAI, FAISS, FastAPI, and Streamlit

</div>
