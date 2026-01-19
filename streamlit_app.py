"""
Support Assistant

AI-powered support ticket resolution system for domain services.
Powered by RAG (Retrieval-Augmented Generation), MCP and LLM (OpenAI).

Deploy directly to Streamlit Cloud via GitHub.
"""

import streamlit as st
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import base64
import urllib.parse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Load .env file for local development
from dotenv import load_dotenv
load_dotenv()

# Load secrets from Streamlit Cloud if available
try:
    if hasattr(st, 'secrets') and len(st.secrets) > 0:
        for key in ['OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_TEMPERATURE',
                    'OPENAI_MAX_TOKENS', 'TOP_K_RESULTS', 'SIMILARITY_THRESHOLD']:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
except Exception:
    pass  # No secrets file, use environment variables

# Set page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Knowledge Assistant - RAG Support System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Brand Colors
BRAND_PRIMARY = "#2F59A3"
BRAND_PRIMARY_DARK = "#254A8D"
BRAND_PRIMARY_LIGHT = "#E9F1FF"
BRAND_ACCENT = "#1EC4FF"
BRAND_SUCCESS = "#28A745"
BRAND_WARNING = "#F5A623"
BRAND_ERROR = "#E53935"

# Custom CSS for professional theme
st.markdown("""
<style>
    /* Import professional fonts and Material Icons */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@400;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Fix all broken Material Symbols ligatures globally */
    .material-symbols-rounded,
    [class*="material-symbols"],
    span[style*="font-family: Material Symbols"] {
        font-family: 'Material Symbols Rounded', sans-serif !important;
        -webkit-font-feature-settings: 'liga' 1 !important;
        font-feature-settings: 'liga' 1 !important;
    }

    /* Hide sidebar collapse button text if icon fails */
    [data-testid="collapsedControl"] span,
    button[kind="headerNoPadding"] span {
        font-family: 'Material Symbols Rounded', sans-serif !important;
        font-size: 0 !important;
    }

    /* Replace broken collapse button with CSS icon */
    [data-testid="collapsedControl"]::after {
        content: '☰';
        font-size: 1.25rem;
        color: #2F59A3;
    }

    /* Root variables */
    :root {
        --brand-primary: #2F59A3;
        --brand-primary-dark: #254A8D;
        --brand-primary-light: #E9F1FF;
        --brand-accent: #1EC4FF;
        --text-primary: #1A1A2E;
        --text-secondary: #4A5568;
        --bg-white: #FFFFFF;
        --bg-light: #F8FAFC;
        --border-color: #E2E8F0;
    }

    /* Main app background */
    .stApp {
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
    }

    /* Professional Headers */
    h1 {
        font-family: 'Inter', sans-serif !important;
        color: #2F59A3 !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    h2, h3 {
        font-family: 'Inter', sans-serif !important;
        color: #1A1A2E !important;
        font-weight: 600 !important;
    }

    h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #4A5568 !important;
        font-weight: 500 !important;
    }

    /* Body text - More specific to avoid breaking icons */
    .stMarkdown p,
    .stMarkdown span,
    .stText,
    [data-testid="stCaptionContainer"],
    .stAlert p {
        font-family: 'Source Sans Pro', sans-serif !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2F59A3 0%, #254A8D 100%) !important;
        border-right: none;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Hide default radio buttons */
    [data-testid="stSidebar"] .stRadio {
        display: none;
    }

    /* Custom navigation buttons */
    .nav-button {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s ease;
        color: rgba(255, 255, 255, 0.85) !important;
        font-weight: 500;
        font-size: 0.95rem;
        border: none;
        width: 100%;
        text-align: left;
        background: transparent;
    }

    .nav-button:hover {
        background: rgba(255, 255, 255, 0.15);
        color: #FFFFFF !important;
    }

    .nav-button.active {
        background: rgba(255, 255, 255, 0.2);
        color: #FFFFFF !important;
        font-weight: 600;
        border-left: 3px solid #1EC4FF;
    }

    .nav-icon {
        font-size: 1.1rem;
        width: 24px;
        text-align: center;
    }

    /* Sidebar button styling */
    [data-testid="stSidebar"] .stButton button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: rgba(255, 255, 255, 0.9) !important;
        text-align: left !important;
        padding: 0.75rem 1rem !important;
        font-weight: 500 !important;
        justify-content: flex-start !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255, 255, 255, 0.18) !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
        color: #FFFFFF !important;
        transform: translateX(4px) !important;
    }

    [data-testid="stSidebar"] .stButton button:focus {
        box-shadow: none !important;
    }

    /* Action card styles */
    .action-card {
        border-radius: 12px;
        padding: 1.25rem;
        margin: 1rem 0;
        border-left: 5px solid;
    }

    .action-card-success {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        border-left-color: #28A745;
    }

    .action-card-warning {
        background: linear-gradient(135deg, rgba(245, 166, 35, 0.1) 0%, rgba(245, 166, 35, 0.05) 100%);
        border-left-color: #F5A623;
    }

    .action-card-error {
        background: linear-gradient(135deg, rgba(229, 57, 53, 0.1) 0%, rgba(229, 57, 53, 0.05) 100%);
        border-left-color: #E53935;
    }

    .action-card-info {
        background: linear-gradient(135deg, rgba(47, 89, 163, 0.1) 0%, rgba(47, 89, 163, 0.05) 100%);
        border-left-color: #2F59A3;
    }

    .action-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .action-icon {
        font-size: 1.5rem;
    }

    .action-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #1A1A2E;
    }

    .action-description {
        color: #4A5568;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }

    .action-steps {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
    }

    .action-steps-title {
        font-weight: 600;
        color: #2F59A3;
        margin-bottom: 0.5rem;
    }

    /* Stats cards */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .stat-card:hover {
        border-color: #2F59A3;
        box-shadow: 0 4px 12px rgba(47, 89, 163, 0.15);
        transform: translateY(-2px);
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2F59A3;
        font-family: 'Inter', sans-serif;
    }

    .stat-label {
        color: #4A5568;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Response card */
    .response-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .response-card:hover {
        border-color: #2F59A3;
        box-shadow: 0 4px 12px rgba(47, 89, 163, 0.1);
    }

    /* Reference tags */
    .reference-tag {
        display: inline-block;
        background: #E9F1FF;
        border: 1px solid #2F59A3;
        border-radius: 6px;
        padding: 0.25rem 0.75rem;
        margin: 0.25rem;
        font-size: 0.85rem;
        color: #2F59A3;
        font-weight: 500;
    }

    /* RAG step cards */
    .rag-step {
        background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .rag-step-header {
        font-weight: 600;
        color: #2F59A3;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }

    /* Context document card */
    .context-doc {
        background: #FFFFFF;
        border-left: 4px solid #2F59A3;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .context-score {
        display: inline-block;
        background: #28A745;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .context-score-medium {
        background: #F5A623;
    }

    .context-score-low {
        background: #E53935;
    }

    /* Text area */
    .stTextArea textarea {
        background: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #1A1A2E !important;
        font-family: 'Source Sans Pro', sans-serif !important;
        font-size: 1rem !important;
    }

    .stTextArea textarea:focus {
        border-color: #2F59A3 !important;
        box-shadow: 0 0 0 3px rgba(47, 89, 163, 0.15) !important;
    }

    .stTextArea textarea::placeholder {
        color: #A0AEC0 !important;
    }

    /* Primary Buttons */
    .stButton > button[kind="primary"] {
        background: #2F59A3 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(47, 89, 163, 0.2) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #254A8D !important;
        box-shadow: 0 4px 12px rgba(47, 89, 163, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Secondary Buttons */
    .stButton > button[kind="secondary"],
    .stButton > button:not([kind="primary"]) {
        background: #FFFFFF !important;
        color: #2F59A3 !important;
        border: 2px solid #2F59A3 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="secondary"]:hover,
    .stButton > button:not([kind="primary"]):hover {
        background: #E9F1FF !important;
        border-color: #254A8D !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 2px dashed #CBD5E0;
        border-radius: 12px;
        padding: 1.5rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #2F59A3;
        background: #F8FAFC;
    }

    /* Expander - Fixed arrow icon display */
    .streamlit-expanderHeader {
        background: #F8FAFC !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Fix broken Material Icons ligatures - hide the _arrow_right text */
    [data-testid="stExpanderToggleIcon"],
    .st-emotion-cache-p5msec {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        font-size: 0 !important;
        overflow: hidden !important;
    }

    /* Add custom CSS arrow indicator */
    [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }

    [data-testid="stExpander"] summary > div:first-child::before {
        content: '▸' !important;
        display: inline-block !important;
        font-size: 1rem !important;
        color: #2F59A3 !important;
        margin-right: 0.5rem !important;
        transition: transform 0.2s ease !important;
    }

    [data-testid="stExpander"] details[open] summary > div:first-child::before {
        content: '▾' !important;
    }

    /* Style the expander content */
    [data-testid="stExpander"] details {
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        padding: 0.75rem 1rem;
        background: #F8FAFC;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.2s ease;
    }

    [data-testid="stExpander"] summary:hover {
        background: #E9F1FF;
    }

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 1rem;
        background: #FFFFFF;
    }

    /* Alert styling */
    .stSuccess {
        background-color: rgba(40, 167, 69, 0.1) !important;
        border-left: 4px solid #28A745 !important;
        border-radius: 8px !important;
    }

    .stError {
        background-color: rgba(229, 57, 53, 0.1) !important;
        border-left: 4px solid #E53935 !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background-color: rgba(245, 166, 35, 0.1) !important;
        border-left: 4px solid #F5A623 !important;
        border-radius: 8px !important;
    }

    .stInfo {
        background-color: rgba(47, 89, 163, 0.1) !important;
        border-left: 4px solid #2F59A3 !important;
        border-radius: 8px !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Logo container */
    .logo-container {
        text-align: center;
        padding: 1.5rem 1rem;
        margin-bottom: 1rem;
    }

    .logo-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .logo-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: 0.02em;
    }

    .logo-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.25rem;
    }

    /* Professional divider */
    hr {
        border: none;
        border-top: 1px solid #E2E8F0;
        margin: 1.5rem 0;
    }

    /* MCP structure display */
    .mcp-section {
        background: linear-gradient(135deg, #E9F1FF 0%, #FFFFFF 100%);
        border: 1px solid #2F59A3;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .mcp-section-title {
        color: #2F59A3;
        font-weight: 700;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Pipeline visualization */
    .pipeline-arrow {
        text-align: center;
        color: #2F59A3;
        font-size: 1.5rem;
        margin: 0.5rem 0;
    }

    /* Metric display */
    .metric-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2F59A3;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #4A5568;
        text-transform: uppercase;
    }

    /* AI Response Card */
    .ai-response-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    .ai-response-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }

    .ai-response-icon {
        font-size: 1.5rem;
        background: linear-gradient(135deg, #2F59A3 0%, #1EC4FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .ai-response-title {
        font-weight: 600;
        font-size: 1rem;
        color: #2F59A3;
    }

    .ai-response-content {
        color: #1A1A2E;
        font-size: 1rem;
        line-height: 1.7;
    }

    /* References section */
    .references-section {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }

    .references-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #4A5568;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .reference-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0;
        font-size: 0.9rem;
        color: #2F59A3;
    }

    .reference-bullet {
        color: #1EC4FF;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'rag_initialized' not in st.session_state:
    st.session_state.rag_initialized = False
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'last_contexts' not in st.session_state:
    st.session_state.last_contexts = []
if 'show_rag_details' not in st.session_state:
    st.session_state.show_rag_details = True


def init_rag_pipeline():
    """Initialize the RAG pipeline."""
    if not st.session_state.rag_initialized:
        try:
            from src.services.rag import initialize_rag_pipeline, reset_rag_pipeline

            # Reset the pipeline singleton to force reinitialization
            # This ensures new uploaded documents are included
            reset_rag_pipeline()

            st.session_state.pipeline = initialize_rag_pipeline()
            st.session_state.rag_initialized = True
            return True
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {e}")
            with st.expander("Show error details"):
                import traceback
                st.code(traceback.format_exc())
            return False
    return True


def render_mermaid_image(mermaid_code):
    """Render Mermaid diagram as an image using mermaid.ink service."""
    # Encode the mermaid code to base64
    graphbytes = mermaid_code.encode("utf8")
    base64_bytes = base64.b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")

    # Create the mermaid.ink URL
    img_url = f"https://mermaid.ink/img/{base64_string}"

    # Display the image
    st.image(img_url, use_container_width=True)


def render_pipeline_diagram(diagram_type="complete"):
    """Render detailed Mermaid pipeline diagrams as images."""

    if diagram_type == "complete":
        st.markdown("### 🔄 Complete System Architecture")
        st.markdown("*End-to-end flow from document upload to ticket resolution*")

        mermaid_code = """
graph TB
    START[User Interaction]

    subgraph "Document Ingestion Pipeline"
        direction TB
        UPLOAD[Document Upload<br/>.txt .md files]
        DOCPROC[Document Processor<br/>Extract metadata & category]
        SEMANTIC[Semantic Chunker<br/>Topic-based splitting]
        SIMPLE[Simple Chunker<br/>Fallback chunking]
        CHUNKS[Document Chunks<br/>with metadata]

        UPLOAD --> DOCPROC
        DOCPROC --> SEMANTIC
        SEMANTIC -->|Success| CHUNKS
        SEMANTIC -->|Error| SIMPLE
        SIMPLE --> CHUNKS
    end

    subgraph "Embedding & Storage"
        direction TB
        EMBED[Embedding Service<br/>all-MiniLM-L6-v2<br/>384 dimensions]
        VECTORS[Vector Embeddings<br/>L2 Normalized]
        FAISS[FAISS Index<br/>IndexFlatIP<br/>Cosine Similarity]
        PERSIST[Disk Storage<br/>./data/vector_store/]

        CHUNKS --> EMBED
        EMBED --> VECTORS
        VECTORS --> FAISS
        FAISS <--> PERSIST
    end

    subgraph "Ticket Resolution Pipeline"
        direction TB
        QUERY[Customer Query]
        MEMCHECK{Check Memory?}
        MEMORY[Session Memory<br/>Last 3 turns]
        MEMCTX[Memory Context]
        QEMBED[Query Embedding<br/>384-dim vector]

        QUERY --> MEMCHECK
        MEMCHECK -->|Has History| MEMORY
        MEMORY --> MEMCTX
        MEMCHECK --> QEMBED
    end

    subgraph "Hybrid Search System"
        direction TB
        HYBRIDSVC[Hybrid Search Service]
        SEM[Semantic Search<br/>FAISS Vector<br/>Weight: 70%]
        BM25[BM25 Keyword<br/>TF-IDF Score<br/>Weight: 30%]
        MERGE[Score Fusion<br/>Normalize & Combine]
        RERANK[Cross-Encoder<br/>Reranking]
        TOPK[Top-K Results<br/>Default: 5 docs]

        QEMBED --> HYBRIDSVC
        HYBRIDSVC --> SEM
        HYBRIDSVC --> BM25
        FAISS -.-> SEM
        FAISS -.-> BM25
        SEM --> MERGE
        BM25 --> MERGE
        MERGE --> RERANK
        RERANK --> TOPK
    end

    subgraph "MCP Prompt Building"
        direction TB
        MCPBUILD[MCP Prompt Builder]
        ROLE[ROLE Section<br/>Expert Identity]
        MEMSEC[MEMORY Section<br/>Past 3 turns]
        CTXSEC[CONTEXT Section<br/>Retrieved docs]
        TASKSEC[TASK Section<br/>Current query]
        SCHEMA[OUTPUT SCHEMA<br/>JSON format]
        MESSAGES[Structured Messages]

        TOPK --> MCPBUILD
        MEMCTX -.-> MCPBUILD
        MCPBUILD --> ROLE
        ROLE --> MEMSEC
        MEMSEC --> CTXSEC
        CTXSEC --> TASKSEC
        TASKSEC --> SCHEMA
        SCHEMA --> MESSAGES
    end

    subgraph "LLM Generation"
        direction TB
        LLMSVC[LLM Service<br/>OpenAI API]
        GPT[GPT-4o-mini<br/>JSON mode<br/>temp=0.3]
        PARSE[JSON Parser<br/>Validation]
        RESPONSE[Ticket Response<br/>answer, refs, action]

        MESSAGES --> LLMSVC
        LLMSVC --> GPT
        GPT --> PARSE
        PARSE --> RESPONSE
    end

    RESPONSE --> MEMORY
    RESPONSE --> START
    START --> UPLOAD
    START --> QUERY

    classDef uploadClass fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    classDef embedClass fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef searchClass fill:#E53935,stroke:#C62828,stroke-width:2px,color:#fff
    classDef memoryClass fill:#00ACC1,stroke:#00838F,stroke-width:2px,color:#fff
    classDef mcpClass fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef llmClass fill:#3F51B5,stroke:#303F9F,stroke-width:2px,color:#fff

    class UPLOAD,DOCPROC,SEMANTIC,SIMPLE,CHUNKS uploadClass
    class EMBED,VECTORS,FAISS,PERSIST embedClass
    class HYBRIDSVC,SEM,BM25,MERGE,RERANK,TOPK,QEMBED searchClass
    class MEMCHECK,MEMORY,MEMCTX memoryClass
    class MCPBUILD,ROLE,MEMSEC,CTXSEC,TASKSEC,SCHEMA,MESSAGES mcpClass
    class LLMSVC,GPT,PARSE,RESPONSE llmClass
"""
        render_mermaid_image(mermaid_code)

    elif diagram_type == "indexing":
        st.markdown("### 📝 Document Indexing Pipeline")
        st.markdown("*From raw documents to searchable vector embeddings*")

        mermaid_code = """
graph TD
    START[Document Upload<br/>txt, md files]

    subgraph "Document Processing"
        direction TB
        EXTRACT[Extract Metadata<br/>Title, Category, Section]
        DETECT[Category Detection<br/>Policy, FAQ, Guide, etc.]
        RAW[Raw Document Text]

        START --> EXTRACT
        EXTRACT --> DETECT
        DETECT --> RAW
    end

    subgraph "Semantic Chunking Primary"
        direction TB
        TOKENIZE[1 Tokenize Sentences<br/>NLTK sentence splitter<br/>regex pattern matching]
        EMBED_SENT[2 Embed Each Sentence<br/>SentenceTransformer<br/>all-MiniLM-L6-v2]
        SIMILARITY[3 Compute Similarities<br/>Cosine between adjacent<br/>sentences buffer_size=1]
        BREAKPOINTS[4 Find Topic Breaks<br/>threshold < 0.5<br/>local minima < 0.6]
        CREATE[5 Create Chunks<br/>Split at breakpoints<br/>500-1000 chars]

        RAW --> TOKENIZE
        TOKENIZE --> EMBED_SENT
        EMBED_SENT --> SIMILARITY
        SIMILARITY --> BREAKPOINTS
        BREAKPOINTS --> CREATE
    end

    FALLBACK[Simple Chunker<br/>Character-based<br/>Sentence-aware]
    CREATE -->|Success| CHUNKS
    CREATE -->|Error| FALLBACK
    FALLBACK --> CHUNKS

    CHUNKS[Document Chunks<br/>with metadata<br/>coherent topics]

    subgraph "Embedding Generation"
        direction TB
        EMBED_SERVICE[Embedding Service<br/>all-MiniLM-L6-v2<br/>Batch processing]
        NORMALIZE[L2 Normalization<br/>For cosine similarity]
        VECTORS[384-dim Vectors<br/>Normalized embeddings]

        CHUNKS --> EMBED_SERVICE
        EMBED_SERVICE --> NORMALIZE
        NORMALIZE --> VECTORS
    end

    subgraph "Index Creation"
        direction TB
        FAISS_BUILD[FAISS IndexFlatIP<br/>Inner product index<br/>Cosine similarity]
        BM25_BUILD[BM25 Index<br/>TF-IDF scoring<br/>Keyword matching]
        DUAL[Dual Indexes<br/>Vector + Keyword]

        VECTORS --> FAISS_BUILD
        CHUNKS --> BM25_BUILD
        FAISS_BUILD --> DUAL
        BM25_BUILD --> DUAL
    end

    subgraph "Persistence Layer"
        direction TB
        SERIALIZE[Serialize Data<br/>Pickle format]
        SAVE_FAISS[Save faiss.index<br/>Binary format]
        SAVE_DOCS[Save documents.pkl<br/>Python objects]
        DISK[Disk Storage<br/>./data/vector_store/]

        DUAL --> SERIALIZE
        SERIALIZE --> SAVE_FAISS
        SERIALIZE --> SAVE_DOCS
        SAVE_FAISS --> DISK
        SAVE_DOCS --> DISK
    end

    DISK --> READY[Ready for Search<br/>Hybrid retrieval enabled]

    classDef processClass fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    classDef chunkClass fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef embedClass fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef indexClass fill:#E91E63,stroke:#C2185B,stroke-width:2px,color:#fff
    classDef storageClass fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff

    class START,EXTRACT,DETECT,RAW processClass
    class TOKENIZE,EMBED_SENT,SIMILARITY,BREAKPOINTS,CREATE,FALLBACK,CHUNKS chunkClass
    class EMBED_SERVICE,NORMALIZE,VECTORS embedClass
    class FAISS_BUILD,BM25_BUILD,DUAL indexClass
    class SERIALIZE,SAVE_FAISS,SAVE_DOCS,DISK,READY storageClass
"""
        render_mermaid_image(mermaid_code)

    elif diagram_type == "hybrid":
        st.markdown("### 🔍 Hybrid Search Pipeline")
        st.markdown("*Combining semantic understanding with keyword matching for superior retrieval*")

        mermaid_code = """
graph TD
    QUERY[User Query<br/>How to renew domain?]

    EMBED[Query Embedding<br/>SentenceTransformer<br/>384-dim vector]

    QUERY --> EMBED

    subgraph "Parallel Search Stage"
        direction LR

        subgraph "Semantic Search Path"
            direction TB
            SEM_START[Semantic Search]
            SEM_VEC[Vector Similarity<br/>FAISS IndexFlatIP]
            SEM_COS[Cosine Distance<br/>Inner product]
            SEM_TOP[Get top_k × 3<br/>candidates 15 docs]
            SEM_WEIGHT[Apply Weight<br/>70 semantic]

            SEM_START --> SEM_VEC
            SEM_VEC --> SEM_COS
            SEM_COS --> SEM_TOP
            SEM_TOP --> SEM_WEIGHT
        end

        subgraph "Keyword Search Path"
            direction TB
            KEY_START[BM25 Keyword Search]
            KEY_TOK[Tokenize Query<br/>Split into terms]
            KEY_TFIDF[Calculate TF-IDF<br/>Term frequency × IDF]
            KEY_BM25[BM25 Scoring<br/>Okapi BM25 formula]
            KEY_TOP[Get top_k × 3<br/>candidates 15 docs]
            KEY_WEIGHT[Apply Weight<br/>30 keyword]

            KEY_START --> KEY_TOK
            KEY_TOK --> KEY_TFIDF
            KEY_TFIDF --> KEY_BM25
            KEY_BM25 --> KEY_TOP
            KEY_TOP --> KEY_WEIGHT
        end
    end

    EMBED --> SEM_START
    EMBED --> KEY_START

    subgraph "Score Fusion Stage"
        direction TB
        NORM[Normalize Scores<br/>Min-Max scaling<br/>to 0-1 range]
        COMBINE[Weighted Combination<br/>0.7 × semantic + 0.3 × keyword]
        MERGE[Merge Unique Docs<br/>Remove duplicates<br/>Keep best scores]
        SORT[Sort by Combined Score<br/>Descending order]

        SEM_WEIGHT --> NORM
        KEY_WEIGHT --> NORM
        NORM --> COMBINE
        COMBINE --> MERGE
        MERGE --> SORT
    end

    subgraph "Reranking Stage"
        direction TB
        SELECT[Select top_k × 2<br/>Top 10 candidates]
        CONCAT[Concatenate<br/>query + SEP + title + content]
        RE_EMBED[Re-embed Combined<br/>Cross-encoder approach]
        RE_SCORE[Re-score Similarity<br/>Query vs combined embedding]
        FINAL_SORT[Sort by Rerank Score]
        TOPK[Select Final top_k<br/>Top 5 results]

        SORT --> SELECT
        SELECT --> CONCAT
        CONCAT --> RE_EMBED
        RE_EMBED --> RE_SCORE
        RE_SCORE --> FINAL_SORT
        FINAL_SORT --> TOPK
    end

    TOPK --> RESULTS[Top-K Results<br/>5 most relevant docs<br/>Ready for LLM context]

    classDef queryClass fill:#2196F3,stroke:#1976D2,stroke-width:3px,color:#fff
    classDef semClass fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    classDef keyClass fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef fusionClass fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef rerankClass fill:#E91E63,stroke:#C2185B,stroke-width:2px,color:#fff
    classDef resultClass fill:#00BCD4,stroke:#0097A7,stroke-width:3px,color:#fff

    class QUERY,EMBED queryClass
    class SEM_START,SEM_VEC,SEM_COS,SEM_TOP,SEM_WEIGHT semClass
    class KEY_START,KEY_TOK,KEY_TFIDF,KEY_BM25,KEY_TOP,KEY_WEIGHT keyClass
    class NORM,COMBINE,MERGE,SORT fusionClass
    class SELECT,CONCAT,RE_EMBED,RE_SCORE,FINAL_SORT,TOPK rerankClass
    class RESULTS resultClass
"""
        render_mermaid_image(mermaid_code)

    elif diagram_type == "memory":
        st.markdown("### 🧠 Session Memory System")
        st.markdown("*Conversation tracking for contextual multi-turn interactions*")

        mermaid_code = """
graph TD
    START[Ticket Resolved<br/>Response generated]

    subgraph "Memory Storage"
        direction TB
        CREATE[Create Turn Object<br/>query, answer, refs, action, timestamp]
        APPEND[Append to Deque<br/>Python collections.deque]
        CHECK{At Capacity?<br/>10 turns max}
        REMOVE[Remove Oldest<br/>FIFO automatic]
        STORE[Stored in Memory<br/>In-memory only]

        START --> CREATE
        CREATE --> APPEND
        APPEND --> CHECK
        CHECK -->|Yes| REMOVE
        CHECK -->|No| STORE
        REMOVE --> STORE
    end

    STORE --> WAIT[Wait for Next Query]

    subgraph "Memory Retrieval Flow"
        direction TB
        NEW_QUERY[New Customer Query<br/>Follow-up question]
        GET_LAST[Get Last N Turns<br/>Default: 3 turns]
        FORMAT[Format as Markdown<br/>Turn structure with Q&A]
        MEMORY_CTX[Memory Context String<br/>Ready for injection]

        NEW_QUERY --> GET_LAST
        GET_LAST --> FORMAT
        FORMAT --> MEMORY_CTX
    end

    WAIT --> NEW_QUERY

    subgraph "MCP Prompt Injection"
        direction TB
        BUILD[Build MCP Prompt]
        INJECT_MEM[Inject MEMORY Section<br/>First in user message]
        INJECT_CTX[Add CONTEXT Section<br/>RAG retrieved docs]
        INJECT_TASK[Add TASK Section<br/>Current query]
        FINAL_PROMPT[Complete Prompt<br/>with conversation history]

        MEMORY_CTX --> BUILD
        BUILD --> INJECT_MEM
        INJECT_MEM --> INJECT_CTX
        INJECT_CTX --> INJECT_TASK
        INJECT_TASK --> FINAL_PROMPT
    end

    subgraph "LLM Understanding"
        direction TB
        SEND[Send to LLM<br/>GPT-4o-mini]
        UNDERSTAND[LLM Understands Context<br/>Pronoun resolution<br/>Topic continuity]
        GENERATE[Generate Contextual Response<br/>References previous turns]
        STORE_NEW[Store New Turn<br/>Back to memory]

        FINAL_PROMPT --> SEND
        SEND --> UNDERSTAND
        UNDERSTAND --> GENERATE
        GENERATE --> STORE_NEW
    end

    STORE_NEW -.->|Loop| STORE

    BENEFITS[Benefits<br/>• Handles 'that', 'it' references<br/>• Maintains topic across turns<br/>• Prevents repetition<br/>• Natural conversation flow]

    GENERATE --> BENEFITS

    classDef storageClass fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    classDef retrievalClass fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef injectionClass fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef llmClass fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef benefitClass fill:#00BCD4,stroke:#0097A7,stroke-width:3px,color:#fff

    class START,CREATE,APPEND,CHECK,REMOVE,STORE,WAIT storageClass
    class NEW_QUERY,GET_LAST,FORMAT,MEMORY_CTX retrievalClass
    class BUILD,INJECT_MEM,INJECT_CTX,INJECT_TASK,FINAL_PROMPT injectionClass
    class SEND,UNDERSTAND,GENERATE,STORE_NEW llmClass
    class BENEFITS benefitClass
"""
        render_mermaid_image(mermaid_code)

    elif diagram_type == "mcp":
        st.markdown("### 📋 MCP (Model Context Protocol)")
        st.markdown("*Structured prompt engineering for reliable, consistent LLM outputs*")

        mermaid_code = """
graph TD
    START[MCP Prompt Builder]

    subgraph "System Message"
        direction TB
        ROLE_START[ROLE Section]
        ROLE_IDENTITY[Define Identity<br/>Expert support assistant<br/>Domain services specialist]
        ROLE_EXPERTISE[Areas of Expertise<br/>• DNS & domains<br/>• Billing & payments<br/>• Security policies<br/>• Technical support]
        ROLE_GUIDELINES[Response Guidelines<br/>• Professional tone<br/>• Policy-compliant<br/>• Clear & actionable]

        ROLE_START --> ROLE_IDENTITY
        ROLE_IDENTITY --> ROLE_EXPERTISE
        ROLE_EXPERTISE --> ROLE_GUIDELINES
    end

    subgraph "User Message Part 1"
        direction TB
        MEM_START[MEMORY Section<br/>Optional if history exists]
        MEM_CHECK{Has Previous<br/>Conversations?}
        MEM_GET[Get Last 3 Turns<br/>from session memory]
        MEM_FORMAT[Format Each Turn<br/>Turn N:<br/>Q: query<br/>A: answer<br/>Action: action_required]
        MEM_INJECT[Inject Memory Context<br/>Recent conversation history]

        MEM_START --> MEM_CHECK
        MEM_CHECK -->|Yes| MEM_GET
        MEM_CHECK -->|No| MEM_SKIP[Skip Memory Section]
        MEM_GET --> MEM_FORMAT
        MEM_FORMAT --> MEM_INJECT
    end

    subgraph "User Message Part 2"
        direction TB
        CTX_START[CONTEXT Section]
        CTX_DOCS[Retrieved Documents<br/>From hybrid search<br/>Top-K results default: 5]
        CTX_FORMAT[Format Each Doc<br/>Document N:<br/>Title<br/>Category > Section<br/>Similarity: XX%<br/>Content snippet]
        CTX_INJECT[Inject RAG Context<br/>Grounded knowledge base]

        CTX_START --> CTX_DOCS
        CTX_DOCS --> CTX_FORMAT
        CTX_FORMAT --> CTX_INJECT
    end

    subgraph "User Message Part 3"
        direction TB
        TASK_START[TASK Section]
        TASK_QUERY[Customer Ticket<br/>Current query text]
        TASK_INST[Analysis Instructions<br/>1 Analyze query intent<br/>2 Use MEMORY for context<br/>3 Reference CONTEXT docs<br/>4 Cite specific policies<br/>5 Determine action]
        TASK_INJECT[Inject Task Definition<br/>What to do with inputs]

        TASK_START --> TASK_QUERY
        TASK_QUERY --> TASK_INST
        TASK_INST --> TASK_INJECT
    end

    subgraph "User Message Part 4"
        direction TB
        SCHEMA_START[OUTPUT SCHEMA Section]
        SCHEMA_FORMAT[JSON Format Spec<br/>answer: string<br/>references: string array<br/>action_required: enum]
        SCHEMA_ACTIONS[Valid Actions<br/>• none<br/>• escalate_to_abuse_team<br/>• escalate_to_billing<br/>• escalate_to_technical<br/>• customer_action_required<br/>• follow_up_required]
        SCHEMA_INJECT[Inject Schema Requirement<br/>Ensures structured output]

        SCHEMA_START --> SCHEMA_FORMAT
        SCHEMA_FORMAT --> SCHEMA_ACTIONS
        SCHEMA_ACTIONS --> SCHEMA_INJECT
    end

    START --> ROLE_START
    ROLE_GUIDELINES --> MEM_START
    MEM_INJECT --> CTX_START
    MEM_SKIP --> CTX_START
    CTX_INJECT --> TASK_START
    TASK_INJECT --> SCHEMA_START
    SCHEMA_INJECT --> MESSAGES

    MESSAGES[Complete Prompt<br/>System + User messages<br/>Structured & complete]

    subgraph "LLM Processing"
        direction TB
        SEND_LLM[Send to GPT-4o-mini<br/>JSON mode enabled<br/>Temperature: 0.3]
        PROCESS[LLM Processes<br/>• Understands role<br/>• Uses memory context<br/>• Grounds in RAG docs<br/>• Follows instructions<br/>• Outputs valid JSON]
        PARSE[Parse JSON Response<br/>Validate schema<br/>Extract fields]
        OUTPUT[Ticket Response<br/>answer, references, action]

        MESSAGES --> SEND_LLM
        SEND_LLM --> PROCESS
        PROCESS --> PARSE
        PARSE --> OUTPUT
    end

    OUTPUT --> RESULT[Structured Output<br/>• Consistent format<br/>• Policy-compliant<br/>• Contextually aware<br/>• Actionable]

    classDef roleClass fill:#9C27B0,stroke:#7B1FA2,stroke-width:2px,color:#fff
    classDef memClass fill:#00BCD4,stroke:#0097A7,stroke-width:2px,color:#fff
    classDef ctxClass fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#fff
    classDef taskClass fill:#FF9800,stroke:#F57C00,stroke-width:2px,color:#fff
    classDef schemaClass fill:#E91E63,stroke:#C2185B,stroke-width:2px,color:#fff
    classDef llmClass fill:#2196F3,stroke:#1976D2,stroke-width:2px,color:#fff
    classDef resultClass fill:#607D8B,stroke:#455A64,stroke-width:3px,color:#fff

    class ROLE_START,ROLE_IDENTITY,ROLE_EXPERTISE,ROLE_GUIDELINES roleClass
    class MEM_START,MEM_CHECK,MEM_GET,MEM_FORMAT,MEM_INJECT,MEM_SKIP memClass
    class CTX_START,CTX_DOCS,CTX_FORMAT,CTX_INJECT ctxClass
    class TASK_START,TASK_QUERY,TASK_INST,TASK_INJECT taskClass
    class SCHEMA_START,SCHEMA_FORMAT,SCHEMA_ACTIONS,SCHEMA_INJECT schemaClass
    class MESSAGES,SEND_LLM,PROCESS,PARSE,OUTPUT llmClass
    class START,RESULT resultClass
"""
        render_mermaid_image(mermaid_code)

    else:
        st.warning(f"Unknown diagram type: {diagram_type}")


def get_score_class(score):
    """Get CSS class based on similarity score."""
    if score >= 0.7:
        return "context-score"
    elif score >= 0.5:
        return "context-score context-score-medium"
    else:
        return "context-score context-score-low"


def render_retrieved_contexts(contexts):
    """Render the retrieved documents with similarity scores."""
    if not contexts:
        st.info("No relevant documents retrieved.")
        return

    st.markdown("##### Retrieved Documents")
    for i, ctx in enumerate(contexts, 1):
        doc = ctx.document
        score = ctx.similarity_score
        score_pct = f"{score:.1%}"

        # Determine score color
        if score >= 0.7:
            score_color = "#28A745"
        elif score >= 0.5:
            score_color = "#F5A623"
        else:
            score_color = "#E53935"

        with st.container():
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**{i}. {doc.title}**")
                st.caption(f"_{doc.category} > {doc.section}_")
            with col2:
                st.markdown(f"<span style='background:{score_color};color:white;padding:2px 8px;border-radius:4px;font-weight:600;font-size:0.8rem;'>{score_pct}</span>", unsafe_allow_html=True)

            with st.expander("Show content", expanded=False):
                st.markdown(doc.content[:500] + ("..." if len(doc.content) > 500 else ""))


def render_mcp_structure():
    """Render the MCP prompt structure explanation."""
    st.markdown("""
    <div class="mcp-section">
        <div class="mcp-section-title">Model Context Protocol (MCP)</div>
        <p style="font-size: 0.9rem; color: #4A5568; margin: 0;">
            Structured prompt engineering pattern with 4 sections:
        </p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    sections = [
        ("ROLE", "Expert support assistant identity", "🎭"),
        ("CONTEXT", "Retrieved documents from RAG", "📚"),
        ("TASK", "Customer ticket + instructions", "📋"),
        ("OUTPUT", "JSON schema specification", "📤")
    ]

    for col, (title, desc, icon) in zip(cols, sections):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div style="font-weight: 600; color: #2F59A3;">{title}</div>
                <div style="font-size: 0.75rem; color: #4A5568;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


def render_response(response, show_details=True):
    """Render a ticket response with optional details."""

    # Render AI Response in a clean card
    st.markdown(f'''
    <div class="ai-response-card">
        <div class="ai-response-header">
            <span class="ai-response-icon">🤖</span>
            <span class="ai-response-title">AI Generated Response</span>
        </div>
        <div class="ai-response-content">
            {response.answer.replace(chr(10), '<br>')}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Render references separately using Streamlit components
    st.markdown("**📚 Knowledge Base References**")
    if response.references:
        for i, ref in enumerate(response.references, 1):
            st.markdown(f"  {i}. {ref}")
    else:
        st.caption("_No specific policy references for this response._")

    # Action required - Enhanced with detailed guidance
    st.markdown("##### Workflow Action")

    action_details = {
        'none': {
            'icon': '✅',
            'title': 'Ticket Resolved',
            'description': 'The AI response fully addresses the customer inquiry. No further action required.',
            'type': 'success'
        },
        'escalate_to_abuse_team': {
            'icon': '🚨',
            'title': 'Escalate to Abuse Team',
            'description': 'This ticket involves policy violations, suspicious activity, or security concerns requiring abuse team review.',
            'type': 'error'
        },
        'escalate_to_billing': {
            'icon': '💳',
            'title': 'Escalate to Billing',
            'description': 'This involves payment disputes, refunds, or billing adjustments requiring billing team attention.',
            'type': 'error'
        },
        'escalate_to_technical': {
            'icon': '🔧',
            'title': 'Escalate to Technical',
            'description': 'Complex technical issue requiring engineering investigation and resolution.',
            'type': 'error'
        },
        'customer_action_required': {
            'icon': '👤',
            'title': 'Awaiting Customer',
            'description': 'The customer needs to complete specific actions mentioned in the response before we can proceed.',
            'type': 'warning'
        },
        'follow_up_required': {
            'icon': '📞',
            'title': 'Follow-up Required',
            'description': 'This ticket needs a follow-up after the customer completes the required actions or after a waiting period.',
            'type': 'info'
        }
    }

    action = action_details.get(response.action_required, {
        'icon': '❓',
        'title': 'Manual Review Needed',
        'description': 'Please review this ticket manually to determine the appropriate action.',
        'type': 'info'
    })

    # Render action card with HTML for better styling
    card_class = f"action-card action-card-{action['type']}"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="action-header">
            <span class="action-icon">{action['icon']}</span>
            <span class="action-title">{action['title']}</span>
        </div>
        <div class="action-description">{action['description']}</div>
    </div>
    """, unsafe_allow_html=True)


def render_output_json(response):
    """Render the MCP-compliant JSON output."""
    output = {
        "answer": response.answer,
        "references": response.references,
        "action_required": response.action_required
    }
    st.code(json.dumps(output, indent=2), language="json")


# Initialize page state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Ticket Resolution"

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">🧠</div>
        <div class="logo-text">Knowledge</div>
        <div class="logo-subtitle">Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Navigation with styled buttons
    nav_items = [
        ("Ticket Resolution", "🎫", "Resolve support tickets"),
        ("Memory Viewer", "🧠", "View conversation memory"),
        ("Knowledge Base", "📄", "Manage documents"),
        ("Analytics", "📊", "View system stats"),
        ("Pipeline Explorer", "🔍", "Explore RAG & Memory"),
        ("Settings", "⚙️", "Configure system"),
    ]

    # Create lookup dict for page icons
    page_icons = {name: icon for name, icon, _ in nav_items}

    for name, icon, tooltip in nav_items:
        if st.button(
            f"{icon}  {name}",
            key=f"nav_{name}",
            use_container_width=True,
            help=tooltip
        ):
            st.session_state.current_page = name
            st.rerun()

    # Set page variable for main content routing
    current_page = st.session_state.current_page
    page = f"{page_icons.get(current_page, '📄')} {current_page}"

    st.markdown("---")

    # Status indicator
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key and len(api_key) > 10:
        st.success("✓ System Online")
    else:
        st.warning("⚠ API Key Required")

    st.markdown("---")

    # Quick stats
    st.markdown("**Session Stats**")
    st.caption(f"Tickets Resolved: {len(st.session_state.conversation_history)}")
    st.caption(f"RAG Pipeline: {'Active' if st.session_state.rag_initialized else 'Not Initialized'}")



# Main content
if page == "🎫 Ticket Resolution":
    st.title("🎫 Support Ticket Resolution")
    st.markdown("*AI-powered Knowledge Assistant using RAG and MCP*")

    # Quick examples
    st.markdown("##### Quick Examples")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🔒 Domain Suspended", use_container_width=True):
            st.session_state.ticket_input = "My domain was suspended and I didn't get any notice. How can I reactivate it?"
            st.rerun()
    with col2:
        if st.button("💳 Refund Request", use_container_width=True):
            st.session_state.ticket_input = "I want a refund for my domain renewal. It was renewed automatically without my permission."
            st.rerun()
    with col3:
        if st.button("🌐 DNS Help", use_container_width=True):
            st.session_state.ticket_input = "How do I update my DNS records to point to a new web host?"
            st.rerun()
    with col4:
        if st.button("🔄 Transfer Domain", use_container_width=True):
            st.session_state.ticket_input = "I want to transfer my domain to another registrar. What do I need to do?"
            st.rerun()

    st.markdown("---")

    # Input area
    ticket_text = st.text_area(
        "Customer Support Ticket",
        height=120,
        placeholder="Enter the customer's support inquiry here...",
        key="ticket_input"
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit = st.button("🚀 Resolve Ticket", type="primary", use_container_width=True)
    with col2:
        show_details = st.checkbox("Show RAG Details", value=True)
    with col3:
        clear = st.button("🗑️ Clear", use_container_width=True)

    if clear:
        st.session_state.conversation_history = []
        st.session_state.current_response = None
        st.session_state.last_contexts = []
        st.rerun()

    if submit and ticket_text.strip():
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            st.error("❌ OPENAI_API_KEY not configured. Please set it in your .env file.")
            st.code("OPENAI_API_KEY=sk-your-key-here")
        else:
            # Show progress
            progress_placeholder = st.empty()

            with progress_placeholder.container():
                st.markdown("### Processing Pipeline")

                # Step 1: Initialize
                with st.spinner("Step 1/4: Initializing RAG Pipeline..."):
                    if not init_rag_pipeline():
                        st.stop()
                st.success("✓ RAG Pipeline Ready")

                # Step 2: Retrieve Context
                with st.spinner("Step 2/4: Retrieving relevant documents..."):
                    try:
                        pipeline = st.session_state.pipeline
                        if pipeline is None:
                            st.error("Pipeline not initialized")
                            st.stop()
                        contexts = pipeline.retrieve_context(ticket_text)
                        st.session_state.last_contexts = contexts
                    except Exception as e:
                        st.error(f"Error retrieving context: {e}")
                        st.stop()
                st.success(f"✓ Retrieved {len(contexts)} relevant documents")

                # Step 3: Generate Response
                with st.spinner("Step 3/4: Generating response with LLM..."):
                    try:
                        pipeline = st.session_state.pipeline
                        if pipeline is None:
                            st.error("Pipeline not initialized")
                            st.stop()
                        response = pipeline.resolve_ticket(ticket_text)
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                        st.stop()
                st.success("✓ Response generated")

                # Step 4: Store
                with st.spinner("Step 4/4: Storing in memory..."):
                    st.session_state.current_response = {
                        'query': ticket_text,
                        'response': response,
                        'contexts': contexts,
                        'timestamp': datetime.now().isoformat()
                    }
                    st.session_state.conversation_history.append(st.session_state.current_response)
                st.success("✓ Complete!")

            # Clear progress and show results
            progress_placeholder.empty()

    # Display current response
    if st.session_state.current_response:
        st.markdown("---")
        st.markdown("### Resolution Result")

        current = st.session_state.current_response

        # Show the query
        st.info(f"**Customer Query:** {current['query']}")

        # Memory Status Indicator
        if st.session_state.rag_initialized and st.session_state.pipeline:
            memory_stats = st.session_state.pipeline.get_memory_stats()
            if memory_stats.get("memory_enabled"):
                total_turns = memory_stats.get("total_turns", 0)
                if total_turns > 0:
                    st.success(f"🧠 **Memory Active:** {total_turns} conversation turn(s) in memory (last 3 used for context)")
                else:
                    st.info("🧠 **Memory Active:** First query - no previous context yet")

        # Main response and details in columns
        if show_details and 'contexts' in current:
            col1, col2 = st.columns([3, 2])

            with col1:
                st.markdown("#### AI Response")
                render_response(current['response'])

            with col2:
                st.markdown("#### RAG Context")
                render_retrieved_contexts(current.get('contexts', []))

                with st.expander("📤 MCP JSON Output", expanded=False):
                    render_output_json(current['response'])

                # Memory Debug Panel
                if st.session_state.pipeline:
                    with st.expander("🧠 Memory Debug Panel", expanded=False):
                        memory_stats = st.session_state.pipeline.get_memory_stats()
                        st.json(memory_stats)

                        # Show what was sent to LLM
                        if memory_stats.get("total_turns", 0) > 0:
                            st.markdown("**What LLM Sees (Memory Context):**")
                            memory_obj = st.session_state.pipeline._get_memory()
                            if memory_obj and not memory_obj.is_empty():
                                memory_preview = memory_obj.get_context_for_prompt()
                                st.text_area("Memory Section in Prompt", memory_preview, height=200)
        else:
            render_response(current['response'])

    # Previous conversations
    if len(st.session_state.conversation_history) > 1:
        st.markdown("---")
        st.markdown("### Previous Tickets")

        for i, item in enumerate(reversed(st.session_state.conversation_history[:-1])):
            with st.expander(f"🎫 {item['query'][:60]}...", expanded=False):
                st.markdown(f"**Query:** {item['query']}")
                render_response(item['response'], show_details=False)


elif page == "🧠 Memory Viewer":
    st.title("🧠 Session Memory Viewer")
    st.markdown("*Monitor conversation context and memory system*")

    # Check if pipeline and memory are available
    if st.session_state.pipeline:
        memory_stats = st.session_state.pipeline.get_memory_stats()

        if memory_stats.get("memory_enabled"):
            # Memory Statistics
            st.markdown("### 📊 Memory Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem;">💬</div>
                    <div class="stat-value">{memory_stats.get('total_turns', 0)}</div>
                    <div class="stat-label">Total Turns</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem;">📦</div>
                    <div class="stat-value">{memory_stats.get('max_capacity', 10)}</div>
                    <div class="stat-label">Max Capacity</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem;">🔍</div>
                    <div class="stat-value">{memory_stats.get('context_window', 3)}</div>
                    <div class="stat-label">Context Window</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                duration = memory_stats.get('session_duration_seconds', 0)
                duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
                st.markdown(f"""
                <div class="stat-card">
                    <div style="font-size: 2rem;">⏱️</div>
                    <div class="stat-value" style="font-size: 1.3rem;">{duration_str}</div>
                    <div class="stat-label">Session Duration</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Memory Content
            memory_obj = st.session_state.pipeline._get_memory()
            if memory_obj and not memory_obj.is_empty():
                st.markdown("### 💾 Stored Conversations")
                st.info(f"""
                **How Memory Works:**
                - Last **{memory_stats.get('context_window', 3)} turns** are included in LLM prompts
                - Maximum **{memory_stats.get('max_capacity', 10)} turns** stored in memory
                - Oldest conversations automatically removed when limit reached
                - Memory cleared on page refresh (session-based)
                """)

                # Show all turns
                turns_list = memory_obj.get_turns_list()
                for i, turn in enumerate(reversed(turns_list), 1):
                    with st.expander(f"**Turn {len(turns_list) - i + 1}:** {turn['query'][:80]}...", expanded=(i == 1)):
                        st.markdown(f"**🗓️ Timestamp:** {turn['timestamp']}")
                        st.markdown(f"**❓ Customer Query:**")
                        st.info(turn['query'])

                        st.markdown(f"**💡 AI Response:**")
                        st.success(turn['answer'][:500] + ("..." if len(turn['answer']) > 500 else ""))

                        if turn['references']:
                            st.markdown(f"**📚 References:**")
                            for ref in turn['references']:
                                st.caption(f"• {ref}")

                        st.markdown(f"**⚡ Action Required:** `{turn['action_required']}`")

                st.markdown("---")

                # What LLM Sees
                st.markdown("### 👁️ What the LLM Actually Sees")
                st.markdown("""
                This is the **MEMORY SECTION** that gets injected into the MCP prompt for the next query.
                The LLM uses this to maintain conversation continuity and answer follow-up questions.
                """)

                memory_context = memory_obj.get_context_for_prompt()
                st.code(memory_context, language="markdown")

                st.markdown("---")

                # Clear Memory Button
                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("🗑️ Clear Memory", type="secondary"):
                        memory_obj.clear()
                        st.session_state.conversation_history = []
                        st.session_state.current_response = None
                        st.success("Memory cleared!")
                        st.rerun()
                with col2:
                    st.caption("⚠️ This will clear all conversation history from the current session")

            else:
                st.info("🔍 **No conversations in memory yet.**\n\nStart by resolving a ticket in the **Ticket Resolution** page. Each resolved ticket will be stored in memory for context in follow-up queries.")

                # Example of how memory works
                st.markdown("---")
                st.markdown("### 📖 How Memory Enables Follow-Up Questions")

                st.markdown("""
                **Example Conversation:**

                1️⃣ **First Query:**
                   - User: "How do I reactivate my suspended domain?"
                   - AI: "Log into portal, update WHOIS, verify email..."
                   - ✅ Stored in memory

                2️⃣ **Follow-Up Query (uses memory):**
                   - User: "How long will **that** take?"
                   - Memory helps AI understand "that" = domain reactivation
                   - AI: "Domain reactivation typically takes 24-48 hours..."
                   - ✅ Also stored in memory

                3️⃣ **Another Follow-Up:**
                   - User: "What if I don't receive the verification email?"
                   - Memory provides full context from previous turns
                   - AI: Provides specific answer about verification emails...
                """)

        else:
            st.warning("Memory is not enabled in the RAG pipeline.")
    else:
        st.warning("⚠️ **Pipeline not initialized.** Please resolve a ticket first in the Ticket Resolution page to initialize the system.")

        st.markdown("---")
        st.markdown("### 🧠 About Session Memory")
        st.info("""
        **Session Memory** is a lightweight, in-memory conversation tracking system designed for Streamlit Cloud.

        **Key Features:**
        - 🚀 **Fast**: No file I/O overhead
        - 💡 **Smart**: Last 3 turns used for context
        - 🔄 **Auto-managed**: Oldest turns removed automatically
        - 🛡️ **Private**: Cleared on session end
        - ☁️ **Cloud-ready**: Works on Streamlit Cloud

        **Memory Flow:**
        ```
        Turn 1 → Store → Turn 2 (uses Turn 1 context) → Store → Turn 3 (uses Turn 1-2 context)
        ```

        **Technical Details:**
        - Storage: Python `deque` (max 10 turns)
        - Context Window: Last 3 turns
        - Injection Point: MCP Prompt MEMORY section
        - Lifecycle: Session-scoped (no persistence)
        """)


elif page == "📄 Knowledge Base":
    st.title("📄 Knowledge Base Management")
    st.markdown("*Manage support documentation, policies, and FAQs*")

    tab1, tab2 = st.tabs(["📤 Upload Documents", "📚 Browse Documents"])

    with tab1:
        st.markdown("### Upload New Documents")
        uploaded_files = st.file_uploader(
            "Drop files here or click to upload",
            type=['txt', 'md'],
            accept_multiple_files=True,
            help="Supports .txt and .md files"
        )

        if uploaded_files:
            if st.button("📤 Process & Index", type="primary"):
                with st.spinner("Processing documents..."):
                    try:
                        from src.services.document_processor import get_document_processor
                        from src.services.vector_store import get_vector_store

                        processor = get_document_processor()
                        vector_store = get_vector_store()

                        total_chunks = 0
                        for file in uploaded_files:
                            try:
                                content = file.read()
                                if not content:
                                    st.warning(f"⚠️ {file.name} is empty, skipping...")
                                    continue

                                docs, file_path = processor.process_uploaded_file(file.name, content)
                                vector_store.add_documents(docs)
                                total_chunks += len(docs)
                                st.success(f"✅ {file.name}: {len(docs)} chunks indexed")

                            except Exception as file_error:
                                st.error(f"❌ Failed to process {file.name}: {file_error}")

                        if total_chunks > 0:
                            st.session_state.rag_initialized = False
                            st.success(f"🎉 Total: {total_chunks} chunks added!")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with tab2:
        st.markdown("### Knowledge Base Documents")

        # Sub-tabs for different document sources
        source_tab1, source_tab2 = st.tabs(["📚 Base Knowledge", "📤 Uploaded Documents"])

        with source_tab1:
            st.markdown("**Static knowledge base documents (built-in)**")
            try:
                from src.data.knowledge_base import get_knowledge_base
                docs = get_knowledge_base()

                # Group by category
                categories = {}
                for doc in docs:
                    cat = doc.category
                    if cat not in categories:
                        categories[cat] = []
                    categories[cat].append(doc)

                for cat, cat_docs in categories.items():
                    with st.expander(f"📁 {cat} ({len(cat_docs)} documents)", expanded=False):
                        for doc in cat_docs:
                            st.markdown(f"**{doc.title}** - _{doc.section}_")
                            st.caption(doc.content[:200] + "...")
                            st.markdown("---")

            except Exception as e:
                st.error(f"Error loading knowledge base: {e}")

        with source_tab2:
            st.markdown("**Documents you've uploaded and indexed**")
            try:
                from src.services.vector_store import get_vector_store
                from src.services.document_processor import get_document_processor

                vector_store = get_vector_store()
                processor = get_document_processor()

                # Get documents from vector store
                indexed_docs = vector_store.documents

                # Filter to show only uploaded docs (not base knowledge)
                # Base knowledge has IDs like "policy-001", uploaded has "filename-chunk-0"
                uploaded_docs = [d for d in indexed_docs if "-chunk-" in d.id]

                if uploaded_docs:
                    st.success(f"✅ {len(uploaded_docs)} document chunks indexed in vector store")

                    # Group by title (original filename)
                    doc_groups = {}
                    for doc in uploaded_docs:
                        title = doc.title
                        if title not in doc_groups:
                            doc_groups[title] = []
                        doc_groups[title].append(doc)

                    for title, chunks in doc_groups.items():
                        with st.expander(f"📄 {title} ({len(chunks)} chunks)", expanded=False):
                            st.caption(f"Category: {chunks[0].category}")
                            for chunk in chunks:
                                st.markdown(f"**{chunk.section}**")
                                st.text(chunk.content[:300] + ("..." if len(chunk.content) > 300 else ""))
                                st.markdown("---")
                else:
                    st.info("No uploaded documents yet. Use the 'Upload Documents' tab to add documents.")

                # Show uploaded files list
                st.markdown("---")
                st.markdown("**Uploaded Files on Disk:**")
                uploaded_files_list = processor.list_uploaded_files()
                if uploaded_files_list:
                    for f in uploaded_files_list:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.text(f"📄 {f['filename']} ({f['size_bytes']} bytes)")
                        with col2:
                            if st.button("🗑️", key=f"del_{f['filename']}", help="Delete file"):
                                processor.delete_file(f['filename'])
                                st.rerun()
                else:
                    st.caption("No files uploaded to disk yet.")

            except Exception as e:
                st.error(f"Error loading uploaded documents: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Reindex button
    st.markdown("---")
    st.warning("**Note:** Reindexing will clear all uploaded documents and reset to base knowledge only.")
    if st.button("🔄 Reindex All Documents (Reset to Base)", type="secondary"):
        with st.spinner("Reindexing..."):
            try:
                from src.services.vector_store import initialize_vector_store
                from src.data.knowledge_base import get_knowledge_base

                docs = get_knowledge_base()
                # Force reinit clears uploaded docs and resets to base knowledge
                initialize_vector_store(docs, force_reinit=True)
                st.session_state.rag_initialized = False
                st.success(f"✅ Reindexed {len(docs)} base documents! Uploaded documents cleared.")
            except Exception as e:
                st.error(f"Error: {e}")


elif page == "📊 Analytics":
    st.title("📊 System Analytics")
    st.markdown("*Monitor system performance and usage*")

    # Stats cards
    col1, col2, col3, col4 = st.columns(4)

    try:
        from src.services.vector_store import get_vector_store
        from src.services.document_processor import get_document_processor

        vector_store = get_vector_store()
        processor = get_document_processor()

        doc_count = vector_store.get_document_count() if hasattr(vector_store, 'get_document_count') else 0
        uploaded_count = len(processor.list_uploaded_files())
        history_count = len(st.session_state.conversation_history)
    except Exception:
        doc_count = 0
        uploaded_count = 0
        history_count = 0

    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">📚</div>
            <div class="stat-value">{doc_count}</div>
            <div class="stat-label">Indexed Documents</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">📁</div>
            <div class="stat-value">{uploaded_count}</div>
            <div class="stat-label">Uploaded Files</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">🎫</div>
            <div class="stat-value">{history_count}</div>
            <div class="stat-label">Tickets Resolved</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div style="font-size: 2rem;">🤖</div>
            <div class="stat-value">GPT-4o</div>
            <div class="stat-label">LLM Model</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # System configuration
    st.markdown("### System Configuration")

    config_data = [
        ("Embedding Model", "all-MiniLM-L6-v2", "Sentence Transformers"),
        ("Embedding Dimension", "384", "Vector size"),
        ("Vector Store", "FAISS (IndexFlatIP)", "Cosine similarity"),
        ("LLM Provider", "OpenAI", "GPT-4o-mini"),
        ("Search Mode", "Hybrid", "Semantic + BM25"),
        ("Top-K Results", os.getenv("TOP_K_RESULTS", "5"), "Documents retrieved"),
        ("Similarity Threshold", os.getenv("SIMILARITY_THRESHOLD", "0.3"), "Minimum score"),
    ]

    for name, value, desc in config_data:
        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            st.markdown(f"**{name}**")
        with col2:
            st.code(value)
        with col3:
            st.caption(desc)


elif page == "🔍 Pipeline Explorer":
    st.title("🔍 AI Pipeline Explorer")
    st.markdown("*Explore how RAG retrieval and memory work together*")

    # Add tabs for different pipeline views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📝 Document Indexing",
        "🔍 Hybrid Search",
        "✂️ Semantic Chunking",
        "📋 MCP Prompts",
        "🧠 Session Memory"
    ])

    # ==================== TAB 1: DOCUMENT INDEXING ====================
    with tab1:
        st.markdown("### 📝 Document Processing & Indexing")
        st.markdown("*How documents are processed, chunked, and prepared for retrieval*")

        # Overview
        st.info("""
        **Document Processing Pipeline:**
        Raw documents → Metadata Extraction → Semantic Chunking → Embedding → Vector Storage
        """)

        st.markdown("---")

        # Supported Document Types
        st.markdown("### 📄 Supported Document Formats")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="metric-box">
                <div style="font-size: 2rem; margin-bottom: 12px;">📝</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Text Files (.txt)</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    • Plain text documents<br/>
                    • Policies and procedures<br/>
                    • FAQs and guides<br/>
                    • Best for: Structured content
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-box">
                <div style="font-size: 2rem; margin-bottom: 12px;">📋</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Markdown Files (.md)</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    • Formatted documentation<br/>
                    • Technical guides<br/>
                    • README files<br/>
                    • Best for: Rich formatting
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Metadata Extraction
        st.markdown("### 🏷️ Automatic Metadata Extraction")

        st.code("""
# Metadata extracted from each document:

1. Title Detection
   - First line of document
   - Or filename (if no clear title)
   - Example: "Domain Renewal Policy"

2. Category Detection (Rule-based)
   - Keywords in title/content:
     • "policy", "policies" → Policy
     • "faq", "question" → FAQ
     • "guide", "how to" → Guide
     • "billing", "payment" → Billing
     • "domain", "dns" → Domain Management
   - Default: "General"

3. Section Identification
   - Extracted from headers
   - Subsection hierarchy
   - Example: "Renewal > Auto-Renewal Settings"

4. File Metadata
   - Source filename
   - Upload timestamp
   - File size
   - Chunk count
        """, language="python")

        st.markdown("---")

        

    # ==================== TAB 2: HYBRID SEARCH ====================
    with tab2:
        st.markdown("### 🔍 Hybrid Search System")
        st.markdown("*Combining semantic understanding with keyword matching for superior retrieval*")

        # Why Hybrid Search?
        st.info("""
        **Why Hybrid Search?**
        - 🎯 **Semantic Search**: Finds conceptually similar content (understands meaning)
        - 🔤 **Keyword Search (BM25)**: Finds exact term matches (catches specific terminology)
        - 🏆 **Hybrid**: Best of both worlds
        """)

        st.markdown("---")

        # Hybrid Search Visualization
        st.markdown("### Three-Stage Hybrid Search Process")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-box" style="min-height: 280px;">
                <div style="font-size: 2rem; margin-bottom: 12px;">🎯</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Semantic Search</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    <strong>Process:</strong><br/>
                    1. Embed query → 384-dim vector<br/>
                    2. FAISS similarity search<br/>
                    3. Cosine similarity scoring<br/>
                    4. Get top_k × 3 candidates<br/><br/>
                    <strong>Weight:</strong> 70%<br/>
                    <strong>Good for:</strong> Conceptual matches
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-box" style="min-height: 280px;">
                <div style="font-size: 2rem; margin-bottom: 12px;">📊</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">BM25 Keyword</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    <strong>Process:</strong><br/>
                    1. Tokenize query<br/>
                    2. Calculate TF-IDF scores<br/>
                    3. BM25 ranking formula<br/>
                    4. Get top_k × 3 candidates<br/><br/>
                    <strong>Weight:</strong> 30%<br/>
                    <strong>Good for:</strong> Exact terms
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-box" style="min-height: 280px;">
                <div style="font-size: 2rem; margin-bottom: 12px;">🏆</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Reranking</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    <strong>Process:</strong><br/>
                    1. Merge both result sets<br/>
                    2. Combine weighted scores<br/>
                    3. Cross-encoder reranking<br/>
                    4. Return final top_k<br/><br/>
                    <strong>Benefit:</strong> Precision boost<br/>
                    <strong>Output:</strong> 5 best docs
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Score Combination Formula
        st.markdown("### 🧮 Score Combination Formula")

        st.code("""
# 1. Normalize scores to [0, 1]
semantic_score_norm = (score - min) / (max - min)
keyword_score_norm = (score - min) / (max - min)

# 2. Weighted combination
combined_score = (0.7 × semantic_score_norm) + (0.3 × keyword_score_norm)

# 3. Cross-encoder reranking
for doc in top_candidates:
    combined_text = query + "[SEP]" + doc.title + doc.content[:500]
    combined_emb = embed(combined_text)
    rerank_score = cosine_similarity(query_emb, combined_emb)

# 4. Sort by rerank_score and return top_k
        """, language="python")

        st.markdown("---")

        # Example Search Results
        with st.expander("📝 Example: Query Processing", expanded=False):
            st.markdown("""
            **Query:** "How do I renew my domain?"

            **Semantic Search Results (Top 5):**
            1. "Domain Renewal Procedure" → 0.92
            2. "Account Management Guide" → 0.88
            3. "Billing and Payment FAQ" → 0.75
            4. "Auto-Renewal Settings" → 0.71
            5. "Domain Registration Process" → 0.68

            **BM25 Keyword Results (Top 5):**
            1. "Domain Renewal Procedure" → 8.5 (raw) → 1.0 (normalized)
            2. "Renew Premium Domains" → 5.2 → 0.61
            3. "Domain Management" → 3.1 → 0.36
            4. "Renewal Pricing" → 2.8 → 0.33
            5. "FAQ: Renewals" → 2.1 → 0.25

            **Combined Scores (70% semantic + 30% keyword):**
            1. "Domain Renewal Procedure" → 0.94
            2. "Account Management Guide" → 0.62
            3. "Renew Premium Domains" → 0.68
            4. "Auto-Renewal Settings" → 0.50
            5. "Billing and Payment FAQ" → 0.53

            **After Reranking (Final Top 5):**
            1. "Domain Renewal Procedure" → 0.96
            2. "Renew Premium Domains" → 0.89
            3. "Account Management Guide" → 0.82
            4. "Billing and Payment FAQ" → 0.74
            5. "Auto-Renewal Settings" → 0.71
            """)

    # ==================== TAB 3: SEMANTIC CHUNKING ====================
    with tab3:
        st.markdown("### ✂️ Semantic Chunking Process")
        st.markdown("*Intelligent text splitting based on topic boundaries*")

        # Why Semantic Chunking?
        st.info("""
        **Why Semantic Chunking over Character-based?**
        - ✅ Preserves topic boundaries (doesn't break mid-topic)
        - ✅ No mid-sentence breaks (complete thoughts)
        - ✅ Better retrieval accuracy (coherent context)
        - ✅ Maintains document structure
        """)

        st.markdown("---")

        # 5-Step Process
        st.markdown("### Five-Step Chunking Algorithm")

        st.code("""
📄 Raw Document Text
    ↓
1️⃣ Tokenize into Sentences
    • Use regex: r'(?<=[.!?])\\s+(?=[A-Z])'
    • Handle newlines
    • Filter empty sentences
    • Returns: List of sentences
    ↓
2️⃣ Embed Each Sentence
    • SentenceTransformer (all-MiniLM-L6-v2)
    • Batch embedding for efficiency
    • Output: n_sentences × 384 matrix
    ↓
3️⃣ Compute Adjacent Similarities
    • For each sentence boundary:
      - Average left window embeddings (buffer_size=1)
      - Average right window embeddings
      - Cosine similarity between windows
    • Output: Similarity scores (0-1)
    ↓
4️⃣ Find Topic Breakpoints
    • Threshold method: similarity < 0.5
    • Local minima: Lower than neighbors AND < 0.6
    • Output: List of boundary indices
    ↓
5️⃣ Create and Optimize Chunks
    • Split text at breakpoints
    • Merge small chunks (< 100 chars)
    • Split large chunks (> 1500 chars)
    • Add 1-sentence overlap
    • Attach metadata (category, section)
    ↓
✅ Semantic Document Chunks
    • Topic-coherent segments
    • Complete sentences
    • Optimal size for retrieval
    • With metadata
        """, language="text")

        st.markdown("---")

        # Visual Example
        st.markdown("### 📖 Example: Document Chunking")

        with st.expander("See detailed example", expanded=False):
            st.markdown("""
            **Input Document:**
            ```
            To renew a domain, you need to log into your account.
            Click the 'Domains' tab in the navigation menu.
            Select the domain you want to renew from the list.
            Click the 'Renew' button next to the domain name.
            Choose your renewal period and proceed to payment.
            For technical support, contact our help desk.
            We're available 24/7 via phone and email.
            ```

            **Step 1: Tokenize → 7 sentences**

            **Step 2: Embed each sentence**

            **Step 3: Compute similarities between adjacent sentences:**
            - Sent 0 ↔ Sent 1: 0.78 (both about navigation)
            - Sent 1 ↔ Sent 2: 0.82 (still about domain selection)
            - Sent 2 ↔ Sent 3: 0.75 (selecting → renewing)
            - Sent 3 ↔ Sent 4: 0.71 (renewal → payment)
            - Sent 4 ↔ Sent 5: **0.38** ← BREAKPOINT (payment → support)
            - Sent 5 ↔ Sent 6: 0.89 (both about support)

            **Step 4: Find breakpoints:**
            - Index 4 (similarity 0.38 < 0.5 threshold) ✓

            **Step 5: Create chunks:**
            - **Chunk 1 (sentences 0-4):** "To renew a domain... proceed to payment." (145 chars)
            - **Chunk 2 (sentences 5-6):** "For technical support... phone and email." (78 chars)

            **Optimization:**
            - Chunk 2 < 100 chars → Merge with Chunk 1
            - **Final: 1 chunk** covering the complete renewal + support info

            """)

    # ==================== TAB 4: MCP PROMPTS ====================
    with tab4:
        st.markdown("### 📋 MCP (Model Context Protocol)")
        st.markdown("*Structured prompt engineering for reliable LLM outputs*")

        # MCP Structure
        render_mcp_structure()

        st.markdown("---")

        # Detailed Section Breakdown
        st.markdown("### Prompt Section Details")

        # 4-Section Architecture
        section_details = [
            ("🎭 ROLE Section", "System Message", """
            • **Purpose**: Define assistant identity and expertise
            • **Content**:
              - "You are an expert customer support assistant..."
              - Domain expertise areas (DNS, billing, domains, security)
              - Response guidelines and tone
            • **Why it matters**: Sets the context for professional, policy-compliant responses
            """),
            ("💬 MEMORY Section", "User Message Part 1", """
            • **Purpose**: Maintain conversation continuity
            • **Content**:
              - Last 3 conversation turns (if available)
              - Previous Q&A pairs
              - Actions taken in past responses
            • **Why it matters**: Enables follow-up questions and prevents repetition
            • **Format**: "## Recent Conversation History\\n### Turn 1: ..."
            """),
            ("📚 CONTEXT Section", "User Message Part 2", """
            • **Purpose**: Provide grounded knowledge from RAG
            • **Content**:
              - Retrieved documents (top 5)
              - Similarity scores (transparency)
              - Document metadata (category, section)
            • **Why it matters**: Prevents hallucinations, enables citations
            • **Format**: "### Document 1: [title]\\n**Similarity:** 92.5%\\n[content]"
            """),
            ("📋 TASK + SCHEMA", "User Message Part 3 & 4", """
            • **TASK Purpose**: Provide current query and instructions
            • **TASK Content**:
              - Customer ticket text
              - 5-step analysis instructions
            • **SCHEMA Purpose**: Ensure structured JSON output
            • **SCHEMA Content**:
              - JSON format specification
              - Field descriptions (answer, references, action_required)
              - Valid action types (6 options)
            • **Why it matters**: Consistent, parseable, actionable responses
            """)
        ]

        for i, (title, subtitle, details) in enumerate(section_details):
            with st.expander(f"{title}: {subtitle}", expanded=(i == 0)):
                st.markdown(details)

        st.markdown("---")

        # What the LLM Sees
        st.markdown("### 👁️ What the LLM Actually Sees")
        st.markdown("*Example prompt with memory context*")

        with st.expander("📝 Scenario: Follow-up Question", expanded=False):
            st.markdown("""
**Previous conversation:**
1. User: "How do I reactivate my suspended domain?"
2. AI: "Log into portal, update WHOIS, verify email..."

**Current question:**
3. User: "How long will that take?"

**What the LLM receives:**
            """)

            st.code("""
SYSTEM MESSAGE (ROLE):
---
You are an expert customer support assistant...

USER MESSAGE:
---
================================================================================
                             MEMORY SECTION
                  (Relevant Past Conversations for Context)
================================================================================

## Recent Conversation History

### Turn 1:
**Customer Query:** How do I reactivate my suspended domain?
**Your Previous Response:** To reactivate your domain, log into your portal
at example.com/login, navigate to 'My Domains' and select the suspended domain.
Update your WHOIS information and verify your email...
**Action Taken:** customer_action_required

Use this conversation history to maintain continuity and avoid repeating information.

================================================================================
                              CONTEXT SECTION
                    (Retrieved from Knowledge Base via RAG)
================================================================================

### Document 1: Policy - Domain Suspension Guidelines
**Similarity Score:** 95.00%

Domains suspended for WHOIS verification failure can be reactivated...
Reactivation typically completes within 24-48 hours after email verification.

================================================================================
                               TASK SECTION
================================================================================

Customer Ticket: "How long will that take?"

[Analysis instructions...]

================================================================================
                            OUTPUT SCHEMA SECTION
================================================================================

{
  "answer": "...",
  "references": [...],
  "action_required": "..."
}
            """, language="text")

            st.success("""
**Result:** The LLM understands:
- ✅ "that" refers to domain reactivation (from memory)
- ✅ Previous context about WHOIS and verification
- ✅ Can provide specific timeline (24-48 hours) from docs
- ✅ Maintains conversation continuity
            """)

        st.markdown("---")

        # Action Types
        st.markdown("### 🎯 Action Types (6 Options)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Resolution Actions:**
            - `none` - Ticket fully resolved
            - `follow_up_required` - Need to check back later
            - `customer_action_required` - Customer must act
            """)

        with col2:
            st.markdown("""
            **Escalation Actions:**
            - `escalate_to_abuse_team` - Security/policy violation
            - `escalate_to_billing` - Payment/refund issue
            - `escalate_to_technical` - Complex technical problem
            """)

    # ==================== TAB 5: SESSION MEMORY ====================
    with tab5:
        st.markdown("### 🧠 Session Memory System")
        st.markdown("*In-memory conversation tracking for contextual responses*")

        # Why Session Memory?
        st.info("""
        **Why Session Memory over File-based?**
        - ✅ Streamlit Cloud compatible (no file system needed)
        - ✅ Simple implementation (in-memory deque)
        - ✅ Fast (no I/O overhead)
        - ✅ Sufficient (10 turns covers most conversations)
        - ✅ Automatically cleared on session end (privacy)
        """)

        st.markdown("---")

        # Memory Architecture
        st.markdown("### 💾 Memory Architecture Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-box">
                <div style="font-size: 2rem; margin-bottom: 12px;">💾</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Storage</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    <strong>Type:</strong> In-memory deque<br/>
                    <strong>Max Capacity:</strong> 10 turns<br/>
                    <strong>Lifecycle:</strong> Session-scoped<br/>
                    <strong>Persistence:</strong> None (ephemeral)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-box">
                <div style="font-size: 2rem; margin-bottom: 12px;">🔄</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Context Window</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    <strong>Size:</strong> Last 3 turns<br/>
                    <strong>Injected into:</strong> MCP prompt<br/>
                    <strong>Position:</strong> MEMORY section<br/>
                    <strong>Format:</strong> Markdown string
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-box">
                <div style="font-size: 2rem; margin-bottom: 12px;">📝</div>
                <div style="font-weight: 700; color: #2F59A3; font-size: 1.1rem; margin-bottom: 8px;">Stored Data</div>
                <div style="font-size: 0.85rem; color: #4A5568; text-align: left; line-height: 1.6;">
                    • Customer query<br/>
                    • AI response answer<br/>
                    • References used<br/>
                    • Action taken<br/>
                    • Timestamp
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Memory Flow Process
        st.markdown("### 🔄 Memory Flow Process")

        st.code("""
┌─────────────────────────────────────────────────────────────┐
│                    MEMORY LIFECYCLE                          │
└─────────────────────────────────────────────────────────────┘

1️⃣ TICKET RESOLUTION
   Customer Query → RAG Pipeline → LLM → Response Generated
                                           ↓
2️⃣ STORAGE
   Create Turn Object:
   {
     query: "...",
     answer: "...",
     references: [...],
     action_required: "...",
     timestamp: datetime
   }
   ↓
   Append to Deque (max 10 turns)
   - If at capacity → Remove oldest (FIFO)
   - Store in session state
                                           ↓
3️⃣ NEXT QUERY (Follow-up)
   New Customer Query Received
   ↓
   Get Last 3 Turns from Memory
   ↓
   Format as Markdown String:
   '''
   ### Turn 1:
   **Query:** ...
   **Answer:** ...
   **Action:** ...
   '''
                                           ↓
4️⃣ PROMPT INJECTION
   Build MCP Prompt:
   - ROLE: System message
   - MEMORY: Last 3 turns ← Injected here
   - CONTEXT: RAG documents
   - TASK: Current query
   - SCHEMA: JSON format
                                           ↓
5️⃣ LLM UNDERSTANDING
   GPT-4o-mini receives complete context:
   ✅ Understands pronouns ("that", "it")
   ✅ Maintains topic continuity
   ✅ Avoids repetition
   ✅ Provides contextual answers
                                           ↓
6️⃣ STORE NEW TURN
   Loop back to step 2 with new response
        """, language="text")

        st.markdown("---")

        # Example Memory Formatting
        st.markdown("### 📝 Memory Formatting Example")

        with st.expander("See how memory is formatted for LLM", expanded=False):
            st.markdown("**Raw Memory (3 turns):**")
            st.code("""
Turn 1: {
  query: "My domain was suspended. How do I fix it?",
  answer: "Log into portal, update WHOIS, verify email...",
  references: ["Domain Suspension Policy"],
  action_required: "customer_action_required"
}

Turn 2: {
  query: "How long will that take?",
  answer: "Reactivation completes within 24-48 hours...",
  references: ["Reactivation Timeline"],
  action_required: "customer_action_required"
}

Turn 3: {
  query: "Can I speed it up?",
  answer: "Contact priority support for expedited review...",
  references: ["Premium Support Options"],
  action_required: "customer_action_required"
}
            """, language="text")

            st.markdown("**Formatted Memory Context (String for Prompt):**")
            st.code("""
================================================================================
                             MEMORY SECTION
                  (Relevant Past Conversations for Context)
================================================================================

## Recent Conversation History

### Turn 1:
**Customer Query:** My domain was suspended. How do I fix it?
**Your Previous Response:** Log into portal, update WHOIS, verify email...
**Action Taken:** customer_action_required

### Turn 2:
**Customer Query:** How long will that take?
**Your Previous Response:** Reactivation completes within 24-48 hours...
**Action Taken:** customer_action_required

### Turn 3:
**Customer Query:** Can I speed it up?
**Your Previous Response:** Contact priority support for expedited review...
**Action Taken:** customer_action_required

Use this conversation history to maintain continuity and avoid repeating information.
            """, language="text")

        st.markdown("---")

        # Live Memory Inspector
        st.markdown("### 🔍 Live Memory Inspector")
        st.markdown("*View the current session's conversation memory*")

        # Get memory statistics
        if init_rag_pipeline():
            try:
                pipeline = st.session_state.pipeline
                if pipeline is None:
                    st.error("Pipeline not initialized")
                else:
                    stats = pipeline.get_memory_stats()

                    if not stats.get("memory_enabled"):
                        st.info("✨ Memory is disabled. No conversation history is being tracked.")
                    else:
                        # Memory Status
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "💬 Total Turns",
                                stats["total_turns"],
                                help="Number of conversation turns in current session"
                            )

                        with col2:
                            st.metric(
                                "🔄 Context Window",
                                stats["context_window"],
                                help="Recent turns included in LLM prompts"
                            )

                        with col3:
                            duration = int(stats["session_duration_seconds"])
                            st.metric(
                                "⏱️ Session Duration",
                                f"{duration // 60}m {duration % 60}s",
                                help="How long this session has been active"
                            )

                        # Show Recent Conversations
                        if stats["total_turns"] > 0:
                            st.markdown("#### 📜 Recent Conversation History")

                            # Get the memory object to access turns
                            memory = pipeline._get_memory()
                            if memory and not memory.is_empty():
                                for i, turn in enumerate(reversed(list(memory.turns)), 1):
                                    with st.expander(f"Turn {stats['total_turns'] - i + 1}: {turn.query[:60]}...", expanded=(i == 1)):
                                        st.markdown(f"**Query:** {turn.query}")
                                        st.markdown(f"**Answer:** {turn.answer[:300]}{'...' if len(turn.answer) > 300 else ''}")
                                        st.markdown(f"**Action:** `{turn.action_required}`")
                                        if turn.references:
                                            st.markdown(f"**References:** {len(turn.references)} documents")
                                        st.caption(f"🕐 {turn.timestamp.strftime('%H:%M:%S')}")
                        else:
                            st.info("📭 No conversations yet. Start by resolving a ticket!")

                        # Clear Memory Button
                        st.markdown("---")
                        if st.button("🗑️ Clear Session Memory", help="Clear all conversation history"):
                            pipeline.clear_session_memory()
                            st.success("✅ Session memory cleared!")
                            st.rerun()

            except Exception as e:
                st.error(f"Error loading memory information: {e}")
                import traceback
                with st.expander("Show error details"):
                    st.code(traceback.format_exc())

    # ==================== END OF TABS ====================


elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.markdown("*Configure the Knowledge Assistant*")

    # API Key status
    api_key = os.getenv("OPENAI_API_KEY", "")
    # if api_key and len(api_key) > 10:
    #     st.success(f"✓ OpenAI API Key configured (ending in ...{api_key[-4:]})")
    # else:
    #     st.error("❌ OpenAI API Key not configured")
    #     st.info("Set `OPENAI_API_KEY` in your `.env` file or Streamlit Cloud secrets.")

    st.markdown("---")

    # Model settings
    st.markdown("### LLM Settings")

    col1, col2 = st.columns(2)

    with col1:
        model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )

    with col2:
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
            step=256
        )

    st.markdown("---")

    # RAG settings
    st.markdown("### RAG Settings")

    col1, col2 = st.columns(2)

    with col1:
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=int(os.getenv("TOP_K_RESULTS", "5")),
            help="Number of documents to retrieve"
        )

    with col2:
        threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(os.getenv("SIMILARITY_THRESHOLD", "0.3")),
            step=0.05,
            help="Minimum similarity score"
        )

    if st.button("💾 Save Settings", type="primary"):
        os.environ["OPENAI_MODEL"] = model
        os.environ["OPENAI_TEMPERATURE"] = str(temperature)
        os.environ["OPENAI_MAX_TOKENS"] = str(max_tokens)
        os.environ["TOP_K_RESULTS"] = str(top_k)
        os.environ["SIMILARITY_THRESHOLD"] = str(threshold)
        st.session_state.rag_initialized = False
        st.success("✅ Settings saved! RAG pipeline will reinitialize.")

    st.markdown("---")

    # Reset options
    st.markdown("### Reset Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reset RAG Pipeline"):
            st.session_state.rag_initialized = False
            st.session_state.pipeline = None
            st.success("RAG pipeline will reinitialize on next use.")

    with col2:
        if st.button("🗑️ Clear Session"):
            st.session_state.conversation_history = []
            st.session_state.current_response = None
            st.session_state.last_contexts = []
            st.success("Session cleared!")
