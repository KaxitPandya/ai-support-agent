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
        ("Knowledge Base", "📄", "Manage documents"),
        ("Analytics", "📊", "View system stats"),
        ("RAG Inspector", "🔬", "Explore the pipeline"),
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

    st.markdown("---")
    st.caption("RAG + MCP + LLM")
    st.caption("FAISS Vector Store")


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


elif page == "🔬 RAG Inspector":
    st.title("🔬 RAG Pipeline Inspector")
    st.markdown("*Understand how the RAG system works*")

    # MCP Structure
    st.markdown("### MCP (Model Context Protocol)")
    render_mcp_structure()

    st.markdown("---")

    # RAG Pipeline Visualization
    st.markdown("### RAG Pipeline Flow")

    col1, col2, col3, col4, col5 = st.columns(5)

    steps = [
        ("1. Input", "Customer ticket text", "📝"),
        ("2. Embed", "Generate query vector", "🔢"),
        ("3. Search", "Find similar docs", "🔍"),
        ("4. Augment", "Build MCP prompt", "📋"),
        ("5. Generate", "LLM response", "🤖"),
    ]

    for col, (title, desc, icon) in zip([col1, col2, col3, col4, col5], steps):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: 600; color: #2F59A3; font-size: 0.9rem;">{title}</div>
                <div style="font-size: 0.75rem; color: #4A5568;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Test the pipeline
    st.markdown("### Test Pipeline")

    test_query = st.text_input("Enter a test query:", "How do I reactivate my suspended domain?")

    if st.button("🔍 Analyze", type="primary"):
        if init_rag_pipeline():
            with st.spinner("Analyzing..."):
                try:
                    # Retrieve context
                    pipeline = st.session_state.pipeline
                    if pipeline is None:
                        st.error("Pipeline not initialized")
                        st.stop()
                    contexts = pipeline.retrieve_context(test_query)

                    # Display results
                    st.markdown("#### Retrieved Documents")

                    for i, ctx in enumerate(contexts, 1):
                        doc = ctx.document
                        score = ctx.similarity_score

                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{i}. {doc.title}**")
                            st.caption(f"Category: {doc.category} | Section: {doc.section}")
                        with col2:
                            if score >= 0.7:
                                st.success(f"{score:.1%}")
                            elif score >= 0.5:
                                st.warning(f"{score:.1%}")
                            else:
                                st.error(f"{score:.1%}")

                        with st.expander("Show content"):
                            st.text(doc.content)

                    # Show MCP prompt structure
                    st.markdown("#### MCP Prompt Preview")
                    from src.prompts.mcp_prompt import build_mcp_prompt
                    messages = build_mcp_prompt(test_query, contexts)

                    with st.expander("System Message (ROLE)"):
                        st.text(messages[0]["content"])

                    with st.expander("User Message (CONTEXT + TASK + OUTPUT)"):
                        st.text(messages[1]["content"][:2000] + "...")

                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    st.markdown("---")

    # Output Schema
    st.markdown("### MCP Output Schema")
    st.code("""{
  "answer": "Clear response with specific steps",
  "references": ["List of policy documents used"],
  "action_required": "none | escalate_to_abuse_team | escalate_to_billing | escalate_to_technical | customer_action_required | follow_up_required"
}""", language="json")


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


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #4A5568; font-size: 0.85rem; padding: 1rem 0;'>"
    "🧠 <strong>Knowledge Assistant</strong> v1.0 | "
    "RAG Pipeline + MCP Protocol + OpenAI LLM | "
    "FAISS Vector Database | "
    "<a href='http://localhost:8000/docs' style='color: #2F59A3; text-decoration: none;'>API Docs</a>"
    "</div>",
    unsafe_allow_html=True
)
