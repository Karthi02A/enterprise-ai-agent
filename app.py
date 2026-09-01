import streamlit as st
import os
import uuid
import numpy as np
import time
import gc
import json
import datetime
import hashlib
import importlib
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Page Configuration (Must be the very first Streamlit command)
st.set_page_config(
    page_title="Enterprise AI Agent",
    layout="wide"
)

# Force reloading of local modules so that signature changes are actively loaded by Streamlit
import services.document_service
importlib.reload(services.document_service)
import services.chunk_service
importlib.reload(services.chunk_service)
import services.embedding_service
importlib.reload(services.embedding_service)
import services.vector_service
importlib.reload(services.vector_service)
import services.retrieval_service
importlib.reload(services.retrieval_service)
import services.agent_service
importlib.reload(services.agent_service)
import services.cache_service
importlib.reload(services.cache_service)
import services.translation_service
importlib.reload(services.translation_service)
import services.eval_service
importlib.reload(services.eval_service)
import services.llm_service
importlib.reload(services.llm_service)

import services.document_service as doc_service
import services.chunk_service as chunk_service
import services.embedding_service as embedding_service
import services.vector_service as vector_service
import services.retrieval_service as retrieval_service
import services.agent_service as agent_service
import services.cache_service as cache_service
import services.translation_service as translation_service
import services.eval_service as eval_service
import services.llm_service as llm_service


# Background Preloader for PyTorch Models
@st.cache_resource
def start_model_preloading():
    import threading
    status = {"embedding": "loading", "reranker": "loading"}
    
    def preload_worker():
        try:
            from services.embedding_service import get_embedding_model
            get_embedding_model()
            status["embedding"] = "ready"
        except Exception as e:
            status["embedding"] = f"error: {str(e)}"
            
        try:
            from services.retrieval_service import get_reranker
            get_reranker()
            status["reranker"] = "ready"
        except Exception as e:
            status["reranker"] = f"error: {str(e)}"
            
    thread = threading.Thread(target=preload_worker, daemon=True)
    thread.start()
    return status

# Initialize preloading asynchronously
preload_status = start_model_preloading()

# Custom minimalist styling (ChatGPT inspired)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Smooth rendering & touch optimization globally */
* {
    -webkit-tap-highlight-color: transparent;
    -webkit-font-smoothing: antialiased;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #f1f5f9;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    text-align: center;
    margin-bottom: 1.5rem;
}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #334155;
}

.sidebar-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #f8fafc;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
}

/* Primary Button overrides */
.stButton>button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 6px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background-color 0.15s ease-in-out !important;
    touch-action: manipulation !important;
}

.stButton>button:hover {
    background-color: #1d4ed8 !important;
    box-shadow: none !important;
}

/* Compact Feedback Buttons Overrides */
div[data-testid="column"] button {
    min-height: 34px !important;
    height: 34px !important;
    padding: 4px 12px !important;
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    color: #f8fafc !important;
    font-size: 0.88rem !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-shadow: none !important;
    width: auto !important;
    touch-action: manipulation !important;
}

div[data-testid="column"] button:hover {
    background-color: #334155 !important;
    border-color: #475569 !important;
}

/* ============================================= */
/* TABLET BREAKPOINT (@media <= 768px)           */
/* ============================================= */
@media (max-width: 768px) {

    /* === GLOBAL LAYOUT === */
    .block-container {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        padding-top: 0.75rem !important;
        max-width: 100% !important;
    }

    /* === HERO HEADER === */
    .hero-title {
        font-size: 1.45rem !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.15rem !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        font-size: 0.85rem !important;
        margin-bottom: 0.8rem !important;
        line-height: 1.35 !important;
    }

    /* === SIDEBAR (slide-out panel) === */
    section[data-testid="stSidebar"] {
        width: 85vw !important;
        max-width: 300px !important;
    }
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stSelectbox select {
        font-size: 0.88rem !important;
        min-height: 38px !important;
    }

    /* === NATIVE-FEEL SWIPEABLE TAB BAR === */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        gap: 2px !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        padding-bottom: 4px !important;
    }
    .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        flex-shrink: 0 !important;
        padding: 6px 10px !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
    }

    /* === FILE UPLOADER (Tab 1) === */
    div[data-testid="stFileUploader"] {
        padding: 0.5rem !important;
    }
    div[data-testid="stFileUploader"] section {
        padding: 0.75rem !important;
    }

    /* === METADATA FORM - Stack 2-col to 1-col === */
    div[data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    /* Compact form inputs */
    .stTextInput input, .stSelectbox select, .stDateInput input {
        font-size: 0.88rem !important;
        min-height: 36px !important;
    }
    .stSlider {
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* === EXPANDERS (metadata cards per file) === */
    div[data-testid="stExpander"] {
        border-radius: 8px !important;
    }
    div[data-testid="stExpander"] summary {
        font-size: 0.88rem !important;
        padding: 0.5rem 0.75rem !important;
    }

    /* === METRIC CARDS (Telemetry KPI tiles) === */
    div[data-testid="stMetric"] {
        padding: 0.5rem !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.72rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }

    /* === DATA TABLES (Catalog + Telemetry Logs) === */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* === CHAT MESSAGES & INPUT (Research Workspace) === */
    .stChatMessage {
        padding: 0.55rem 0.65rem !important;
        border-radius: 8px !important;
        margin-bottom: 0.4rem !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
    }
    div[data-testid="stChatInput"] {
        padding: 0.4rem !important;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 0.9rem !important;
        min-height: 40px !important;
    }

    /* === RADIO BUTTONS (Research Depth) === */
    .stRadio > div {
        gap: 0.5rem !important;
    }
    .stRadio label {
        font-size: 0.85rem !important;
        padding: 4px 8px !important;
    }

    /* === BUTTONS (general) === */
    .stButton>button {
        font-size: 0.88rem !important;
        padding: 0.45rem 0.75rem !important;
        min-height: 38px !important;
    }

    /* === ALERTS / INFO / WARNING / SUCCESS bars === */
    div[data-testid="stAlert"] {
        padding: 0.5rem 0.75rem !important;
        font-size: 0.85rem !important;
    }
}

/* ============================================= */
/* SMALL PHONE BREAKPOINT (@media <= 480px)      */
/* ============================================= */
@media (max-width: 480px) {
    .hero-title {
        font-size: 1.2rem !important;
        margin-top: 0.3rem !important;
    }
    .hero-subtitle {
        font-size: 0.78rem !important;
        margin-bottom: 0.6rem !important;
    }
    .block-container {
        padding-left: 0.35rem !important;
        padding-right: 0.35rem !important;
    }

    /* Ultra-compact tab labels */
    .stTabs [data-baseweb="tab"] {
        padding: 5px 7px !important;
        font-size: 0.72rem !important;
    }

    /* Feedback buttons tighter for small screens */
    div[data-testid="column"] button {
        padding: 2px 6px !important;
        font-size: 0.78rem !important;
        min-height: 30px !important;
    }

    /* Telemetry metrics - 2x2 grid friendly */
    div[data-testid="stMetric"] label {
        font-size: 0.65rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 0.95rem !important;
    }

    /* Chat messages even more compact */
    .stChatMessage {
        padding: 0.4rem 0.5rem !important;
    }
    .stChatMessage [data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem !important;
    }
}

/* ============================================= */
/* VIEWPORT META (prevent zooming on double-tap) */
/* ============================================= */
@media (hover: none) and (pointer: coarse) {
    /* Touch device only: larger touch targets */
    .stButton>button, div[data-testid="column"] button {
        min-height: 40px !important;
    }
    input, select, textarea {
        font-size: 16px !important; /* Prevents iOS zoom on focus */
    }
}
</style>
""", unsafe_allow_html=True)

# Read Groq API Key silently from .env
api_key = os.getenv("GROQ_API_KEY", "")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []  # List of metadata dicts representing indexed files
if "db_stats" not in st.session_state:
    st.session_state.db_stats = {"chunks": 0, "pdfs": 0, "docx": 0, "xlsx": 0, "csv": 0}
if "processing" not in st.session_state:
    st.session_state.processing = False
if "current_response" not in st.session_state:
    st.session_state.current_response = ""
if "current_metadata" not in st.session_state:
    st.session_state.current_metadata = None
if "ingestion_results" not in st.session_state:
    st.session_state.ingestion_results = None
if "cleanup_done" not in st.session_state:
    vector_service.cleanup_expired_collections(max_age_seconds=7200)
    st.session_state.cleanup_done = True

# ----------------------------------------------------
# SIDEBAR CONTROL PANEL
# ----------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-title'>Enterprise Controls</div>", unsafe_allow_html=True)
    
    # Organization/Tenant ID Context
    tenant_id = st.text_input(
        "Workspace / Organization Context", 
        value="AcmeCorp",
        key="tenant_id",
        help="Namespace context for semantic caching and document partitioning."
    )
    
    # Department Context
    department_id = st.text_input(
        "Department Context",
        value="General Corporate",
        key="department_id",
        help="Department context for partition filter of document search."
    )

# Calculate scope-aware collection name (session-isolated for zero-default clean start)
raw_col_name = f"rag_{st.session_state.tenant_id}_{st.session_state.department_id}_{st.session_state.session_id[:8]}"
st.session_state.collection_name = vector_service.sanitize_collection_name(raw_col_name)

# Update active collection counts and synchronize local state with ChromaDB
coll = vector_service.get_collection(st.session_state.collection_name)
count = coll.count()
st.session_state.db_stats["chunks"] = count

# Auto-sync st.session_state.indexed_docs catalog entries with ChromaDB
try:
    existing_chunks = coll.get(include=["metadatas"])
    if existing_chunks and "metadatas" in existing_chunks and existing_chunks["metadatas"]:
        docs_map = {}
        for m in existing_chunks["metadatas"]:
            if not m:
                continue
            doc_identity = m.get("doc_identity")
            if doc_identity and doc_identity not in docs_map:
                docs_map[doc_identity] = {
                    "document_id": m.get("document_id"),
                    "doc_identity": doc_identity,
                    "filename": m.get("filename"),
                    "source_type": m.get("source_type"),
                    "category": m.get("category"),
                    "department": m.get("department"),
                    "industry": m.get("industry"),
                    "version": m.get("version"),
                    "priority": int(m.get("priority", 3)),
                    "ingestion_date": m.get("ingestion_date"),
                    "document_date": m.get("document_date"),
                    "doc_hash": m.get("doc_hash"),
                    "chunks_count": 0
                }
            if doc_identity:
                docs_map[doc_identity]["chunks_count"] += 1
        st.session_state.indexed_docs = list(docs_map.values())
    else:
        st.session_state.indexed_docs = []
except Exception:
    st.session_state.indexed_docs = []

with st.sidebar:
    # Models preloader status
    emb_state = preload_status["embedding"]
    rerank_state = preload_status["reranker"]
    if emb_state == "ready" and rerank_state == "ready":
        st.success("✓ AI models loaded and active")
    elif "error" in emb_state or "error" in rerank_state:
        st.error(f"⚠️ Model load failed: Emb: {emb_state}, Reranker: {rerank_state}")
    else:
        st.info("⏳ AI models initializing... (~60s on first startup)")
        import time as _time
        _time.sleep(3)
        st.rerun()
        
    # Model Choice
    model_choice = st.selectbox(
        "Model Selection",
        options=[
            "qwen/qwen3.6-27b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "groq/compound-beta"
        ],
        index=0,
        help="qwen/qwen3.6-27b is the confirmed free model on this account."
    )
    
    # Parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("Reset Session & Purge Cache"):
        vector_service.delete_collection(st.session_state.collection_name)
        cache_service.invalidate_cache(
            tenant_id=st.session_state.tenant_id,
            session_id=st.session_state.session_id,
            department=st.session_state.department_id
        )
        if hasattr(retrieval_service.get_bm25_index, "clear"):
            retrieval_service.get_bm25_index.clear()
        st.session_state.messages = []
        st.session_state.indexed_docs = []
        st.session_state.doc_metadata = {}
        st.session_state.ingestion_results = None
        st.session_state.db_stats = {"chunks": 0, "pdfs": 0, "docx": 0, "xlsx": 0, "csv": 0}
        st.session_state.session_id = str(uuid.uuid4())
        gc.collect()
        st.success("Wiped active workspace session.")
        st.rerun()

# ----------------------------------------------------
# MAIN CONTENT WORKSPACE
# ----------------------------------------------------
st.markdown("<div class='hero-title'>Enterprise AI Agent</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Decision-ready research syntheses over corporate knowledge bases.</div>", unsafe_allow_html=True)

if not api_key or not api_key.strip():
    st.error("GROQ_API_KEY is missing from environment. Please verify your .env configuration.")

# Configure Main 4 Navigation Workspaces in request section order:
# Ingest Documents -> Knowledge Catalog -> Research Workspace -> Telemetry & Analytics
tab_ingestion, tab_catalog, tab_workspace, tab_telemetry = st.tabs([
    "📥 Ingest Documents", 
    "📁 Knowledge Catalog", 
    "🔍 Research Workspace", 
    "📊 Telemetry & Analytics"
])

# ====================================================
# TAB 1: INGEST DOCUMENTS
# ====================================================
with tab_ingestion:
    st.subheader("Add Corporate Knowledge Files")
    st.write("Upload multiple files sequentially or in batches. Supported primary formats: **PDF**, **DOCX**, **XLSX**, **CSV**.")
    
    # Ingestion results view (AUTO-CLOSE / COLLAPSE the editor on success)
    if st.session_state.ingestion_results:
        st.markdown("### Latest Indexing Operations Summary")
        
        success_count = st.session_state.ingestion_results.get("success_count", 0)
        fail_count = st.session_state.ingestion_results.get("fail_count", 0)
        total_chunks = sum(st.session_state.ingestion_results.get("chunks_indexed", []))
        
        st.markdown(f"**✅ Ingestion Complete**")
        st.markdown(f"- **{success_count}** documents processed successfully")
        st.markdown(f"- **{total_chunks}** chunks indexed")
        st.markdown(f"- **{fail_count}** failed")
        
        for msg in st.session_state.ingestion_results.get("success", []):
            st.success(msg)
        for msg in st.session_state.ingestion_results.get("warnings", []):
            st.warning(msg)
        for msg in st.session_state.ingestion_results.get("errors", []):
            st.error(msg)
            
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("Configure New Files / Clear Status"):
                st.session_state.ingestion_results = None
                st.rerun()
        with col_c2:
            show_catalog = st.checkbox("View Knowledge Catalog Here", value=False)
            if show_catalog:
                st.markdown("---")
                st.subheader("Current Workspace Catalog")
                if not st.session_state.indexed_docs:
                    st.info("No corporate assets ingested in this session.")
                else:
                    registry_data = []
                    for doc in st.session_state.indexed_docs:
                        registry_data.append({
                            "Filename": doc.get("filename", "Doc"),
                            "Format": doc.get("source_type", "txt").upper(),
                            "Category": doc.get("category", "General"),
                            "Department": doc.get("department", "All"),
                            "Version": doc.get("version", "1.0"),
                            "Chunks Count": doc.get("chunks_count", 0),
                        })
                    import pandas as pd
                    st.dataframe(pd.DataFrame(registry_data), use_container_width=True)
                    
    # Only display the file uploader and metadata review editor if we don't have ingestion results displayed,
    # OR if the user explicitly clicked clear / is not done.
    if not st.session_state.ingestion_results:
        # Multi-file uploader
        uploaded_files = st.file_uploader(
            "Upload Corporate Documents",
            type=["pdf", "docx", "xlsx", "csv", "xls", "txt", "json"],
            accept_multiple_files=True,
            label_visibility="visible"
        )
        
        if uploaded_files:
            if "doc_metadata" not in st.session_state:
                st.session_state.doc_metadata = {}
                
            st.markdown("### Review & Setup Document Metadata")
            
            # Check for non-enterprise document names and display corporate warning banner
            non_ent_files = [u.name for u in uploaded_files if any(w in u.name.lower() for w in ["resume", "cv", "cover", "letter", "personal", "applicant", "candidate", "bio"])]
            if non_ent_files:
                st.warning(
                    f"⚠️ **Non-Enterprise Document Warning**: The uploaded file(s) `{', '.join(non_ent_files)}` appear to be personal/non-corporate documents.\n\n"
                    "This system is designed specifically for **Enterprise Corporate Research Documents** (e.g. Policies, Strategy Plans, Financial Reports, Operations Metrics). "
                    "Please ensure you upload enterprise-related corporate documents for business research synthesis."
                )
            else:
                st.info("Set independent metadata parameters for each uploaded asset below:")
            
            # Populate file metadata states if missing
            for u_file in uploaded_files:
                fname = u_file.name
                if fname not in st.session_state.doc_metadata:
                    # Heuristic defaults mapping
                    fname_lower = fname.lower()
                    default_cat = "Operations"
                    default_dept = "General Corporate"
                    default_ind = "General"
                    
                    if any(w in fname_lower for w in ["resume", "cv", "cover", "letter", "applicant", "candidate", "profile", "bio"]):
                        default_cat = "HR / Career"
                        default_dept = "HR"
                    elif "strategy" in fname_lower:
                        default_cat = "Strategy"
                        default_dept = "Strategy"
                    elif "financial" in fname_lower or "revenue" in fname_lower or "budget" in fname_lower:
                        default_cat = "Financial"
                        default_dept = "Finance"
                        default_ind = "Finance"
                    elif "hr" in fname_lower or "workforce" in fname_lower:
                        default_cat = "HR"
                        default_dept = "HR"
                    elif "governance" in fname_lower or "policy" in fname_lower:
                        default_cat = "Policy"
                        default_dept = "Compliance"
                    elif "operations" in fname_lower or "metrics" in fname_lower:
                        default_cat = "Operations"
                        default_dept = "Operations"
                    elif any(w in fname_lower for w in ["tech", "technology", "systems", "cloud"]):
                        default_cat = "Technology"
                        default_dept = "Technology"
                        
                    st.session_state.doc_metadata[fname] = {
                        "category": default_cat,
                        "department": default_dept,
                        "industry": default_ind,
                        "version": "1.0",
                        "priority": 1,
                        "doc_date": datetime.date(2026, 8, 31)
                    }
                    
                m = st.session_state.doc_metadata[fname]
                
                # Draw metadata configurators in standalone expanders per file
                with st.expander(f"📄 {fname} Ingestion Parameters", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        # Category choices
                        category_choice = st.selectbox(
                            "Category",
                            options=["Financial", "HR", "Strategy", "Operations", "Technology", "Policy", "Compliance", "Custom"],
                            index=["Financial", "HR", "Strategy", "Operations", "Technology", "Policy", "Compliance", "Custom"].index(m["category"]) if m["category"] in ["Financial", "HR", "Strategy", "Operations", "Technology", "Policy", "Compliance", "Custom"] else 3,
                            key=f"cat_{fname}"
                        )
                        if category_choice == "Custom":
                            custom_val = st.text_input("Enter Custom Category", value="Operations", key=f"cust_cat_{fname}")
                            m["category"] = custom_val
                        else:
                            m["category"] = category_choice
                            
                        # Target Department choices
                        dept_options = ["Finance", "HR", "Technology", "Operations", "Strategy", "Management", "Compliance", "General Corporate"]
                        dept_choice = st.selectbox(
                            "Target Department",
                            options=dept_options,
                            index=dept_options.index(m["department"]) if m["department"] in dept_options else dept_options.index("General Corporate"),
                            key=f"dept_{fname}"
                        )
                        m["department"] = dept_choice
                        
                        # Target Industry choices
                        ind_options = ["General", "Finance", "Healthcare", "Retail", "Manufacturing", "Energy", "Technology", "Public Sector"]
                        ind_choice = st.selectbox(
                            "Target Industry",
                            options=ind_options,
                            index=ind_options.index(m["industry"]) if m["industry"] in ind_options else ind_options.index("General"),
                            key=f"ind_{fname}"
                        )
                        m["industry"] = ind_choice
                        
                    with col2:
                        m["version"] = st.text_input("Document Version", value=m["version"], key=f"ver_{fname}")
                        m["priority"] = st.slider("Priority Level (1=High, 3=Low)", min_value=1, max_value=3, value=m["priority"], key=f"pri_{fname}")
                        m["doc_date"] = st.date_input("Document Date", value=m["doc_date"], key=f"date_{fname}")
                        
            st.markdown("<br>", unsafe_allow_html=True)
            is_ready = (preload_status["embedding"] == "ready")
            btn_label = "Index Selected Documents" if is_ready else "Indexing (Waiting for models to preload...)"
            
            # Display the central Action Index button
            # Display the central Action Index button
            if st.button(btn_label, disabled=not is_ready):
                operation_success = []
                operation_warnings = []
                operation_errors = []
                
                success_count = 0
                fail_count = 0
                chunks_total = 0
                
                # Evict semantic cache on new document ingestion to prevent stale matches
                cache_service.invalidate_cache(
                    tenant_id=st.session_state.tenant_id,
                    session_id=st.session_state.session_id,
                    department=st.session_state.department_id
                )
                
                total_files = len(uploaded_files)
                # Consolidated progress card UI layout
                progress_container = st.container()
                with progress_container:
                    st.markdown("### 📥 Indexing Document Batch")
                    batch_progress = st.progress(0.0)
                    overall_status = st.empty()
                    steps_list = st.empty()
                
                for file_idx, u_file in enumerate(uploaded_files):
                    fname = u_file.name
                    m = st.session_state.doc_metadata.get(fname)
                    if not m:
                        continue
                    
                    # Update progress bar
                    batch_progress.progress(file_idx / total_files)
                    
                    def update_steps(step_num, status_txt="In Progress"):
                        icons = ["⏳", "⏳", "⏳", "⏳", "⏳"]
                        for s in range(step_num):
                            if s < len(icons):
                                icons[s] = "✓"
                        if step_num < len(icons):
                            icons[step_num] = "⚡"
                        steps_list.markdown(
                            f"**Currently Processing ({file_idx + 1}/{total_files}):** `{fname}`\n"
                            f"* {icons[0]} **File received & validated**\n"
                            f"* {icons[1]} **Text extraction & parsing** ({status_txt if step_num == 1 else ('Done' if step_num > 1 else 'Waiting')})\n"
                            f"* {icons[2]} **Chunk splitting** ({status_txt if step_num == 2 else ('Done' if step_num > 2 else 'Waiting')})\n"
                            f"* {icons[3]} **AI Embeddings generation** ({status_txt if step_num == 3 else ('Done' if step_num > 3 else 'Waiting')})\n"
                            f"* {icons[4]} **ChromaDB indexing** ({status_txt if step_num == 4 else ('Done' if step_num > 4 else 'Waiting')})"
                        )
                        
                    try:
                        # Step 1: File received & enterprise document validation
                        MAX_SIZE = 30 * 1024 * 1024
                        if u_file.size > MAX_SIZE:
                            raise ValueError("File exceeds maximum 30 MB size limit.")
                        
                        # Strict Enterprise Document Rejection Guard
                        if any(w in fname.lower() for w in ["resume", "cv", "cover", "letter", "personal", "applicant", "candidate", "bio"]):
                            raise ValueError("Non-enterprise document rejected. Only corporate enterprise documents (Policies, Strategy Plans, Financial Reports, Operations Metrics) are permitted.")
                            
                        update_steps(1, "Parsing text content...")
                        
                        # Calculate stable content hash
                        file_bytes = u_file.getvalue()
                        doc_hash = hashlib.sha256(file_bytes).hexdigest()
                        doc_id = f"doc_{doc_hash}"
                        version = m["version"]
                        doc_identity = f"{fname}_v{version}"
                        
                        # Search for existing matched catalog item
                        match_doc = None
                        for doc in st.session_state.indexed_docs:
                            if doc["filename"] == fname and doc["version"] == version:
                                match_doc = doc
                                break
                                
                        # Force re-indexing: purge existing chunks if same filename/version is requested
                        if match_doc:
                            try:
                                coll.delete(where={"doc_identity": doc_identity})
                            except Exception:
                                pass
                            st.session_state.indexed_docs = [d for d in st.session_state.indexed_docs if d.get("doc_identity") != doc_identity]
                            
                        # Step 2: Extract text
                        pages = doc_service.extract_document_pages(
                            u_file,
                            filename=fname,
                            timeout_seconds=120
                        )
                        update_steps(2, "Generating parent-child splits...")
                        
                        doc_meta_payload = {
                            "document_id": doc_id,
                            "doc_identity": doc_identity,
                            "filename": fname,
                            "source_type": os.path.splitext(fname.lower())[1][1:],
                            "category": m["category"],
                            "department": m["department"],
                            "industry": m["industry"],
                            "version": m["version"],
                            "priority": m["priority"],
                            "ingestion_date": time.strftime("%Y-%m-%d"),
                            "document_date": m["doc_date"].strftime("%Y-%m-%d"),
                            "doc_hash": doc_hash
                        }
                        
                        # Step 3: Chunk text
                        chunks, metas = chunk_service.create_parent_child_chunks(
                            pages=pages,
                            source_type=doc_meta_payload["source_type"],
                            filename=fname,
                            additional_metadata=doc_meta_payload
                        )
                        
                        if not chunks:
                            raise ValueError("Extractable text context is empty.")
                        update_steps(3, "Encoding text to vector space...")
                        
                        # Step 4: Embed text
                        embeddings = embedding_service.generate_embeddings(
                            chunks,
                            progress_callback=None
                        )
                        update_steps(4, "Saving vectors to ChromaDB...")
                        
                        # Step 5: Index chunks
                        deterministic_ids = [f"{doc_id}_v{version}_chunk_{j}" for j in range(len(chunks))]
                        vector_service.store_chunks(st.session_state.collection_name, chunks, embeddings, metas, ids=deterministic_ids)
                        update_steps(5, "Completed!")
                        
                        # Store locally in session registry
                        doc_meta_payload["chunks_count"] = len(chunks)
                        st.session_state.indexed_docs.append(doc_meta_payload)
                        operation_success.append(f"✓ '{fname}' (v{version}) successfully parsed and indexed into {len(chunks)} chunks.")
                        success_count += 1
                        chunks_total += len(chunks)
                        
                    except Exception as e:
                        operation_errors.append(f"❌ '{fname}': {str(e)}")
                        fail_count += 1
                        
                    time.sleep(0.5)
                
                # Mark final progress state complete
                batch_progress.progress(1.0)
                overall_status.success("Batch processing complete!")
                time.sleep(1.0)
                progress_container.empty()
                
                # Wipe Streamlit BM25 cache if present
                if hasattr(retrieval_service.get_bm25_index, "clear"):
                    retrieval_service.get_bm25_index.clear()
                
                # Save logs to session display state and reload
                st.session_state.ingestion_results = {
                    "success": operation_success,
                    "warnings": operation_warnings,
                    "errors": operation_errors,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "chunks_indexed": [chunks_total]
                }
                st.rerun()

# ====================================================
# TAB 2: KNOWLEDGE CATALOG
# ====================================================
with tab_catalog:
    col_cat_h1, col_cat_h2 = st.columns([3, 1])
    with col_cat_h1:
        st.subheader("Asset Registry")
    with col_cat_h2:
        if st.button("🗑️ Clear All Assets", help="Purge all indexed documents and demo files from database"):
            vector_service.delete_collection(st.session_state.collection_name)
            cache_service.invalidate_cache(
                tenant_id=st.session_state.tenant_id,
                session_id=st.session_state.session_id,
                department=st.session_state.department_id
            )
            if hasattr(retrieval_service.get_bm25_index, "clear"):
                retrieval_service.get_bm25_index.clear()
            st.session_state.indexed_docs = []
            st.session_state.doc_metadata = {}
            st.session_state.db_stats = {"chunks": 0, "pdfs": 0, "docx": 0, "xlsx": 0, "csv": 0}
            st.toast("Knowledge base purged!")
            st.rerun()
    
    if not st.session_state.indexed_docs:
        st.info("No corporate assets ingested. Upload files using the 'Ingest Documents' tab.")
    else:
        # Construct dynamic table of files metadata details
        registry_data = []
        for doc in st.session_state.indexed_docs:
            registry_data.append({
                "Filename": doc.get("filename", "Doc"),
                "Format": doc.get("source_type", "txt").upper(),
                "Category": doc.get("category", "General"),
                "Department": doc.get("department", "All"),
                "Industry": doc.get("industry", "General"),
                "Version": doc.get("version", "1.0"),
                "Priority": f"P{doc.get('priority', 3)}",
                "Ingestion Date": doc.get("ingestion_date", "-"),
                "Chunks Count": doc.get("chunks_count", 0),
                "Status": "Active"
            })
            
        import pandas as pd
        df_registry = pd.DataFrame(registry_data)
        st.dataframe(
            df_registry,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Filename": st.column_config.TextColumn("Filename", width="medium"),
                "Format": st.column_config.TextColumn("Format", width="small"),
                "Category": st.column_config.TextColumn("Category", width="small"),
                "Department": st.column_config.TextColumn("Department", width="small"),
                "Industry": st.column_config.TextColumn("Industry", width="small"),
                "Version": st.column_config.TextColumn("Version", width="small"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "Ingestion Date": st.column_config.TextColumn("Ingestion Date", width="small"),
                "Chunks Count": st.column_config.NumberColumn("Chunks", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small")
            }
        )
        
    st.subheader("Vector Database stats")
    st.metric(label="Total Chunks in Database", value=st.session_state.db_stats["chunks"])

# ====================================================
# TAB 3: RESEARCH WORKSPACE
# ====================================================
with tab_workspace:
    # Workspace configurations options
    research_depth = st.radio(
            "Research Depth",
            options=["Quick Research", "Deep Research"],
            index=0,
            horizontal=True,
            help="Quick Research = 2 API calls (faster, free tier friendly). Deep Research = 4+ calls (slower, may hit rate limits)."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Active Collection Chunks check warning
    if st.session_state.db_stats["chunks"] == 0:
        st.info("No active corporate documents found in the database. Please go to the 'Ingest Documents' tab to upload some.")
        
    # Render active messages
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.messages):
            role_symbol = "🧑‍💻" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=role_symbol):
                st.markdown(msg["content"])
                
                # Feedback options for assistant message
                if msg["role"] == "assistant" and "latency" in msg:
                    # Provide feedback voting buttons
                    v_up, v_down, _ = st.columns([1, 1, 14])
                    with v_up:
                        if st.button("👍", key=f"feed_up_{idx}"):
                            eval_service.update_feedback(msg.get("log_id", ""), 1)
                            st.toast("Thank you for your positive feedback!")
                    with v_down:
                        if st.button("👎", key=f"feed_down_{idx}"):
                            eval_service.update_feedback(msg.get("log_id", ""), 0)
                            st.toast("Feedback recorded.")
                            
    # Lock query box if building or database is empty
    prompt = st.chat_input(
        "Ask a business/research question...",
        disabled=(st.session_state.db_stats["chunks"] == 0) or st.session_state.processing
    )
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.current_prompt = prompt
        st.session_state.processing = True
        st.rerun()
        
    # Main Agent Process Execution
    if st.session_state.processing and "current_prompt" in st.session_state:
        current_prompt = st.session_state.current_prompt
        
        # Intercept conversational greetings to route straight to multilingual static responses
        from services.llm_service import is_conversational_query
        if is_conversational_query(current_prompt):
            prompt_lower = current_prompt.lower()
            is_thanks = any(w in prompt_lower for w in ["thank", "thx", "ty"])
            
            if is_thanks:
                final_response = "You're very welcome! Please let me know if you have any other questions about your corporate documents."
            else:
                final_response = "Hello! I am here to help you research and analyze your business documents and files. Please feel free to ask any research questions you have about your data."
            
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(final_response)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "latency": 0.0,
                "log_id": f"greet_{int(time.time())}"
            })
            
            st.session_state.processing = False
            if "current_prompt" in st.session_state:
                del st.session_state.current_prompt
            st.rerun()
            
        with chat_container:
            with st.chat_message("assistant", avatar="🤖"):
                progress_stages = st.empty()
                response_placeholder = st.empty()
                
                def draw_status(pct_text):
                    progress_stages.markdown(f"🤖 **Status:** *{pct_text}*")
                    
                t_start = time.time()
                
                # Cache lookup
                draw_status("Retrieving semantic cache index calculations...")
                q_emb = embedding_service.generate_embeddings([current_prompt])[0]
                cached_ans, cached_src, cache_type = cache_service.get_cached_result(
                    tenant_id=st.session_state.tenant_id,
                    query_text=current_prompt,
                    query_embedding=q_emb,
                    similarity_threshold=0.92,
                    session_id=st.session_state.session_id,
                    department=st.session_state.department_id
                )
                
                if cached_ans:
                    draw_status(f"Found cache result! Returning cached synthesis...")
                    response_placeholder.markdown(cached_ans)
                    progress_stages.empty()
                    
                    t_total = time.time() - t_start
                    
                    # Log cache hit telemetry
                    log_id = eval_service.log_query_execution(
                        query=current_prompt,
                        complexity="simple",
                        sub_query_count=1,
                        retrieval_latency=t_total,
                        llm_latency=0.0,
                        total_latency=t_total,
                        cache_hit_or_miss=f"HIT ({cache_type})",
                        sources_retrieved=cached_src,
                        session_id=st.session_state.session_id,
                        tenant_id=st.session_state.tenant_id,
                        department=st.session_state.department_id
                    )
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": cached_ans, 
                        "latency": t_total, 
                        "log_id": log_id
                    })
                    st.session_state.processing = False
                    if "current_prompt" in st.session_state:
                         del st.session_state.current_prompt
                    st.rerun()
                    
                # Streaming Agent Loop
                streamer = agent_service.perform_research_stream(
                    query=current_prompt,
                    collection_name=st.session_state.collection_name,
                    api_key=api_key,
                    tenant_id=st.session_state.tenant_id,
                    research_depth=research_depth,
                    use_hybrid=True,
                    use_reranker=True,
                    temperature=temperature,
                    model=model_choice,
                    progress_callback=lambda stage_text: draw_status(stage_text)
                )
                
                agent_response = ""
                metadata_header = None
                
                for chunk in streamer:
                    if chunk.startswith("__METADATA__:"):
                        metadata_header = json.loads(chunk[13:].strip())
                        t_retrieval_done = time.time()
                        retrieval_latency = t_retrieval_done - t_start
                        continue
                    agent_response += chunk
                    response_placeholder.markdown(agent_response + "▌")
                    
                # Clean reasoning text from agent responses if reasoning model was selected
                agent_response = translation_service.clean_think_tags(agent_response)
                    
                response_placeholder.markdown(agent_response)
                progress_stages.empty()
                
                # Fetch telemetry tags
                complexity_tag = metadata_header.get("complexity", "simple") if metadata_header else "simple"
                sub_queries = metadata_header.get("sub_queries", [current_prompt]) if metadata_header else [current_prompt]
                sources_retrieved = metadata_header.get("sources", []) if metadata_header else []
                
                t_total = time.time() - t_start
                llm_latency = t_total - retrieval_latency
                
                # Save output to cache (using english representation)
                if not agent_response.startswith("Insufficient") and not agent_response.startswith("Error"):
                    cache_service.set_cached_result(
                        tenant_id=st.session_state.tenant_id,
                        query_text=current_prompt,
                        query_embedding=q_emb,
                        response_text=agent_response,
                        sources=sources_retrieved,
                        session_id=st.session_state.session_id,
                        department=st.session_state.department_id
                    )
                    
                # Log telemetry output
                log_id = eval_service.log_query_execution(
                    query=current_prompt,
                    complexity=complexity_tag,
                    sub_query_count=len(sub_queries),
                    retrieval_latency=retrieval_latency,
                    llm_latency=llm_latency,
                    total_latency=t_total,
                    cache_hit_or_miss="MISS",
                    sources_retrieved=sources_retrieved,
                    session_id=st.session_state.session_id,
                    tenant_id=st.session_state.tenant_id,
                    department=st.session_state.department_id
                )
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": agent_response, 
                    "latency": t_total, 
                    "log_id": log_id
                })
                
                st.session_state.processing = False
                if "current_prompt" in st.session_state:
                    del st.session_state.current_prompt
                st.rerun()

# ====================================================
# TAB 4: TELEMETRY & ANALYTICS
# ====================================================
with tab_telemetry:
    st.subheader("Performance & Telemetry Analytics Dashboard")
    
    logs = eval_service.read_evaluation_data(session_id=st.session_state.session_id)
    if not logs:
        st.info("No query logs compiled. Run retrieval requests in the Research Workspace to view metrics.")
    else:
        # Compute database aggregate calculations
        total_q = len(logs)
        cache_hits = sum(1 for log in logs if log.get("cache_hit_or_miss") == "HIT")
        cache_misses = total_q - cache_hits
        hit_ratio = (cache_hits / total_q) * 100 if total_q > 0 else 0
        
        # Latency statistics
        latencies = [log["total_latency"] for log in logs if log["total_latency"] > 0]
        avg_lat = np.mean(latencies) if latencies else 0.0
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.metric("Total Queries", value=total_q)
        with col_t2:
            st.metric("Cache Hit Ratio", value=f"{hit_ratio:.1f}%")
        
        col_t3, col_t4 = st.columns(2)
        with col_t3:
            st.metric("Avg Latency", value=f"{avg_lat:.2f}s")
        with col_t4:
            # Positive ratings count
            positive_votes = sum(1 for log in logs if log.get("feedback") == 1)
            negative_votes = sum(1 for log in logs if log.get("feedback") == 0)
            st.metric("Positive Feedbacks", value=f"{positive_votes} 👍 / {negative_votes} 👎")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Query logs registry")
        
        log_records = []
        for idx, log in enumerate(reversed(logs)):
            log_records.append({
                "Index": total_q - idx - 1, 
                "Timestamp": log.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                "Query": log.get("query", ""),
                "Complexity": log.get("complexity", "simple"),
                "Queries Count": log.get("sub_query_count", 1),
                "Total Latency (s)": f"{log.get('total_latency', 0.0):.2f}s",
                "Cache State": log.get("cache_hit_or_miss", "MISS"),
                "Feedback": "👍 Positive" if log.get("feedback") == 1 else ("👎 Negative" if log.get("feedback") == 0 else "None")
            })
            
        import pandas as pd
        st.dataframe(pd.DataFrame(log_records), use_container_width=True)