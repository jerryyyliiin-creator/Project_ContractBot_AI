import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
#from langchain.schema.output_parser import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Other Tools
import pandas as pd
import json
from datetime import datetime
from difflib import get_close_matches
from streamlit_option_menu import option_menu

# Page Configuration
st.set_page_config(page_title="Homepage", layout="wide")
st.logo("logo.png")

# Environment Variables and Core Configurations
load_dotenv()
INDEX_NAME = "contract-assistant"

# Session State Initialization
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = []
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "core_points_text" not in st.session_state:
    st.session_state.core_points_text = ""

# Core Functions
@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def fetch_pinecone_namespaces(index_name):
    pc = get_pinecone_client()
    try:
        index_stats = pc.describe_index(index_name).stats
        return list(index_stats.namespaces.keys()) if index_stats and index_stats.namespaces else []
    except Exception:
        return []

if not st.session_state.processed_namespaces:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME)

def process_and_ingest_reference_file(uploaded_file):
    namespace = uploaded_file.name
    with st.spinner(f"Processing reference document '{namespace}' and storing it in the permanent knowledge base..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        PineconeVectorStore.from_documents(docs, embedding=embeddings, index_name=INDEX_NAME, namespace=namespace)
        os.remove(tmp_file_path)
    st.success(f"Reference document '{namespace}' has been successfully stored in the knowledge base!")
    if namespace not in st.session_state.processed_namespaces:
        st.session_state.processed_namespaces.append(namespace)
    st.cache_data.clear()

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    os.remove(tmp_file_path)
    return vectorstore.as_retriever(search_kwargs={'k': 2})

# UI Rendering
def draw_main_app():
    st.markdown("""
    <style>
    :root { --content-gutter: 8px; } /* small gutter next to the sidebar */

    /* Remove Streamlit's default left padding on the main content.
        Use high-specificity selectors + !important to win. */
    [data-testid="stAppViewContainer"] .main .block-container,
    section.main > div.block-container,
    .main .block-container {
        max-width: 1180px;
        padding-top: 0 !important;
        padding-left: var(--content-gutter) !important;
        padding-right: 1rem !important;  /* keep a right gutter */
    }

    /* HERO: do NOT pull under the sidebar. Remove the negative left margin. */
    .hero-wrap{
        margin: 0  -2rem 1.25rem 0;   /* top right bottom left (left is now 0) */
        padding: 3.5rem 2rem 2.25rem;
        background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,.25), transparent 60%),
                    radial-gradient(900px 600px at 110% 10%, rgba(16,185,129,.24), transparent 60%),
                    linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
        border-bottom: 1px solid rgba(255,255,255,.08);
    }

    .kicker{
        display:inline-flex; gap:.5rem; align-items:center;
        padding:.35rem .7rem; border-radius:999px;
        background: rgba(99,102,241,.15);
        border:1px solid rgba(99,102,241,.35);
        font-size:.82rem; letter-spacing:.02em;
    }
    h1.hero{margin:.6rem 0 .2rem; font-size:2.1rem; line-height:1.15;}
    p.sub{margin:.3rem 0 0; opacity:.9}
    .btn-row{display:flex; gap:.6rem; margin-top:1rem; flex-wrap:wrap}
    .btn{
        padding:.6rem .95rem; border-radius:12px; text-decoration:none;
        border:1px solid rgba(255,255,255,.14);
        background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
        transition:.2s;
    }
    .btn:hover{transform:translateY(-1px); border-color:rgba(255,255,255,.28)}
    .btn.primary{background:linear-gradient(180deg, rgba(99,102,241,.55), rgba(99,102,241,.35)); border-color:rgba(99,102,241,.65)}
    .btn.success{background:linear-gradient(180deg, rgba(16,185,129,.55), rgba(16,185,129,.35)); border-color:rgba(16,185,129,.65)}

    /* Cards */
    .glass{
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 10px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .section-h {font-size:1.05rem; font-weight:600; opacity:.95; margin-bottom:.35rem}
    .chip{display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .55rem; border-radius:999px; font-size:.78rem;
            border:1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.06);}

    /* History rows */
    .row{padding:.5rem 0;}
    .row + .row{border-top:1px solid rgba(255,255,255,.08)}

    /* Mobile/tablet: add a bit more left gutter for comfort */
    @media (max-width: 992px){
        [data-testid="stAppViewContainer"] .main .block-container { padding-left: 1rem !important; }
        .hero-wrap{ margin: 0 -1rem 1rem 0; }
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    :root { --content-gutter: 8px; } /* small gutter next to the sidebar */

    /* Remove Streamlit's default left padding on the main content.
        Use high-specificity selectors + !important to win. */
    [data-testid="stAppViewContainer"] .main .block-container,
    section.main > div.block-container,
    .main .block-container {
        max-width: 1180px;
        padding-top: 0 !important;
        padding-left: var(--content-gutter) !important;
        padding-right: 1rem !important;  /* keep a right gutter */
    }

    /* HERO: do NOT pull under the sidebar. Remove the negative left margin. */
    .hero-wrap{
        margin: 0  -2rem 1.25rem 0;   /* top right bottom left (left is now 0) */
        padding: 3.5rem 2rem 2.25rem;
        background: radial-gradient(1200px 600px at 10% -10%, rgba(99,102,241,.25), transparent 60%),
                    radial-gradient(900px 600px at 110% 10%, rgba(16,185,129,.24), transparent 60%),
                    linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
        border-bottom: 1px solid rgba(255,255,255,.08);
    }

    .kicker{
        display:inline-flex; gap:.5rem; align-items:center;
        padding:.35rem .7rem; border-radius:999px;
        background: rgba(99,102,241,.15);
        border:1px solid rgba(99,102,241,.35);
        font-size:.82rem; letter-spacing:.02em;
    }
    h1.hero{margin:.6rem 0 .2rem; font-size:2.1rem; line-height:1.15;}
    p.sub{margin:.3rem 0 0; opacity:.9}
    .btn-row{display:flex; gap:.6rem; margin-top:1rem; flex-wrap:wrap}
    .btn{
        padding:.6rem .95rem; border-radius:12px; text-decoration:none;
        border:1px solid rgba(255,255,255,.14);
        background:linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
        transition:.2s;
    }
    .btn:hover{transform:translateY(-1px); border-color:rgba(255,255,255,.28)}
    .btn.primary{background:linear-gradient(180deg, rgba(99,102,241,.55), rgba(99,102,241,.35)); border-color:rgba(99,102,241,.65)}
    .btn.success{background:linear-gradient(180deg, rgba(16,185,129,.55), rgba(16,185,129,.35)); border-color:rgba(16,185,129,.65)}

    /* Cards */
    .glass{
        background: rgba(255,255,255,.05);
        border: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 10px 30px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.04);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .section-h {font-size:1.05rem; font-weight:600; opacity:.95; margin-bottom:.35rem}
    .chip{display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .55rem; border-radius:999px; font-size:.78rem;
            border:1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.06);}

    /* History rows */
    .row{padding:.5rem 0;}
    .row + .row{border-top:1px solid rgba(255,255,255,.08)}

    /* Mobile/tablet: add a bit more left gutter for comfort */
    @media (max-width: 992px){
        [data-testid="stAppViewContainer"] .main .block-container { padding-left: 1rem !important; }
        .hero-wrap{ margin: 0 -1rem 1rem 0; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Main Function
    params = st.query_params   # Verifying Route Query Parameters
    if params.get("goto") == "review":
        st.switch_page("pages/4_Review_Parameters.py")   

    st.markdown("""
        <div style="text-align: center; padding: 2rem 1rem;">
        <div class="hero-wrap">
        <span class="kicker">Beta Version Private Preview - Version 1.2</span> 
            <h1 class="hero">Accelerate Contract Analysis with Innovative Turbocharged AI</h1>
        <p class="sub">Upload Comparison Baseline ➝ Upload for Review ➝ Render real-time differences, clause risks, and revision suggestions.</p>
        </div>
        """, unsafe_allow_html=True)


    # Helper Functions
    def _find_by_id(qid):
        for it in st.session_state.search_history:
            if it["id"] == qid: return it
        return None

    # Instructions
    st.markdown('<div id="how"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("### Instructions")
        st.markdown(
            """
            1. **Establish Baseline**: Upload a “Reference Contract”, which the system will store permanently as the baseline for comparison.
            2. **Upload Draft**: Upload a “Document Pending Review”, and the system will perform paragraph-level dynamic differencing and clause-level semantic alignment.
            3. **Smart Analysis**: Enter a query on the feature page (e.g., Liability Cap, Termination Clause) to obtain key differences and revision suggestions.
            4. **History and Pinning**: All analysis operations are recorded, with support for pinning and tagging to facilitate auditing.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # Enterprise Value
    st.markdown('<div id="how"></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown("### Enterprise Value")
        st.markdown("""
                    <style>
                    /* Layout */
                    .main > div { padding-top: 0rem; }
                    .section { max-width: 1200px; margin: 0 auto; padding: 2rem 1rem 4rem 1rem; }

                    /* Features grid */
                        .features {
                        display: grid;
                        grid-template-columns: repeat(4, minmax(0, 1fr));
                        gap: 0.7rem;
                        margin-top: -0.4rem;
                    }
                    @media (max-width: 980px) {
                        .features { grid-template-columns: repeat(2, minmax(0, 1fr)); }
                    }
                    @media (max-width: 560px) {
                        .features { grid-template-columns: 1fr; }
                    }
                    .card {
                        background: linear-gradient(90deg, #0B2343 0%, #0A7C4A 100%);
                        border-radius: 18px; padding: 1.35rem; height: 100%;
                        box-shadow: 0 6px 16px rgba(11,35,67,.06);
                    }
                    .card h3 {
                    margin: .6rem 0 .35rem 0;
                    font-size: 1.1rem; line-height: 1.2; color: #0b2343;
                    color: white;
                    }
                    .card p { margin: 0; font-size: 0.9rem; color: white; }

                    /* Simple icon pill */
                    .icon {
                    width: 42px; height: 42px; display:grid; place-items:center;
                    border-radius: 10px; background: #eef5ff; color: #0b2343; font-size: 1.2rem;
                    box-shadow: inset 0 0 0 1px rgba(11,35,67,.06);
                    }""", unsafe_allow_html=True)
        
    st.markdown("""
    <div class="features">
    <div class="card">
        <div class="icon">🔘</div>
        <h3>Comprehensive Clause Tracking</h3>
        <p>Enable the creation of customized AI models to monitor all issues of importance to your organization.</p>
    </div>

    <div class="card">
        <div class="icon">🌐</div>
        <h3>Real-time Access to AI Insights</h3>
        <p>Enable a Comprehensive Smart Contract Database with Speed and Ease.</p>
    </div>

    <div class="card">
        <div class="icon">🧭</div>
        <h3>Eliminate Manual Processes</h3>
        <p>Standardized and optimized processes minimize errors, expedite review and approval, and ensure that documentation adheres to regulatory and internal control requirements.</p>
    </div>

    <div class="card">
        <div class="icon">📈</div>
        <h3>Robust Internal Controls & Oversight</h3>
        <p>Compliance First, Results Oriented — Empowering Every Contract to Create Greater Value.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        """
        <div style="opacity:.5; text-align:center; padding:1.1rem 0;">
          Built with Streamlit. Designed for Legal Ops. ©2025 Ernst & Young LLP. All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

# Main Logic
draw_main_app()
