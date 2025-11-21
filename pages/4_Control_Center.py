import os, tempfile, json
from datetime import datetime
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from dotenv import load_dotenv

# LangChain / Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Page Setting
st.set_page_config(page_title="Control Center", layout="wide")
st.logo("logo.png")

load_dotenv()
INDEX_NAME = "contract-assistant"

# Session Default Parameters
if "search_history" not in st.session_state:
    st.session_state.search_history = [
        {"id":"q-001","query":"Starlux Airlines Contract Analysis","timestamp":"2025-06-28T10:34:00",
         "results":[{"title":"Q2 Summary","snippet":"YOY Revenue Growth 12%","path":"reports/q2_2024.pdf"}],
         "tags":["finance","q2"],"pinned":True,"notes":"for Board Presentation"},
        {"id":"q-002","query":"Customer Churn Dashboard","timestamp":"2025-07-03T14:09:20",
         "results":[{"title":"Cohort Analysis","snippet":"Jan–Jun","path":"dashboards/churn.html"}],
         "tags":["product","retention"],"pinned":False,"notes":""},
    ]
if "processed_namespaces" not in st.session_state:
    st.session_state.processed_namespaces = []
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = None
if "core_points_text" not in st.session_state:
    st.session_state.core_points_text = ""
if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = None
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 256

# Helper Functions
def _find_by_id(qid):
    for it in st.session_state.search_history:
        if it["id"] == qid: return it
    return None

def _df(items):
    return pd.DataFrame([{
        "id": it["id"],
        "query": it["query"],
        "timestamp": it["timestamp"],
        "tags": ", ".join(it.get("tags", [])),
        "pinned": it.get("pinned", False),
        "top_title": (it["results"][0]["title"] if it.get("results") else ""),
        "top_path": (it["results"][0]["path"] if it.get("results") else ""),
    } for it in items])

@st.cache_resource
def get_pinecone_client():
    return Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def fetch_pinecone_namespaces(index_name):
    pc = get_pinecone_client()
    try:
        stats = pc.describe_index(index_name).stats
        return list(stats.namespaces.keys()) if stats and stats.namespaces else []
    except Exception:
        return []

if not st.session_state.processed_namespaces:
    st.session_state.processed_namespaces = fetch_pinecone_namespaces(INDEX_NAME)

def process_and_ingest_reference_file(uploaded_file):
    namespace = uploaded_file.name
    with st.spinner(f"Processing reference document ‘{namespace}’ and storing it in the persistent knowledge base…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name
        loader = PyPDFLoader(path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        docs = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        PineconeVectorStore.from_documents(docs, embedding=embeddings,
                                           index_name=INDEX_NAME, namespace=namespace)
        os.remove(path)
    st.success(f"Reference document ‘{namespace}’ has been successfully stored in the knowledge base!")
    if namespace not in st.session_state.processed_namespaces:
        st.session_state.processed_namespaces.append(namespace)
    st.cache_data.clear()

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_uploaded_file.getvalue())
        path = tmp.name
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(split_docs, embeddings)
    os.remove(path)
    return vs.as_retriever(search_kwargs={'k': 2})

def run_comparison(template_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    tpl = """
        You are a top legal expert. Your task is to precisely compare the same clause in two contracts, with the highest priority being to protect our company’s interests.
        **Our company’s standard template clause:**
        ```{template_clause}```
        **Corresponding clause in the submitted document:**
        ```{uploaded_clause}```
        Please focus on {topic} as the review point and complete the following tasks:
        1. **Clause Summary**, 2. **Differential Analysis**, 3. **Risk Alerts and Recommendation**。
        Present your analysis report clearly in Markdown format.
        """
    prompt = PromptTemplate.from_template(tpl)
    chain = prompt | llm | StrOutputParser()
    results = {}
    progress = st.progress(0, text="Initiate Comparison...")
    for i, topic in enumerate(review_points):
        t_docs = template_retriever.invoke(topic)
        u_docs = uploaded_retriever.invoke(topic)
        t_text = "\n---\n".join([d.page_content for d in t_docs])
        u_text = "\n---\n".join([d.page_content for d in u_docs])
        results[topic] = chain.invoke({"topic": topic,
                                       "template_clause": t_text,
                                       "uploaded_clause": u_text})
        progress.progress((i + 1) / len(review_points), text=f"Analyzing: {topic}")
    progress.empty()
    return results

# UI
st.header("Control Center")
pins = sum(1 for it in st.session_state.search_history if it.get("pinned"))

# Menu Tabs
sub = option_menu(
    None, ["Query History", f"Pinned ({pins})", "Other Tools"],
    icons=["clock-history", "pin-angle-fill", "wrench-adjustable-circle"],
    menu_icon="cast", default_index=0, orientation="horizontal",
)

# Helper Functions ("Query History", "Pinned")
def _toggle_pin(_id: str):
    item = _find_by_id(_id)
    if not item:
        return
    item["pinned"] = not item.get("pinned", False)
    st.session_state[f"pin_{_id}"] = item["pinned"]  # Keep the UI toggle state consistent when re-executing.

# Query History
if sub == "Query History":
    q = st.text_input("Search using Keywords")
    c1, c2 = st.columns([3, 1])
    tag_filter = c1.text_input("Filter by Tag")
    only_pinned = c2.toggle("Pinned Only", value=False)

    items = list(st.session_state.search_history)  

    # Text filter
    if q:
        ql = q.lower()
        items = [it for it in items if ql in it.get("query", "").lower()]

    # Tag filter (case-insensitive)
    if tag_filter:
        tf = tag_filter.lower()
        items = [
            it for it in items
            if tf in [str(t).lower() for t in it.get("tags", [])]
        ]

    # Pinned-only filter
    if only_pinned:
        items = [it for it in items if it.get("pinned")]

    # Newest first
    items = sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True)

    if not items:
        st.info("No matching history.")
    else:
        for it in items:
            ts_raw = it.get("timestamp", "")
            try:
                ts_disp = datetime.fromisoformat(ts_raw).strftime("%b %d, %Y %H:%M")
            except Exception:
                ts_disp = ts_raw or "—"

            with st.expander(f"**{it.get('query','(no query)')}** · {ts_disp}"):
                st.toggle(
                    "Pinned",
                    key=f"pin_{it['id']}",
                    value=it.get("pinned", False),
                    on_change=_toggle_pin,
                    args=(it["id"],),
                )

# Pinned Tab
elif sub.startswith("Pinned"):
    pinned_items = [it for it in st.session_state.search_history if it.get("pinned")]
    if pinned_items:
        df = _df(pinned_items)
        df = df.drop(columns=["pinned"], errors="ignore")
        if "timestamp" in df.columns:     # If the DataFrame contains a timestamp column, display the most recent data first.
            try:
                df = df.sort_values("timestamp", ascending=False)
            except Exception:
                pass
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No pinned items yet.")

# Tools Tab
elif sub == "Other Tools":
    st.markdown("##### Quick Exports")

    items = list(st.session_state.search_history)
    all_df = _df(items)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    c1, c2, c3 = st.columns([1, 1, 1])  #equal width columns

    # Export All (JSON)
    c1.download_button(
        "Export all (JSON)",
        json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"search_history_{ts}.json",
        mime="application/json",
        disabled=not items,
        use_container_width=True
    )

    # Export All (CSV)
    c2.download_button(
        "Export all (CSV)",
        (all_df.to_csv(index=False).encode("utf-8") if not all_df.empty else b""),
        file_name=f"search_history_{ts}.csv",
        mime="text/csv",
        disabled=all_df.empty,
        use_container_width=True
    )

    # Export Pinned (CSV)
    pinned_items = [it for it in items if it.get("pinned")]
    pinned_df = _df(pinned_items)

    c3.download_button(
        "Export pinned (CSV)",
        (pinned_df.to_csv(index=False).encode("utf-8") if not pinned_df.empty else b""),
        file_name=f"search_history_pinned_{ts}.csv",
        mime="text/csv",
        disabled=pinned_df.empty,
        use_container_width=True
    )

    st.markdown("##### Maintenance")
    m1, m2, m3 = st.columns(3)

    # Clear All History
    if m1.button("Clear all history", use_container_width=True, type="secondary", disabled=not items):
        st.session_state.search_history = []
        st.success("History cleared. Rerun to see changes.")
        st.stop()

    # Unpin All
    if m2.button("Unpin all", use_container_width=True, type="secondary", disabled=not pinned_items):
        for it in st.session_state.search_history:
            it["pinned"] = False
        st.success("All items unpinned.")

    # Provide a small preview so users can see the content they are about to export.
    with st.expander("Preview (first 100 rows)"):
        if all_df.empty:
            st.info("No history to preview.")
        else:
            preview_df = all_df.copy()
            preview_df = preview_df.drop(columns=["pinned"], errors="ignore")
            st.dataframe(preview_df.head(100), use_container_width=True, hide_index=True)

# Results
if st.session_state.get("comparison_results"):
    st.subheader("Contract Comparison Analysis Report")
    for topic, result in st.session_state.comparison_results.items():
        with st.expander(f"**審查項目：{topic}**", expanded=True):
            st.markdown(result, unsafe_allow_html=True)
    st.session_state.comparison_results = None
