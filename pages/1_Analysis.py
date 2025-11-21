import os
import tempfile
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# LangChain/Vector DB
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader, TextLoader
# #from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser
# from langchain_community.vectorstores import FAISS
# from langchain.retrievers.multi_query import MultiQueryRetriever
# LangChain / Vector DB
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


# Import Utility Libraries for S3 Storage and Pinecone Learning
import storage_utils as storage
from utils import ingest_docs_to_pinecone

# Page Settings
st.set_page_config(page_title="AI-Assisted Contract Review and Risk Analysis", layout="wide")
st.logo("logo.png")

load_dotenv()

# Core Config for AI Training 
LEARNING_NAMESPACE = "approved-analyses"
INDEX_NAME = "contract-assistant"

# Initialize Session State for Model Parameters
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3     
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 3072 

# Helper Functions
@st.cache_data(show_spinner=False)
def get_language(text_snippet: str, _llm):
    """Use LLM to quickly detect the text language"""
    if not text_snippet.strip():
        return "unknown"
    try:
        prompt = PromptTemplate.from_template("Detect the primary language of the following text. Respond with only the two-letter ISO 639-1 code (e.g., 'en' for English, 'zh' for Chinese). Text: ```{text}```")
        chain = prompt | _llm | StrOutputParser()
        sample = text_snippet[:200]
        lang_code = chain.invoke({"text": sample})
        return lang_code.lower()
    except Exception:
        return "unknown"

@st.cache_data(show_spinner=False)
def translate_to_chinese(text_to_translate: str, _llm):
    """Use LLM to translate text into Traditional Chinese"""
    if not text_to_translate.strip():
        return ""
    try:
        prompt = PromptTemplate.from_template("Please translate the following legal text into Traditional Chinese. Only return the translated text, without any explanation or preamble. Text: ```{text}```")
        chain = prompt | _llm | StrOutputParser()
        return chain.invoke({"text": text_to_translate})
    except Exception as e:
        st.markdown(f"<span style='color:white'>An error occurred during translation: {e}</span>", unsafe_allow_html=True)
        return text_to_translate

# Refactored Core Comparison Function
def run_comparison(template_retriever, uploaded_retriever, review_points, temperature, max_tokens):
    """Execute contract comparison using a two-step generation process (analyze then summarize)"""
    llm = ChatOpenAI(model_name='gpt-4o', temperature=temperature, max_tokens=max_tokens)
    
    # Step 1: Optimized high-quality detailed report prompt
    tpl = """
    **Role:** You are a seasoned Senior Legal Counsel at EY. Your primary duty is to protect EY's interests. Your review must be commercially-aware, risk-focused, and provide immediately actionable advice for our internal non-lawyer project teams.
    **Objective:** Conduct a detailed preliminary review of a counterparty's contract clause ("Clause B") against our standard template ("Clause A") on the specific topic of "**{topic}**". Assume "Our Company" is EY.

---
**Context 1: Past High-Quality Analysis Examples (for style and depth reference)**
```{approved_examples}```
---
**Context 2: Clause A (Our Company's Standard Template - Normalized to Traditional Chinese)**
```{clause_A}```
---
**Context 3: Clause B (Counterparty's Draft - Normalized to Traditional Chinese)**
```{clause_B}```
---

**Task & Formatting Rules:**
1.  **Language:** The entire report MUST be written in the language of original clause.
2.  **Headings:** Use Markdown level 3 headings (`###`) for the two main sections (e.g., `### 1. Core differences and risks to our company (EY).`).
3.  **Bullet Points:** Use a single dash (`- `) for all bullet points. Do not use asterisks (`*`) or circles (`o`).
4.  **Content:** Address all points with insightful, concise analysis based on the provided clauses.

### 1. Key Differences & Risks to EY
-   **Material Differences**: Directly compare Clause A and B. Instead of just listing facts, synthesize the differences.
    * *Good Example:* `The counterparty’s draft extends the confidentiality obligation to 5 years after contract termination, whereas our template specifies only 2 years. This significantly increases our company’s long-term compliance costs and legal risks.`
    * *Bad Example:* `Clause A is 2 years, Clause B is 5 years.`
-   **Potential Risks to EY**: For each key difference, explicitly state the commercial, legal, or operational risk. Frame it as "This exposes us to the risk of...". Be specific to EY's business model (e.g., regulatory duties, data handling, global firm structure).

### 2. Revision & Negotiation Strategy
-   **Primary Redline Suggestion**: Provide a direct, copy-pasteable revision to Clause B to mitigate the risks. If no change is truly needed, state "建議接受 (Acceptable as is)".
-   **Negotiation Strategy & Bottom Line**:
    * **Goal:** Clearly state our main goal (e.g., "The primary objective is to shorten the confidentiality period to no more than 3 years.").
    * **Rationale:** Provide a brief, commercially-sound reason we can use in negotiations (e.g., "Explain to the counterparty that 2–3 years is the industry standard, and an excessively long period is disproportionate and increases management costs for both parties.").
    * **Fallback Position:** Offer a potential compromise if our primary suggestion is rejected (e.g., "If the counterparty insists on 5 years, we can accept, but we will require the addition of an exemption clause stating: “This does not include data that our company must retain in order to comply with legal or professional standards.").
"""
    analysis_chain = PromptTemplate.from_template(tpl) | llm | StrOutputParser()
    
    detailed_results = {}
    progress = st.progress(0, text="Conducting in-depth contract review....")

    # mq_template_retriever = MultiQueryRetriever.from_llm(retriever=template_retriever, llm=llm)
    # mq_uploaded_retriever = MultiQueryRetriever.from_llm(retriever=uploaded_retriever, llm=llm)

    compressor = LLMChainExtractor.from_llm(llm)
    mq_template_retriever = ContextualCompressionRetriever(
        base_retriever=template_retriever,
        base_compressor=compressor
    )
    mq_uploaded_retriever = ContextualCompressionRetriever(
        base_retriever=uploaded_retriever,
        base_compressor=compressor
    )

    
    # Execute Analysis Loop
    for i, topic in enumerate(review_points):
        display_topic = topic.split(' (')[0].replace('&nbsp;', ' ').strip()
        search_query = topic.replace('&nbsp;', ' ')
        progress.progress((i + 0.5) / len(review_points), text=f"Analysis in Progress: {display_topic}")
        a_text = "No relevant examples found."

        t_docs = mq_template_retriever.get_relevant_documents(search_query)
        u_docs = mq_uploaded_retriever.get_relevant_documents(search_query)
        
        t_text_original = "\n---\n".join([d.page_content for d in t_docs])
        u_text_original = "\n---\n".join([d.page_content for d in u_docs])
        
        lang_t = get_language(t_text_original, llm)
        lang_u = get_language(u_text_original, llm)
        
        with st.spinner(f"Language normalization in progress ({display_topic})..."):
            t_text_final = translate_to_chinese(t_text_original, llm) if 'en' in lang_t else t_text_original
            u_text_final = translate_to_chinese(u_text_original, llm) if 'en' in lang_u else u_text_original

        if not t_text_final.strip(): t_text_final = "No relevant clause found in the document."
            
        report = analysis_chain.invoke({
            "topic": display_topic,
            "approved_examples": a_text,
            "clause_A": t_text_final,
            "clause_B": u_text_final
        })
        detailed_results[topic] = report
        
    # Step 2: Produce a high-quality summary after all detailed reports are generated
    progress.progress(1.0, text="Refining the overall risk summary...")

    full_detailed_report_context = "\n\n---\n\n".join(
        f"### Review Items：{topic.split(' (')[0]}\n\n{report}" 
        for topic, report in detailed_results.items()
    )

    # Prompt (for generating summaries from high-quality reports)
    final_summary_tpl = """
    You are a top legal associate. Your task is to read the following full itemized review report and prepare an extremely concise Overall Risk Summary for senior executives.

    **Task Instructions:**
    1.  **Focus on the core essence**: From each report, extract the most important core differences and risks as well as the most critical preferred revision recommendations.
    2.  **Results-oriented**: The summary should be clear and direct, enabling readers to immediately grasp the issues and proposed solutions.
    3.  **Strict formatting**: Your answer must be only one line of text. Use ||| to separate “Topic”, “Key Differences & Risks”, and “Core Revision Suggestions”. Use ;;; to separate different review items. Within each field, you may use Markdown bullet points (- ) and line breaks (\n).

    **Example Format:**
    `Confidentiality Period ||| - Counterparty draft sets 5 years, significantly increasing our long-term compliance risk.\n- Commencement point is after contract termination, unfavorable to us. ||| - Recommend shortening to EY’s 2-year standard.\n- Recommend changing commencement point to “date of disclosure.” ;;; Definition of Confidential Information ||| - Counterparty definition is overly broad and may include publicly available information. ||| - Recommend adding the five standard exceptions from our template.`

    ---
    **Full itemized review report:**
    ```{full_report}```
    ---
    **Please immediately generate summary content that meets all of the above requirements:**
    """
    final_summary_prompt = PromptTemplate.from_template(final_summary_tpl)
    # Use LLM with a lower temperature (recommended 0.1–0.3) to ensure summary stability for testing
    summary_llm = ChatOpenAI(model_name='gpt-4o', temperature=0.1, max_tokens=2048)
    summary_chain = final_summary_prompt | summary_llm | StrOutputParser()
    
    summary_raw = summary_chain.invoke({"full_report": full_detailed_report_context})
    
    # Parse the summary in the new format
    summary_points = []
    try:
        items = summary_raw.strip().split(';;;')
        for item in items:
            if not item.strip(): continue
            parts = item.strip().split('|||')
            if len(parts) == 3:
                topic, difference, suggestion = [p.strip().replace('\\n', '\n') for p in parts]
                summary_points.append({
                    'topic': topic,
                    'difference': difference,
                    'suggestion': suggestion
                })
            else: # If the format does not match, perform basic fallback processing
                 summary_points.append({'topic': item, 'difference': 'Format parsing failed.', 'suggestion': 'Please review the detailed report.'})
    except Exception as e:
        st.error(f"An error occurred while generating the summary: {e}")
        summary_points.append({'topic': 'Summary generation failed', 'difference': 'Unable to parse AI response', 'suggestion': str(e)})

    progress.empty()
    return {"summary": summary_points, "details": detailed_results}

@st.cache_resource
def load_and_process_pdf_for_faiss(_uploaded_file):
    if not _uploaded_file:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(_uploaded_file.getvalue())
        path = tmp.name
    loader = PyPDFLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    if not split_docs:
        os.remove(path)
        st.info(f"No processable text content found in document ‘{_uploaded_file.name}’.")
        return None
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_documents(split_docs, embeddings)
    os.remove(path)
    return vs.as_retriever(search_kwargs={'k': 3})

def process_and_store_reference_file(uploaded_file):
    filename = uploaded_file.name
    with st.spinner(f"Processing reference document ‘{filename}’ and loading into memory..."):
        retriever = load_and_process_pdf_for_faiss(uploaded_file)
        if retriever:
            st.session_state.reference_retrievers[filename] = retriever
            st.success(f"Reference document ‘{filename}’ has been successfully loaded!")

# UI
if "reference_retrievers" not in st.session_state:
    st.session_state.reference_retrievers = {}

st.header("AI-Assisted Contract Review and Risk Analysis")

CORE_REVIEW_POINTS = [
    "Confidentiality Period",
    "Definition of Confidential Information",
    "Permitted Disclosures",
    "Governing Law and Jurisdiction",
    "Return or Destruction of Information",
    "Remedies for Breach",
    "Intellectual Property Rights",
    "Notice of Breach and Cure Period",
    "Limitation of Liability",
    "Force Majeure",
    "Subcontracting Restrictions",
    "Professional Indemnity"
]
with st.expander("Customize Review Parameters", expanded=True):
    cols = st.columns(4)
    for i, point in enumerate(CORE_REVIEW_POINTS):
        with cols[i % 4]:
            st.toggle(point.split(" (")[0], value=True, key=point)
    st.text_area("Add Review Items (one per line)：", key="core_points_text", height=100)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Step 1: Upload Reference Document")
    new_ref_file = st.file_uploader("Upload a PDF as the New Comparison Baseline", type="pdf", key="ref_uploader_faiss")
    if st.button("Process and Load Reference Document"):
        if new_ref_file:
            process_and_store_reference_file(new_ref_file)
        else:
            st.info("Please select a reference document first.")
with col2:
    st.subheader("Step 2: Select Comparison Baseline")
    processed_files = list(st.session_state.reference_retrievers.keys())
    selected_index = None
    if st.session_state.get("selected_namespace") in processed_files:
        selected_index = processed_files.index(st.session_state.get("selected_namespace"))
    selected = st.selectbox(
        "Select a reference document from those already uploaded:",
        options=processed_files,
        index=selected_index,
        placeholder="Please select..."
    )
    if selected is not None: 
        st.session_state.selected_namespace = selected
st.divider()

st.subheader("Step 3: Configure AI Analysis Parameters")
st.session_state.temperature = st.slider(
    "Model Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
    help='Lower values produce more specific and consistent results, while higher values yield more creative and diverse outcomes. A value between 0.1 and 0.4 is recommended for stable and insightful analysis.'
)
st.session_state.max_tokens = st.slider(
    "Max Tokens", 512, 4096, st.session_state.max_tokens, 128,
    help='Limit the length of a single AI response. Since detailed reports may be extensive, it is recommended to set the value above 3000 to avoid report truncation.'
)
st.divider()

st.subheader("Step 4: Upload the Document for Review and Run Analysis")
selected_namespace = st.session_state.get("selected_namespace")
if not selected_namespace:
    st.info("Please upload a reference document in Step 1 and select one as the comparison baseline in Step 2.")
else:
    st.success(f"The current comparison baseline is: {selected_namespace}")

target_file = st.file_uploader("Upload the contract document you want to review", type="pdf", key="target_uploader")

if target_file: 
    st.session_state.target_file_name = target_file.name

start_button = st.button("Initialize In-Depth AI Review", type="primary", use_container_width=True, disabled=(not target_file or not selected_namespace))

if start_button:
    with st.spinner("Preparing comparison environment..."):
        template_retriever = st.session_state.reference_retrievers[selected_namespace]
        uploaded_retriever = load_and_process_pdf_for_faiss(target_file)
        
    if not uploaded_retriever:
        st.info("The review document failed to process or is empty. Please re-upload.")
    else:
        temp = st.session_state.temperature
        max_tok = st.session_state.max_tokens
        
        active_review_points = [p for p in CORE_REVIEW_POINTS if st.session_state.get(p, True)]
        custom_points = [line.strip() for line in st.session_state.get("core_points_text", "").split('\n') if line.strip()]
        final_review_points = active_review_points + custom_points

        if not final_review_points:
            st.info("Please select or add at least one review item.")
        else:
            st.session_state.comparison_results = run_comparison(template_retriever, uploaded_retriever, final_review_points, temp, max_tok)
            st.rerun()

# Report Display & Storage Functionality
if st.session_state.get("comparison_results"):
    st.balloons()
    st.header("AI in-depth review report has been completed")
    st.info("You can review the summary and report below. If you find the report to be of high quality, you may archive it for AI retraining.")
    st.divider()

    st.subheader("Risk Summary Overview")
    
    summary_data = st.session_state.comparison_results.get('summary', [])
    details_data = st.session_state.comparison_results.get('details', {})

    full_report_md = "# AI Contract Review Report\n\n"
    
    summary_table_md = "| **Item** | **Key Differences & Risk** | **Core Revision Recommendations** |\n"
    summary_table_md += "|:---|:---|:---|\n"
    for item in summary_data:
        # Handling line breaks
        topic_display = item.get('topic', 'N/A')
        difference_display = item.get('difference', '').replace('\n', '<br>')
        suggestion_display = item.get('suggestion', '').replace('\n', '<br>')
        summary_table_md += f"| {topic_display} | {difference_display} | {suggestion_display} |\n"
    
    st.markdown(summary_table_md, unsafe_allow_html=True)
    full_report_md += "## Risk Summary Overview\n\n" + summary_table_md.replace('<br>', '\n') + "\n\n"
    st.divider()
    
    st.subheader("Itemized Review Report")
    full_report_md += "## Itemized Review Report\n\n"
    for topic, report_md in details_data.items():
        display_topic = topic.split(' (')[0].replace('&nbsp;', ' ').strip()
        with st.expander(f"**Review Items：{display_topic}**", expanded=False):
            st.markdown(report_md, unsafe_allow_html=True)
        full_report_md += f"### Review Items：{display_topic}\n\n{report_md}\n\n---\n\n"
        
    st.divider()
    
    st.subheader("Analysis Archiving and AI Retraining")
    st.markdown("If you approve the quality of this report’s analysis, you can click the button below. The system will archive it to Amazon S3 and feed its content to the AI as a complete “best practice” example for training.")

    if st.button("I approve the quality of this report. Archive it to the cloud and use it for AI training.", type="primary", use_container_width=True):
        template_name = st.session_state.get("selected_namespace", "template").replace('.pdf', '')
        target_name = st.session_state.get("target_file_name", "target").replace('.pdf', '')
        timestamp = datetime.now().strftime('%Y%m%d')
        
        storage_filename = f"Approved_Report_{template_name}_vs_{target_name}_{timestamp}.md"
        
        upload_success = storage.upload_report_to_storage(full_report_md, filename=storage_filename)

        if upload_success:
            try:
                with st.spinner(f"Converting report knowledge into AI long-term memory..."):
                    learning_content = f"【Best Practice Case: Contract Review Report - {template_name} vs {target_name}】\n\n{full_report_md}"
                    
                    with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8", suffix=".txt") as tmp_file:
                        tmp_file.write(learning_content)
                        tmp_file_path = tmp_file.name
                    
                    loader = TextLoader(tmp_file_path)
                    documents = loader.load()
                    ingest_docs_to_pinecone(documents, INDEX_NAME, LEARNING_NAMESPACE)
                    os.remove(tmp_file_path)
                    
                    st.success(f"AI has successfully learned the analysis pattern from this report!")
                    
                    st.session_state.comparison_results = None
                    st.info("The page is about to refresh...")
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred during the AI learning process: {e}")
