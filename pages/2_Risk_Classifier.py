import os
import re
import io
import json
import time
import textwrap
from typing import List, Literal, Optional

import streamlit as st
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from dotenv import load_dotenv
import fitz

# Assume the `text_splitter` module is located in the project root directory.
from text_splitter import smart_split

# LangChain, spaCy (enables language detection and precise extraction)
from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
import spacy

import Risk_Knowledge

load_dotenv()
st.set_page_config(page_title="Contract Risk Classifier", layout="wide")
st.logo("logo.png")

MODEL_NAME = "gpt-4o"
EXTRACTION_MODEL_NAME = "gpt-4o-mini"

# Pydantic Schema (for Data Validation)
class ClauseRisk(BaseModel):
    """Data structure, including the fields: risk, reason, and suggestion."""
    clause: str
    risk: Literal["HIGH", "MEDIUM", "NOTICEABLE"]
    risk_sentence: Optional[str] = Field(None)
    reason: str
    suggestion: Optional[str] = Field(None, description="A complete, revised version of the clause text to mitigate risk.")
    tags: List[str] = Field(default_factory=list)

RISK_RUBRIC = Risk_Knowledge.get_risk_rubric_string()

# Prompts
SYSTEM_PROMPT_STAGE1 = f"""
You are a senior legal counsel specializing in contract review. Your sole task is to analyze a **single given contract clause**, classify its risk level, and provide a complete, rewritten version of the clause that is ready for use.

**Analysis Steps (Follow Strictly):**
1.  **Analyze ONLY the Provided Text**: Your entire analysis **MUST** be based **exclusively** on the text of the clause given to you. **It is strictly forbidden to invent, assume, or refer to other clause numbers or topics** (e.g., 'Article 5' or 'indemnification') unless they are explicitly written in the provided text. Your analysis must directly correspond to the content of the input clause.
2.  **Consult Rubric**: Compare the clause against the `Risk Classification Rubric` provided below.
3.  **Determine Risk Level**:
    - Classify as **"HIGH"** or **"MEDIUM"** if it matches a corresponding risk description in the rubric.
    - Classify as **"NOTICEABLE"** only if it does NOT match High/Medium risks but pertains to standard matters (e.g., governing law, confidentiality period).
4.  **Formulate Reason**: Write a concise, clear explanation for your risk classification.
5.  **Provide Full Revision (Crucial Task)**:
    - Provide the revised clause in the **same language** as the original clause.
    - For any "HIGH" or "MEDIUM" risk clause, you **MUST** provide a **complete, standalone, and rewritten version of the clause**. This rewritten clause should mitigate all identified risks and be ready to replace the original text. It is not just a comment, but the full revised text.
    - For "NOTICEABLE" clauses or if the original text is already acceptable, respond with the exact phrase "No Modification Required".
6.  **Extract Keywords**: Identify 2-4 relevant keywords from the clause.
7.  **Construct JSON**: Assemble your entire analysis into a SINGLE, valid JSON object.

**Risk Classification Rubric:**
---
{RISK_RUBRIC}
---

**Output Format Rules (Strictly Enforced):**
- Your output **MUST** be a single, valid JSON object.
- The JSON keys must be **EXACTLY**: `risk`, `reason`, `suggestion`, `tags`.
- `risk` must be one of "HIGH", "MEDIUM", or "NOTICEABLE".
- `reason` must be in English.
- `suggestion` **MUST** contain the full, revised clause text, or the exact phrase "No Modification Required".
- Do **NOT** include the original `clause` in your JSON response.
"""

SYSTEM_PROMPT_STAGE2 = """
You are a legal text analysis assistant. Given a full clause and a reason for its risk, extract the **exact, single sentence (or at most two)** that is the primary source of the risk.
Respond with ONLY the extracted sentence(s). No explanation, no preamble, no quotes.
"""

# Helpers Functions
def extract_text_from_pdf(file_bytes_io: io.BytesIO) -> str:
    """ Use PyMuPDF (fitz) for text extraction. This ensures that the same engine is used for both text extraction and subsequent annotation searches"""
    doc = fitz.open(stream=file_bytes_io, filetype="pdf")
    full_text = []
    for page in doc:
        full_text.append(page.get_text() or "")
    doc.close()
    return "\n".join(full_text)

@st.cache_data(show_spinner=False)      # Identifies primary langauge of the text input
def get_language(text_snippet: str) -> str:
    if not text_snippet.strip(): return "en"
    try:
        llm = ChatOpenAI(model_name=EXTRACTION_MODEL_NAME, temperature=0)
        prompt = PromptTemplate.from_template("Detect the primary language (ISO 639-1 code, 'en' or 'zh') of this text: ```{text}```")
        chain = prompt | llm | StrOutputParser()
        lang_code = chain.invoke({"text": text_snippet[:500]}).lower()
        return 'zh' if 'zh' in lang_code else 'en'
    except Exception:
        return "en"

# Clean risk data: standardize the risk level fields and label formats. If the format is invalid, append an error label.
def sanitize_risk_data(risk_data: dict) -> dict:
    if "tags" not in risk_data or not isinstance(risk_data.get("tags"), list):
        risk_data["tags"] = []
    raw_risk = str(risk_data.get("risk", "MEDIUM")).upper()
    if "HIGH" in raw_risk:
        risk_data["risk"] = "HIGH"
    elif "MEDIUM" in raw_risk:
        risk_data["risk"] = "MEDIUM"
    elif "NOTICEABLE" in raw_risk or "LOW" in raw_risk or "STANDARD" in raw_risk:
        risk_data["risk"] = "NOTICEABLE"
    else:
        risk_data["risk"] = "MEDIUM"
        risk_data["tags"].append("risk_parse_failed")
    return risk_data

# Perform risk classification on clauses, with a two-stage process: 
# first retrieve risk information, then extract risk-related sentences, and organize them into a standardized structure.
def classify_clause(client: OpenAI, clause: str) -> ClauseRisk:
    resp_stage1 = client.chat.completions.create(
        model=MODEL_NAME, temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT_STAGE1}, {"role": "user", "content": clause}],
        response_format={"type": "json_object"},
    )
    if not resp_stage1.choices or not resp_stage1.choices[0].message or not resp_stage1.choices[0].message.content:
        raise ValueError("AI model returned an empty or invalid response in Stage 1.")
    content_str = resp_stage1.choices[0].message.content
    try:
        match = re.search(r"\{.*\}", content_str, re.DOTALL)
        if match:
            risk_data = json.loads(match.group(0))
        else:
            risk_data = json.loads(content_str)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to parse JSON from model response: {content_str}")

    risk_data = sanitize_risk_data(risk_data)
    risk_data["clause"] = clause
    risk_data.setdefault("risk_sentence", None)
    risk_data.setdefault("suggestion", "Unable to generate suggestions.")

    if risk_data.get("risk") in ["HIGH", "MEDIUM"]:
        user_prompt_stage2 = f"Full Clause:\n```\n{clause}\n```\n\nReason for Risk:\n{risk_data.get('reason', '')}"
        resp_stage2 = client.chat.completions.create(
            model=EXTRACTION_MODEL_NAME, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT_STAGE2}, {"role": "user", "content": user_prompt_stage2}],
        )
        if resp_stage2.choices and resp_stage2.choices[0].message and resp_stage2.choices[0].message.content:
            risk_sentence = resp_stage2.choices[0].message.content.strip()
            if risk_sentence and risk_sentence in clause:
                risk_data["risk_sentence"] = risk_sentence
    return ClauseRisk.model_validate(risk_data)

# Enable batch processing for multiple clause risk classifications, with progress tracking and error handling to ensure uninterrupted workflow.
def classify_batch(clauses: List[str]) -> List[ClauseRisk]:
    client = OpenAI()
    out: List[ClauseRisk] = []
    total = max(len(clauses), 1)
    progress = st.progress(0, text="Initializing AI analysis...")
    for i, c in enumerate(clauses, start=1):
        progress.progress(min(i / total, 1.0), text=f"Analyzing clause {i} of {total}...")
        try:
            out.append(classify_clause(client, c))
        except (ValueError, ValidationError) as e:
            out.append(ClauseRisk(clause=c, risk="NOTICEABLE", reason=f"An error occurred during model analysis or validation: {str(e)[:100]}", suggestion="N/A", tags=["error", "parsing_failed"]))
        except Exception as e:
            out.append(ClauseRisk(clause=c, risk="NOTICEABLE", reason=f"An unexpected system error occurred: {str(e)[:100]}", suggestion="N/A", tags=["error", "system_error"]))
        time.sleep(0.05)
    progress.empty()
    return out

def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _candidate_snippets(text: str, min_len: int = 15, max_len: int = 100) -> List[str]:
    """
    Newly upgraded text-snippet generation function with enhanced Chinese support.
    1. Punctuation-aware segmentation: Uses full-width Chinese punctuation (e.g., “。；！？”) to achieve more natural sentence breaks.
    2. Sliding-window splitting: If a long sentence contains no punctuation, applies an overlapping sliding window so the entire sentence is covered, rather than only the beginning or end.
    3. Optimized length parameters: Adjusts the minimum and maximum lengths to better fit the information density of Chinese text.
    """
    t = _normalize_spaces(text)
    if not t:
        return []
    if len(t) <= max_len:
        return [t]

    # Prefer splitting sentences using both Chinese and English punctuation marks
    # (?<=[...]) is a "positive lookbehind" in regex, ensuring the punctuation itself is not removed
    sentences = re.split(r'(?<=[。？！；!?;\.])\s*', t)
    valid_sentences = [s.strip() for s in sentences if s and s.strip() and len(s.strip()) >= min_len]

    # If punctuation-based splitting produces multiple sentences, use this result
    if len(valid_sentences) > 1 and any(len(s) < max_len for s in valid_sentences):
         return valid_sentences

    # If punctuation-based splitting is not possible (e.g., an extremely long clause without punctuation),
    # then use the overlapping sliding-window method to ensure full content coverage.
    chunks = []

    # Apply an overlap of 20 characters to ensure smoother annotation continuity.
    overlap = 20
    step = max_len - overlap

    for i in range(0, len(t), step):
        chunk = t[i:i + max_len]
        if chunk and len(chunk.strip()) > min_len:
            chunks.append(chunk.strip())

    return chunks if chunks else [t]

def _inset(rect, margin: float) -> "fitz.Rect":
    return fitz.Rect(rect.x0 + margin, rect.y0 + margin, rect.x1 - margin, rect.y1 - margin)

def build_highlighted_pdf(src_pdf_bytes: bytes, items: List[ClauseRisk], include_resolved: bool = False, resolved_idx: set = None) -> bytes:
    """PDF generation logic ensures that the full revised clauses are included in the annotations."""
    if resolved_idx is None: resolved_idx = set()
    doc = fitz.open(stream=src_pdf_bytes, filetype="pdf")
    not_found = []
    for idx, item in enumerate(items):
        if (not include_resolved and idx in resolved_idx) or item.risk not in ["HIGH", "MEDIUM"]:
            continue
        text_to_search = item.risk_sentence if item.risk_sentence else item.clause
        snippets = _candidate_snippets(text_to_search)

        # Defining a flag to track whether this risk item has already been identified and annotated.
        item_found_and_annotated = False

        for page in doc:
            # Checking all text snippets
            for snip in snippets:
                if not isinstance(snip, str) or len(snip) < 5: continue
                rects = page.search_for(snip, quads=True)

                if rects:
                    annot = page.add_highlight_annot(rects)    # When a segment is detected, execute annotation and insert an associated comment.
                    annot.set_colors(stroke=(1, 0, 0) if item.risk == "HIGH" else (1, 0.55, 0))

                    info_content = f"【Risk Factor】\n{item.reason}"
                    if item.suggestion and item.suggestion != "No changes needed.":
                        info_content += f"\n\n【Suggested revision】\n{item.suggestion}"

                    annot.set_info(content=f"[{item.risk}] {info_content}")
                    annot.update()

                    item_found_and_annotated = True   # Set the flag to True to indicate that this risk item has been processed.
                    break                             # Break out of the innermost snippet loop.
            
            if item_found_and_annotated:     # If this risk item has already been processed, also break out of the mid-level page loop.
                break

        if not item_found_and_annotated:    
            not_found.append(item)           # If the risk item remains unfound after traversing through all pages.

    if not_found:
        summary_page = doc.new_page()
        header = "Clauses Not Found For Highlighting"
        text_blocks = [header, ""]
        for miss in not_found:
            text_blocks.append(f"- {miss.risk} · {', '.join(miss.tags) or 'untagged'}")
            text_blocks.append(f"  Reason: {miss.reason}")
            text_blocks.append("  " + textwrap.shorten(_normalize_spaces(miss.clause), width=220))
            text_blocks.append("")
        inner_rect = _inset(summary_page.rect, 36)
        summary_page.insert_textbox(inner_rect, "\n".join(text_blocks), fontsize=10, align=0,fontname="china-tc")
    out = io.BytesIO()
    doc.save(out, deflate=True, garbage=4)
    doc.close()
    return out.getvalue()

# UI 
st.header("Contract Risk Classifier")
st.markdown("Upload a contract PDF, and the AI will automatically highlight high/medium-risk clauses, precisely locate the risk sentences, and provide analysis explanations along with revision suggestions.")
if "results" not in st.session_state: st.session_state.results = []
if "resolved" not in st.session_state: st.session_state.resolved = set()
if "source_pdf_bytes" not in st.session_state: st.session_state.source_pdf_bytes = None
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader("Please upload your contract (.pdf)", type=["pdf"], key="contract_pdf")
    process_clicked = st.button("Process and perform AI risk analysis", type="primary", use_container_width=True, disabled=uploaded is None)
with right:
    cap = st.number_input("Maximum Clauses for Analysis", min_value=5, max_value=50, value=20, step=5)
    split_method = st.selectbox("Text Splitting Method", options=["semantic", "regex", "recursive"], index=0)
    MODEL_NAME = st.selectbox("Language Model", options=["gpt-4o", "gpt-4-turbo"], index=0)

if process_clicked:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY unconfigured.")
        st.stop()
    if uploaded is None:
        st.warning("Upload a PDF file before proceeding.")
        st.stop()

    try:
        pdf_bytes = uploaded.getvalue()
        st.session_state.source_pdf_bytes = pdf_bytes
        text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    except Exception as e:
        st.error(f"An error occurred while reading or parsing the PDF document: {e}")
        st.stop()

    lang = get_language(text)
    st.caption(f"Detected primary language of the document: **{'中文 (zh)' if lang == 'zh' else 'English (en)'}**")
    clauses = smart_split(text, method=split_method)[:cap]
    st.caption(f"Using the '{split_method}' method, {len(clauses)} clause blocks were generated.")
    if not clauses:
        st.warning("Failed to extract valid clauses from the document. Please check the document content or try another extraction method.")
        st.stop()

    all_results = classify_batch(clauses)

    high_medium_results = [r for r in all_results if r.risk in ["HIGH", "MEDIUM"]]
    noticeable_count = len(all_results) - len(high_medium_results)

    st.session_state.results = high_medium_results
    st.session_state.resolved = set()

    st.session_state.last_run_message = f"Analysis complete! {len(high_medium_results)} high/medium-risk items identified."
    if noticeable_count > 0:
        st.session_state.last_run_message += f"Automatically filtered {noticeable_count} low-risk items."

    st.rerun()

if "last_run_message" in st.session_state:
    st.success(st.session_state.last_run_message)
    del st.session_state.last_run_message

if st.session_state.results:
    results: List[ClauseRisk] = st.session_state.results
    resolved: set = st.session_state.resolved
    def mark_resolved(idx: int): st.session_state.resolved.add(idx)
    def undo_resolved(idx: int): st.session_state.resolved.discard(idx)
    active_indices = [i for i in range(len(results)) if i not in resolved]
    active_results = [results[i] for i in active_indices]

    counts = {"HIGH": sum(1 for r in active_results if r.risk == "HIGH"),
              "MEDIUM": sum(1 for r in active_results if r.risk == "MEDIUM")}

    st.divider()
    st.subheader("Risk Summary")
    st.write(f"🔴 **High Risk: {counts['HIGH']}** · 🟠 **Medium Risk: {counts['MEDIUM']}** · ✅ **Resolved: {len(resolved)}**")

    show_resolved = st.checkbox("Show resolved items", value=False)

    badge_map = {"HIGH": "🔴 HIGH RISK", "MEDIUM": "🟠 MEDIUM RISK"}

    def render_card(idx: int, item: ClauseRisk, is_resolved: bool):
        with st.container(border=True):
            left_col, right_col = st.columns([8, 2])
            with left_col:
                status = "✅ RESOLVED" if is_resolved else badge_map.get(item.risk, "🟠 MEDIUM RISK")
                st.markdown(f"**{status}** · *Tags: {', '.join(item.tags) or 'N/A'}*")
                if item.risk_sentence and not is_resolved:
                    st.markdown("##### 🔑 Risk Root-Cause Sentence")
                    st.markdown(f"> {item.risk_sentence}")

                st.markdown("##### 💬 AI Analysis & Reason")
                st.info(f"{item.reason}")

                if item.suggestion and item.suggestion != "No changes required." and not is_resolved:
                    st.markdown("##### ✍️ Suggested Full Revision")
                    st.success(f"{item.suggestion}")

                with st.expander("View Full Clause Context"):
                    st.text_area("Clause Text", value=item.clause, height=150, disabled=True, key=f"clause_text_{idx}")

            with right_col:
                if not is_resolved:
                    st.button("Resolved", key=f"resolve_btn_{idx}", help="Mark this item as resolved", on_click=mark_resolved, args=(idx,), use_container_width=True)
                else:
                    st.button("Undo", key=f"undo_btn_{idx}", help="Return this item to the pending list", on_click=undo_resolved, args=(idx,), use_container_width=True)

    for i in active_indices:
        render_card(i, results[i], is_resolved=False)
    if show_resolved and resolved:
        st.markdown("### Resolved Items")
        for i in sorted(list(resolved)):
            render_card(i, results[i], is_resolved=True)

    st.divider()
    st.subheader("Export with Highlights (PDF)")
    if st.session_state.source_pdf_bytes:
        include_resolved_pdf = st.toggle("Include resolved items in the PDF", value=False)
        try:
            pdf_bytes = build_highlighted_pdf(
                st.session_state.source_pdf_bytes,
                items=results,
                include_resolved=include_resolved_pdf,
                resolved_idx=resolved
            )
            st.download_button(
                "Export PDF with highlighted annotations",
                data=pdf_bytes, file_name="contract_highlighted.pdf",
                mime="application/pdf", use_container_width=True,
            )
        except Exception as e:
            st.error(f"An error occurred while generating PDF annotations: {e}")
