import streamlit as st
import os, boto3
import tempfile
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from text_splitter import recursive_split

# Import functions from the shared library
from utils import (
    extract_revisions_from_single_doc,
    extract_comments_from_docx,
    ingest_docs_to_pinecone,
    get_pinecone_client
)
load_dotenv()
st.set_page_config(page_title="Admin Console", layout="wide")
st.logo("logo.png")

st.header("Knowledge Base Admin Dashboard")
st.markdown("Here, you can manage GCO experiences or perform system maintenance.")

INDEX_NAME = "contract-assistant"

# Ingest approved report from Amazon S3
from dotenv import load_dotenv
load_dotenv()

def ingest_reports_from_s3(bucket_name, prefix="", namespace="approved_reports"):
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION_NAME")
    )
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if "Contents" not in objects:
        st.warning("No reports were found in S3.")
        return
    for obj in objects["Contents"]:
        key = obj["Key"]
        if not key.endswith(".md"):
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
            s3.download_fileobj(bucket_name, key, tmp)
            tmp_path = tmp.name

        loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()

        if docs:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs_split = splitter.split_documents(docs)
            ingest_docs_to_pinecone(
                docs_split,
                os.getenv("PINECONE_INDEX"),
                namespace
            )

        os.remove(tmp_path)
        st.success(f"Successfully Learned Report: {key}")

# Feature 1: Upload GCO Experience Document
st.subheader("Upload GCO Review Experience Document (.docx)")
gco_file = st.file_uploader("Select a Word file containing tracked changes or comments", type="docx", key="gco_uploader")

gco_namespace = st.text_input(
    "Assign a namespace to this GCO experience",
    value="gco-case-studies",
    help="It is recommended to store all GCO experience documents in the same namespace to facilitate unified retrieval."
)

if st.button("Extract and store experience from GCO document"):
    if gco_file and gco_namespace:
        with st.spinner(f"Extracting GCO experience from '{gco_file.name}'..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(gco_file.getvalue())
                tmp_file_path = tmp_file.name

            # Extracting Experience
            word_nsmap = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            revision_chunks = extract_revisions_from_single_doc(tmp_file_path, word_nsmap)
            comment_chunks = extract_comments_from_docx(tmp_file_path)
            all_wisdom_chunks = revision_chunks + comment_chunks

            os.remove(tmp_file_path)

        if all_wisdom_chunks:
            # Convert the extracted experience text into a LangChain Document object
            with tempfile.NamedTemporaryFile(delete=False, mode="w+", encoding="utf-8") as wisdom_txt:
                wisdom_txt.write("\n".join(all_wisdom_chunks))
                wisdom_txt_path = wisdom_txt.name

            loader = TextLoader(wisdom_txt_path)
            documents = loader.load()

            # Upload to Pinecone
            ingest_docs_to_pinecone(documents, INDEX_NAME, gco_namespace)
            os.remove(wisdom_txt_path)
            st.success(f"Successfully extracted and stored {len(all_wisdom_chunks)} GCO experience entries!")
        else:
            st.warning("No extractable tracked changes or comments were found in the document.")
    else:
        st.error("Please upload a file and specify a namespace.")

st.divider()

st.subheader("Report Learning (S3)")
if st.button("Relearn S3 Report"):
    bucket_name = st.secrets["aws"]["s3_bucket_name"]
    ingest_reports_from_s3(bucket_name)

# Feature 3: Index Management (Danger Zone)
with st.expander("Danger Zone: Index Management"):
    st.warning("Warning: The following actions will permanently delete data from the knowledge base.")

    if st.button("Delete All Vectors"):
        try:
            pc = get_pinecone_client()
            index = pc.Index(INDEX_NAME)
            index.delete(delete_all=True)
            st.success(f"Successfully cleared all data in index '{INDEX_NAME}'!")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"An error occurred while clearing the index: {e}")
