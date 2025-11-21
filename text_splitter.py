import re
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from semantic_text_splitter import TextSplitter

def regex_split(text: str):
    """Clause-Oriented Splitting"""
    pattern = r"(?=\n\d+\.\s)|(?=\nArticle\s+\d+)|(?=\nSection\s+\d+)"
    chunks = re.split(pattern, text)
    return [c.strip() for c in chunks if c.strip()]

def semantic_split(text: str, max_chunk_size=800):
    """Semantic-Oriented Splitting"""
    splitter = TextSplitter(max_chunk_size)
    return splitter.chunks(text)

def recursive_split(text: str, chunk_size=1000, overlap=100):
    """Recursive Character Splitting"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    return splitter.split_text(text)

def smart_split(text: str, method="semantic"):
    """Unified interface with selectable splitting methods"""
    if method == "regex":
        return regex_split(text)
    elif method == "recursive":
        return recursive_split(text)
    elif method == "semantic":
        return semantic_split(text)
    else:
        raise ValueError(f"Unknown split method: {method}")
