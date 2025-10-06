import os
import io
import hashlib
import requests
from typing import List, Tuple
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import get_embeddings, get_splitter

def _clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def _hash_path(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]

def load_pdfs_from_paths(paths: List[str]) -> List[Document]:
    docs = []
    for p in paths:
        try:
            loader = PyPDFLoader(p)
            loaded = loader.load()
            # Ensure metadata includes a stable source
            for d in loaded:
                d.metadata = d.metadata or {}
                d.metadata["source"] = os.path.basename(p)
            docs.extend(loaded)
        except Exception as e:
            print(f"[WARN] Failed to read {p}: {e}")
    return docs

def load_urls(urls: List[str]) -> List[Document]:
    docs = []
    for u in urls:
        try:
            resp = requests.get(u, timeout=15)
            resp.raise_for_status()
            text = _clean_html(resp.text)
            meta = {"source": u}
            docs.append(Document(page_content=text, metadata=meta))
        except Exception as e:
            print(f"[WARN] Failed to fetch {u}: {e}")
    return docs

def build_faiss_index(documents: List[Document], persist_dir: str) -> Tuple[FAISS, int]:
    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    embeddings = get_embeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vs.save_local(persist_dir)
    return vs, len(chunks)

def load_or_build_index(pdf_paths: List[str], urls: List[str], persist_dir: str) -> Tuple[FAISS, int]:
    # Try load existing
    try:
        vs = FAISS.load_local(persist_dir, get_embeddings(), allow_dangerous_deserialization=True)
        return vs, -1
    except Exception:
        pass
    # Build new
    docs = []
    docs.extend(load_pdfs_from_paths(pdf_paths))
    docs.extend(load_urls(urls))
    return build_faiss_index(docs, persist_dir)
