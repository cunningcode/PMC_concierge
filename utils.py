import os
import time
from typing import List, Dict, Any, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, separators=["\n\n", "\n", ".", " "]
    )

def get_embeddings():
    # You can swap to "text-embedding-3-small" for lower cost.
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    return OpenAIEmbeddings(model=model)

def get_llm(model: str = None, temperature: float = 0.1):
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=temperature)

def with_retry(fn, retries: int = 3, delay: float = 1.2):
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(delay)

def format_sources(docs, max_chars: int = 2600) -> str:
    out = []
    seen = set()
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source", "document")
        page = meta.get("page", None)
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        label = f"{src}" + (f" (p.{page+1})" if isinstance(page, int) else "")
        snippet = d.page_content[:max_chars]
        out.append(f"- **{label}** — {snippet}")
    return "\n".join(out)
