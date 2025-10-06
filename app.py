import os
import streamlit as st
import pandas as pd
from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from utils import get_llm, format_sources
from scripts.ingest import load_or_build_index, build_faiss_index, load_pdfs_from_paths, load_urls

st.set_page_config(page_title="PMC Concierge", page_icon="🚴", layout="wide")

PERSIST_DIR = "storage/faiss-index"
DEFAULT_SYSTEM_PROMPT = """You are the PMC Rider Concierge, a helpful, concise assistant for Pan-Mass Challenge riders, donors, and volunteers.
- Use the provided knowledge base to answer. If unsure, ask a clarifying question.
- Always cite your sources at the end with short bullets: (Source: filename or URL, page if available).
- If a question is about personal accounts or registration actions, provide links and clear steps rather than pretending to perform actions.
- Keep answers brief and actionable. When appropriate, offer related tips.
"""

with st.sidebar:
    st.title("🚴 PMC Concierge")
    st.caption("Streamlit + RAG (PDFs + URLs)")

    api_ok = bool(os.getenv("OPENAI_API_KEY"))
    st.write("🔑 OpenAI key:", "✅ found" if api_ok else "❌ missing — set OPENAI_API_KEY")

    st.subheader("Knowledge Base")
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    url_list = st.text_area("Add URLs (one per line)", placeholder="https://www.pmc.org/ride\nhttps://www.pmc.org/about")

    if st.button("Rebuild Index"):
        paths = []
        # Save uploaded PDFs into ./data and collect paths
        if pdf_files:
            os.makedirs("data", exist_ok=True)
            for f in pdf_files:
                path = os.path.join("data", f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                paths.append(path)
        urls = [u.strip() for u in (url_list or "").splitlines() if u.strip()]
        with st.spinner("Building vector index..."):
            docs = []
            if paths:
                docs.extend(load_pdfs_from_paths(paths))
            if urls:
                docs.extend(load_urls(urls))
            vs, n = build_faiss_index(docs, PERSIST_DIR)
        st.success("Index built.")

    st.divider()
    st.subheader("Settings")
    system_prompt = st.text_area("System Prompt", value=DEFAULT_SYSTEM_PROMPT, height=180)
    model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    top_k = st.slider("Top-K documents", 2, 8, 4)

st.title("PMC Concierge")
tabs = st.tabs(["💬 Chat", "📊 Survey Insights (beta)"])

# Try load FAISS (if not present, create an empty placeholder)
vectorstore = None
try:
    vectorstore = FAISS.load_local(PERSIST_DIR, None, allow_dangerous_deserialization=True)
except Exception:
    pass

with tabs[0]:
    st.subheader("Ask a question")
    user_q = st.text_input("Try: “What is the fundraising minimum?” or “Where do I check-in on Saturday?”")
    ask = st.button("Ask")
    if ask and user_q.strip():
        if vectorstore is None:
            st.warning("No index found yet. Upload PDFs and/or add URLs in the sidebar, then click Rebuild Index.")
        else:
            # Temporarily attach embeddings to the loaded store (Streamlit reload quirk)
            from utils import get_embeddings
            vectorstore.embedding_function = get_embeddings()
            docs = vectorstore.similarity_search(user_q, k=top_k)
            sources_md = format_sources(docs)
            context = "\n\n".join([d.page_content for d in docs])
            llm = get_llm(model_name, temperature=0.1)
            messages = [
                SystemMessage(content=system_prompt),
                SystemMessage(content=f"Context from the knowledge base:\n\n{context}"),
                HumanMessage(content=user_q),
            ]
            with st.spinner("Thinking..."):
                resp = llm.invoke(messages)
            st.markdown(resp.content)
            st.markdown("#### Sources")
            st.markdown(sources_md)

with tabs[1]:
    st.subheader("Upload an XLSX with survey responses")
    f = st.file_uploader("Choose a .xlsx", type=["xlsx"], key="xlsx")
    text_col = st.text_input("Name of free-text column (e.g., 'Why did you stop riding?')")
    if f is not None and text_col:
        try:
            df = pd.read_excel(f)
            st.write("Rows:", len(df))
            if text_col not in df.columns:
                st.error(f"Column '{text_col}' not found. Available: {list(df.columns)[:10]}...")
            else:
                # Very simple keyword stats as a baseline
                texts = df[text_col].astype(str).str.lower().fillna("")
                keywords = {
                    "time": ["time", "busy", "schedule", "conflict"],
                    "fundraising": ["fundraising", "donor", "donation", "ask"],
                    "training": ["train", "fitness", "injury", "health"],
                    "logistics": ["travel", "hotel", "route", "parking", "check-in"],
                    "team": ["team", "captain", "group"],
                }
                counts = {}
                for k, words in keywords.items():
                    counts[k] = int(texts.apply(lambda t: any(w in t for w in words)).sum())
                st.bar_chart(pd.DataFrame.from_dict(counts, orient="index", columns=["count"]))

                # Optional: quick theme summary via LLM
                if os.getenv("OPENAI_API_KEY"):
                    from utils import get_llm
                    llm = get_llm(temperature=0.2)
                    sample = "\n".join(texts.sample(min(200, len(texts)), random_state=42).tolist())
                    prompt = f"Read these rider survey comments and list 5-7 concise themes with one-sentence insights each:\n\n{sample}"
                    with st.spinner("Summarizing themes with AI..."):
                        s = llm.invoke([HumanMessage(content=prompt)]).content
                    st.markdown("#### AI Theme Summary")
                    st.markdown(s)
        except Exception as e:
            st.exception(e)
