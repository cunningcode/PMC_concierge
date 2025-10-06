# PMC Concierge (Streamlit + RAG)

A rider-facing prototype chatbot for the Pan-Mass Challenge that supports:
- Retrieval-Augmented Generation (RAG) over uploaded PDFs and optional website pages
- Citations back to source documents
- Basic survey insights (upload an XLSX and explore free-text responses)
- Ready for Streamlit Cloud or local run

## Quickstart (Local)
1. Install Python 3.10+
2. `pip install -r requirements.txt`
3. Set your environment variable:
   - On macOS/Linux: `export OPENAI_API_KEY=sk-...`
   - On Windows (Powershell): `$env:OPENAI_API_KEY="sk-..."`
4. Run the app: `streamlit run app.py`

## Deploy on Streamlit Cloud
- Create a new app from this repo
- Add `OPENAI_API_KEY` in the app's Secrets
- (Optional) Preload PDFs by committing them under `data/` and clicking **Rebuild Index** in the sidebar once deployed

## Files
- `app.py` — main Streamlit app (chat + survey insights)
- `scripts/ingest.py` — helper routines to build a local FAISS vectorstore from PDFs and URLs
- `utils.py` — small helpers for chunking, caching, and LLM calls
- `requirements.txt` — Python dependencies
- `data/` — put your PDFs here (or upload via the UI)
- `storage/` — persisted vector index

## Notes
- The web crawler ingests a list of URLs you provide; it fetches each page and strips boilerplate. For a full crawl later, consider a sitemap-based crawler.
- This prototype stores embeddings locally (FAISS). For production, consider Supabase pgvector or managed vector DB.
