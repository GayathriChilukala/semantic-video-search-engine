# Semantic Video Search Engine (Prototype)
 
This repository provides a minimal prototype for building a semantic video search engine using OpenAI embeddings and FAISS for local vector search.

Quick start

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Create a virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Ingest a video:

```bash
python -m src.ingest --video path/to/video.mp4 --index outputs/index.faiss --meta outputs/meta.json
```

Notes on ASR / embeddings
- By default the project will use local models when available (set `ASR_MODE=local` and `LOCAL_SENTENCE_MODEL` in `.env`).
- To use OpenAI-hosted Whisper/embeddings set `OPENAI_API_KEY` and `ASR_MODE=openai`.
- You can also target an Azure-compatible multimodal inference endpoint (for example `models.github.ai`) by setting
	`ASR_MODE=azure` and supplying `GITHUB_TOKEN` (or `AZURE_KEY`) plus `AZURE_INFERENCE_ENDPOINT` and `AZURE_MODEL_NAME`.
	The multimodal endpoint may accept audio directly; check your provider docs for required request format and scopes.

4. Run the API server:

```bash
uvicorn src.api:app --reload --port 8000
```

5. Open the Streamlit UI:

```bash
streamlit run src.frontend.py
```

Files
- `src/ingest.py`: Extracts audio, transcribes, chunks text, computes embeddings, builds FAISS index.
- `src/search_index.py`: Helper to load and query FAISS index and metadata.
- `src/api.py`: FastAPI search endpoint.
- `src/frontend.py`: Streamlit prototype UI.

Notes
- This is a prototype: tune chunking, embeddings, and indexing for production use.
- Using OpenAI APIs requires an API key; calls may incur cost.
