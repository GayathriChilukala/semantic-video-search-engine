import os
import numpy as np
import faiss
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
LOCAL_SENTENCE_MODEL = os.environ.get("LOCAL_SENTENCE_MODEL", "all-MiniLM-L6-v2")

_sentence_model = None

def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise ImportError("sentence-transformers not installed. Install requirements or set EMBED_MODEL to an API provider.")
        _sentence_model = SentenceTransformer(LOCAL_SENTENCE_MODEL)
    return _sentence_model

class SearchIndex:
    def __init__(self, index_path: str, meta_path: str):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.meta = None
        self._load()

    def _load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError("Index or metadata not found. Run ingestion first.")
        self.index = faiss.read_index(self.index_path)
        self.meta = np.load(self.meta_path, allow_pickle=True)

    def embed_query(self, query: str):
        # Use OpenAI if API key present, else local sentence-transformers
        if openai.api_key:
            resp = openai.Embedding.create(model=EMBED_MODEL, input=[query])
            return np.array(resp['data'][0]['embedding'], dtype=np.float32)
        else:
            model = _get_sentence_model()
            emb = model.encode([query], convert_to_numpy=True)
            return np.array(emb[0], dtype=np.float32)

    def search(self, query: str, k: int = 5):
        qvec = self.embed_query(query).reshape(1, -1)
        D, I = self.index.search(qvec, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            meta = self.meta[idx].item() if hasattr(self.meta[idx], 'item') else self.meta[idx]
            results.append({"score": float(score), "meta": meta})
        return results
