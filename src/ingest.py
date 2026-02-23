import os
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from moviepy.editor import VideoFileClip
import numpy as np
import ffmpeg
import openai
import base64
from tqdm import tqdm
from dotenv import load_dotenv
import faiss

load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_KEY

# Defaults
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
LOCAL_SENTENCE_MODEL = os.environ.get("LOCAL_SENTENCE_MODEL", "all-MiniLM-L6-v2")
ASR_MODE = os.environ.get("ASR_MODE", "openai")  # or 'local' or 'azure'

# Lazy imports for local models
_sentence_model = None
_whisper_model = None

def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise ImportError("sentence-transformers not installed. Install requirements or set EMBED_MODEL to an API provider.")
        _sentence_model = SentenceTransformer(LOCAL_SENTENCE_MODEL)
    return _sentence_model

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
        except Exception:
            raise ImportError("whisper not installed or not available. Install the 'whisper' package or use OpenAI ASR.")
        _whisper_model = whisper.load_model("small")
    return _whisper_model


def transcribe_audio_azure(audio_path: str) -> str:
    """Transcribe audio using an Azure-compatible multimodal inference endpoint.

    Environment variables used:
    - GITHUB_TOKEN or AZURE_KEY: credential for the inference endpoint
    - AZURE_INFERENCE_ENDPOINT: URL for the inference endpoint (defaults to models.github.ai)
    - AZURE_MODEL_NAME: model id (defaults to microsoft/Phi-4-multimodal-instruct)

    NOTE: The exact ingestion format for audio depends on the provider. This helper encodes the
    audio as base64 and sends it inside a message; adjust to your provider's preferred file/attachments API.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("AZURE_KEY")
    if not token:
        raise EnvironmentError("GITHUB_TOKEN or AZURE_KEY required for Azure inference mode")
    endpoint = os.environ.get("AZURE_INFERENCE_ENDPOINT", "https://models.github.ai/inference")
    model_name = os.environ.get("AZURE_MODEL_NAME", "microsoft/Phi-4-multimodal-instruct")

    try:
        from azure.ai.inference import ChatCompletionsClient
        from azure.ai.inference.models import UserMessage
        from azure.core.credentials import AzureKeyCredential
    except Exception:
        raise ImportError("azure.ai.inference SDK not installed. Install the package or use a different ASR_MODE")

    client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token))

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "Transcribe the following audio. The audio is provided as base64 after the marker 'AUDIO_BASE64:'.\n"
        "Return only the transcript text with no extra commentary.\n\n"
        f"AUDIO_BASE64:{audio_b64}"
    )

    resp = client.complete(
        messages=[UserMessage(prompt)],
        temperature=0.0,
        top_p=1.0,
        max_tokens=20000,
        model=model_name,
    )

    try:
        return resp.choices[0].message.content
    except Exception:
        return str(resp)

def extract_audio(video_path: str, out_audio: str):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_audio, fps=16000, nbytes=2)

def transcribe_audio_whisper(audio_path: str) -> str:
    # Choose between OpenAI-hosted Whisper or local Whisper
    mode = ASR_MODE
    if mode == "azure":
        return transcribe_audio_azure(audio_path)
    if mode == "openai":
        with open(audio_path, "rb") as f:
            resp = openai.Audio.transcribe(model="whisper-1", file=f)
        return resp.get("text", "")
    else:
        model = _get_whisper_model()
        result = model.transcribe(audio_path)
        return result.get("text", "")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

def embed_texts(texts, embed_mode: str = "auto"):
    """
    embed_mode: 'auto'|'openai'|'local'
    - 'auto' uses OpenAI if key present, else local sentence-transformers
    """
    mode = embed_mode
    if mode == "auto":
        mode = "openai" if OPENAI_KEY else "local"

    if mode == "openai":
        resp = openai.Embedding.create(model=EMBED_MODEL, input=texts)
        embs = [d["embedding"] for d in resp["data"]]
        return np.array(embs, dtype=np.float32)
    else:
        model = _get_sentence_model()
        embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(embs, dtype=np.float32)

def build_faiss_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def save_index_and_meta(index, meta, index_path: str, meta_path: str):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(meta_path, np.array(meta, dtype=object), allow_pickle=True)

def main(video, index_path, meta_path):
    tmp_audio = tempfile.mktemp(suffix=".wav")
    print("Extracting audio...")
    extract_audio(video, tmp_audio)
    print("Transcribing audio (OpenAI Whisper)...")
    text = transcribe_audio_whisper(tmp_audio)
    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    print("Embedding chunks...")
    embeddings = embed_texts(chunks)
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    meta = [{"text": chunks[i], "source": video, "chunk_id": i} for i in range(len(chunks))]
    print("Saving index and metadata...")
    save_index_and_meta(index, meta, index_path, meta_path)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--index", default="outputs/index.faiss")
    parser.add_argument("--meta", default="outputs/meta.npy")
    args = parser.parse_args()
    main(args.video, args.index, args.meta)
