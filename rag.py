import time
import numpy as np
import faiss
from google import genai
from pypdf import PdfReader
from fastembed import TextEmbedding

MODELS = ["gemini-flash-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _embedder


def embed(texts):
    vecs = np.array(list(get_embedder().embed(texts)), dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs


def load_pdf(file):
    reader = PdfReader(file)
    return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]


def chunk(pages, size=800, overlap=150):
    chunks = []
    for page_no, text in pages:
        for start in range(0, len(text), size - overlap):
            piece = text[start:start + size].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
    return chunks


def build_index(chunks):
    vecs = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def search(index, chunks, question, k=4):
    _, ids = index.search(embed([question]), k)
    return [chunks[i] for i in ids[0] if i != -1]


def answer(client, question, retrieved):
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in retrieved)
    prompt = (
        "Answer the question using ONLY the context below. "
        "Cite the pages you used like [Page 3]. "
        "If the answer is not in the context, say you cannot find it.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    last_error = None
    for model in MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                return response.text
            except Exception as e:
                last_error = e
                time.sleep(2)
    return f"The AI service is busy right now. Please try again in a minute. (Details: {str(last_error)[:200]})"