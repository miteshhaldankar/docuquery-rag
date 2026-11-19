"""Core RAG pipeline: load -> chunk -> retrieve -> prompt -> answer.

Retrieval uses TF-IDF (scikit-learn) so the project runs offline with no model
download. The LLM call is isolated in `call_llm`, so swapping providers
(OpenAI, Gemini, a local model) means changing one function.
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MODEL = os.getenv("LLM_MODEL", "claude-haiku-5-5")

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the provided "
    "context. If the answer is not in the context, say you could not find it in "
    "the document. Keep answers concise and mention which source number(s) you used."
)


# ---------- 1. Loading ----------
def load_text(filename: str, data: bytes) -> str:
    """Extract text from a PDF or plain-text upload."""
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


# ---------- 2. Chunking ----------
def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text into overlapping chunks of `chunk_size` words."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


# ---------- 3. Retrieval ----------
@dataclass
class Hit:
    index: int
    score: float
    text: str


class Retriever:
    def __init__(self, chunks: list[str]):
        if not chunks:
            raise ValueError("No text to index")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [Hit(int(i), float(scores[i]), self.chunks[i]) for i in order if scores[i] > 0]


# ---------- 4. Prompting + generation ----------
def build_prompt(question: str, hits: list[Hit]) -> str:
    context = "\n\n".join(f"[Source {n}] {h.text}" for n, h in enumerate(hits, 1))
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer using only the context above."


def call_llm(prompt: str, model: str = DEFAULT_MODEL) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model=model,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def answer(question: str, retriever: Retriever, k: int = 3) -> tuple[str, list[Hit]]:
    hits = retriever.search(question, k)
    if not hits:
        return "I could not find anything relevant in the document.", []
    if not os.getenv("ANTHROPIC_API_KEY"):
        # Graceful fallback so the app still demos without an API key.
        top = hits[0].text
        return (f"(No API key set, showing the most relevant passage instead)\n\n{top}", hits)
    try:
        return call_llm(build_prompt(question, hits)), hits
    except Exception as exc:  # network / quota / auth errors
        return f"LLM call failed ({type(exc).__name__}). Most relevant passage:\n\n{hits[0].text}", hits
