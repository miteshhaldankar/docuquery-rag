import pathlib
import pytest
from rag import Retriever, build_prompt, chunk_text, load_text, answer

DOC = pathlib.Path(__file__).parent.parent / "sample_docs" / "company_handbook.txt"


def test_chunk_overlap_and_size():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert all(len(c.split()) <= 40 for c in chunks)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert chunks[-1].split()[-1] == "w99"  # nothing dropped at the end


def test_chunk_edge_cases():
    assert chunk_text("") == []
    assert chunk_text("short text", 50, 10) == ["short text"]
    with pytest.raises(ValueError):
        chunk_text("a b c", 5, 5)


@pytest.fixture
def retriever():
    text = load_text("handbook.txt", DOC.read_bytes())
    return Retriever(chunk_text(text, 60, 15))


@pytest.mark.parametrize("question,expected", [
    ("How many days of annual leave do I get?", "18 days"),
    ("What is the daily meal allowance when travelling?", "800 rupees"),
    ("How long is the training programme?", "6-month"),
    ("How soon must a security incident be reported?", "within one hour"),
])
def test_retrieval_finds_right_chunk(retriever, question, expected):
    hits = retriever.search(question, k=3)
    assert any(expected in h.text for h in hits)


def test_irrelevant_question_returns_nothing(retriever):
    assert retriever.search("xyzzy quantum zebra", k=3) == []


def test_prompt_contains_sources_and_question(retriever):
    hits = retriever.search("annual leave", 2)
    p = build_prompt("annual leave?", hits)
    assert "[Source 1]" in p and "Question: annual leave?" in p


def test_answer_falls_back_without_api_key(retriever, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reply, hits = answer("How many sick leave days?", retriever)
    assert "No API key" in reply and hits


def test_answer_uses_llm_when_key_set(retriever, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr("rag.call_llm", lambda prompt, model=None: "LLM says 18 days [Source 1]")
    reply, _ = answer("How many annual leave days?", retriever)
    assert reply.startswith("LLM says")


def test_answer_survives_llm_error(retriever, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    def boom(*a, **k): raise RuntimeError("network")
    monkeypatch.setattr("rag.call_llm", boom)
    reply, hits = answer("annual leave days", retriever)
    assert "LLM call failed" in reply and hits
