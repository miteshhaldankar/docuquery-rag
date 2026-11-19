# DocuQuery: Document Q&A with Retrieval-Augmented Generation (RAG)

Upload a PDF or text file and ask questions in plain English. The app finds the most relevant
passages, sends only those to an LLM, and shows the answer with its sources.

**Tech:** Python, Streamlit, scikit-learn (TF-IDF retrieval), pypdf, Anthropic API, pytest

## How RAG works here
1. **Load**: extract text from the PDF/TXT (`load_text`)
2. **Chunk**: split into overlapping word windows so answers aren't cut at boundaries (`chunk_text`)
3. **Retrieve**: TF-IDF + cosine similarity ranks chunks against the question (`Retriever`)
4. **Prompt**: build a prompt that tells the model to answer *only* from the context (`build_prompt`)
5. **Generate**: LLM answers and cites sources (`call_llm`)

If the answer isn't in the document, the model is instructed to say so (reduces hallucination).
Without an API key the app still runs and shows the best-matching passage.

## Run it
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here      # optional; Windows: set ANTHROPIC_API_KEY=...
streamlit run app.py
```
Upload `sample_docs/company_handbook.txt` and try: *"How many days of annual leave do I get?"*

## Test
```bash
pytest -q        # 11 tests: chunking, retrieval accuracy, prompt, fallback, LLM error handling
```

## Design notes / possible upgrades
- TF-IDF keeps it runnable offline. To upgrade to semantic search, swap `Retriever` for
  sentence-transformers embeddings + FAISS/ChromaDB (same `search()` interface).
- To use another LLM provider, only `call_llm` needs to change.
- Responsible AI: answers are grounded in the document, sources are shown, and the prompt
  forbids guessing. Don't upload confidential data to third-party APIs.
