# Document Q&A with Citations (RAG)

Upload a PDF, ask a question, and get an answer with page references. Built with Streamlit, FAISS, fastembed embeddings, and the Gemini API.

**Live demo:** https://rag-document-app-pvzclgkmsu7cyc298ew4sn.streamlit.app/(access code: DIYA10022025)

![App screenshot](screenshot.png)

## How it works
1. The PDF is split into overlapping text chunks, each tagged with its page number.
2. Each chunk is embedded (BAAI/bge-small-en-v1.5) and stored in a FAISS index.
3. A question is embedded and the top-k most similar chunks are retrieved.
4. Gemini answers using only those chunks and cites the pages. If the answer isn't in the context, it says so.
5. The app retries and falls back to other models if the API is busy.

## Evaluation
I tested retrieval on a 25-question set written from my own IEEE paper. Each question has a keyword that must appear in the retrieved text. The metric is hit rate (how often the right passage is retrieved).

| Chunk size | top-k 2 | top-k 4 | top-k 6 |
|---|---|---|---|
| 400 | 80% | 84% | **92%** |
| 800 | 76% | 88% | **92%** |
| 1200 | 76% | 88% | **92%** |

**Findings**
- Retrieving more chunks (k=2 to k=6) improved hit rate by 12-16 points.
- Chunk size mattered less; all sizes reached 92% at k=6. I use 400-character chunks with k=6, which reaches 92% while sending the model less text than 1200-character chunks at k=4.
- With 25 questions, each question is worth 4 points, so small differences are within noise.

**Failure analysis (2 of 25 missed)**
- A one-clause fact about which library implements the autoencoder, buried in a longer paragraph.
- A vague question about the worst-performing baseline, answered by one sentence in a dense results paragraph.

Both are short facts inside text that many other chunks resemble. Likely fixes: hybrid keyword + vector search or a reranker.

## Limitations
- The metric checks retrieval, not whether the generated answer is correct.
- The test set is small and from a single document.
- Scanned PDFs without selectable text are not supported.

## Run locally
```bash
pip install -r requirements.txt
```
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = ""
ACCESS_CODE = "your-code"
```
```bash
streamlit run app.py
```