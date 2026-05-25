# TrueAI RAG Assistant

> A production-grade GenAI-powered Chat Assistant that answers user questions using **Retrieval-Augmented Generation (RAG)**.

---

## Architecture Diagram

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                       │
│                                                         │
│  POST /api/chat                                         │
│       │                                                 │
│       ├─► [1] Embed Query (SentenceTransformer)         │
│       │         all-MiniLM-L6-v2 → 384-dim vector       │
│       │                                                 │
│       ├─► [2] Cosine Similarity Search (FAISS)          │
│       │         IndexFlatIP on normalised vectors        │
│       │         Top-K=3, Threshold=0.35                 │
│       │                                                 │
│       ├─► [3] Threshold Check                           │
│       │         Below threshold → Fallback response     │
│       │                                                 │
│       ├─► [4] Build Prompt                              │
│       │         Context + History + Question            │
│       │                                                 │
│       └─► [5] Gemini 2.5 Flash (LLM)                   │
│                 Temperature=0.2                         │
│                 Retry on rate-limit (3x backoff)        │
│                 Token usage logged                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
{ reply, tokensUsed, retrievedChunks }
```

---

## RAG Workflow Explanation

1. **Indexing (startup)**
   - Load `docs.json` (10 rich documents)
   - Chunk each document at ~400 tokens with 50-token overlap
   - Generate embeddings via `SentenceTransformer (all-MiniLM-L6-v2)`
   - **Normalise** all vectors to unit length
   - Store in FAISS `IndexFlatIP` (inner product = cosine on unit vectors)
   - Store chunk metadata: `{title, chunk_id, source, text}`

2. **Querying (per request)**
   - Embed user query (same model, normalised)
   - Run FAISS inner product search → Top-3 chunks
   - Filter by cosine similarity threshold (0.35)
   - If no chunks pass → return safe fallback message
   - Build context string from retrieved chunks
   - Construct grounded prompt (context + history + question)
   - Call Gemini 2.5 Flash → return reply

---

## Embedding Strategy

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` (SentenceTransformer) |
| Dimension | 384 |
| Normalisation | L2-normalised (unit vectors) |
| Library | `sentence-transformers` |

The model runs **locally** — no external embedding API cost. It produces semantically rich dense vectors suitable for capturing meaning over keywords.

---

## Similarity Search Explanation

- **Method**: Cosine Similarity via FAISS `IndexFlatIP`
- **Formula**: `cosine(q, d) = q · d / (|q| × |d|)` — simplified to inner product since both vectors are unit-normalised
- **Threshold**: 0.35 (scores below this are treated as "not relevant")
- **Top-K**: 3 chunks retrieved per query

This ensures retrieval is **semantic** (not keyword-based). A query like *"How do I secure my login?"* correctly retrieves the **Two-Factor Authentication** chunk even without exact keyword overlap.

---

## Prompt Design Reasoning

```
You are a helpful, accurate, and concise customer support assistant.
Your ONLY source of truth is the context provided below.
Do NOT use any prior knowledge outside the context.

Context:
{retrieved_context}

Conversation History:
{history}

Question:
{user_question}

Answer:
```

**Design decisions:**
- `temperature=0.2` — keeps answers factual and deterministic
- Context injected first — the LLM is anchored to retrieved facts before seeing the question
- History included — enables multi-turn follow-up questions
- Explicit instruction to not use prior knowledge — prevents hallucination
- Fallback instruction — model says "I don't know" when context is insufficient

---

## Project Structure

```
trueai-rag-assistant/
│
├── app/
│   ├── routes/
│   │   └── chat.py          # POST /api/chat, DELETE /api/session/{id}
│   ├── services/
│   │   ├── rag_service.py   # RAG pipeline orchestration
│   │   └── llm_service.py   # Gemini API wrapper with retry/error handling
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── vectorstore/
│   │   └── store.py         # FAISS cosine similarity, chunking, metadata
│   ├── prompts/
│   │   └── templates.py     # Prompt builder (context + history + question)
│   ├── utils/
│   │   └── logger.py        # Structured logging (scores, tokens)
│   └── main.py              # FastAPI app factory, startup indexing
│
├── frontend/
│   ├── index.html           # Chat UI
│   ├── styles.css           # Dark-mode premium CSS
│   └── app.js               # Session management, API calls, markdown
│
├── docs.json                # 10-document knowledge base
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- A Gemini API key ([get one here](https://makersuite.google.com/app/apikey))

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/trueai-rag-assistant.git
cd trueai-rag-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 5. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the app

Visit [http://localhost:8000](http://localhost:8000)

---

## API Reference

### `POST /api/chat`

```json
Request:
{
  "sessionId": "abc123",
  "message": "How can I reset my password?"
}

Response:
{
  "reply": "You can reset your password from Settings > Security...",
  "tokensUsed": 312,
  "retrievedChunks": 3,
  "chunks": [
    { "title": "Reset Password", "chunk_id": "reset_password_chunk_0", "score": 0.8721 }
  ]
}
```

### `GET /health`

```json
{ "status": "healthy" }
```

### `DELETE /api/session/{sessionId}`

Clears server-side conversation history for the session.

---

## Deployment (Render)

1. Push repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
6. Add environment variable: `GEMINI_API_KEY=your_key`
7. Deploy

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python + FastAPI |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) |
| Vector Store | FAISS (cosine similarity) |
| Frontend | HTML + Vanilla CSS + JavaScript |
| Config | python-dotenv |
