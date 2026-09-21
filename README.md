# Agentic AI eBook RAG Chatbot

A Python RAG chatbot built for the Appening Infotech AI Engineer Intern interview task.

## Architecture

PDF -> text extraction -> overlapping chunks -> Sentence Transformer embeddings -> Chroma vector DB -> LangGraph retrieval node -> Gemini generation node -> FastAPI response.

## Why Chroma?

The assignment allows "Pinecone (or any Vector DB of your choice)". Chroma is used because it is lightweight, persistent, and easy to run locally without an external database account.

## Tech Stack

- Python
- LangGraph
- ChromaDB
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Gemini API
- FastAPI
- PyPDF

## Setup

### 1. Create virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Add Gemini API key

Copy `.env.example` to `.env` and set:

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.5-flash
```

Never commit `.env`.

### 3. Ingest the eBook

```powershell
python -m app.ingest
```

This downloads the supplied Agentic AI eBook, extracts the text, chunks it, creates embeddings, and stores them in Chroma.

### 4. Start the API

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

### 5. Example request

POST `/chat`

```json
{
  "question": "What is agentic AI?"
}
```

The response contains `answer`, `retrieved_context`, and `confidence`.

## Sample Queries

1. What is agentic AI?
2. What are the key characteristics of an AI agent?
3. How are agentic AI systems different from traditional chatbots?
4. What role does planning play in agentic AI?
5. What are the major components of an agentic AI system?
6. What challenges are associated with agentic AI?

## Grounding

The generation prompt restricts the model to retrieved eBook context. If the retrieved context is insufficient, the assistant is instructed to say that the information was not found in the eBook.

## Confidence

The confidence value is a retrieval similarity signal derived from Chroma cosine distance; it is not a calibrated probability.
