from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag import ask

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    version="1.0.0",
    description="Grounded RAG API built with LangGraph, Chroma and Gemini."
)

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "Agentic AI eBook RAG API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return ask(request.question.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
