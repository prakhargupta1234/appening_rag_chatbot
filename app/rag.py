from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from langgraph.graph import StateGraph, START, END
from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GEMINI_API_KEY, GEMINI_MODEL

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Put it in .env")

embedder = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_collection(COLLECTION_NAME)
gemini = genai.Client(api_key=GEMINI_API_KEY)

class State(TypedDict, total=False):
    question: str
    contexts: list[dict]
    answer: str
    confidence: float

def retrieve(state: State):
    q_embedding = embedder.encode([state["question"]], normalize_embeddings=True).tolist()
    result = collection.query(
        query_embeddings=q_embedding,
        n_results=4,
        include=["documents", "metadatas", "distances"]
    )
    contexts = []
    for doc, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        contexts.append({
            "text": doc,
            "page": meta.get("page"),
            "source": meta.get("source"),
            "score": round(score, 4),
        })
    confidence = sum(x["score"] for x in contexts) / len(contexts) if contexts else 0.0
    return {"contexts": contexts, "confidence": round(confidence, 4)}

def generate(state: State):
    context = "\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in state["contexts"]
    )
    prompt = f"""You are a strict RAG assistant for the Agentic AI eBook.

Answer the user's question ONLY using the supplied context from the eBook.
Do not use outside knowledge. If the context does not contain enough information,
say exactly: "I couldn't find enough information about that in the Agentic AI eBook."

Do not invent facts, examples, page numbers, or citations.

Context:
{context}

User question:
{state["question"]}
"""
    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    return {"answer": response.text.strip()}

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
rag_app = graph.compile()

def ask(question: str):
    result = rag_app.invoke({"question": question})
    return {
        "answer": result["answer"],
        "retrieved_context": result["contexts"],
        "confidence": result["confidence"],
    }
