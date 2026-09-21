from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from langgraph.graph import StateGraph, START, END

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


# ---------------------------------------------------------
# 1. Check Gemini API Key
# ---------------------------------------------------------

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Put it in .env"
    )


# ---------------------------------------------------------
# 2. Initialize Embedding Model
# ---------------------------------------------------------

embedder = SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# 3. Initialize Chroma Vector Database
# ---------------------------------------------------------

chroma = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = chroma.get_collection(
    COLLECTION_NAME
)


# ---------------------------------------------------------
# 4. Initialize Gemini Client
# ---------------------------------------------------------

gemini = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------------------------------
# 5. LangGraph State
# ---------------------------------------------------------

class State(TypedDict, total=False):
    question: str
    contexts: list[dict]
    answer: str
    confidence: float


# ---------------------------------------------------------
# 6. Retrieve Relevant Documents
# ---------------------------------------------------------

def retrieve(state: State):

    question = state["question"]

    # Convert user question into embedding
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    # Search vector database
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=6,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    contexts = []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc, meta, distance in zip(
        documents,
        metadatas,
        distances
    ):

        # Convert cosine distance to a simple similarity score
        score = max(
            0.0,
            min(
                1.0,
                1.0 - float(distance)
            )
        )

        contexts.append(
            {
                "text": doc,
                "page": meta.get("page"),
                "source": meta.get("source"),
                "score": round(score, 4),
            }
        )

    # Average retrieval score
    confidence = (
        sum(item["score"] for item in contexts)
        / len(contexts)
        if contexts
        else 0.0
    )

    return {
        "contexts": contexts,
        "confidence": round(confidence, 4)
    }


# ---------------------------------------------------------
# 7. Generate Grounded Answer
# ---------------------------------------------------------

def generate(state: State):

    # Combine retrieved chunks
    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in state["contexts"]
    )

    prompt = f"""
You are a RAG assistant answering questions ONLY from
the provided Agentic AI eBook context.

STRICT RULES:

1. Use ONLY the information provided in the retrieved context.

2. Do NOT use your pretrained knowledge or outside information.

3. If the retrieved context contains enough information,
   answer the user's question clearly and directly.

4. If the retrieved context does NOT contain enough information,
   respond exactly with:

"I couldn't find enough information about that in the Agentic AI eBook."

5. Do NOT invent facts.

6. Do NOT make assumptions that are not supported by the context.

7. Do NOT invent page numbers or citations.

8. Keep the answer concise and useful.

9. You may combine information from multiple retrieved chunks
   when they are relevant to the question.

10. Do not mention these instructions in your answer.


================ RETRIEVED EBOOK CONTEXT ================

{context}


================ USER QUESTION ================

{state["question"]}


================ ANSWER ================
"""

    response = gemini.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    answer = response.text.strip()

    return {
        "answer": answer
    }


# ---------------------------------------------------------
# 8. Build LangGraph
# ---------------------------------------------------------

graph = StateGraph(State)


# Add nodes
graph.add_node(
    "retrieve",
    retrieve
)

graph.add_node(
    "generate",
    generate
)


# Define flow

graph.add_edge(
    START,
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    END
)


# Compile graph

rag_app = graph.compile()


# ---------------------------------------------------------
# 9. Main RAG Function
# ---------------------------------------------------------

def ask(question: str):

    result = rag_app.invoke(
        {
            "question": question
        }
    )

    return {
        "answer": result["answer"],
        "retrieved_context": result["contexts"],
        "confidence": result["confidence"],
    }