import os
from dotenv import load_dotenv

load_dotenv()

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
PDF_PATH = "data/Ebook-Agentic-AI.pdf"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "agentic_ai_ebook"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
