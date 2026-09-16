import os
from dotenv import load_dotenv

load_dotenv()


# ============================================================
# Models
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

LLM_MODEL = os.getenv("LLM_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ============================================================
# RAG Settings
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

RETRIEVAL_K = 10
RERANK_TOP_N = 4


# ============================================================
# Paths
# ============================================================

PDF_PATH = "data/orionworks_handbook.pdf"

VECTORSTORE_PATH = "data/vectorstore"