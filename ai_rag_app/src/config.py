from __future__ import annotations
from pathlib import Path

# folders
BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "data" / "docs"
VSTORE_DIR = BASE_DIR / "vectorstore"

# chroma
COLLECTION_NAME = "ai_docs"

# embeddings and chunking
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast
CHUNK_SIZE = 900  # characters per chunk (approx ~225 tokens)
CHUNK_OVERLAP = 150  # characters to overlap (context continuity)
BATCH_SIZE = 128  # batch size for upserts
