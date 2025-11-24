from __future__ import annotations
import os
from pathlib import Path

# folders
BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = BASE_DIR / "data" / "docs"
# prefer a writable local cache by default; allow override via env
_ENV_VSTORE = os.environ.get("AI_RAG_VSTORE_DIR")
if _ENV_VSTORE:
    VSTORE_DIR = Path(_ENV_VSTORE).expanduser()
else:
    VSTORE_DIR = Path.home() / ".cache" / "ai_rag_app" / "vectorstore"

# chroma
COLLECTION_NAME = "ai_docs"

# embeddings and chunking
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
BATCH_SIZE = 128
