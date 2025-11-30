from __future__ import annotations
import os
import tempfile
from pathlib import Path

# point vector store to a temp dir for the *entire* test session
_tmp = Path(tempfile.mkdtemp(prefix="rag_store_"))
os.environ.setdefault("AI_RAG_VSTORE_DIR", str(_tmp / "vectorstore"))
