from __future__ import annotations
from pathlib import Path

def build_index(docs_dir: str | Path = "ai_rag_app/data/docs",
                vs_dir: str | Path = "ai_rag_app/vectorstore") -> None:
    # Day 9: chunk docs, embed with sentence-transformers, persist to Chroma.
    print(f"[index] would read from: {Path(docs_dir).resolve()}")
    print(f"[index] would write vector store to: {Path(vs_dir).resolve()}")
    print("[index] Stub complete (add chunking + embeddings on Day 9).")

def main() -> None:
    build_index()

if __name__ == "__main__":
    main()
