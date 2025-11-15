from __future__ import annotations
from pathlib import Path
import re
from typing import Iterable, List

import chromadb
from sentence_transformers import SentenceTransformer

from .config import (
    DOCS_DIR,
    VSTORE_DIR,
    COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def _read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Install pypdf to read PDFs") from e
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def _normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _chunk(text: str, size: int, overlap: int) -> List[str]:
    # simple sliding window over paragraphs
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= size:
            cur = f"{cur}\n\n{p}"
        else:
            chunks.append(cur)
            # overlap tail of previous chunk
            tail = cur[-overlap:]
            cur = (tail + "\n\n" + p) if tail else p
    if cur:
        chunks.append(cur)
    # enforce max size
    return [c[:size] for c in chunks if c]


def _iter_docs(root: Path) -> Iterable[Path]:
    for ext in ("*.md", "*.markdown", "*.txt", "*.pdf"):
        yield from root.rglob(ext)


def build_index(docs_dir: str | Path = DOCS_DIR, persist_dir: str | Path = VSTORE_DIR) -> None:
    docs_dir, persist_dir = Path(docs_dir), Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_or_create_collection(COLLECTION_NAME)

    model = SentenceTransformer(DEFAULT_EMBED_MODEL)

    added = 0
    for path in _iter_docs(docs_dir):
        try:
            text = _read_pdf(path) if path.suffix.lower() == ".pdf" else _read_md(path)
        except Exception:
            continue
        text = _normalize_ws(text)
        chunks = _chunk(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        ids = [f"{path.name}:{i}" for i in range(len(chunks))]
        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
        metadatas = [{"source": str(path), "chunk_index": i} for i in range(len(chunks))]

        # upsert in batches to avoid huge payloads
        B = 128
        for i in range(0, len(chunks), B):
            col.upsert(
                ids=ids[i : i + B],
                documents=chunks[i : i + B],
                embeddings=embeddings[i : i + B],
                metadatas=metadatas[i : i + B],
            )
        added += len(chunks)

    print(f"[index] added {added} chunks to collection '{COLLECTION_NAME}' at {persist_dir}")


def main() -> None:
    build_index()


if __name__ == "__main__":
    main()
