from typing import Iterable, List, Optional
from pathlib import Path
import re
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from .config import (
    DOCS_DIR,
    VSTORE_DIR,
    COLLECTION_NAME,
    DEFAULT_EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    BATCH_SIZE,
)
from filelock import FileLock


# ---------- file reading ----------


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # ensure pypdf installed

    reader = PdfReader(str(path))
    out = []
    for page in reader.pages:
        out.append(page.extract_text() or "")
    return "\n".join(out)


def _read_doc(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    return _read_text_file(path)


# ---------- cleaning and chunking ----------

_ws_multi = re.compile(r"[ \t]+")
_parabreaks = re.compile(r"\n{3,}")


def _normalize_ws(text: str) -> str:
    # collapse excessive whitespace while keeping paragraph breaks
    text = _parabreaks.sub("\n\n", text.strip())
    return _ws_multi.sub(" ", text)


def _chunk_paragraphs(text: str, size: int, overlap: int) -> List[str]:
    """Simple char-length window over paragraphs with overlap tail."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= size:
            cur = f"{cur}\n\n{p}"
        else:
            chunks.append(cur[:size])
            tail = cur[-overlap:] if overlap > 0 else ""
            cur = f"{tail}\n\n{p}" if tail else p
    if cur:
        chunks.append(cur[:size])
    return [c for c in chunks if c]


def _est_tokens(chars: int) -> int:
    # rough heuristic ~4 chars/token for English
    return max(1, round(chars / 4))


# ---------- ids and hashes ----------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _chunk_ids(path: Path, n: int) -> List[str]:
    # include file stem and index so re-indexing idempotent
    base = path.stem
    return [f"{base}:{i}" for i in range(n)]


# ---------- iter docs ----------


def _iter_docs(root: Path) -> Iterable[Path]:
    exts = ("*.md", "*.markdown", "*.txt", "*.pdf")
    for ext in exts:
        yield from root.rglob(ext)


# ⬇️ NEW: parameterized builder
def build_index_with_params(
    docs_dir: str | Path = DOCS_DIR,
    persist_dir: str | Path = VSTORE_DIR,
    embed_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> None:
    docs_dir, persist_dir = Path(docs_dir), Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_or_create_collection(COLLECTION_NAME)

    lock = FileLock(str(Path(persist_dir) / ".chroma.lock"))
    with lock:
        client = chromadb.PersistentClient(path=str(persist_dir))
        col = client.get_or_create_collection(COLLECTION_NAME)
        # ... rest of indexing & upserts ...

    # use overrides or defaults
    _embed_model = embed_model or DEFAULT_EMBED_MODEL
    _chunk_size = chunk_size or CHUNK_SIZE
    _chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    _batch = batch_size or BATCH_SIZE

    model = SentenceTransformer(_embed_model)

    total_docs = 0
    total_chunks = 0
    total_chars = 0

    for path in _iter_docs(docs_dir):
        raw = _read_doc(path)
        if not raw.strip():
            continue
        text = _normalize_ws(raw)
        chunks = _chunk_paragraphs(text, _chunk_size, _chunk_overlap)
        if not chunks:
            continue

        ids = _chunk_ids(path, len(chunks))
        metadatas = []
        for i, c in enumerate(chunks):
            metadatas.append(
                {
                    "source": str(path),
                    "chunk_index": i,
                    "chars": len(c),
                    "tokens_est": _est_tokens(len(c)),
                    "content_sha256": _sha256(c),
                    "embed_model": _embed_model,
                    "chunk_size": _chunk_size,
                    "chunk_overlap": _chunk_overlap,
                }
            )

        embeddings = model.encode(chunks, normalize_embeddings=True).tolist()

        for i in range(0, len(chunks), _batch):
            col.upsert(
                ids=ids[i : i + _batch],
                documents=chunks[i : i + _batch],
                embeddings=embeddings[i : i + _batch],
                metadatas=metadatas[i : i + _batch],
            )

        total_docs += 1
        total_chunks += len(chunks)
        total_chars += sum(len(c) for c in chunks)

    avg_tokens = _est_tokens(total_chars / total_chunks) if total_chunks else 0
    print(
        f"[index] docs={total_docs} chunks={total_chunks} "
        f"avg_tokens_per_chunk≈{avg_tokens} store={persist_dir} "
        f"model={_embed_model} size={_chunk_size} overlap={_chunk_overlap}"
    )


# keep the original name as a thin wrapper
def build_index(docs_dir: str | Path = DOCS_DIR, persist_dir: str | Path = VSTORE_DIR) -> None:
    build_index_with_params(docs_dir, persist_dir)
