#!/usr/bin/env python3
"""
Build FAISS + SQLite index compatible with unified_arxiv_retriever.E5DirectRetriever.

Reads local text / JSONL under a source directory, preprocesses and chunks text,
embeds with the same E5 model as config.py, writes:
  - faiss_index       (FAISS binary, IndexFlatIP, normalized vectors)
  - index_store.db    (SQLite: document + meta_document)
Removes faiss_document_mapping.pkl if present so the retriever rebuilds the pickle on first load.

Usage (from repository root):
  python build_index.py --source_dir ./my_corpus --output_dir ./squai_data/faiss_index
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from config import DB_PATH, E5_INDEX_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, USE_GPU
except ImportError:
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from config import DB_PATH, E5_INDEX_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, USE_GPU

from sqlite_compat import open_db

logger = logging.getLogger("build_index")


# ---------------------------------------------------------------------------
# Configuration defaults (override via CLI or build_index_from_directory kwargs)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_BATCH_SIZE = 32
DEFAULT_EXTENSIONS = (".txt", ".md", ".jsonl")


# ---------------------------------------------------------------------------
# Text preprocessing & chunking
# ---------------------------------------------------------------------------


def preprocess_text(text: str) -> str:
    """
    Normalize whitespace and strip control characters; keep content semantics.
    """
    if not text:
        return ""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _slide_window(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split a long string into overlapping windows."""
    if chunk_size <= 0:
        return [text] if text.strip() else []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)
    out: List[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - chunk_overlap)
    while i < n:
        piece = text[i : i + chunk_size].strip()
        if piece:
            out.append(piece)
        if i + chunk_size >= n:
            break
        i += step
    return out


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Paragraph-aware chunking with sliding windows for long paragraphs.
    """
    text = preprocess_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    buf = ""
    for p in paragraphs:
        if len(p) > chunk_size:
            if buf:
                chunks.extend(_slide_window(buf, chunk_size, chunk_overlap))
                buf = ""
            chunks.extend(_slide_window(p, chunk_size, chunk_overlap))
            continue
        added = f"{buf}\n\n{p}" if buf else p
        if len(added) <= chunk_size:
            buf = added
        else:
            if buf:
                chunks.extend(_slide_window(buf, chunk_size, chunk_overlap))
            buf = p

    if buf:
        chunks.extend(_slide_window(buf, chunk_size, chunk_overlap))

    seen = set()
    unique: List[str] = []
    for c in chunks:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------------------------
# Loading local documents
# ---------------------------------------------------------------------------


def _paper_id_from_path(path: Path) -> str:
    return path.stem or path.name


def _load_jsonl_line(obj: Dict[str, Any], source_hint: str) -> Optional[Tuple[str, str]]:
    """
    Map a JSON object to (paper_id, text). Supports common SQuAI / unarXive-like keys.
    """
    pid = (
        obj.get("paper_id")
        or obj.get("id")
        or obj.get("doc_id")
        or source_hint
    )
    pid = str(pid)

    if "abstract" in obj and obj["abstract"]:
        title = ""
        meta = obj.get("metadata") or {}
        if isinstance(meta, dict):
            title = meta.get("title") or ""
        parts = [title, obj.get("abstract", "")]
        body = ". ".join(p for p in parts if p)
        sections = obj.get("sections") or {}
        if isinstance(sections, dict):
            for sec, val in sections.items():
                if isinstance(val, dict) and val.get("text"):
                    body = f"{body}\n\n{sec}: {val['text']}"
        return pid, body

    for key in ("text", "content", "body", "passage"):
        if obj.get(key):
            return pid, str(obj[key])

    return None


def iter_documents_from_file(
    path: Path, extensions: Sequence[str]
) -> Iterable[Tuple[str, str, str]]:
    """
    Yield (logical_source_id, paper_id, raw_text) per document or JSONL row.
    """
    ext = path.suffix.lower()
    if ext not in extensions:
        return

    if ext == ".jsonl":
        hint = _paper_id_from_path(path)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("Skip JSONL %s:%s: %s", path, line_no, e)
                        continue
                    if not isinstance(obj, dict):
                        continue
                    parsed = _load_jsonl_line(obj, f"{hint}:{line_no}")
                    if not parsed:
                        continue
                    pid, raw = parsed
                    src_id = f"{path.as_posix()}#{line_no}"
                    yield src_id, pid, raw
        except OSError as e:
            logger.error("Cannot read %s: %s", path, e)
        return

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.error("Cannot read %s: %s", path, e)
        return

    pid = _paper_id_from_path(path)
    yield path.as_posix(), pid, raw


def discover_files(source_dir: Path, extensions: Sequence[str]) -> List[Path]:
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    files: List[Path] = []
    for root, _, names in os.walk(source_dir):
        for name in names:
            p = Path(root) / name
            if p.suffix.lower() in exts:
                files.append(p)
    files.sort()
    return files


def load_corpus_records(
    source_dir: Path,
    extensions: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """
    Flatten corpus into chunk records:
      doc_row_id, paper_id, content (chunk text), source_file
    """
    records: List[Dict[str, Any]] = []
    files = discover_files(source_dir, extensions)
    if not files:
        logger.warning("No files matching %s under %s", extensions, source_dir)
        return records

    for fp in files:
        for src_id, paper_id, raw in iter_documents_from_file(fp, extensions):
            chunks = chunk_text(raw, chunk_size, chunk_overlap)
            path_tag = hashlib.md5(str(fp.resolve()).encode("utf-8")).hexdigest()[:10]
            for i, chunk in enumerate(chunks):
                safe_pid = re.sub(r"[^\w\-.:]+", "_", paper_id)[:120]
                doc_row_id = f"{safe_pid}_{path_tag}_c{i}"
                records.append(
                    {
                        "doc_row_id": doc_row_id,
                        "paper_id": paper_id,
                        "content": chunk,
                        "source_file": str(fp),
                    }
                )
    return records


def load_corpus_records_from_full_text_db(
    kv_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    max_papers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Read (key, value) pairs from full_text_db KV store, chunk each value,
    and return the same record shape as load_corpus_records.
    Keys must be UTF-8 paper_id strings; values are full document text.

    If ``max_papers`` is set, papers are sorted by ``paper_id`` and only the
    first ``max_papers`` are indexed (reproducible subset for smaller / faster builds).
    """
    if max_papers is not None and max_papers < 1:
        raise ValueError("max_papers must be >= 1 when set")

    pairs: List[Tuple[str, str]] = []
    db = open_db(str(kv_path), create_if_missing=False)
    try:
        for key, value in db:
            paper_id = key.decode("utf-8", errors="replace")
            text = value.decode("utf-8", errors="replace")
            pairs.append((paper_id, text))
    finally:
        db.close()

    total_in_store = len(pairs)
    pairs.sort(key=lambda x: x[0])
    if max_papers is not None:
        pairs = pairs[:max_papers]
    logger.info(
        "full_text_db: %d papers will be chunked (total keys in store: %d)%s",
        len(pairs),
        total_in_store,
        f", max_papers={max_papers}" if max_papers is not None else "",
    )

    records: List[Dict[str, Any]] = []
    for paper_id, text in pairs:
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        path_tag = hashlib.md5(paper_id.encode("utf-8")).hexdigest()[:10]
        for i, chunk in enumerate(chunks):
            safe_pid = re.sub(r"[^\w\-.:]+", "_", paper_id)[:120]
            doc_row_id = f"{safe_pid}_{path_tag}_c{i}"
            records.append(
                {
                    "doc_row_id": doc_row_id,
                    "paper_id": paper_id,
                    "content": chunk,
                    "source_file": f"full_text_db:{paper_id}",
                }
            )
    return records


def build_index_from_full_text_db(
    full_text_db_path: str | Path | None = None,
    output_dir: Optional[str | Path] = None,
    *,
    embedding_model: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: Optional[str] = None,
    max_papers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build FAISS + index_store from SQuAI full_text_db (same layout as build_index_from_directory).

    Args:
        full_text_db_path: LevelDB/SQLite KV directory (default: config.DB_PATH).
        output_dir: FAISS output directory (default: config.E5_INDEX_DIR).
        max_papers: If set, only index this many papers (sorted by paper_id), for smaller/faster builds.
    """
    kv = Path(full_text_db_path or DB_PATH).resolve()
    out = Path(output_dir or E5_INDEX_DIR).resolve()
    model_name = embedding_model or EMBEDDING_MODEL

    if not kv.is_dir():
        raise FileNotFoundError(f"full_text_db directory does not exist: {kv}")

    records = load_corpus_records_from_full_text_db(
        kv, chunk_size, chunk_overlap, max_papers=max_papers
    )
    if not records:
        raise RuntimeError(
            f"No chunks produced from {kv}. Ensure keys/values exist in full_text_db."
        )

    contents = [r["content"] for r in records]
    model = load_embedding_model(model_name, device=device)
    dim_model = model.get_sentence_embedding_dimension()
    if EMBEDDING_DIM and dim_model != EMBEDDING_DIM:
        logger.warning(
            "config.EMBEDDING_DIM=%s but model reports %s — using model dimension %s",
            EMBEDDING_DIM,
            dim_model,
            dim_model,
        )

    vectors = encode_chunks(model, contents, batch_size)
    index = build_faiss_ip_index(vectors)

    rows: List[Tuple[str, int, str, str]] = []
    for i, rec in enumerate(records):
        rows.append(
            (
                rec["doc_row_id"],
                i,
                rec["content"],
                rec["paper_id"],
            )
        )

    write_faiss_artifacts(out, index, rows)

    stats: Dict[str, Any] = {
        "n_vectors": int(index.ntotal),
        "dimension": int(vectors.shape[1]),
        "n_source_chunks": len(records),
        "output_dir": str(out),
        "faiss_path": str(out / "faiss_index"),
        "sqlite_path": str(out / "index_store.db"),
        "embedding_model": model_name,
        "source_full_text_db": str(kv),
    }
    if max_papers is not None:
        stats["max_papers"] = max_papers
    return stats


# ---------------------------------------------------------------------------
# Embedding & FAISS
# ---------------------------------------------------------------------------


def load_embedding_model(
    model_name: str, device: Optional[str] = None
) -> SentenceTransformer:
    dev = device or ("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    logger.info("Loading SentenceTransformer: %s on %s", model_name, dev)
    model = SentenceTransformer(model_name, device=dev)
    return model


def e5_passage_prefix(text: str) -> str:
    t = text.strip()
    low = t.lower()
    if low.startswith("passage:") or low.startswith("query:"):
        return t
    return f"passage: {t}"


def encode_chunks(
    model: SentenceTransformer,
    contents: Sequence[str],
    batch_size: int,
) -> np.ndarray:
    texts = [e5_passage_prefix(c) for c in contents]
    logger.info("Encoding %d passages (batch_size=%d)...", len(texts), batch_size)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_faiss_ip_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# SQLite (schema aligned with unified_arxiv_retriever.E5DirectRetriever)
# ---------------------------------------------------------------------------


def write_index_store(
    db_path: Path,
    rows: Sequence[Tuple[str, int, str, str]],
) -> None:
    """
    Args:
        rows: (document_id, faiss_vector_id, content, paper_id)
    """
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE document (
                id TEXT PRIMARY KEY,
                vector_id TEXT,
                content TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE meta_document (
                document_id TEXT,
                name TEXT,
                value TEXT,
                PRIMARY KEY (document_id, name)
            )
            """
        )
        for doc_id, vid, content, paper_id in rows:
            cur.execute(
                "INSERT INTO document (id, vector_id, content) VALUES (?, ?, ?)",
                (doc_id, str(vid), content),
            )
            cur.execute(
                """
                INSERT INTO meta_document (document_id, name, value)
                VALUES (?, 'paper_id', ?)
                """,
                (doc_id, paper_id),
            )
        conn.commit()
    finally:
        conn.close()


def write_faiss_artifacts(
    output_dir: Path,
    index: faiss.IndexFlatIP,
    rows: Sequence[Tuple[str, int, str, str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = output_dir / "faiss_index"
    db_path = output_dir / "index_store.db"
    pkl_path = output_dir / "faiss_document_mapping.pkl"

    faiss.write_index(index, str(faiss_path))
    write_index_store(db_path, rows)

    if pkl_path.exists():
        pkl_path.unlink()
        logger.info("Removed stale %s (will be rebuilt on first retrieval load)", pkl_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index_from_directory(
    source_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    embedding_model: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    batch_size: int = DEFAULT_BATCH_SIZE,
    extensions: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end index build. Returns a stats dict with n_docs, dimension, paths.

    Compatible with config.E5_INDEX_DIR layout expected by E5DirectRetriever.
    """
    src = Path(source_dir).resolve()
    out = Path(output_dir or E5_INDEX_DIR).resolve()
    ext = tuple(extensions) if extensions else DEFAULT_EXTENSIONS
    model_name = embedding_model or EMBEDDING_MODEL

    if not src.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    records = load_corpus_records(src, ext, chunk_size, chunk_overlap)
    if not records:
        raise RuntimeError(
            f"No chunks produced from {src}. Check extensions {ext} and file contents."
        )

    contents = [r["content"] for r in records]
    model = load_embedding_model(model_name, device=device)
    dim_model = model.get_sentence_embedding_dimension()
    if EMBEDDING_DIM and dim_model != EMBEDDING_DIM:
        logger.warning(
            "config.EMBEDDING_DIM=%s but model reports %s — using model dimension %s",
            EMBEDDING_DIM,
            dim_model,
            dim_model,
        )

    vectors = encode_chunks(model, contents, batch_size)
    index = build_faiss_ip_index(vectors)

    rows: List[Tuple[str, int, str, str]] = []
    for i, rec in enumerate(records):
        rows.append(
            (
                rec["doc_row_id"],
                i,
                rec["content"],
                rec["paper_id"],
            )
        )

    write_faiss_artifacts(out, index, rows)

    stats = {
        "n_vectors": int(index.ntotal),
        "dimension": int(vectors.shape[1]),
        "n_source_chunks": len(records),
        "output_dir": str(out),
        "faiss_path": str(out / "faiss_index"),
        "sqlite_path": str(out / "index_store.db"),
        "embedding_model": model_name,
    }
    return stats


def print_index_summary(stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("Index build complete (E5DirectRetriever-compatible layout)")
    print("=" * 60)
    print(f"  Documents (vectors): {stats['n_vectors']}")
    print(f"  Vector dimension:    {stats['dimension']}")
    print(f"  Embedding model:     {stats['embedding_model']}")
    print(f"  Output directory:    {stats['output_dir']}")
    print(f"  FAISS file:          {stats['faiss_path']}")
    print(f"  SQLite store:        {stats['sqlite_path']}")
    if stats.get("max_papers") is not None:
        print(f"  max_papers (subset): {stats['max_papers']}")
    print("=" * 60 + "\n")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build FAISS + SQLite index for SQuAI (E5 + index_store.db)"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Root directory containing .txt / .md / .jsonl files (recursive)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"Index output directory (default: config.E5_INDEX_DIR = {E5_INDEX_DIR})",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help=f"HuggingFace embedding model (default: config.EMBEDDING_MODEL)",
    )
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--extensions",
        type=str,
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated file extensions, e.g. .txt,.md,.jsonl",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device override, e.g. cuda or cpu",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.verbose)
    ext_tuple = tuple(x.strip() for x in args.extensions.split(",") if x.strip())

    try:
        stats = build_index_from_directory(
            args.source_dir,
            output_dir=args.output_dir,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            extensions=ext_tuple,
            device=args.device,
        )
    except Exception as e:
        logger.exception("Index build failed: %s", e)
        return 1

    print_index_summary(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
