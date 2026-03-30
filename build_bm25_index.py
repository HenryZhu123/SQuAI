#!/usr/bin/env python3
"""
Build a LlamaIndex BM25 persisted index from full_text_db (same layout as bm25_retrieval / FastLlamaIndexBM25Retriever).

Writes under output_dir (default: config BM25_INDEX_DIR), e.g. squai_data/bm25_retriever.
Each document is one paper; indexed text is a prefix of full text (BM25 retrieval returns this as the "abstract" channel).
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import Stemmer
    from llama_index.core import Document
    from llama_index.core.storage.docstore import SimpleDocumentStore
    from llama_index.retrievers.bm25 import BM25Retriever
except ImportError as e:
    raise ImportError(
        "BM25 index build requires llama-index and llama-index-retrievers-bm25. "
        "Install: pip install llama-index llama-index-retrievers-bm25 PyStemmer"
    ) from e

try:
    from config import BM25_INDEX_DIR, DB_PATH
except ImportError:
    _ROOT = Path(__file__).resolve().parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from config import BM25_INDEX_DIR, DB_PATH

from sqlite_compat import open_db

logger = logging.getLogger("build_bm25_index")

# Max characters per paper indexed into BM25 (keeps memory and index size reasonable)
DEFAULT_INDEX_TEXT_CHARS = 20000
DEFAULT_SIMILARITY_TOP_K = 5


def load_paper_pairs_from_full_text_db(
    kv_path: Path,
    max_papers: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """
    Load (paper_id, full_text) from KV store, sorted by paper_id.
    If max_papers is set, keep only the first max_papers after sorting.
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

    total = len(pairs)
    pairs.sort(key=lambda x: x[0])
    if max_papers is not None:
        pairs = pairs[:max_papers]

    logger.info(
        "full_text_db: using %d papers for BM25 (total keys in store: %d)%s",
        len(pairs),
        total,
        f", max_papers={max_papers}" if max_papers is not None else "",
    )
    return pairs


def build_bm25_from_full_text_db(
    full_text_db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    max_papers: Optional[int] = None,
    index_text_chars: int = DEFAULT_INDEX_TEXT_CHARS,
    similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
) -> Dict[str, Any]:
    """
    Build and persist LlamaIndex BM25 under output_dir.

    Args:
        full_text_db_path: KV directory (default: config.DB_PATH).
        output_dir: Persist directory (default: config.BM25_INDEX_DIR).
        max_papers: Subset size after sorting by paper_id; None = all papers.
        index_text_chars: Only first N characters of each full text are indexed.
        similarity_top_k: BM25Retriever default top_k at persist time.
    """
    kv = Path(full_text_db_path or DB_PATH).resolve()
    out = Path(output_dir or BM25_INDEX_DIR).resolve()

    if not kv.is_dir():
        raise FileNotFoundError(f"full_text_db directory does not exist: {kv}")

    pairs = load_paper_pairs_from_full_text_db(kv, max_papers=max_papers)
    if not pairs:
        raise RuntimeError(f"No papers loaded from {kv}")

    docs: List[Document] = []
    for paper_id, text in pairs:
        body = text.strip()
        if index_text_chars > 0 and len(body) > index_text_chars:
            body = body[:index_text_chars]
        docs.append(
            Document(
                text=body,
                metadata={"paper_id": paper_id},
            )
        )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building BM25 over %d documents (similarity_top_k=%s)...",
        len(docs),
        similarity_top_k,
    )
    docstore = SimpleDocumentStore()
    docstore.add_documents(docs)
    retriever = BM25Retriever.from_defaults(
        docstore=docstore,
        similarity_top_k=similarity_top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )
    retriever.persist(str(out))
    logger.info("BM25 index persisted to %s", out)

    stats: Dict[str, Any] = {
        "n_documents": len(docs),
        "output_dir": str(out),
        "source_full_text_db": str(kv),
        "index_text_chars": index_text_chars,
        "similarity_top_k": similarity_top_k,
    }
    if max_papers is not None:
        stats["max_papers"] = max_papers
    return stats


def print_bm25_summary(stats: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("BM25 index build complete (LlamaIndex persist layout)")
    print("=" * 60)
    print(f"  Documents:        {stats['n_documents']}")
    print(f"  Output directory: {stats['output_dir']}")
    print(f"  Source DB:        {stats['source_full_text_db']}")
    print(f"  Indexed chars/doc (max): {stats['index_text_chars']}")
    if stats.get("max_papers") is not None:
        print(f"  max_papers:       {stats['max_papers']}")
    print("=" * 60 + "\n")
