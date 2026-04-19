import logging
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class ChunkScore:
    text: str
    dense_score: float
    sparse_score: float
    hybrid_score: float
    idx: int


def chunk_text(text: str, chunk_size: int = 1400, chunk_overlap: int = 250) -> List[str]:
    """
    Character-window chunking with overlap.
    Falls back to paragraph chunks when text is short.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunk_size = max(200, int(chunk_size))
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size - 1))

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end >= len(text):
            break
        start += step

    # Deduplicate exact duplicates while preserving order.
    seen = set()
    uniq = []
    for c in chunks:
        if c in seen:
            continue
        seen.add(c)
        uniq.append(c)
    return uniq


class HybridChunkRanker:
    """Chunk-level sparse + dense scoring with optional sentence-transformers backend."""

    def __init__(self, alpha: float = 0.65, embedding_model: str = "intfloat/e5-large-v2"):
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.embedding_model = embedding_model
        self._encoder = None
        self._encoder_failed = False

    def rank(
        self, query: str, chunks: Sequence[str], top_k: int = 3
    ) -> List[ChunkScore]:
        if not chunks:
            return []
        dense = self._dense_scores(query, chunks)
        sparse = self._sparse_scores(query, chunks)
        dense_n = _normalize(dense)
        sparse_n = _normalize(sparse)
        out: List[ChunkScore] = []
        for i, chunk in enumerate(chunks):
            hybrid = self.alpha * dense_n[i] + (1.0 - self.alpha) * sparse_n[i]
            out.append(
                ChunkScore(
                    text=chunk,
                    dense_score=float(dense_n[i]),
                    sparse_score=float(sparse_n[i]),
                    hybrid_score=float(hybrid),
                    idx=i,
                )
            )
        out.sort(key=lambda x: x.hybrid_score, reverse=True)
        return out[: max(1, int(top_k))]

    def _dense_scores(self, query: str, chunks: Sequence[str]) -> List[float]:
        encoder = self._load_encoder()
        if encoder is None:
            # Fallback: sparse-only mode (dense score neutral).
            return [0.0 for _ in chunks]
        try:
            query_vec = encoder.encode(
                [f"query: {query}"],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            doc_vecs = encoder.encode(
                [f"passage: {c}" for c in chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sims = np.dot(doc_vecs, query_vec)
            return [float(x) for x in sims.tolist()]
        except Exception as exc:
            logger.warning("Dense chunk scoring failed; fallback to sparse-only: %s", exc)
            return [0.0 for _ in chunks]

    def _load_encoder(self):
        if self._encoder is not None:
            return self._encoder
        if self._encoder_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
            return self._encoder
        except Exception as exc:
            logger.warning("Could not initialize sentence-transformers model: %s", exc)
            self._encoder_failed = True
            return None

    def _sparse_scores(self, query: str, chunks: Sequence[str]) -> List[float]:
        # Lightweight BM25-like scoring.
        q_tokens = _tokenize(query)
        if not q_tokens:
            return [0.0 for _ in chunks]
        docs_tokens = [_tokenize(c) for c in chunks]
        n_docs = len(chunks)
        avg_dl = sum(len(toks) for toks in docs_tokens) / max(1, n_docs)
        if avg_dl <= 0:
            return [0.0 for _ in chunks]

        # Document frequency.
        df = {}
        for tok in set(q_tokens):
            df[tok] = sum(1 for toks in docs_tokens if tok in toks)

        k1 = 1.5
        b = 0.75
        scores = []
        for toks in docs_tokens:
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            score = 0.0
            for q in q_tokens:
                freq = tf.get(q, 0)
                if freq <= 0:
                    continue
                idf = math.log(1.0 + (n_docs - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
                denom = freq + k1 * (1.0 - b + b * dl / avg_dl)
                score += idf * (freq * (k1 + 1.0) / max(1e-9, denom))
            scores.append(score)
        return scores


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _normalize(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [1.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def compact_text_for_judge(text: str, max_chars: int = 1800) -> str:
    """Build short summary text for Agent3 stability."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)].rstrip()
    tail = text[-int(max_chars * 0.3) :].lstrip()
    return f"{head}\n...\n{tail}"


def format_ranked_chunks_for_agent2(
    paper_id: str,
    title: Optional[str],
    snippet: str,
    ranked_chunks: Sequence[ChunkScore],
) -> str:
    title_text = title or "Unknown title"
    lines: List[str] = [
        f"Paper ID: {paper_id}",
        f"Title: {title_text}",
    ]
    if snippet:
        lines.append(f"Search snippet: {snippet}")
    lines.append("Retrieved chunks:")
    for i, chunk in enumerate(ranked_chunks, start=1):
        lines.append(
            f"[Chunk {i} | hybrid={chunk.hybrid_score:.3f} | dense={chunk.dense_score:.3f} | sparse={chunk.sparse_score:.3f}]"
        )
        lines.append(chunk.text)
    return "\n".join(lines)
