import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from config import (
    EMBEDDING_MODEL,
    PAPERCLIP_AGENT2_ENABLED,
    PAPERCLIP_AGENT2_TOP_CHUNKS,
    PAPERCLIP_CACHE_MAX_BYTES,
    PAPERCLIP_CACHE_MAX_ITEMS,
    PAPERCLIP_CAT_TIMEOUT_SEC,
    PAPERCLIP_CAT_WORKERS,
    PAPERCLIP_CHUNK_OVERLAP,
    PAPERCLIP_CHUNK_SIZE,
    PAPERCLIP_SEARCH_TIMEOUT_SEC,
    PAPERCLIP_SEARCH_TOP_K,
)
from paperclip_chunk_hybrid import (
    HybridChunkRanker,
    chunk_text,
    compact_text_for_judge,
    format_ranked_chunks_for_agent2,
)
from paperclip_client import PaperclipClient, PaperclipSearchResult

logger = logging.getLogger(__name__)


@dataclass
class Agent2DocInput:
    doc_id: str
    agent2_text: str
    judge_text: str
    full_text: str


class _LRUTextCache:
    def __init__(self, max_items: int, max_bytes: int):
        self.max_items = max(1, int(max_items))
        self.max_bytes = max(1024, int(max_bytes))
        self._items: "OrderedDict[str, str]" = OrderedDict()
        self._size_bytes = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def put(self, key: str, value: str) -> None:
        if key is None:
            return
        value = value or ""
        new_bytes = len(value.encode("utf-8", errors="ignore"))
        with self._lock:
            if key in self._items:
                old = self._items.pop(key)
                self._size_bytes -= len(old.encode("utf-8", errors="ignore"))
            self._items[key] = value
            self._size_bytes += new_bytes
            self._trim()

    def _trim(self):
        while self._items and (
            len(self._items) > self.max_items or self._size_bytes > self.max_bytes
        ):
            _, value = self._items.popitem(last=False)
            self._size_bytes -= len(value.encode("utf-8", errors="ignore"))


class PaperclipBackedRetriever:
    """
    Wrapper retriever:
    - keeps existing Retriever interface for callers
    - optionally uses paperclip for Agent2 candidate papers + chunk retrieval
    """

    def __init__(self, base_retriever, alpha: float = 0.65):
        self.base = base_retriever
        self.paperclip_enabled = bool(PAPERCLIP_AGENT2_ENABLED)
        self.alpha = float(alpha)
        self.paperclip = PaperclipClient(
            search_timeout_sec=PAPERCLIP_SEARCH_TIMEOUT_SEC,
            cat_timeout_sec=PAPERCLIP_CAT_TIMEOUT_SEC,
        )
        self.ranker = HybridChunkRanker(alpha=self.alpha, embedding_model=EMBEDDING_MODEL)
        self.chunk_size = PAPERCLIP_CHUNK_SIZE
        self.chunk_overlap = PAPERCLIP_CHUNK_OVERLAP
        self.top_chunks = max(1, int(PAPERCLIP_AGENT2_TOP_CHUNKS))
        self.search_top_k = max(1, int(PAPERCLIP_SEARCH_TOP_K))
        self.cat_workers = max(1, int(PAPERCLIP_CAT_WORKERS))
        self.content_cache = _LRUTextCache(
            max_items=PAPERCLIP_CACHE_MAX_ITEMS, max_bytes=PAPERCLIP_CACHE_MAX_BYTES
        )
        self._meta_cache: Dict[str, dict] = {}

    def retrieve_abstracts(self, query: str, top_k: int = None) -> List[Tuple[str, str]]:
        if not self.paperclip_enabled:
            return self.base.retrieve_abstracts(query, top_k=top_k)

        top = max(1, int(top_k or self.search_top_k))
        try:
            rows = self.paperclip.search(query, limit=top)
        except Exception as exc:
            logger.warning("paperclip search failed; fallback to local retriever: %s", exc)
            return self.base.retrieve_abstracts(query, top_k=top_k)

        if not rows:
            logger.warning("paperclip search returned no rows; fallback to local retriever")
            return self.base.retrieve_abstracts(query, top_k=top_k)

        out: List[Tuple[str, str]] = []
        for row in rows[:top]:
            snippet = row.snippet or ""
            title = row.title or ""
            summary = f"Title: {title}\nSnippet: {snippet}".strip()
            out.append((summary, row.paper_id))
        logger.info("paperclip retrieval: %d candidates for query", len(out))
        return out

    def get_full_texts(self, doc_ids: List[str], db=None) -> List[Tuple[str, str]]:
        if not self.paperclip_enabled:
            return self.base.get_full_texts(doc_ids, db=db)
        out: List[Tuple[str, str]] = []
        for doc_id in doc_ids:
            content = self._get_content_lines(doc_id)
            if not content:
                continue
            out.append((content, doc_id))
        if not out:
            logger.warning("paperclip full-text fetch failed; fallback to local retriever")
            return self.base.get_full_texts(doc_ids, db=db)
        return out

    def build_agent2_documents(
        self, query: str, retrieved_abstracts: Sequence[Tuple[str, str]]
    ) -> Dict[str, Agent2DocInput]:
        """
        Build Agent2 inputs from paperclip full text chunks.
        Returns mapping doc_id -> Agent2DocInput
        """
        if not self.paperclip_enabled:
            return {}
        if not retrieved_abstracts:
            return {}

        # Pull full text concurrently.
        wanted_ids = [doc_id for _, doc_id in retrieved_abstracts]
        full_by_id: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.cat_workers) as executor:
            future_to_id = {
                executor.submit(self._get_content_lines, doc_id): doc_id for doc_id in wanted_ids
            }
            for future in as_completed(future_to_id):
                doc_id = future_to_id[future]
                try:
                    text = future.result()
                except Exception as exc:
                    logger.warning("paperclip cat failed for %s: %s", doc_id, exc)
                    text = ""
                if text:
                    full_by_id[doc_id] = text

        out: Dict[str, Agent2DocInput] = {}
        for abstract_or_snippet, doc_id in retrieved_abstracts:
            full = full_by_id.get(doc_id, "")
            if not full:
                # Fallback for this doc: keep old abstract behavior.
                fallback_text = abstract_or_snippet or ""
                out[doc_id] = Agent2DocInput(
                    doc_id=doc_id,
                    agent2_text=fallback_text,
                    judge_text=compact_text_for_judge(fallback_text, max_chars=1200),
                    full_text=fallback_text,
                )
                continue

            chunks = chunk_text(
                full,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            ranked = self.ranker.rank(query, chunks, top_k=self.top_chunks)
            meta = self._get_meta(doc_id) or {}
            title = None
            if isinstance(meta, dict):
                title = meta.get("title")

            agent2_doc = format_ranked_chunks_for_agent2(
                paper_id=doc_id,
                title=title,
                snippet=abstract_or_snippet,
                ranked_chunks=ranked,
            )
            out[doc_id] = Agent2DocInput(
                doc_id=doc_id,
                agent2_text=agent2_doc,
                judge_text=compact_text_for_judge(agent2_doc, max_chars=1800),
                full_text=full,
            )
            logger.info(
                "paperclip chunks selected for %s: %d/%d",
                doc_id,
                len(ranked),
                len(chunks),
            )
        return out

    def _get_content_lines(self, doc_id: str) -> str:
        cached = self.content_cache.get(doc_id)
        if cached is not None:
            return cached
        text = self.paperclip.get_content_lines(doc_id)
        text = (text or "").strip()
        if text:
            self.content_cache.put(doc_id, text)
            logger.info("paperclip cat fetched %s (%d chars)", doc_id, len(text))
        return text

    def _get_meta(self, doc_id: str) -> Optional[dict]:
        if doc_id in self._meta_cache:
            return self._meta_cache[doc_id]
        try:
            meta = self.paperclip.get_meta_json(doc_id)
        except Exception:
            meta = None
        self._meta_cache[doc_id] = meta or {}
        return meta

    def close(self):
        if hasattr(self.base, "close"):
            self.base.close()
