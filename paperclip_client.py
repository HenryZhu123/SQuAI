import json
import logging
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

_ARXIV_ID_RE = re.compile(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b")
_PAPERS_PATH_RE = re.compile(r"/papers/([^/\s]+)")
_NUMBERED_TITLE_RE = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")
_ID_KEYVAL_RE = re.compile(
    r"\b(?:paper[_ -]?id|doc(?:ument)?[_ -]?id|id)\s*[:=]\s*([A-Za-z0-9._:-]+)",
    re.IGNORECASE,
)
_PMID_RE = re.compile(r"\bpmid\s*[:#]?\s*([0-9]{6,10})\b", re.IGNORECASE)
_PMCID_RE = re.compile(r"\bpmcid\s*[:#]?\s*(PMC[0-9]+)\b", re.IGNORECASE)
# Standalone PMC article ids (e.g. "PMC10867117 · Frontiers ...")
_PMC_STANDALONE_RE = re.compile(r"\b(PMC\d+)\b", re.IGNORECASE)
# bioRxiv / medRxiv style ids (e.g. "bio_998895f972a1 · bioRxiv · ...")
_BIO_PREPRINT_RE = re.compile(r"\b(bio_[A-Za-z0-9]+)\b")
_MEDRXIV_PREPRINT_RE = re.compile(r"\b(medrxiv_[A-Za-z0-9]+)\b", re.IGNORECASE)
# Middle dot (U+00B7) metadata line: "<paper_id> · venue · date"
_MIDDLE_DOT_SPLIT_RE = re.compile(r"\s*[\u00b7]\s*")


@dataclass
class PaperclipSearchResult:
    paper_id: str
    snippet: str
    title: Optional[str] = None
    raw_line: str = ""


class PaperclipClient:
    """Thin subprocess wrapper for paperclip CLI with robust output parsing."""

    def __init__(self, search_timeout_sec: int = 20, cat_timeout_sec: int = 30):
        self.search_timeout_sec = max(1, int(search_timeout_sec))
        self.cat_timeout_sec = max(1, int(cat_timeout_sec))

    def _run(self, args: List[str], timeout_sec: int) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout_sec,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("paperclip command not found in PATH") from exc
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            msg = stderr or stdout or f"paperclip exited with code {proc.returncode}"
            raise RuntimeError(msg)
        return proc

    def search(self, query: str, limit: int = 5) -> List[PaperclipSearchResult]:
        """
        Search papers and parse ids from diverse stdout formats.

        Supported/assumed output styles:
        1) JSON array or JSONL with fields like id/paper_id/title/snippet.
        2) tab/pipe separated plain lines containing an id-like token.
        3) free text lines containing '/papers/<id>' or arXiv-like ids.
        """
        cmd = ["paperclip", "search", query]
        if int(limit) > 0:
            # Keep optional to avoid breaking installations that ignore this flag.
            cmd.extend(["--limit", str(int(limit))])
        proc = self._run(cmd, timeout_sec=self.search_timeout_sec)
        return self._parse_search_output(proc.stdout, limit=max(1, int(limit)))

    def cat(self, path: str) -> str:
        proc = self._run(["paperclip", "cat", path], timeout_sec=self.cat_timeout_sec)
        return proc.stdout or ""

    def get_meta_json(self, paper_id: str) -> Optional[dict]:
        raw = self.cat(f"/papers/{paper_id}/meta.json")
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("paperclip meta.json parse failed for %s", paper_id)
            return None

    def get_content_lines(self, paper_id: str) -> str:
        return self.cat(f"/papers/{paper_id}/content.lines")

    def _parse_search_output(self, text: str, limit: int) -> List[PaperclipSearchResult]:
        text = (text or "").strip()
        if not text:
            return []

        parsed = self._try_parse_json_or_jsonl(text)
        if not parsed:
            parsed = self._parse_lines(text)

        # Deduplicate by paper_id preserving order.
        uniq = []
        seen = set()
        for item in parsed:
            pid = (item.paper_id or "").strip()
            if not pid or pid in seen:
                continue
            seen.add(pid)
            uniq.append(item)
            if len(uniq) >= limit:
                break
        return uniq

    def _try_parse_json_or_jsonl(self, text: str) -> List[PaperclipSearchResult]:
        out: List[PaperclipSearchResult] = []
        # Try full JSON first.
        try:
            data = json.loads(text)
            if isinstance(data, list):
                for obj in data:
                    item = self._from_mapping(obj)
                    if item:
                        out.append(item)
            elif isinstance(data, dict):
                maybe_rows = data.get("results") or data.get("data")
                if isinstance(maybe_rows, list):
                    for obj in maybe_rows:
                        item = self._from_mapping(obj)
                        if item:
                            out.append(item)
            if out:
                return out
        except json.JSONDecodeError:
            pass

        # Try JSONL.
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                return []
            item = self._from_mapping(obj)
            if item:
                out.append(item)
        return out

    def _from_mapping(self, obj: object) -> Optional[PaperclipSearchResult]:
        if not isinstance(obj, dict):
            return None
        paper_id = (
            obj.get("paper_id")
            or obj.get("id")
            or obj.get("doc_id")
            or obj.get("document_id")
        )
        if not isinstance(paper_id, str):
            path_hint = str(obj.get("path", ""))
            m = _PAPERS_PATH_RE.search(path_hint)
            if m:
                paper_id = m.group(1)
        if not isinstance(paper_id, str) or not paper_id.strip():
            return None
        snippet = str(
            obj.get("snippet")
            or obj.get("abstract")
            or obj.get("summary")
            or obj.get("text")
            or ""
        )
        title = obj.get("title")
        return PaperclipSearchResult(
            paper_id=paper_id.strip(),
            snippet=snippet.strip(),
            title=title.strip() if isinstance(title, str) else None,
            raw_line=json.dumps(obj, ensure_ascii=False)[:1000],
        )

    def _parse_lines(self, text: str) -> List[PaperclipSearchResult]:
        # Parse "numbered result blocks" first (matches paperclip's rich text output):
        # Found 92 papers ...
        # 1. <title>
        #    <authors>
        #    <venue/date>
        #    <url>
        #    "<snippet>"
        blocks = self._split_numbered_blocks(text)
        if blocks:
            block_items = self._parse_numbered_blocks(blocks)
            if block_items:
                return block_items

        # Fallback for simpler outputs: parse line-by-line but with strict ID rules only.
        out: List[PaperclipSearchResult] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            # Skip known non-result header lines.
            if line.lower().startswith("found ") and " paper" in line.lower():
                continue
            paper_id = self._extract_id_from_line(line)
            if not paper_id:
                continue
            snippet = line
            out.append(
                PaperclipSearchResult(
                    paper_id=paper_id,
                    snippet=snippet,
                    title=None,
                    raw_line=raw,
                )
            )
        return out

    def _split_numbered_blocks(self, text: str) -> List[List[str]]:
        blocks: List[List[str]] = []
        current: List[str] = []
        for raw in text.splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            if _NUMBERED_TITLE_RE.match(line):
                if current:
                    blocks.append(current)
                current = [line]
            elif current:
                current.append(line)
        if current:
            blocks.append(current)
        return blocks

    def _parse_numbered_blocks(self, blocks: List[List[str]]) -> List[PaperclipSearchResult]:
        out: List[PaperclipSearchResult] = []
        for block in blocks:
            head = block[0].strip()
            m = _NUMBERED_TITLE_RE.match(head)
            if not m:
                continue
            title = m.group(2).strip()
            body = "\n".join(block)
            paper_id = self._extract_id_from_text(body)
            if not paper_id:
                # Intentionally skip result block without trusted id to avoid false positives.
                logger.debug("paperclip parse: skip block without trusted id: %s", title[:80])
                continue

            snippet = ""
            for ln in block[1:]:
                s = ln.strip()
                if s.startswith('"') and s.endswith('"') and len(s) >= 2:
                    snippet = s.strip('"')
                    break
            if not snippet:
                snippet = title
            out.append(
                PaperclipSearchResult(
                    paper_id=paper_id,
                    snippet=snippet,
                    title=title,
                    raw_line=body[:1000],
                )
            )
        return out

    def _id_from_middle_dot_metadata_line(self, text: str) -> Optional[str]:
        """Paperclip human output: '<paper_id> · journal · YYYY-MM-DD' on one line."""
        for raw in text.splitlines():
            line = raw.strip()
            if "\u00b7" not in line:
                continue
            parts = _MIDDLE_DOT_SPLIT_RE.split(line, maxsplit=1)
            head = parts[0].strip() if parts else ""
            if not head:
                continue
            m_pmc = re.fullmatch(r"(PMC)(\d+)", head, flags=re.IGNORECASE)
            if m_pmc:
                return "PMC" + m_pmc.group(2)
            if _BIO_PREPRINT_RE.fullmatch(head):
                return head
            if _MEDRXIV_PREPRINT_RE.fullmatch(head):
                return head.lower() if head.lower().startswith("medrxiv_") else head
        return None

    def _extract_id_from_text(self, text: str) -> Optional[str]:
        m = _PAPERS_PATH_RE.search(text)
        if m:
            return m.group(1)

        m = _ID_KEYVAL_RE.search(text)
        if m:
            return m.group(1)

        m = _ARXIV_ID_RE.search(text)
        if m:
            return m.group(0)

        m = _PMC_STANDALONE_RE.search(text)
        if m:
            return m.group(1).upper()

        m = _BIO_PREPRINT_RE.search(text)
        if m:
            return m.group(1)

        m = _MEDRXIV_PREPRINT_RE.search(text)
        if m:
            return m.group(1)

        m = _PMCID_RE.search(text)
        if m:
            return m.group(1)

        m = _PMID_RE.search(text)
        if m:
            return m.group(1)

        mid = self._id_from_middle_dot_metadata_line(text)
        if mid:
            return mid

        return None

    def _extract_id_from_line(self, line: str) -> Optional[str]:
        # Strict extraction only (no generic "first token" fallback).
        return self._extract_id_from_text(line)
