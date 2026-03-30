#!/usr/bin/env python3
"""
Build SQuAI full-text key-value store (LevelDB or SQLite compat) from local JSONL corpora.

Expected JSONL format: unarXive-like records with paper_id, metadata.title, abstract,
and optional body_text[]. Keys match BM25 metadata paper_id for retrieval alignment.

Usage (from repo root):
  python scripts/build_full_text_db.py
  python scripts/build_full_text_db.py --jsonl squai_data/test/arXiv_src_2212_086.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from config import DB_PATH
from sqlite_compat import open_db

logger = logging.getLogger("build_full_text_db")


def _abstract_text(paper: Dict[str, Any]) -> str:
    a = paper.get("abstract")
    if isinstance(a, dict):
        return (a.get("text") or "").strip()
    if isinstance(a, str):
        return a.strip()
    meta = paper.get("metadata") or {}
    ma = meta.get("abstract")
    if isinstance(ma, str):
        return ma.strip()
    return ""


def _title_text(paper: Dict[str, Any]) -> str:
    meta = paper.get("metadata") or {}
    t = meta.get("title") or paper.get("title")
    if isinstance(t, str):
        return t.strip()
    return ""


def _body_text(paper: Dict[str, Any]) -> str:
    parts: List[str] = []
    for block in paper.get("body_text") or []:
        if not isinstance(block, dict):
            continue
        sec = (block.get("section") or "").strip()
        txt = (block.get("text") or "").strip()
        if not txt:
            continue
        if sec:
            parts.append(f"{sec}\n{txt}")
        else:
            parts.append(txt)
    return "\n\n".join(parts)


def format_full_text(paper_id: str, title: str, abstract: str, body: str) -> str:
    """Layout compatible with run_SQuAI PaperTitleExtractor / LevelDB patterns."""
    lines = [
        f"Content for {paper_id}:",
        title,
        "",
        f"abstract: {abstract}",
        "",
    ]
    if body:
        lines.append(body)
    return "\n".join(lines).strip()


def paper_to_full_text(paper: Dict[str, Any]) -> Optional[str]:
    pid = paper.get("paper_id")
    if not pid:
        return None
    pid = str(pid).strip()
    title = _title_text(paper)
    abstract = _abstract_text(paper)
    body = _body_text(paper)
    return format_full_text(pid, title, abstract, body)


def ingest_jsonl_file(path: str, db, stats: Dict[str, int]) -> None:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                paper = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip %s line %s: %s", path, line_no, e)
                stats["skipped"] += 1
                continue
            if not isinstance(paper, dict):
                stats["skipped"] += 1
                continue
            full = paper_to_full_text(paper)
            if not full:
                stats["skipped"] += 1
                continue
            pid = str(paper.get("paper_id", "")).strip()
            db.put(pid.encode("utf-8"), full.encode("utf-8"))
            stats["written"] += 1


def build_full_text_db(
    jsonl_paths: List[str],
    output_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Write all papers from given JSONL files into the KV store at config DB_PATH
    (or output_dir if provided).
    """
    out = output_dir or DB_PATH
    os.makedirs(out, exist_ok=True)

    # Remove misplaced corpus files inside DB dir (e.g. *.jsonl copied by mistake)
    for stray in glob.glob(os.path.join(out, "*.jsonl")):
        try:
            os.remove(stray)
            logger.info("Removed stray JSONL from DB directory: %s", stray)
        except OSError as e:
            logger.warning("Could not remove %s: %s", stray, e)

    stats = {"written": 0, "skipped": 0, "files": 0}
    db = open_db(out, create_if_missing=True)
    try:
        for jp in jsonl_paths:
            if not os.path.isfile(jp):
                logger.warning("File not found: %s", jp)
                continue
            stats["files"] += 1
            logger.info("Ingesting %s", jp)
            ingest_jsonl_file(jp, db, stats)
    finally:
        db.close()

    return stats


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Build full_text_db from JSONL corpus")
    parser.add_argument(
        "--jsonl",
        action="append",
        default=None,
        help="JSONL file (repeatable). Default: squai_data/test/*.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=f"KV store directory (default: config DB_PATH = {DB_PATH})",
    )
    args = parser.parse_args()

    if args.jsonl:
        paths = args.jsonl
    else:
        default_glob = os.path.join(_REPO_ROOT, "squai_data", "test", "*.jsonl")
        paths = sorted(glob.glob(default_glob))

    if not paths:
        logger.error("No JSONL files found. Pass --jsonl or place *.jsonl under squai_data/test/")
        return 1

    stats = build_full_text_db(paths, output_dir=args.output_dir)
    print("\n=== full_text_db build complete ===")
    print(f"  JSONL files processed: {stats['files']}")
    print(f"  Records written:        {stats['written']}")
    print(f"  Lines skipped:          {stats['skipped']}")
    print(f"  Output directory:       {args.output_dir or DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
