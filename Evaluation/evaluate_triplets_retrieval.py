#!/usr/bin/env python3
"""
Evaluate retrieval quality on triplets:
  (query, ground_truth_doc_id, answer)

Metrics:
  - Recall@K
  - Precision@K
  - MRR

Usage:
  python Evaluation/evaluate_triplets_retrieval.py \
    --triplets Evaluation/evaluation_triplets_50.jsonl \
    --retriever_type hybrid \
    --top_k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple



@dataclass
class Triplet:
    query: str
    ground_truth_doc_id: str
    answer: str


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_triplets(path: str, max_samples: Optional[int] = None) -> List[Triplet]:
    items: List[Triplet] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
            if not all(k in obj for k in ("query", "ground_truth_doc_id", "answer")):
                raise ValueError(
                    f"Missing required keys at line {line_no}. "
                    "Required: query, ground_truth_doc_id, answer"
                )
            items.append(
                Triplet(
                    query=str(obj["query"]),
                    ground_truth_doc_id=str(obj["ground_truth_doc_id"]),
                    answer=str(obj["answer"]),
                )
            )
            if max_samples and len(items) >= max_samples:
                break
    return items


def _candidate_gt_ids(ground_truth_doc_id: str) -> Tuple[str, str]:
    """
    Return (raw_id, paper_level_id).
    Triplets are currently chunk-level ids like:
      2212.11739_4be59cf0bb_c1
    while run_SQuAI retriever returns paper-level ids like:
      2212.11739
    """
    raw = str(ground_truth_doc_id).strip()
    paper_level = raw.split("_", 1)[0] if "_" in raw else raw
    return raw, paper_level


def evaluate(
    triplets: List[Triplet],
    retriever_type: str,
    top_k: int,
    alpha: float,
) -> dict:
    # Late imports so script can be run from Evaluation/
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from config import E5_INDEX_DIR, BM25_INDEX_DIR, DB_PATH
    from run_SQuAI import initialize_retriever

    # Build retriever through run_SQuAI API as requested.
    retriever = initialize_retriever(
        retriever_type=retriever_type,
        e5_index_dir=E5_INDEX_DIR,
        bm25_index_dir=BM25_INDEX_DIR,
        db_path=DB_PATH,
        top_k=top_k,
        alpha=alpha,
    )

    recalls = []
    precisions = []
    reciprocal_ranks = []
    misses = []

    try:
        for t in triplets:
            try:
                results = retriever.retrieve_abstracts(t.query, top_k=top_k)
                ranked_doc_ids = [doc_id for _, doc_id in results]
            except Exception as e:
                ranked_doc_ids = []
                misses.append(
                    {"query": t.query, "doc_id": t.ground_truth_doc_id, "error": str(e)}
                )

            gt_raw, gt_paper = _candidate_gt_ids(t.ground_truth_doc_id)
            candidate_ids = [gt_raw]
            if gt_paper != gt_raw:
                candidate_ids.append(gt_paper)

            hit_positions = [ranked_doc_ids.index(cid) for cid in candidate_ids if cid in ranked_doc_ids]
            hit = len(hit_positions) > 0

            recall_at_k = 1.0 if hit else 0.0
            precision_at_k = (1.0 / float(top_k)) if hit else 0.0

            if hit:
                rank = min(hit_positions) + 1  # 1-based
                rr = 1.0 / float(rank)
            else:
                rr = 0.0
                misses.append({"query": t.query, "doc_id": gt_raw, "paper_id": gt_paper})

            recalls.append(recall_at_k)
            precisions.append(precision_at_k)
            reciprocal_ranks.append(rr)
    finally:
        if hasattr(retriever, "close"):
            try:
                retriever.close()
            except Exception:
                pass

    n = len(triplets)
    metrics = {
        "samples": n,
        "retriever_type": retriever_type,
        "top_k": top_k,
        "alpha": alpha,
        "ground_truth_matching": "raw chunk id OR paper-level id parsed from chunk id",
        f"recall@{top_k}": (sum(recalls) / n) if n else 0.0,
        f"precision@{top_k}": (sum(precisions) / n) if n else 0.0,
        "MRR": (sum(reciprocal_ranks) / n) if n else 0.0,
        "hits": int(sum(1 for x in recalls if x > 0)),
        "misses": n - int(sum(1 for x in recalls if x > 0)),
        "miss_examples": misses[:20],
    }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on (query, doc_id, answer) triplets")
    parser.add_argument(
        "--triplets",
        type=str,
        default=os.path.join(_repo_root(), "Evaluation", "evaluation_triplets_50.jsonl"),
        help="Path to triplets JSONL",
    )
    parser.add_argument("--retriever_type", choices=["e5", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.65, help="Hybrid alpha (ignored for pure e5/bm25)")
    parser.add_argument("--max_samples", type=int, default=0, help="Debug mode limit; 0 means all")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    triplets = _load_triplets(args.triplets, max_samples=(args.max_samples or None))
    if not triplets:
        print("No triplets loaded.")
        return 1

    metrics = evaluate(
        triplets=triplets,
        retriever_type=args.retriever_type,
        top_k=args.top_k,
        alpha=args.alpha,
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
