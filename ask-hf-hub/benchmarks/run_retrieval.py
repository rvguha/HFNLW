"""Retrieval benchmark: flat BM25 against fielded BM25F, over 100 queries.

Gold sets come from the query predicates in retrieval-queries.yaml, evaluated
against the served corpus. That keeps the benchmark free to run and independent
of any judge, but it constrains what can honestly be measured:

* A query whose gold set is a large fraction of the corpus cannot separate two
  rankings -- P@10 is near 1.0 for anything. Those are reported but excluded
  from the headline, and the cut is printed rather than hidden.
* `keyword_any` predicates define relevance partly by term presence, which
  flatters lexical retrieval. They are never the sole predicate on the narrow
  queries here, and known-item queries use `repo_any`, which has no such bias.

Usage: python benchmarks/run_retrieval.py [--sweep]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import asyncio  # noqa: E402

from predicates import gold_set, load_queries, trap_set  # noqa: E402

from askhub.adapters import load_file  # noqa: E402
from askhub.catalog import MemoryCatalog  # noqa: E402
from askhub.config import (  # noqa: E402
    DEFAULT_BM25_FIELD_B,
    DEFAULT_BM25_FIELD_WEIGHTS,
)
from askhub.models import RETRIEVAL_FIELDS  # noqa: E402

CORPUS = ROOT / "src/askhub/corpus/huggingface.jsonl"
QUERIES = Path(__file__).resolve().parent / "retrieval-queries.yaml"
# Above this share of the corpus a query is relevant to almost everything and
# stops discriminating between rankings.
BROAD_GOLD_SHARE = 0.05
FLAT = ({"name": 1.0, "description": 1.0, "body": 1.0},
        {"name": 0.75, "description": 0.75, "body": 0.75})
# Candidate weightings for combining the three embedding channels.
VECTOR_GRID = [
    {"name": 1.0, "description": 1.0, "body": 1.0},
    {"name": 2.0, "description": 1.0, "body": 1.0},
    {"name": 1.0, "description": 2.0, "body": 1.0},
    {"name": 1.0, "description": 1.0, "body": 2.0},
    {"name": 3.0, "description": 1.0, "body": 1.0},
    {"name": 1.0, "description": 3.0, "body": 2.0},
    {"name": 0.5, "description": 1.0, "body": 1.0},
    {"name": 2.0, "description": 2.0, "body": 1.0},
]


def category(query: dict) -> str:
    if "repo_any" in (query.get("requires") or {}):
        return "known-item"
    return "descriptive"


def evaluate(catalog, documents, queries, golds, traps, depth=10, score=None,
             phrasing="question"):
    """Per-query P@10, MRR and nDCG@10 under binary relevance.

    `score` maps a query to an array of per-document scores; it defaults to
    BM25F so the lexical and vector channels are measured by identical code.

    `phrasing` selects which wording of the query to send: `question` is the
    terse search-box form, `chat` is how someone actually asks -- a scenario,
    constraints, and a question wrapped in ordinary prose. The gold set is the
    same either way, which is the point: it isolates phrasing as a variable
    rather than confounding it with a different notion of relevance.
    """
    score = score or (lambda q: catalog._bm25(q))
    rows = []
    for query in queries:
        gold, trap = golds[query["id"]], traps[query["id"]]
        scores = score(query[phrasing] if query.get(phrasing) else query["question"])
        # Top-k by partition rather than a full sort: the sweep runs this a few
        # thousand times over ten thousand records, and only the head matters.
        head = np.argpartition(-scores, min(depth, len(scores) - 1))[:depth]
        order = sorted(head.tolist(), key=lambda i: (-float(scores[i]), documents[i].id))
        hits = [1 if i in gold else 0 for i in order]
        first = next((rank for rank, hit in enumerate(hits, 1) if hit), 0)
        ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(gold), depth) + 1))
        dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
        rows.append({
            "id": query["id"],
            "category": category(query),
            "gold": len(gold),
            "precision": sum(hits) / depth,
            "rr": 1 / first if first else 0.0,
            "ndcg": dcg / ideal if ideal else 0.0,
            "trap_rate": sum(1 for i in order if i in trap) / depth if trap else 0.0,
        })
    return rows


def summarise(rows, keep):
    out = {}
    for name, subset in (("known-item", [r for r in rows if r["category"] == "known-item"]),
                         ("descriptive", [r for r in rows if r["category"] == "descriptive"
                                          and r["id"] in keep]),
                         ("all (discriminating)", [r for r in rows if r["id"] in keep])):
        if not subset:
            continue
        n = len(subset)
        out[name] = {
            "n": n,
            "P@10": sum(r["precision"] for r in subset) / n,
            "MRR": sum(r["rr"] for r in subset) / n,
            "nDCG@10": sum(r["ndcg"] for r in subset) / n,
        }
    trapped = [r for r in rows if r["trap_rate"] or True]
    out["trap rate"] = sum(r["trap_rate"] for r in trapped) / len(trapped)
    return out


def union_recall(bm25, vector, gold, depth=10):
    """Did a relevant record reach the ranking model at all?

    The served pipeline takes `depth` from each channel and hands the union --
    twice `depth` candidates -- to the LLM ranker, which then reorders them. So
    the question retrieval has to answer is coverage, not order: nDCG over a
    merged list of `depth` items measures two channels fighting for ten slots,
    which is not a stage that exists.
    """
    reached = set()
    for scores in (bm25, vector):
        head = np.argpartition(-scores, depth)[:depth]
        reached.update(head.tolist())
    return 1.0 if reached & gold else 0.0, len(reached & gold)


def _phrasing_report(documents, queries, golds, traps, keep):
    """The same hundred questions, asked twice: as a search box, and as a chat."""
    catalog = MemoryCatalog(
        documents, np.empty((len(documents), 1), dtype=np.float32),
        field_weights=DEFAULT_BM25_FIELD_WEIGHTS, field_b=DEFAULT_BM25_FIELD_B,
    )
    print("\n-- phrasing: search-box wording vs how someone actually asks " + "-" * 8)
    header = f"{'phrasing':<16}" + "".join(f"{h:>26}" for h in ("known-item", "descriptive"))
    print(header)
    print(f"{'':<16}" + "".join(f"{'P@10   MRR  nDCG':>26}" for _ in range(2)))
    per_phrasing = {}
    for label, key in (("search box", "question"), ("chat", "chat")):
        rows = evaluate(catalog, documents, queries, golds, traps, phrasing=key)
        per_phrasing[key] = {r["id"]: r for r in rows}
        line = f"{label:<16}"
        for name in ("known-item", "descriptive"):
            subset = [r for r in rows if r["category"] == name
                      and (name == "known-item" or r["id"] in keep)]
            n = len(subset)
            line += (f"{sum(r['precision'] for r in subset) / n:>10.3f}"
                     f"{sum(r['rr'] for r in subset) / n:>7.3f}"
                     f"{sum(r['ndcg'] for r in subset) / n:>9.3f}")
        print(line)

    hurt = sorted(
        (per_phrasing["chat"][q["id"]]["ndcg"] - per_phrasing["question"][q["id"]]["ndcg"],
         q["id"], q["question"])
        for q in queries if q["id"] in keep or q["id"] in per_phrasing["chat"]
    )
    print("\n  worst regressions when asked conversationally:")
    for delta, qid, text in hurt[:6]:
        print(f"    {delta:+.3f}  {qid}  {text[:52]}")
    print("  biggest gains:")
    for delta, qid, text in reversed(hurt[-3:]):
        print(f"    {delta:+.3f}  {qid}  {text[:52]}")


def _vector_report(documents, queries, golds, traps, keep):
    from query_vectors import embed_queries

    from askhub.config import Config
    from askhub.providers import OpenRouterProvider

    config = Config.from_env()
    provider = OpenRouterProvider(
        config.openrouter_api_key, config.llm_model, config.embedding_model,
        config.openrouter_base_url, config.app_url, config.app_title,
        config.llm_reasoning_effort,
    )
    catalog = asyncio.run(MemoryCatalog.load(
        CORPUS.parent, provider, cache_directory=CORPUS.parent / ".cache",
        field_weights=DEFAULT_BM25_FIELD_WEIGHTS, field_b=DEFAULT_BM25_FIELD_B,
    ))
    if not catalog.field_matrices:
        print("no per-field embedding cache; run the embedding build first")
        return
    vectors = embed_queries([q["question"] for q in queries], config)

    def summarise_rows(rows):
        ki = [r for r in rows if r["category"] == "known-item"]
        de = [r for r in rows if r["category"] == "descriptive" and r["id"] in keep]
        return (sum(r["ndcg"] for r in ki) / len(ki),
                sum(r["ndcg"] for r in de) / len(de),
                sum(r["precision"] for r in de) / len(de))

    def report(label, score):
        rows = evaluate(catalog, documents, queries, golds, traps, score=score)
        known, desc, precision = summarise_rows(rows)
        print(f"{label:<34} known {known:.3f}   desc {desc:.3f}   desc P@10 {precision:.3f}")
        return rows

    print("\n-- embedding channels " + "-" * 46)
    report("BM25F (lexical only)", lambda q: catalog._bm25(q))
    for name in RETRIEVAL_FIELDS:
        matrix = catalog.field_matrices[name]
        report(f"vector: {name} only", lambda q, m=matrix: m @ vectors[q])

    # The vector channel's job is what BM25F is worst at, so it is tuned on the
    # union's coverage rather than on its own standalone ranking quality.
    def coverage(weights):
        catalog.vector_weights = weights
        hit = {"known-item": [], "descriptive": []}
        depth_hits = []
        for query in queries:
            gold = golds[query["id"]]
            if not gold or (category(query) == "descriptive" and query["id"] not in keep):
                continue
            found, count = union_recall(
                catalog._bm25(query["question"]),
                catalog._vector_scores(vectors[query["question"]]),
                gold,
            )
            hit[category(query)].append(found)
            depth_hits.append(count)
        return (sum(hit["known-item"]) / len(hit["known-item"]),
                sum(hit["descriptive"]) / len(hit["descriptive"]),
                sum(depth_hits) / len(depth_hits))

    print("\n-- union coverage: did a relevant record reach the ranker? " + "-" * 10)
    lexical_only = {"name": 0.0, "description": 0.0, "body": 0.0}
    known, desc, mean = coverage(lexical_only)
    print(f"{'BM25F alone (10 candidates)':<44} known {known:.3f}  desc {desc:.3f}  "
          f"mean relevant {mean:.2f}")
    best = None
    for weights in VECTOR_GRID:
        known, desc, mean = coverage(weights)
        label = ",".join(f"{k[0]}{v:g}" for k, v in weights.items())
        print(f"{'+ vector ' + label:<44} known {known:.3f}  desc {desc:.3f}  "
              f"mean relevant {mean:.2f}")
        if best is None or known + desc > best[0]:
            best = (known + desc, weights, mean)
    print(f"\nbest union weights: {best[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="scan name-field weights")
    parser.add_argument("--chat", action="store_true",
                        help="also score the conversational phrasing of every query")
    parser.add_argument("--vector", action="store_true",
                        help="also score the per-field embedding channels and the union "
                             "the pipeline actually serves (needs the embedding cache)")
    parser.add_argument("--json", type=Path, help="write full per-query results here")
    args = parser.parse_args()

    documents = tuple(load_file(CORPUS))
    items = [d.schema_object for d in documents]
    queries = load_queries(QUERIES)
    golds = {q["id"]: gold_set(items, q) for q in queries}
    traps = {q["id"]: trap_set(items, q) for q in queries}

    limit = int(len(items) * BROAD_GOLD_SHARE)
    broad = sorted(qid for qid, gold in golds.items() if len(gold) > limit)
    keep = {q["id"] for q in queries} - set(broad)
    print(f"{len(queries)} queries over {len(items)} records")
    print(f"excluded as too broad to discriminate (gold > {limit} records, "
          f"{BROAD_GOLD_SHARE:.0%} of corpus): {len(broad)} -> {', '.join(broad)}\n")

    empty = np.empty((len(documents), 1), dtype=np.float32)
    # The shipped configuration is always the last row, so the table reports
    # what actually serves rather than a nearby setting.
    configs = [("flat BM25", *FLAT)]
    if args.sweep:
        tuned = dict(DEFAULT_BM25_FIELD_WEIGHTS)
        for weight in (4, 8, 16, 24, 32, 48, 64):
            configs.append((
                f"  name={weight}",
                {**tuned, "name": float(weight)},
                dict(DEFAULT_BM25_FIELD_B),
            ))
    configs.append(("BM25F (shipped)", dict(DEFAULT_BM25_FIELD_WEIGHTS),
                    dict(DEFAULT_BM25_FIELD_B)))

    report = {}
    header = f"{'config':<20}" + "".join(
        f"{h:>26}" for h in ("known-item", "descriptive", "all (discriminating)")
    )
    print("known-item gold sets hold one record, so P@10 there tops out at 0.100; "
          "read MRR and nDCG.\n")
    print(header)
    print(f"{'':<20}" + "".join(f"{'P@10   MRR  nDCG':>26}" for _ in range(3)))
    for label, field_weights, field_b in configs:
        catalog = MemoryCatalog(documents, empty, field_weights=field_weights, field_b=field_b)
        rows = evaluate(catalog, documents, queries, golds, traps)
        stats = summarise(rows, keep)
        report[label] = {"summary": stats, "per_query": rows}
        line = f"{label:<20}"
        for name in ("known-item", "descriptive", "all (discriminating)"):
            s = stats[name]
            line += f"{s['P@10']:>10.3f}{s['MRR']:>7.3f}{s['nDCG']:>9.3f}" if False else \
                    f"{s['P@10']:>10.3f}{s['MRR']:>7.3f}{s['nDCG@10']:>9.3f}"
        print(line + f"   trap {stats['trap rate']:.3f}")

    if args.chat:
        _phrasing_report(documents, queries, golds, traps, keep)

    if args.vector:
        _vector_report(documents, queries, golds, traps, keep)

    if args.json:
        args.json.write_text(json.dumps({"broad_excluded": broad, "results": report}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
