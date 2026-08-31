"""Build a reference relevance set by judging a wide retrieval with strong models.

The problem this solves: precision@k as computed in evaluate.py grades the
retriever against predicates checked on the very fields the retriever indexes.
A record tagged `biomedicine` by enrichment is retrieved *because* of that tag
and judged relevant *because* of that tag. The number moves with the corpus and
says little about whether an answer was good.

A reference set breaks the circle. Retrieve far more than a user would ever see,
have models too slow and expensive to serve judge every one of them, and keep
what they call relevant. The judgement then comes from reading the record, not
from matching the field that retrieved it.

Two properties worth being explicit about:

* **Pooled, not exhaustive.** Only retrieved records are judged, so a relevant
  record no retrieval configuration ever surfaces cannot appear in the set. This
  is the standard limitation of pooled evaluation and the reason the pool is
  made wide.
* **Two judges, kept separate.** Agreement between independent models is itself
  a measurement. Collapsing them into one list before recording that agreement
  would throw away the only evidence of how reliable the set is.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Judgement:
    query_id: str
    judge: str
    relevant: list[dict[str, Any]] = field(default_factory=list)
    retrieved: int = 0
    seconds: float = 0.0
    cost_usd: float = 0.0
    tokens: int = 0
    error: str = ""


def judge_query(endpoint: str, question: str, judge: str, width: int,
                full_matrix: bool = False, timeout: float = 1800.0) -> Judgement:
    """Ask one question at evaluation width and keep the judge's verdicts.

    With `full_matrix`, the judge returns a verdict on every retrieved record
    rather than a shortlist, which is what agreement between judges has to be
    computed over.
    """
    payload = {
        "query": question,
        "retrieval_count": width,
        "ranking_count": width,
        "max_results": width,
        "min_score": 0,
        "ranking_model": judge,
        "include_excluded": full_matrix,
    }
    request = urllib.request.Request(
        endpoint, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"})

    started = time.time()
    relevant: list[dict[str, Any]] = []
    candidates: set[str] = set()
    usage: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as stream:
            for raw in stream:
                line = raw.decode().strip()
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                kind, content = event.get("message_type"), event.get("content")
                if kind == "candidate":
                    candidates.update(i.get("url", "") for i in (content or []))
                elif kind == "result":
                    for item in content or []:
                        relevant.append({
                            "url": item.get("url"),
                            "name": item.get("name"),
                            "score": item.get("score"),
                            "relevance": item.get("relevance"),
                        })
                elif kind == "usage":
                    usage = content
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return Judgement("", judge, error=f"{type(exc).__name__}: {exc}"[:200],
                         seconds=round(time.time() - started, 1))

    models = (usage or {}).get("models", [])
    cost = round(sum(m.get("cost", 0.0) for m in models), 6)
    ranking_calls = sum(m.get("calls", 0) for m in models if m.get("model") == judge)

    # A judge that was never actually asked looks exactly like a judge that
    # found nothing relevant -- an empty list either way. It cost this run 38 of
    # 45 queries: OpenRouter began returning 402 Payment Required, the server
    # degraded to no results, and the harness recorded "0 relevant" for each.
    # A reference set quietly full of empty judgements is worse than one that
    # stops, so an unbilled judge is an error and not a verdict.
    return Judgement(
        query_id="",
        judge=judge,
        relevant=relevant,
        retrieved=len(candidates),
        seconds=round(time.time() - started, 1),
        cost_usd=cost,
        tokens=sum(m.get("total_tokens", 0) for m in models),
        error=unusable_reason(judge, cost, ranking_calls),
    )


def judgement_key(question: str, judge: str, width: int, corpus: str,
                  mode: str = "shortlist") -> str:
    """What makes two judgements the same purchase.

    `mode` is part of the key because the two are not the same purchase at all:
    "shortlist" asks the judge to select at most a few records, "full" asks for a
    verdict on every one. Sharing a key would let a full-matrix run silently read
    back a shortlist and report nine verdicts as if they were a hundred.
    """
    digest = hashlib.sha256()
    for part in (question, judge, str(width), corpus, mode, CAPTURE_VERSION):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()[:32]


# Bumped whenever what gets captured changes, so a cached judgement from an
# earlier capture cannot be mistaken for a current one. Dropping deduplication
# in full-matrix mode changed the record count per query from ~96 to 100.
CAPTURE_VERSION = "2"

LABELS = ("strong", "relevant", "exclude")


def label_alignment(judgements: dict[str, Judgement]) -> dict[str, Any]:
    """Per-record label agreement between exactly two judges.

    Set overlap answers "did they pick the same records"; this answers "did they
    say the same thing about each record", which is the question a reference set
    rests on. Cohen's kappa corrects for the agreement two judges would reach by
    chance -- with three labels and a skewed distribution, raw agreement alone
    flatters heavily.
    """
    usable = [j for j in judgements.values() if not j.error]
    if len(usable) != 2:
        return {}
    left, right = usable
    by_url = [
        ({r["url"]: r.get("relevance") for r in j.relevant if r.get("url")})
        for j in (left, right)
    ]
    shared = sorted(set(by_url[0]) & set(by_url[1]))
    if not shared:
        return {}

    matrix: dict[str, dict[str, int]] = {a: dict.fromkeys(LABELS, 0) for a in LABELS}
    same = 0
    for url in shared:
        a, b = by_url[0][url], by_url[1][url]
        if a in LABELS and b in LABELS:
            matrix[a][b] += 1
            same += a == b

    n = sum(sum(row.values()) for row in matrix.values())
    observed = same / n if n else 0.0
    expected = sum(
        (sum(matrix[a].values()) / n) * (sum(matrix[x][a] for x in LABELS) / n)
        for a in LABELS
    ) if n else 0.0
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else 0.0
    return {
        "compared_records": n,
        "label_agreement": round(observed, 3),
        "cohens_kappa": round(kappa, 3),
        "confusion": {f"{left.judge.split('/')[-1]}={a}": {
            f"{right.judge.split('/')[-1]}={b}": matrix[a][b] for b in LABELS} for a in LABELS},
    }


def unusable_reason(judge: str, cost_usd: float, ranking_calls: int) -> str:
    """Why a judgement cannot be trusted, or "" if it can.

    A judge that was never asked looks exactly like a judge that read 200
    records and rejected them all: an empty list either way. That cost one run
    38 of 45 queries -- OpenRouter began returning 402, the server degraded to
    no results, and every query was recorded as "no relevant records found". An
    unbilled judge is an error, not a verdict.
    """
    if ranking_calls == 0:
        return f"judge {judge} was never called; ranking unavailable"
    if cost_usd == 0.0:
        return f"judge {judge} made {ranking_calls} calls at zero cost; likely rejected"
    return ""


def build(endpoint: str, queries: list[dict[str, Any]], judges: list[str], width: int,
          store: Any = None, corpus: str = "", full_matrix: bool = False,
          log=print) -> dict[str, Any]:
    """Judge every query with every judge; report agreement alongside the set.

    A judgement already bought is read from cache. Only usable verdicts are
    cached -- a judge that was never billed must be retried, not remembered.
    """
    per_query: list[dict[str, Any]] = []
    total_cost = 0.0
    reused = 0

    for index, query in enumerate(queries, 1):
        judgements: dict[str, Judgement] = {}
        for judge in judges:
            key = judgement_key(query["question"], judge, width, corpus,
                                "full" if full_matrix else "shortlist")
            path = store.judgement(key) if store is not None else None

            if path is not None and path.exists():
                cached = json.loads(path.read_text())
                result = Judgement(query["id"], judge, cached["relevant"],
                                   cached.get("retrieved", 0), cached.get("seconds", 0.0),
                                   cached.get("cost_usd", 0.0), cached.get("tokens", 0))
                reused += 1
                log(f"  {query['id']} {judge:28} {len(result.relevant):>3} relevant  "
                    f"(cached)")
            else:
                result = judge_query(endpoint, query["question"], judge, width, full_matrix)
                result.query_id = query["id"]
                total_cost += result.cost_usd
                note = result.error or f"{len(result.relevant):>3} relevant"
                log(f"  {query['id']} {judge:28} {note}  {result.seconds:5.1f}s  "
                    f"${result.cost_usd:.3f}")
                # Validate before caching, not after: an unbilled verdict
                # written to cache would make the failure permanent and free.
                if not result.error and not result.cost_usd:
                    result.error = unusable_reason(judge, result.cost_usd, 0)
                if path is not None and not result.error:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({
                        "query_id": query["id"], "question": query["question"],
                        "judge": judge, "width": width, "corpus": corpus,
                        "mode": "full" if full_matrix else "shortlist",
                        "relevant": result.relevant, "retrieved": result.retrieved,
                        "seconds": result.seconds, "cost_usd": result.cost_usd,
                        "tokens": result.tokens,
                    }, indent=2))
            judgements[judge] = result

        # Only judges that actually ran contribute; a failed judge must not be
        # read as unanimous agreement that nothing is relevant.
        # A cached judgement was validated before it was written, so anything
        # still carrying zero cost here came from this run and already failed.
        # In full-matrix mode `relevant` holds every verdict, exclusions
        # included, so set membership means "was judged", not "was accepted".
        # Agreement has to be computed over the labels.
        def accepted(verdicts):
            return {r["url"] for r in verdicts
                    if r.get("url") and r.get("relevance") in ("strong", "relevant")}

        sets = {j: (accepted(v.relevant) if full_matrix
                    else {r["url"] for r in v.relevant if r.get("url")})
                for j, v in judgements.items() if not v.error}
        union: set[str] = set().union(*sets.values()) if sets else set()
        agreed: set[str] = set.intersection(*sets.values()) if len(sets) > 1 else union

        per_query.append({
            "id": query["id"],
            "question": query["question"],
            "retrieved": max((v.retrieved for v in judgements.values()), default=0),
            "by_judge": {
                j: {"relevant": v.relevant, "seconds": v.seconds,
                    "cost_usd": v.cost_usd, "tokens": v.tokens, "error": v.error}
                for j, v in judgements.items()
            },
            # `agreed` is the conservative set both judges accepted; `union` is
            # the generous one. Which to score against is a choice made when
            # scoring, not here.
            "agreed": sorted(agreed),
            "union": sorted(union),
            "agreement": round(len(agreed) / len(union), 3) if union else None,
            **(label_alignment(judgements) if full_matrix else {}),
        })
        log(f"  {query['id']} [{index}/{len(queries)}] agreed {len(agreed)} of "
            f"{len(union)} union")

    failed = [q["id"] for q in per_query
              if any(v["error"] for v in q["by_judge"].values())]
    scored = [q for q in per_query if q["agreement"] is not None and q["id"] not in failed]
    aligned = [q for q in per_query if q.get("compared_records")]
    total_compared = sum(q["compared_records"] for q in aligned)
    return {
        "compared_records": total_compared,
        "mean_label_agreement": (round(sum(q["label_agreement"] * q["compared_records"]
                                           for q in aligned) / total_compared, 3)
                                 if total_compared else None),
        "mean_cohens_kappa": (round(sum(q["cohens_kappa"] for q in aligned) / len(aligned), 3)
                              if aligned else None),
        "incomplete": failed,
        "endpoint": endpoint,
        "judges": judges,
        "width": width,
        "queries": len(per_query),
        "total_cost_usd": round(total_cost, 4),
        "judgements_reused_from_cache": reused,
        "mean_agreement": (round(sum(q["agreement"] for q in scored) / len(scored), 3)
                           if scored else None),
        "mean_agreed_per_query": (round(sum(len(q["agreed"]) for q in per_query)
                                        / max(1, len(per_query)), 1)),
        "queries_with_no_relevant_record": [q["id"] for q in per_query
                                           if not q["union"] and q["id"] not in failed],
        "per_query": per_query,
    }
