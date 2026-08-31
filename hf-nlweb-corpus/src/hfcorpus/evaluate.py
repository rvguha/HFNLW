"""Golden-query evaluation (design doc s10.3).

Queries state *acceptable result properties*, not a hand-labelled result list,
so the suite survives a corpus rebuild. Each query may also declare traps --
near-matches a keyword search reliably returns and this corpus should not.

Reported per query and in aggregate: precision@k, recall@k, family diversity,
filter compliance, evidence correctness, and the trap rate, each against both
the metadata-only baseline and the full corpus record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .retrieve import Index, tokenize

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


EVIDENCE_PROPERTIES = {
    "trained_on": "hf:trainedOn",
    "fine_tuned_on": "hf:fineTunedOn",
    "evaluated_on": "hf:evaluatedOn",
    "intended_for": "hf:intendedFor",
    "architecture": "hf:architectureFor",
}


def _values(item: dict[str, Any], key: str) -> list[str]:
    value = item.get(key)
    if value is None:
        return []
    return [str(v) for v in (value if isinstance(value, list) else [value])]


def matches(item: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Evaluate a requirement block against one emitted item. Every key present
    must hold; an absent key is not a constraint."""
    if "task_any" in spec:
        haystack = {str(item.get("hf:task") or "").lower()}
        haystack |= {s.lower() for s in _values(item, "hf:category")}
        if not haystack & {str(t).lower() for t in spec["task_any"]}:
            return False

    if "subject_any" in spec:
        terms = {s.lower() for s in _values(item, "hf:subject")}
        if not terms & {str(s).lower() for s in spec["subject_any"]}:
            return False

    if "language_any" in spec:
        langs = {str(x).lower() for x in _values(item, "inLanguage")}
        if not langs & {str(x).lower() for x in spec["language_any"]}:
            return False

    if "library_any" in spec and str(item.get("hf:library") or "").lower() not in {
            str(x).lower() for x in spec["library_any"]}:
        return False

    if "license_status_any" in spec and \
            str(item.get("hf:licenseStatus") or "") not in set(spec["license_status_any"]):
        return False

    if "max_parameters" in spec:
        value = item.get("hf:parameters")
        if not isinstance(value, int) or value > int(spec["max_parameters"]):
            return False

    if "min_parameters" in spec:
        value = item.get("hf:parameters")
        if not isinstance(value, int) or value < int(spec["min_parameters"]):
            return False

    if "discovery_scope" in spec and item.get("hf:discoveryScope") != spec["discovery_scope"]:
        return False

    if "evidence_type_any" in spec:
        present = {rel for rel, key in EVIDENCE_PROPERTIES.items() if item.get(key)}
        if not present & set(spec["evidence_type_any"]):
            return False

    # Confidence is no longer published -- it was the extractor grading its own
    # inference. The constraint degrades to "carries any evidence at all".
    if "min_confidence" in spec and not any(
            item.get(key) for key in EVIDENCE_PROPERTIES.values()):
        return False

    if "keyword_any" in spec:
        quotes = [q for key in EVIDENCE_PROPERTIES.values() for q in _values(item, key)]
        text = " ".join([
            item.get("name", ""), item.get("description", ""),
            " ".join(_values(item, "keywords")),
            " ".join(_values(item, "hf:capability")),
            " ".join(v.replace("-", " ") for v in _values(item, "hf:subject")),
            *quotes,
        ]).lower()
        tokens = set(tokenize(text))
        needles = [str(x).lower() for x in spec["keyword_any"]]
        if not any(n in tokens or n in text for n in needles):
            return False

    return True


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text())
    return list(raw["queries"])


def evaluate(items: list[dict[str, Any]], queries: list[dict[str, Any]],
             k: int = 10) -> dict[str, Any]:
    indexes = {
        "full": Index.build(items, "full"),
        "metadata_only": Index.build(items, "metadata_only"),
    }
    per_query: list[dict[str, Any]] = []
    totals: dict[str, dict[str, float]] = {name: {} for name in indexes}

    for query in queries:
        requires = query.get("requires", {}) or {}
        avoid = query.get("avoid", {}) or {}
        qualifying = [i for i in items if matches(i, requires)]
        row: dict[str, Any] = {
            "id": query["id"],
            "question": query["question"],
            "qualifying_in_corpus": len(qualifying),
            "results": {},
        }
        for name, index in indexes.items():
            hits = [item for item, _score in index.search(query["question"], k)]
            relevant = [i for i in hits if matches(i, requires)]
            traps = [i for i in hits if avoid and matches(i, avoid)]
            families = {i.get("hf:family") for i in hits}
            evidence_backed = [
                i for i in relevant
                if not requires.get("subject_any")
                or any(i.get(key) for key in EVIDENCE_PROPERTIES.values())
            ]
            denominator = min(k, len(qualifying)) or 1
            row["results"][name] = {
                "returned": len(hits),
                "precision_at_k": round(len(relevant) / max(1, len(hits)), 4),
                "recall_at_k": round(len(relevant) / denominator, 4),
                "family_diversity": round(len(families) / max(1, len(hits)), 4),
                "trap_rate": round(len(traps) / max(1, len(hits)), 4),
                "evidence_backed_fraction": round(
                    len(evidence_backed) / max(1, len(relevant)), 4) if relevant else 0.0,
                "top_hits": [i.get("hf:repository", "") for i in hits[:5]],
            }
            for metric, value in row["results"][name].items():
                if isinstance(value, (int, float)):
                    totals[name][metric] = totals[name].get(metric, 0.0) + float(value)
        per_query.append(row)

    count = max(1, len(per_query))
    summary = {
        name: {metric: round(value / count, 4) for metric, value in metrics.items()}
        for name, metrics in totals.items()
    }
    unanswerable = [r["id"] for r in per_query if r["qualifying_in_corpus"] == 0]
    return {
        "k": k,
        "queries": len(per_query),
        "queries_with_no_qualifying_record": unanswerable,
        "summary": summary,
        "per_query": per_query,
    }
