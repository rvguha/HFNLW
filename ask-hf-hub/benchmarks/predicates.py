"""Relevance as checkable predicates over an emitted corpus record.

Ported from hf-nlweb-corpus `evaluate.matches` so this benchmark depends on the
served corpus alone, not on the pipeline that built it. Relevance is stated as
*properties an acceptable result must have* rather than a hand-labelled result
list, which is what makes a hundred-query gold set free to produce and immune to
a corpus rebuild renumbering everything.

`repo_any` is added here: a known-item query has exactly one right answer, and
naming it is more honest than approximating it with a property that happens to
be unique today. `has_limitations` is added for the same reason -- asking for
models that document their risks is a question about whether a field is
populated, not about whether the word "bias" appears somewhere.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _values(item: dict[str, Any], key: str) -> list[str]:
    value = item.get(key)
    if value is None:
        return []
    return [str(v) for v in (value if isinstance(value, list) else [value])]


# Evidence relations, as emitted. Grouped quotes replaced the old Claim nodes.
EVIDENCE_PROPERTIES = {
    "trained_on": "hf:trainedOn",
    "fine_tuned_on": "hf:fineTunedOn",
    "evaluated_on": "hf:evaluatedOn",
    "intended_for": "hf:intendedFor",
    "architecture": "hf:architectureFor",
}


def repo_id(item: dict[str, Any]) -> str:
    return str(item.get("hf:repository")
               or str(item.get("@id", "")).split("huggingface.co/")[-1])


def matches(item: dict[str, Any], spec: dict[str, Any]) -> bool:
    """Every key present must hold; an absent key is not a constraint."""
    if "repo_any" in spec and repo_id(item).lower() not in {
        str(x).lower() for x in spec["repo_any"]
    }:
        return False

    if "has_limitations" in spec and (
        bool(item.get("hf:limitation")) is not bool(spec["has_limitations"])
    ):
        return False

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
        str(x).lower() for x in spec["library_any"]
    }:
        return False

    if "license_status_any" in spec and str(
        item.get("hf:licenseStatus") or ""
    ) not in set(spec["license_status_any"]):
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
        present = {
            relation for relation, key in EVIDENCE_PROPERTIES.items() if item.get(key)
        }
        if not present & set(spec["evidence_type_any"]):
            return False

    # Confidence is no longer published: it was the extractor grading its own
    # inference, and the emitted record keeps only what is checkable. The
    # constraint degrades to "has any evidence at all".
    if "min_confidence" in spec and not any(
        item.get(key) for key in EVIDENCE_PROPERTIES.values()
    ):
        return False

    if "keyword_any" in spec:
        quotes = [q for key in EVIDENCE_PROPERTIES.values() for q in _values(item, key)]
        text = " ".join(
            [
                str(item.get("name") or ""),
                str(item.get("description") or ""),
                " ".join(_values(item, "keywords")),
                " ".join(_values(item, "hf:capability")),
                " ".join(_values(item, "hf:subject")),
                *quotes,
            ]
        ).lower()
        tokens = set(tokenize(text))
        needles = [str(x).lower() for x in spec["keyword_any"]]
        if not any(n in tokens or n in text for n in needles):
            return False

    return True


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    return list(yaml.safe_load(Path(path).read_text())["queries"])


def gold_set(items: list[dict[str, Any]], query: dict[str, Any]) -> set[int]:
    """Indices of the records that satisfy the query.

    A record matching `avoid` is a declared trap -- the near-match a keyword
    search reliably returns -- and is non-relevant even if it also satisfies
    `requires`.
    """
    requires = query.get("requires") or {}
    avoid = query.get("avoid")
    return {
        index
        for index, item in enumerate(items)
        if matches(item, requires) and not (avoid and matches(item, avoid))
    }


def trap_set(items: list[dict[str, Any]], query: dict[str, Any]) -> set[int]:
    avoid = query.get("avoid")
    if not avoid:
        return set()
    return {index for index, item in enumerate(items) if matches(item, avoid)}
