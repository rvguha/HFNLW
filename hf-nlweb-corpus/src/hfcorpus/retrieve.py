"""A small local retriever used to evaluate the corpus (design doc s10.3).

This is not NLWeb. It exists so golden queries can be scored the moment the
JSONL is written, and so the ingested-into-NLWeb numbers have a baseline to beat.
Two profiles are built from the same items:

* ``metadata_only`` -- name, tags, task, publisher: what the Hub already gives a
  keyword search; and
* ``full`` -- plus the grounded description, capabilities, subject areas, and
  evidence quotes the corpus adds.

The gap between them is the measurement that matters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

TOKEN = re.compile(r"[a-z0-9][a-z0-9+.#-]*")
K1 = 1.2
B = 0.75

FIELD_BOOSTS = {
    "name": 3.0,
    "subcategory": 2.5,
    "subject": 3.0,
    "capability": 2.0,
    "keywords": 1.5,
    "description": 1.0,
    "evidence": 1.0,
    "publisher": 1.0,
    "notes": 0.75,
}
METADATA_ONLY_FIELDS = ("name", "keywords", "subcategory", "publisher")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


EVIDENCE_PROPERTIES = ("hf:trainedOn", "hf:fineTunedOn", "hf:evaluatedOn",
                      "hf:intendedFor", "hf:architectureFor")


def _values(item: dict[str, Any], key: str) -> list[str]:
    value = item.get(key)
    if value is None:
        return []
    return [str(v) for v in (value if isinstance(value, list) else [value])]


def fields(item: dict[str, Any]) -> dict[str, str]:
    quotes = [q for key in EVIDENCE_PROPERTIES for q in _values(item, key)]
    notes = [n for key in ("hf:intendedUse", "hf:limitation", "hf:deploymentNote")
             for n in _values(item, key)]
    return {
        "name": " ".join([item.get("name", ""), str(item.get("hf:repository") or "")]),
        "subcategory": " ".join([*_values(item, "hf:category"),
                                 str(item.get("hf:task") or "")]),
        # Subjects are slugs; retrieval wants the words in them.
        "subject": " ".join(v.replace("-", " ") for v in _values(item, "hf:subject")),
        "capability": " ".join(_values(item, "hf:capability")),
        "keywords": " ".join([*_values(item, "keywords"),
                              str(item.get("hf:library") or "")]),
        "description": item.get("description", ""),
        "evidence": " ".join(quotes),
        "publisher": (item.get("creator") or {}).get("name", ""),
        "notes": " ".join(notes),
    }


@dataclass
class Index:
    profile: str = "full"
    items: list[dict[str, Any]] = field(default_factory=list)
    _docs: list[Counter] = field(default_factory=list, repr=False)
    _lengths: list[float] = field(default_factory=list, repr=False)
    _df: Counter = field(default_factory=Counter, repr=False)
    _avg_length: float = 0.0

    @classmethod
    def build(cls, items: list[dict[str, Any]], profile: str = "full") -> Index:
        index = cls(profile=profile, items=list(items))
        allowed = FIELD_BOOSTS if profile == "full" else {
            k: FIELD_BOOSTS[k] for k in METADATA_ONLY_FIELDS}
        for item in index.items:
            counts: Counter = Counter()
            for name, text in fields(item).items():
                boost = allowed.get(name)
                if not boost:
                    continue
                for token in tokenize(text):
                    counts[token] += boost
            index._docs.append(counts)
            index._lengths.append(sum(counts.values()))
            for token in counts:
                index._df[token] += 1
        index._avg_length = (sum(index._lengths) / len(index._lengths)) if index._lengths else 1.0
        return index

    def search(self, query: str, k: int = 10) -> list[tuple[dict[str, Any], float]]:
        terms = tokenize(query)
        if not terms or not self.items:
            return []
        n = len(self.items)
        scored: list[tuple[float, int]] = []
        for i, counts in enumerate(self._docs):
            score = 0.0
            length = self._lengths[i] or 1.0
            for term in terms:
                freq = counts.get(term, 0.0)
                if not freq:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * (freq * (K1 + 1)) / (
                    freq + K1 * (1 - B + B * length / (self._avg_length or 1.0)))
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda pair: (-pair[0], self.items[pair[1]]["@id"]))
        return [(self.items[i], round(score, 4)) for score, i in scored[:k]]
