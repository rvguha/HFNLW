"""Selection policy and taxonomy loading.

The policy is data, not code: every knob that changes which models end up in the
corpus lives in ``config/selection.yaml`` and is versioned by ``policy_version``
so a snapshot can name the exact rules that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Stratum:
    name: str
    description: str
    share: int          # target within the 2,500-record reference allocation
    quota: int          # quota for the profile actually being built
    queries: list[dict[str, Any]]
    # Floor on results per query, for strata whose subject matter sits deep in
    # a popularity ranking. Sorting a language tag by downloads returns the
    # giant multilingual models first: a Yoruba query does not reach an actual
    # Yoruba model until about rank 190, so a shallow fetch collects the same
    # few hundred multilingual repositories 62 times over.
    min_per_query: int = 0


@dataclass
class Policy:
    path: Path
    raw: dict[str, Any]
    profile: str
    target_count: int
    review_sample: int
    strata: list[Stratum]

    @property
    def policy_version(self) -> str:
        return str(self.raw["policy_version"])

    @property
    def seed(self) -> int:
        return int(self.raw["random_seed"])

    @property
    def oversample_factor(self) -> int:
        return int(self.raw["oversample_factor"])

    @property
    def ranking(self) -> dict[str, Any]:
        return self.raw["ranking"]

    @property
    def caps(self) -> dict[str, Any]:
        return self.raw["caps"]

    @property
    def patterns(self) -> dict[str, list[str]]:
        return self.raw["patterns"]

    @property
    def documentation_signals(self) -> dict[str, float]:
        return self.raw["documentation_signals"]

    @property
    def card(self) -> dict[str, Any]:
        return self.raw["card"]

    @property
    def enrichment(self) -> dict[str, Any]:
        return self.raw["enrichment"]

    def stratum(self, name: str) -> Stratum | None:
        return next((s for s in self.strata if s.name == name), None)


def load_policy(path: str | Path, profile: str = "pilot") -> Policy:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    profiles = raw["profiles"]
    if profile not in profiles:
        raise KeyError(f"unknown profile {profile!r}; have {sorted(profiles)}")
    target = int(profiles[profile]["target_count"])

    shares = [int(s["target"]) for s in raw["strata"]]
    total_share = sum(shares)
    strata = [
        Stratum(
            name=s["name"],
            description=s.get("description", ""),
            share=int(s["target"]),
            quota=_scaled_quota(int(s["target"]), total_share, target),
            queries=list(s["queries"]),
            min_per_query=int(s.get("min_per_query", 0)),
        )
        for s in raw["strata"]
    ]
    # Scaling rounds each quota independently; hand the rounding drift to the
    # largest stratum so the quotas always sum to exactly target_count.
    drift = target - sum(s.quota for s in strata)
    if drift:
        biggest = max(range(len(strata)), key=lambda i: strata[i].quota)
        s = strata[biggest]
        strata[biggest] = Stratum(s.name, s.description, s.share, s.quota + drift,
                                  s.queries, s.min_per_query)

    return Policy(
        path=path,
        raw=raw,
        profile=profile,
        target_count=target,
        review_sample=int(profiles[profile]["review_sample"]),
        strata=strata,
    )


def _scaled_quota(share: int, total_share: int, target: int) -> int:
    # At least one record per stratum: an eight-stratum demonstration that drops
    # a whole modality at small profile sizes is not representative.
    return max(1, math.floor(share / total_share * target))


@dataclass
class Taxonomy:
    version: str
    id: str
    name: str
    terms: list[dict[str, Any]]
    evidence_types: dict[str, dict[str, Any]]
    _by_alias: dict[str, str] = field(default_factory=dict, repr=False)

    def resolve(self, label: str) -> dict[str, Any] | None:
        """Map a free-text subject label onto a taxonomy term, or None."""
        key = label.strip().lower()
        term_id = self._by_alias.get(key)
        if term_id is None:
            return None
        return next(t for t in self.terms if t["id"] == term_id)

    def term_uri(self, term_id: str) -> str:
        return f"{self.id}#{term_id}"

    def establishes_specialty(self, evidence_type: str) -> bool:
        return bool(self.evidence_types.get(evidence_type, {}).get("establishes_specialty", False))

    def max_confidence(self, evidence_type: str) -> str:
        return str(self.evidence_types.get(evidence_type, {}).get("max_confidence", "low"))


def load_taxonomy(path: str | Path) -> Taxonomy:
    raw = yaml.safe_load(Path(path).read_text())
    by_alias: dict[str, str] = {}
    for term in raw["terms"]:
        by_alias[term["id"].lower()] = term["id"]
        by_alias[term["name"].lower()] = term["id"]
        for alias in term.get("aliases", []):
            by_alias[alias.lower()] = term["id"]
    return Taxonomy(
        version=str(raw["version"]),
        id=raw["id"],
        name=raw["name"],
        terms=raw["terms"],
        evidence_types=raw["evidence_types"],
        _by_alias=by_alias,
    )
