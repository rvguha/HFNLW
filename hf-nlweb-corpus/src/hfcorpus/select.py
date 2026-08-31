"""Stratified candidate discovery, ranking, and selection (design doc s2).

A pure download ranking overrepresents embeddings, popular base-model
derivatives, and packaging variants. So: build oversized pools per stratum,
score inside each stratum only, then select under publisher and family caps with
explicit reserves for diversity and recency.

Every candidate that is ever considered gets a decision row in
manifests/selection.jsonl. Decisions are appended, never rewritten (s1.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .config import Policy
from .naming import (
    inferred_parameters_from_name,
    matches_any,
    provisional_family_key,
    repo_owner,
    variant_kind,
)

REASON_CODES = {
    "included_ranked": "Selected through stratum ranking",
    "included_diversity": "Added to improve language/domain/publisher/size diversity",
    "included_recent": "Added through the recent/trending reserve",
    "excluded_empty_card": "No useful descriptive model card",
    "excluded_exact_duplicate": "Duplicates a canonical repository",
    "suppressed_variant": "Deployment variant attached to a represented family",
    "excluded_checkpoint": "Intermediate or non-user-facing training artifact",
    "excluded_inaccessible": "Private, disabled, unavailable, or gated without access",
    "excluded_over_quota": "Ranked below the stratum quota or a diversity cap",
    "quarantined_fetch": "Source could not be retrieved; transient failure, retry",
    "quarantined_parse": "Source could not be normalized reliably",
    "quarantined_enrichment": "Structured enrichment or evidence validation failed",
}


@dataclass
class Candidate:
    repo_id: str
    stratum: str
    author: str = ""
    sha: str = ""
    pipeline_tag: str | None = None
    library_name: str | None = None
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    downloads_all_time: int = 0
    likes: int = 0
    trending_score: float = 0.0
    created_at: str = ""
    last_modified: str = ""
    gated: Any = False
    private: bool = False
    disabled: bool = False
    languages: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    license: str | None = None
    base_models: list[str] = field(default_factory=list)
    parameters: int | None = None
    parameters_source: str = "none"
    eval_results: bool = False
    found_by: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> Candidate:
        return cls(**obj)


def candidate_from_model_info(info: Any, stratum: str, query_label: str) -> Candidate:
    card = _card_data_dict(getattr(info, "card_data", None))
    safetensors = getattr(info, "safetensors", None)
    parameters, source = None, "none"
    if safetensors is not None and getattr(safetensors, "total", None):
        parameters, source = int(safetensors.total), "safetensors"
    else:
        guessed = inferred_parameters_from_name(info.id)
        if guessed:
            parameters, source = guessed, "inferred_from_name"

    base = card.get("base_model")
    base_models = ([base] if isinstance(base, str)
                   else [b for b in (base or []) if isinstance(b, str)])
    api_base = getattr(info, "base_models", None)
    if isinstance(api_base, dict):
        base_models += [m["id"] for m in api_base.get("models", [])
                        if isinstance(m, dict) and m.get("id")]

    language = card.get("language")
    languages = [language] if isinstance(language, str) else list(language or [])
    datasets = card.get("datasets")
    datasets = [datasets] if isinstance(datasets, str) else list(datasets or [])

    return Candidate(
        repo_id=info.id,
        stratum=stratum,
        author=getattr(info, "author", None) or repo_owner(info.id),
        sha=getattr(info, "sha", "") or "",
        pipeline_tag=getattr(info, "pipeline_tag", None),
        library_name=getattr(info, "library_name", None),
        tags=list(getattr(info, "tags", None) or []),
        downloads=int(getattr(info, "downloads", 0) or 0),
        downloads_all_time=int(getattr(info, "downloads_all_time", 0) or 0),
        likes=int(getattr(info, "likes", 0) or 0),
        trending_score=float(getattr(info, "trending_score", 0.0) or 0.0),
        created_at=_iso(getattr(info, "created_at", None)),
        last_modified=_iso(getattr(info, "last_modified", None)),
        gated=getattr(info, "gated", False),
        private=bool(getattr(info, "private", False)),
        disabled=bool(getattr(info, "disabled", False)),
        languages=[str(x) for x in languages],
        datasets=[str(x) for x in datasets],
        license=card.get("license"),
        base_models=sorted(set(base_models)),
        parameters=parameters,
        parameters_source=source,
        eval_results=bool(getattr(info, "eval_results", None)) or bool(card.get("model-index")),
        found_by=[query_label],
    )


def discover(policy: Policy, hub: Any, log=print) -> list[Candidate]:
    """Fetch oversized candidate pools for every stratum (s2.4 step 1)."""
    pool: dict[str, Candidate] = {}
    for stratum in policy.strata:
        want = stratum.quota * policy.oversample_factor
        per_query = max(10, stratum.min_per_query,
                        math.ceil(want / max(1, len(stratum.queries))))
        found = 0
        for query in stratum.queries:
            label = f"{stratum.name}:{_query_label(query)}"
            for info in hub.list_models(query, per_query):
                existing = pool.get(info.id)
                if existing is not None:
                    if label not in existing.found_by:
                        existing.found_by.append(label)
                    continue
                pool[info.id] = candidate_from_model_info(info, stratum.name, label)
                found += 1
        log(f"  {stratum.name}: quota {stratum.quota}, pool +{found}")

    candidates = list(pool.values())
    assign_primary_strata(candidates, policy)
    return candidates


def assign_primary_strata(candidates: list[Candidate], policy: Policy) -> None:
    """Give each candidate the stratum that needs it most.

    A repository usually answers to several strata at once -- a Hindi
    instruction model is discovered by both `text_generation_chat` and
    `language_coverage` -- and the doc asks for one primary stratum per record
    (s2.1). Taking whichever query ran first makes quotas depend on the order
    strata happen to appear in the YAML, which starves a narrow stratum listed
    after broad ones: `language_coverage` drew a pool of 102 against a quota of
    1,333 that way, because eight broader strata had already claimed every
    language-tagged model.

    Candidates that only one stratum found are fixed first, then the shared ones
    go to whichever of their strata is furthest from its quota.
    """
    quotas = {s.name: s.quota for s in policy.strata}
    options = {
        c.repo_id: sorted({label.split(":", 1)[0] for label in c.found_by} & quotas.keys())
        for c in candidates
    }
    filled: dict[str, int] = dict.fromkeys(quotas, 0)

    shared: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.repo_id):
        choices = options[candidate.repo_id]
        if len(choices) == 1:
            candidate.stratum = choices[0]
            filled[choices[0]] += 1
        elif choices:
            shared.append(candidate)

    for candidate in shared:
        choices = options[candidate.repo_id]
        best = min(choices, key=lambda name: (filled[name] / max(1, quotas[name]), name))
        candidate.stratum = best
        filled[best] += 1


def deterministic_features(policy: Policy,
                           candidates: list[Candidate]) -> dict[str, dict[str, Any]]:
    """Per-candidate features that do not depend on the model card (s2.2, s2.4
    step 3). Documentation quality here is a metadata-completeness proxy;
    cards.documentation_quality replaces it after fetch."""
    owner_counts: dict[str, int] = {}
    for c in candidates:
        owner_counts[c.author] = owner_counts.get(c.author, 0) + 1

    signals = policy.documentation_signals
    small = policy.caps["small_model_max_parameters"]
    lineage = _lineage_keys(candidates, policy.patterns)
    features: dict[str, dict[str, Any]] = {}
    for c in candidates:
        doc = 0.0
        doc += signals["has_pipeline_tag"] * bool(c.pipeline_tag)
        doc += signals["has_library"] * bool(c.library_name)
        doc += signals["has_license"] * bool(c.license)
        doc += signals["has_language"] * bool(c.languages)
        doc += signals["has_datasets"] * bool(c.datasets)
        doc += signals["has_base_model"] * bool(c.base_models)
        doc += signals["has_eval_results"] * bool(c.eval_results)

        non_english = any(lang.lower() not in ("en", "english") for lang in c.languages)
        is_small = c.parameters is not None and c.parameters <= small
        rare_publisher = owner_counts.get(c.author, 0) <= 3
        diversity = sum([non_english, is_small, rare_publisher, bool(c.datasets)]) / 4.0

        kind = variant_kind(c.repo_id, policy.patterns)
        features[c.repo_id] = {
            "documentation_quality": round(min(1.0, doc), 4),
            "recency_score": _recency(c.last_modified, policy.ranking["recency_half_life_days"]),
            "diversity_bonus": round(diversity, 4),
            "non_english": non_english,
            "small_model": is_small,
            "rare_publisher": rare_publisher,
            "variant_kind": kind,
            "family_key": lineage[c.repo_id],
        }
    return features


def _lineage_keys(candidates: list[Candidate], patterns: dict[str, list[str]]) -> dict[str, str]:
    """Family key for the selection-time cap, resolved along the whole chain.

    The cap exists to stop one base model's derivatives dominating (s2.3), but
    it was computed one hop up the lineage while families.py later resolves the
    chain to its root. The two keys disagreed, so 24 records shipped under
    `qwen--qwen2-5-7b` against a cap of 4: selection saw 21 distinct keys, one
    per intermediate ancestor, and never fired. Walking to the root here makes
    the cap act on the same grouping the record is published with.
    """
    bases = {c.repo_id: c.base_models for c in candidates}

    def root(repo_id: str) -> str:
        """Walk to the end of the declared chain. Iterative with a visited set
        rather than a depth limit: a limit silently stops collapsing exactly the
        long derivative chains the cap is meant to catch."""
        seen = {repo_id}
        current = repo_id
        while True:
            declared = bases.get(current) or []
            if not declared:
                return current
            parent = sorted(declared)[0]
            if parent in seen:          # a cycle in declared lineage
                return current
            if parent not in bases:     # chain leaves the candidate pool
                return parent
            seen.add(parent)
            current = parent

    return {c.repo_id: provisional_family_key(root(c.repo_id), [], patterns) for c in candidates}


def score(policy: Policy, candidates: list[Candidate],
          features: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Rank within each stratum and never across strata (s2.2)."""
    weights = policy.ranking["weights"]
    penalty = float(policy.ranking["duplication_penalty"])
    scores: dict[str, float] = {}
    by_stratum: dict[str, list[Candidate]] = {}
    for c in candidates:
        by_stratum.setdefault(c.stratum, []).append(c)

    for pool in by_stratum.values():
        pct_downloads = _percentiles({c.repo_id: math.log1p(c.downloads) for c in pool})
        pct_likes = _percentiles({c.repo_id: math.log1p(c.likes) for c in pool})
        pct_trending = _percentiles({c.repo_id: c.trending_score for c in pool})
        for c in pool:
            f = features[c.repo_id]
            value = (
                weights["downloads"] * pct_downloads[c.repo_id]
                + weights["likes"] * pct_likes[c.repo_id]
                + weights["trending"] * pct_trending[c.repo_id]
                + weights["recency"] * f["recency_score"]
                + weights["documentation"] * f["documentation_quality"]
                + weights["diversity"] * f["diversity_bonus"]
            )
            if f["variant_kind"] in ("quantization", "format_conversion", "mirror"):
                value -= penalty
            scores[c.repo_id] = round(value, 6)
    return scores


@dataclass
class Decision:
    repo_id: str
    stratum: str
    decision: str      # included | suppressed | excluded
    reason_code: str
    score: float
    features: dict[str, Any]


def select(policy: Policy, candidates: list[Candidate],
           features: dict[str, dict[str, Any]], scores: dict[str, float]) -> list[Decision]:
    """Greedy selection under publisher and family caps, with a diversity
    reserve and the recent/trending reserve filled last (s2.3, s2.4)."""
    decisions: list[Decision] = []

    def record(c: Candidate, decision: str, reason: str) -> None:
        decisions.append(Decision(c.repo_id, c.stratum, decision, reason,
                                  scores.get(c.repo_id, 0.0), features[c.repo_id]))

    eligible: dict[str, list[Candidate]] = {}
    eligible_ids: set[str] = set()
    excluded_owners = policy.patterns.get("excluded_owners", [])
    for c in candidates:
        kind = features[c.repo_id]["variant_kind"]
        if excluded_owners and matches_any(c.author, excluded_owners):
            record(c, "excluded", "excluded_checkpoint")
        elif c.private or c.disabled or c.gated not in (False, None, "", "auto"):
            record(c, "excluded", "excluded_inaccessible")
        elif kind == "checkpoint":
            record(c, "excluded", "excluded_checkpoint")
        elif kind in ("quantization", "format_conversion"):
            # Retained as a deployment variant, kept out of the primary quota.
            record(c, "suppressed", "suppressed_variant")
        elif kind == "mirror":
            record(c, "suppressed", "suppressed_variant")
        else:
            eligible.setdefault(c.stratum, []).append(c)
            eligible_ids.add(c.repo_id)

    publisher_cap = max(1, int(float(policy.caps["publisher_max_fraction"])
                              * policy.target_count))
    family_cap = int(policy.caps["family_max_records"])
    reserve_fraction = float(policy.caps["diversity_reserve_fraction"])

    publisher_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    selected: set[str] = set()

    def can_take(c: Candidate) -> str | None:
        if publisher_counts.get(c.author, 0) >= publisher_cap:
            return "publisher_cap"
        if family_counts.get(features[c.repo_id]["family_key"], 0) >= family_cap:
            return "family_cap"
        return None

    def take(c: Candidate, reason: str) -> None:
        selected.add(c.repo_id)
        publisher_counts[c.author] = publisher_counts.get(c.author, 0) + 1
        key = features[c.repo_id]["family_key"]
        family_counts[key] = family_counts.get(key, 0) + 1
        record(c, "included", reason)

    deferred: list[Candidate] = []
    for stratum in policy.strata:
        pool = sorted(eligible.get(stratum.name, []),
                      key=lambda c: (-scores[c.repo_id], c.repo_id))
        ranked_quota = max(1, round(stratum.quota * (1.0 - reserve_fraction)))
        reason = "included_recent" if stratum.name == "trending_recent" else "included_ranked"
        taken = 0
        for c in pool:
            if taken >= ranked_quota:
                deferred.append(c)
                continue
            if can_take(c) is None:
                take(c, reason)
                taken += 1
            else:
                deferred.append(c)

        # Diversity reserve: prefer candidates that add under-covered properties.
        remaining = stratum.quota - taken
        if remaining > 0:
            reserve_pool = [c for c in pool if c.repo_id not in selected]
            reserve_pool.sort(key=lambda c: (
                -(features[c.repo_id]["non_english"] + features[c.repo_id]["small_model"]
                  + features[c.repo_id]["rare_publisher"]),
                -scores[c.repo_id], c.repo_id,
            ))
            for c in reserve_pool:
                if remaining <= 0:
                    break
                if can_take(c) is not None:
                    continue
                feature = features[c.repo_id]
                is_diverse = (feature["non_english"] or feature["small_model"]
                              or feature["rare_publisher"])
                take(c, "included_diversity" if is_diverse else reason)
                remaining -= 1

    for c in deferred:
        if c.repo_id not in selected:
            record(c, "excluded", "excluded_over_quota")

    # A stratum can under-fill when its pool is exhausted or capped out; top up
    # globally from the best unselected candidates so the profile hits its count.
    shortfall = policy.target_count - len(selected)
    if shortfall > 0:
        rest = sorted((c for c in candidates
                       if c.repo_id in eligible_ids and c.repo_id not in selected),
                      key=lambda c: (-scores[c.repo_id], c.repo_id))
        for c in rest:
            if shortfall <= 0:
                break
            if can_take(c) is None:
                take(c, "included_diversity")
                shortfall -= 1

    return decisions


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    """Rank percentile in [0, 1]; ties share the average rank so identical
    values cannot be ordered by dictionary insertion accident."""
    if not values:
        return {}
    ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(ordered)
    if n == 1:
        return {ordered[0][0]: 1.0}
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            out[ordered[k][0]] = rank / (n - 1)
        i = j + 1
    return out


def _recency(last_modified: str, half_life_days: float) -> float:
    if not last_modified:
        return 0.0
    try:
        when = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    age = (datetime.now(UTC) - when).days
    return round(0.5 ** (max(0, age) / half_life_days), 4)


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    return str(value)


def _card_data_dict(card_data: Any) -> dict[str, Any]:
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return card_data
    for method in ("to_dict", "to_json_dict"):
        fn = getattr(card_data, method, None)
        if callable(fn):
            try:
                out = fn()
                if isinstance(out, dict):
                    return out
            except (TypeError, ValueError):
                pass
    return dict(getattr(card_data, "__dict__", {}) or {})


def _query_label(query: dict[str, Any]) -> str:
    parts = [f"{k}={v}" for k, v in sorted(query.items()) if k != "sort"]
    parts.append(f"sort={query.get('sort', 'downloads')}")
    return ",".join(parts)
