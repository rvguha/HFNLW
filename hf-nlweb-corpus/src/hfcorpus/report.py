"""Quality reporting (design doc s10.1, s10.2).

Three layers get measured: corpus composition, record fidelity, and -- in
evaluate.py -- end-to-end answer quality. This module covers the first two and
writes the stratified review sample a human actually reads.
"""

from __future__ import annotations

import csv
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# The Hub's published pipeline-tag vocabulary, as of the pinned client. Held
# here rather than fetched so a quality report is reproducible offline and a
# coverage number does not silently move when the Hub adds a tag.
HUB_PIPELINE_TAGS = (
    "any-to-any", "audio-classification", "audio-text-to-text", "audio-to-audio",
    "automatic-speech-recognition", "depth-estimation", "document-question-answering",
    "feature-extraction", "fill-mask", "graph-ml", "image-classification",
    "image-feature-extraction", "image-segmentation", "image-text-to-text", "image-to-3d",
    "image-to-image", "image-to-text", "image-to-video", "keypoint-detection",
    "mask-generation", "object-detection", "question-answering", "reinforcement-learning",
    "robotics", "sentence-similarity", "summarization", "table-question-answering",
    "tabular-classification", "tabular-regression", "text-classification",
    "text-generation", "text-ranking", "text-to-3d", "text-to-audio", "text-to-image",
    "text-to-speech", "text-to-video", "text2text-generation", "time-series-forecasting",
    "token-classification", "translation", "unconditional-image-generation",
    "video-classification", "video-text-to-text", "video-to-video",
    "visual-document-retrieval", "visual-question-answering", "voice-activity-detection",
    "zero-shot-classification", "zero-shot-image-classification",
    "zero-shot-object-detection", "image-text-to-video",
)
HUB_PIPELINE_TAG_COUNT = len(HUB_PIPELINE_TAGS)

PARAMETER_BANDS = [
    (0, 150_000_000, "<150M"),
    (150_000_000, 1_000_000_000, "150M-1B"),
    (1_000_000_000, 4_000_000_000, "1B-4B"),
    (4_000_000_000, 15_000_000_000, "4B-15B"),
    (15_000_000_000, 80_000_000_000, "15B-80B"),
    (80_000_000_000, float("inf"), ">80B"),
]


EVIDENCE_PROPERTIES = {
    "trained_on": "hf:trainedOn",
    "fine_tuned_on": "hf:fineTunedOn",
    "evaluated_on": "hf:evaluatedOn",
    "intended_for": "hf:intendedFor",
    "architecture": "hf:architectureFor",
}


def _props(item: dict[str, Any]) -> dict[str, Any]:
    """Facts are direct properties now; this keeps the call sites unchanged."""
    return item


def _band(parameters: Any) -> str:
    if not isinstance(parameters, int):
        return "unknown"
    for low, high, label in PARAMETER_BANDS:
        if low <= parameters < high:
            return label
    return "unknown"


def _age_bucket(date_modified: str | None, now: datetime) -> str:
    if not date_modified:
        return "unknown"
    try:
        when = datetime.fromisoformat(str(date_modified).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    for months, label in ((3, "<=3 months"), (6, "<=6 months"), (12, "<=12 months")):
        if when >= now - timedelta(days=30.4 * months):
            return label
    return ">12 months"


def build(items: list[dict[str, Any]], internals: dict[str, dict[str, Any]],
          decisions: list[dict[str, Any]], families: dict[str, Any],
          enrichment_provenance: list[dict[str, Any]],
          validation: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    total = len(items)
    safe = max(1, total)

    tasks, libraries, languages, publishers, licenses = (Counter() for _ in range(5))
    bands, ages, subjects, scopes, relations, family_counter = (Counter() for _ in range(6))
    claim_count = 0
    evidence_types = Counter()
    coverage = Counter()

    for item in items:
        props = _props(item)
        tasks[props.get("huggingface:pipelineTag") or "unknown"] += 1
        libraries[props.get("huggingface:library") or "unknown"] += 1
        publishers[(item.get("creator") or {}).get("name", "unknown")] += 1
        licenses[props.get("corpus:licenseStatus") or "unknown"] += 1
        bands[_band(props.get("ml:parameterCount"))] += 1
        ages[_age_bucket(item.get("dateModified"), now)] += 1
        scopes[props.get("corpus:discoveryScope") or "unknown"] += 1
        relations[props.get("corpus:familyRelation") or "unknown"] += 1
        family_counter[props.get("corpus:familyId") or "unknown"] += 1
        for language in item.get("inLanguage", []) or ["unspecified"]:
            languages[str(language).lower()] += 1
        for term in item.get("hf:subject", []) or ["unknown"]:
            subjects[str(term)] += 1

        if item.get("license"):
            coverage["license_declared"] += 1
        if item.get("hf:baseModel") or item.get("hf:trainedOnDataset"):
            coverage["lineage_or_dataset"] += 1
        if item.get("hf:trainedOnDataset"):
            coverage["training_dataset"] += 1
        if item.get("hf:limitation"):
            coverage["limitations"] += 1
        if item.get("hf:capability"):
            coverage["capabilities"] += 1
        if item.get("hf:subject"):
            coverage["subject_areas"] += 1
        internal = internals.get(item["@id"], {})
        if internal.get("card_status") == "ok":
            coverage["usable_card"] += 1
        if internal.get("description_source") == "model_card_prose" or item.get("hf:capability"):
            coverage["useful_description"] += 1

        # Confidence is no longer published; evidence is counted by the relation
        # it supports, which is the distinction that actually carries meaning.
        for relation, key in EVIDENCE_PROPERTIES.items():
            quotes = item.get(key) or []
            claim_count += len(quotes)
            evidence_types[relation] += len(quotes)

    reasons = Counter(d["reason_code"] for d in decisions)
    strata = Counter(d["stratum"] for d in decisions if d["decision"] == "included")

    dropped = Counter()
    for internal in internals.values():
        for entry in internal.get("dropped_claims", []):
            dropped[entry.get("reason", "unknown")] += 1

    input_tokens = sum(p.get("input_tokens", 0) for p in enrichment_provenance)
    output_tokens = sum(p.get("output_tokens", 0) for p in enrichment_provenance)
    cost = round(sum(p.get("estimated_cost_usd", 0.0) for p in enrichment_provenance), 4)
    latencies = sorted(p.get("latency_seconds", 0.0) for p in enrichment_provenance)

    top_publisher = publishers.most_common(1)[0] if publishers else ("none", 0)
    top_family = family_counter.most_common(1)[0] if family_counter else ("none", 0)

    return {
        "snapshot": snapshot,
        "generated_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "composition": {
            "records": total,
            "by_stratum": dict(strata.most_common()),
            "by_task": dict(tasks.most_common()),
            "by_library": dict(libraries.most_common(20)),
            "by_language": dict(languages.most_common(30)),
            "by_publisher_top20": dict(publishers.most_common(20)),
            "by_license_status": dict(licenses.most_common()),
            "by_parameter_band": dict(bands.most_common()),
            "by_age": dict(ages.most_common()),
            "by_subject_area": dict(subjects.most_common(40)),
            "distinct_publishers": len(publishers),
            "distinct_families": len(family_counter),
        },
        "hub_coverage": {
            # Breadth against the Hub's own vocabularies, so "representative"
            # is a measurement rather than an assertion. The pipeline-tag
            # denominator is fixed at the Hub's published count; languages are
            # counted as distinct ISO codes actually present.
            "pipeline_tags_covered": len([t for t in tasks if t != "unknown"]),
            "pipeline_tags_available": HUB_PIPELINE_TAG_COUNT,
            "pipeline_tag_fraction": round(
                len([t for t in tasks if t != "unknown"]) / HUB_PIPELINE_TAG_COUNT, 3),
            "language_codes_covered": len([lang for lang in languages if len(lang) <= 3]),
            "subject_terms_covered": len(subjects),
            "uncovered_pipeline_tags": sorted(set(HUB_PIPELINE_TAGS) - set(tasks)),
        },
        "concentration": {
            "top_publisher": top_publisher[0],
            "top_publisher_fraction": round(top_publisher[1] / safe, 4),
            "top_family": top_family[0],
            "top_family_fraction": round(top_family[1] / safe, 4),
        },
        "coverage": {key: round(value / safe, 4) for key, value in coverage.items()},
        "duplication": {
            "by_discovery_scope": dict(scopes.most_common()),
            "by_family_relation": dict(relations.most_common()),
            "families_with_multiple_records": sum(1 for c in family_counter.values() if c > 1),
            "family_collapse_rate": round(1 - len(family_counter) / safe, 4),
            "candidate_duplicates_flagged": sum(
                1 for f in families.values() if f.get("candidate_duplicates")),
        },
        "selection": {
            "candidates_considered": len({d["repo_id"] for d in decisions}),
            "by_reason_code": dict(reasons.most_common()),
        },
        "enrichment": {
            "records_enriched": len(enrichment_provenance),
            "claims": claim_count,
            "claims_per_record": round(claim_count / safe, 2),
            "by_evidence_type": dict(evidence_types.most_common()),
            "dropped_claims_by_reason": dict(dropped.most_common()),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "median_latency_seconds": latencies[len(latencies) // 2] if latencies else 0.0,
        },
        "validation": validation,
    }


REVIEW_COLUMNS = [
    "repo_id", "url", "stratum", "family_id", "family_relation", "discovery_scope",
    "task", "parameters", "languages", "license_status", "subject_areas",
    "top_evidence_type", "evidence_quote", "description",
    "review_factual", "review_evidence_sufficient", "review_omissions",
    "review_family_correct", "review_useful",
]


def review_sample(items: list[dict[str, Any]], decisions: list[dict[str, Any]],
                  size: int, path: Path) -> int:
    """Stratified review sample, oversampling exactly where errors hide (s10.2):
    niche domains, multilingual models, records asserting subjects with no
    supporting quote, and families with more than one record."""
    stratum_by_id = {d["repo_id"]: d["stratum"] for d in decisions if d["decision"] == "included"}
    family_sizes = Counter(i.get("hf:family") for i in items)

    def priority(item: dict[str, Any]) -> tuple:
        # Confidence is no longer published, so "thin evidence" replaces
        # "low confidence" as the signal that a record deserves a human look:
        # a record asserting subjects with no supporting quote at all.
        thin = bool(item.get("hf:subject")) and not any(
            item.get(key) for key in EVIDENCE_PROPERTIES.values())
        niche = int(item.get("hf:downloads") or 0) < 10_000
        multilingual = len(item.get("inLanguage", [])) > 1
        crowded_family = family_sizes.get(item.get("hf:family"), 0) > 1
        # Sort descending on interest; repo id keeps it deterministic.
        return (-(thin + niche + multilingual + crowded_family), item["@id"])

    by_stratum: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        repo_id = item.get("hf:repository", "")
        by_stratum.setdefault(stratum_by_id.get(repo_id, "unknown"), []).append(item)

    chosen: list[dict[str, Any]] = []
    per_stratum = max(1, size // max(1, len(by_stratum)))
    for pool in by_stratum.values():
        chosen += sorted(pool, key=priority)[:per_stratum]
    if len(chosen) < size:
        rest = sorted((i for i in items if i not in chosen), key=priority)
        chosen += rest[: size - len(chosen)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for item in chosen[:size]:
            repo_id = item.get("hf:repository", "")
            evidence = next(((relation, item[key][0])
                             for relation, key in EVIDENCE_PROPERTIES.items()
                             if item.get(key)), ("", ""))
            writer.writerow({
                "repo_id": repo_id,
                "url": item["@id"],
                "stratum": stratum_by_id.get(repo_id, "unknown"),
                "family_id": item.get("hf:family", ""),
                "family_relation": item.get("hf:familyRelation", ""),
                "discovery_scope": item.get("hf:discoveryScope", ""),
                "task": item.get("hf:task", ""),
                "parameters": item.get("hf:parameters", ""),
                "languages": ";".join(item.get("inLanguage", [])),
                "license_status": item.get("hf:licenseStatus", ""),
                "subject_areas": ";".join(item.get("hf:subject", [])),
                "top_evidence_type": evidence[0],
                "evidence_quote": evidence[1],
                "description": item.get("description", "")[:400],
                "review_factual": "",
                "review_evidence_sufficient": "",
                "review_omissions": "",
                "review_family_correct": "",
                "review_useful": "",
            })
    return min(len(chosen), size)
